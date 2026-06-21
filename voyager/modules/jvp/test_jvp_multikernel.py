# SPDX-License-Identifier: Apache-2.0
"""
CPU validation for the multi-kernel JVP front/back-end (``jvp_multikernel.py``).

Ground truth is **finite differences** (an independent check — using
``torch.func.jvp`` as the reference would be circular for the affine ops, which the
module itself implements with ``torch.func.jvp``). For the pure-linear ops we also
assert the exact split↔merge roundtrip.

Run on the server (env ``voyager``; importing the real modules pulls loguru /
attention, which are guarded but present there):

    python -m voyager.modules.jvp.test_jvp_multikernel

All tensors are float64 so finite differences agree to ~1e-8.
"""

import types

import torch
import torch.nn as nn

from voyager.modules.embed_layers import MultiPatchEmbed
from voyager.modules.mlp_layers import MultiFinalLayer
from voyager.modules.models import HYVideoDiffusionTransformer
from voyager.modules.jvp.jvp_multikernel import (
    multikernel_patchify_jvp,
    split_branches_jvp,
    merge_back_jvp,
    final_layer_multi_jvp,
    unpatchify_multi_jvp,
)

torch.manual_seed(0)

# ---- shared tiny config ----
B, C_IN, C_OUT = 1, 4, 3
T, H, W = 2, 4, 4
EMBED = 8
PATCH_SIZES = [(1, 2, 2), (1, 4, 4)]
INDICES = [[0], [1]]            # frame 0 -> kernel 0 (keyframe), frame 1 -> kernel 1
EPS = 1e-5


def _fd(g, inputs, tangents, eps=EPS):
    """Central finite-difference directional derivative of ``g`` along ``tangents``.

    ``g`` may return a tensor or a list/tuple of tensors.
    """
    plus = g(*[x + eps * t for x, t in zip(inputs, tangents)])
    minus = g(*[x - eps * t for x, t in zip(inputs, tangents)])
    if isinstance(plus, (list, tuple)):
        return [(p - m) / (2 * eps) for p, m in zip(plus, minus)]
    return (plus - minus) / (2 * eps)


def _close(a, b, atol=1e-6, rtol=1e-5, msg=""):
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    if not ok:
        diff = (a - b).abs().max().item()
        raise AssertionError(f"{msg}: max abs diff {diff:.3e}")


def _make_stub():
    """A minimal object carrying the real ``merge_back`` / ``unpatchify_multi`` methods."""
    stub = types.SimpleNamespace()
    stub.unpatchify_channels = C_OUT
    stub.merge_back = types.MethodType(HYVideoDiffusionTransformer.merge_back, stub)
    stub.unpatchify_multi = types.MethodType(HYVideoDiffusionTransformer.unpatchify_multi, stub)
    return stub


def test_patchify_jvp():
    img_in = MultiPatchEmbed(PATCH_SIZES, in_chans=C_IN, embed_dim=EMBED).double()
    x = torch.randn(B, C_IN, T, H, W, dtype=torch.float64, requires_grad=True)
    t_x = torch.randn_like(x)

    img, t_img, patch_indices = multikernel_patchify_jvp(img_in, x, t_x, INDICES)

    # primal matches plain forward; patch_indices matches plain forward
    ref_img, ref_pidx = img_in(x, INDICES)
    _close(img, ref_img, msg="patchify primal")
    assert torch.equal(patch_indices, ref_pidx), "patch_indices mismatch"

    # primal keeps grad to params
    img.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in img_in.parameters()), \
        "patchify primal not differentiable to conv params"

    # tangent matches finite differences
    fd = _fd(lambda x_: img_in(x_, INDICES)[0], (x.detach(),), (t_x,))
    _close(t_img, fd, msg="patchify tangent vs FD")
    print(f"[ok] patchify JVP  (N={img.shape[1]}, patch_indices={patch_indices[0].tolist()})")


def test_split_merge_roundtrip():
    img_in = MultiPatchEmbed(PATCH_SIZES, in_chans=C_IN, embed_dim=EMBED).double()
    x = torch.randn(B, C_IN, T, H, W, dtype=torch.float64)
    t_x = torch.randn_like(x)
    img, t_img, patch_indices = multikernel_patchify_jvp(img_in, x, t_x, INDICES)

    N = img.shape[1]
    freqs_cos = torch.randn(N, 4, dtype=torch.float64)
    freqs_sin = torch.randn(N, 4, dtype=torch.float64)

    (b1, (fc1, fs1)), (b2, (fc2, fs2)) = split_branches_jvp(
        img, t_img, freqs_cos, freqs_sin, patch_indices
    )

    # RoPE freqs sliced by the same per-position mask
    key_mask = (patch_indices[0] == 0).cpu()
    nonkey_mask = (patch_indices[0] != 0).cpu()
    _close(fc1, freqs_cos[key_mask], msg="freqs_cos branch1 slice")
    _close(fs2, freqs_sin[nonkey_mask], msg="freqs_sin branch2 slice")

    # split then merge reconstructs the original sequence (value AND tangent), exactly
    stub = _make_stub()
    merged, t_merged = merge_back_jvp(stub, b1, b2, patch_indices)
    _close(merged, img, atol=0, rtol=0, msg="merge(split(img)) value")
    _close(t_merged, t_img, atol=0, rtol=0, msg="merge(split(t_img)) tangent")
    print("[ok] split/merge roundtrip exact (value + tangent)")


def test_merge_back_jvp():
    stub = _make_stub()
    # synthesise two branches consistent with patch_indices = [0,0,0,0,1]
    patch_indices = torch.tensor([[0, 0, 0, 0, 1]])
    img_key = torch.randn(B, 4, EMBED, dtype=torch.float64)
    img_non = torch.randn(B, 1, EMBED, dtype=torch.float64)
    t_key = torch.randn_like(img_key)
    t_non = torch.randn_like(img_non)

    _, t_merged = merge_back_jvp(stub, (img_key, t_key), (img_non, t_non), patch_indices)
    fd = _fd(
        lambda a, b: stub.merge_back(a, b, patch_indices),
        (img_key, img_non), (t_key, t_non),
    )
    _close(t_merged, fd, msg="merge_back tangent vs FD")
    print("[ok] merge_back JVP vs FD")


def _build_final_layer():
    fl = MultiFinalLayer(
        hidden_size=EMBED, patch_sizes=PATCH_SIZES, out_channels=C_OUT,
        act_layer=nn.SiLU,
    ).double()
    # MultiFinalLayer zero-inits its linears/modulation -> trivially zero outputs.
    # Randomise so the affine JVP is actually exercised.
    with torch.no_grad():
        for p in fl.parameters():
            p.normal_(std=0.3)
    return fl


def test_final_layer_jvp():
    fl = _build_final_layer()
    img = torch.randn(B, 5, EMBED, dtype=torch.float64)
    vec = torch.randn(B, EMBED, dtype=torch.float64)
    t_img = torch.randn_like(img)
    t_vec = torch.randn_like(vec)
    patch_indices = torch.tensor([[0, 0, 0, 0, 1]])

    outs, t_outs = final_layer_multi_jvp(fl, (img, t_img), (vec, t_vec), INDICES, patch_indices)

    ref = fl(img, vec, indices=INDICES, patch_indices=patch_indices)
    for k, (o, r) in enumerate(zip(outs, ref)):
        _close(o, r, msg=f"final_layer primal[{k}]")

    fd = _fd(
        lambda x_, c_: fl(x_, c_, indices=INDICES, patch_indices=patch_indices),
        (img, vec), (t_img, t_vec),
    )
    for k, (t, f) in enumerate(zip(t_outs, fd)):
        _close(t, f, msg=f"final_layer tangent[{k}] vs FD")
    print(f"[ok] final_layer JVP vs FD  ({len(outs)} kernel outputs)")


def test_unpatchify_multi_jvp():
    stub = _make_stub()
    # token counts per kernel: k0 -> T_k=1,H_k=2,W_k=2 = 4 ; k1 -> 1,1,1 = 1
    x0 = torch.randn(B, 4, C_OUT * 1 * 2 * 2, dtype=torch.float64)
    x1 = torch.randn(B, 1, C_OUT * 1 * 4 * 4, dtype=torch.float64)
    t0 = torch.randn_like(x0)
    t1 = torch.randn_like(x1)

    vol, t_vol = unpatchify_multi_jvp(
        stub, [(x0, t0), (x1, t1)], T=T, H=H, W=W, patch_sizes=PATCH_SIZES, indices=INDICES
    )
    ref = stub.unpatchify_multi([x0, x1], T=T, H=H, W=W, patch_sizes=PATCH_SIZES, indices=INDICES)
    _close(vol, ref, msg="unpatchify primal")

    fd = _fd(
        lambda a, b: stub.unpatchify_multi([a, b], T=T, H=H, W=W, patch_sizes=PATCH_SIZES, indices=INDICES),
        (x0, x1), (t0, t1),
    )
    _close(t_vol, fd, msg="unpatchify tangent vs FD")
    print(f"[ok] unpatchify_multi JVP vs FD  (vol {tuple(vol.shape)})")


if __name__ == "__main__":
    test_patchify_jvp()
    test_split_merge_roundtrip()
    test_merge_back_jvp()
    test_final_layer_jvp()
    test_unpatchify_multi_jvp()
    print("\nAll multi-kernel JVP front/back-end checks passed.")
