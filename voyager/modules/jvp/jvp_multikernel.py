# SPDX-License-Identifier: Apache-2.0
"""
Multi-kernel (MultiPatchEmbed / MultiFinalLayer) JVP front- and back-end.

The dual-branch architecture is built on top of the multi-kernel patchify: the
front-end ``MultiPatchEmbed`` emits a token sequence plus a per-token
``patch_indices`` label (which kernel / patch-size produced each token), and the
back-end ``MultiFinalLayer`` + ``unpatchify_multi`` invert it. The keyframe tokens
(label 0) feed the first branch; the non-keyframe tokens feed the second branch.

This module threads a forward-mode tangent ``(value, tangent)`` through those four
ops so MeanFlow's ``dudt`` can be computed on the dual-branch model. It is the
shared foundation for BOTH the Option-B (frozen second branch) and Option-A (full
JVP) variants; nothing here is mode-specific.

Two kinds of op, two strategies
--------------------------------
* **Affine ops** — ``MultiPatchEmbed`` (Conv3d + bias + optional norm) and
  ``MultiFinalLayer`` (AdaLN modulation + biased linears). The tangent of an affine
  map drops the bias, so we use ``torch.func.jvp`` (which does that correctly).
  ``MultiPatchEmbed`` additionally returns an integer ``patch_indices`` that has no
  tangent; we differentiate a float-only wrapper and recover ``patch_indices`` from
  a no-grad call (it depends only on ``indices`` + token layout, not on ``x``).
* **Pure-linear ops** — the keyframe/non-keyframe token split, ``merge_back``
  (scatter) and ``unpatchify_multi`` (scatter + normalize by a constant weight map).
  These have no bias, so ``op(value+tangent) = op(value) + op(tangent)`` exactly;
  we simply apply the op to the value and the tangent separately. This is exact AND
  avoids running ``torch.func.jvp`` over their in-place ``index_put`` / accumulation
  logic (which functorch does not always support).

Policy (matches ``jvp_model.py``): primal outputs keep their autograd graph to the
model parameters; all returned tangents are detached (truncated / first-order
forward-mode propagation).
"""

from typing import List, Optional, Sequence, Tuple

import torch

from .jvp_attention import TensorWithT


# --------------------------------------------------------------------------- #
# Front-end: patchify + keyframe/non-keyframe split
# --------------------------------------------------------------------------- #
def multikernel_patchify_jvp(
    img_in,
    x: torch.Tensor,
    t_x: torch.Tensor,
    indices,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """JVP of ``MultiPatchEmbed.forward(x, indices)``.

    Args:
        img_in: the model's ``MultiPatchEmbed`` instance.
        x: latent ``[B, C, T, H, W]`` (primal); carries tangent ``t_x``.
        indices: per-kernel frame-index lists (static; not differentiated).

    Returns ``(img, t_img, patch_indices)`` where ``img`` is ``[B, N, D]`` (keeps
    grad to params), ``t_img`` is the detached tangent, and ``patch_indices`` is the
    integer ``[B, N]`` kernel label (no tangent).
    """
    def f(x_):
        out, _ = img_in(x_, indices)
        return out

    img, t_img = torch.func.jvp(f, (x,), (t_x,))

    # patch_indices is x-value-independent (it is fixed by `indices` + the token
    # layout), so a no-grad forward recovers it without polluting the JVP wrapper
    # with an integer output. The extra conv pass is front-end-only (paid once, not
    # per block) — optimize later if it ever shows up in a profile.
    with torch.no_grad():
        _, patch_indices = img_in(x, indices)

    return img, t_img.detach(), patch_indices


def split_branches_jvp(
    img: torch.Tensor,
    t_img: torch.Tensor,
    freqs_cos: Optional[torch.Tensor],
    freqs_sin: Optional[torch.Tensor],
    patch_indices: torch.Tensor,
) -> Tuple[
    Tuple[TensorWithT, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]],
    Tuple[TensorWithT, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]],
]:
    """Split patchified tokens into keyframe (branch 1) and non-keyframe (branch 2).

    Mirrors ``models.py`` (the ``use_second_branch`` block): branch 1 = tokens with
    ``patch_indices == 0``; branch 2 = the rest. RoPE freqs are sliced by the same
    per-position mask. The split is a pure gather (linear), so the tangent is gathered
    with the identical mask.

    Returns ``(branch1, branch2)`` where each is ``((img, t_img), (freqs_cos, freqs_sin))``.
    """
    B, N, D = img.shape

    # --- branch 2: non-keyframe tokens ---
    nonkey_mask = (patch_indices != 0)                 # [B, N]
    K2 = int(nonkey_mask.sum(dim=1)[0].item())
    assert torch.all(nonkey_mask.sum(dim=1) == K2), \
        "non-keyframe token count differs across batch"
    img2 = img[nonkey_mask].reshape(B, K2, D)
    t_img2 = t_img[nonkey_mask].reshape(B, K2, D)
    fc2 = freqs_cos[nonkey_mask[0].cpu()] if freqs_cos is not None else None
    fs2 = freqs_sin[nonkey_mask[0].cpu()] if freqs_sin is not None else None

    # --- branch 1: keyframe tokens ---
    key_mask = (patch_indices == 0)                    # [B, N]
    K1 = int(key_mask.sum(dim=1)[0].item())
    assert torch.all(key_mask.sum(dim=1) == K1), \
        "keyframe token count differs across batch"
    img1 = img[key_mask].reshape(B, K1, D)
    t_img1 = t_img[key_mask].reshape(B, K1, D)
    fc1 = freqs_cos[key_mask[0].cpu()] if freqs_cos is not None else None
    fs1 = freqs_sin[key_mask[0].cpu()] if freqs_sin is not None else None

    return ((img1, t_img1.detach()), (fc1, fs1)), ((img2, t_img2.detach()), (fc2, fs2))


# --------------------------------------------------------------------------- #
# Back-end: merge_back + final layer + unpatchify
# --------------------------------------------------------------------------- #
def merge_back_jvp(
    model,
    key_withT: TensorWithT,
    nonkey_withT: TensorWithT,
    patch_indices: torch.Tensor,
) -> TensorWithT:
    """JVP of ``HYVideoDiffusionTransformer.merge_back`` (scatter; pure linear).

    Re-interleaves the two branches' tokens into one ``[B, N, D]`` sequence by the
    ``patch_indices`` labels. Linear ⇒ tangent = merge_back(tangents).
    """
    img_key, t_key = key_withT
    img_nonkey, t_nonkey = nonkey_withT
    merged = model.merge_back(img_key, img_nonkey, patch_indices)          # keeps grad
    t_merged = model.merge_back(t_key, t_nonkey, patch_indices)            # exact tangent
    return merged, t_merged.detach()


def final_layer_multi_jvp(
    final_layer,
    img_withT: TensorWithT,
    vec_withT: TensorWithT,
    indices,
    patch_indices: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """JVP of ``MultiFinalLayer.forward(x, c, indices, patch_indices)``.

    Affine (AdaLN modulation + biased per-kernel linears) and returns a *list* of
    per-kernel tensors, so we use ``torch.func.jvp`` over ``(x, c)``; the list output
    structure is preserved in the tangents.
    """
    img, t_img = img_withT
    vec, t_vec = vec_withT

    def f(x_, c_):
        return final_layer(x_, c_, indices=indices, patch_indices=patch_indices)

    outs, t_outs = torch.func.jvp(f, (img, vec), (t_img, t_vec))
    outs = list(outs)
    t_outs = [t.detach() for t in t_outs]
    return outs, t_outs


def unpatchify_multi_jvp(
    model,
    xs_withT: Sequence[TensorWithT],
    T: int,
    H: int,
    W: int,
    patch_sizes,
    indices,
) -> TensorWithT:
    """JVP of ``unpatchify_multi`` (scatter + normalize by a constant weight; linear).

    ``unpatchify_multi`` scatters each kernel's patches into the output volume and
    divides by a token-count weight map that depends only on the patch layout (not on
    ``xs`` values). That is linear in ``xs`` ⇒ tangent = unpatchify_multi(tangents).
    """
    xs = [v for (v, _) in xs_withT]
    t_xs = [t for (_, t) in xs_withT]
    out = model.unpatchify_multi(xs, T=T, H=H, W=W, patch_sizes=patch_sizes, indices=indices)
    t_out = model.unpatchify_multi(t_xs, T=T, H=H, W=W, patch_sizes=patch_sizes, indices=indices)
    return out, t_out.detach()
