# SPDX-License-Identifier: Apache-2.0
"""
CPU validation for the model-level JVP forward (``jvp_model.py``).

A full ``HYVideoDiffusionTransformer`` can't be built on CPU (flash-attn / loguru),
so we assemble a small mock model with the exact attribute surface
``model_forward_jvp`` reads: mock patchify (Conv3d), sinusoidal time-embed,
vector-embed, ``linear`` text projection, the already-validated mock DiT blocks, a
mock final layer, and an ``unpatchify`` matching the real einsum. We then compare
``model_forward_jvp`` against a **monolithic** ``torch.func.jvp`` over the same math
(naive attention, text held CONSTANT — matching the design decision).

This validates the model-level threading: time-embed JVP, patchify JVP, the block
loop, txt cropping + zero-tangent, the img/txt concat-split, and the final+unpatchify
JVP. Exact equivalence to the *real* submodules is the GPU-server check (compare the
primal against ``model.forward``).
"""

import math

import torch
import torch.nn as nn

from ..modulate_layers import modulate
from ..posemb_layers import get_nd_rotary_pos_embed
from .jvp_attention import naive_attention_withT
from .jvp_model import model_forward_jvp
from .test_jvp_blocks import MockDoubleBlock, MockSingleBlock


def _timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half).to(t.device)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class _MockTimeIn(nn.Module):
    def __init__(self, hidden, freq=32):
        super().__init__()
        self.freq = freq
        self.mlp = nn.Sequential(nn.Linear(freq, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, t):
        return self.mlp(_timestep_embedding(t, self.freq).type(self.mlp[0].weight.dtype))


class _MockPatchEmbed(nn.Module):
    """Conv3d patchify matching PatchEmbed: B,C,T,H,W -> B,N,hidden."""

    def __init__(self, in_ch, hidden, patch):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, hidden, kernel_size=patch, stride=patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class _MockFinal(nn.Module):
    def __init__(self, hidden, patch_size, out_ch):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.linear = nn.Linear(hidden, patch_size[0] * patch_size[1] * patch_size[2] * out_ch)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))

    def forward(self, x, c):
        shift, scale = self.adaLN(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift=shift, scale=scale))


class MockModel(nn.Module):
    def __init__(self, hidden=32, heads=4, in_ch=8, out_ch=4, patch=(1, 2, 2),
                 n_double=2, n_single=2, text_dim=16, text2_dim=12):
        super().__init__()
        self.patch_size = patch
        self.patch_sizes = None
        self.use_context_block = False
        self.use_second_branch = False
        self.i2v_condition_type = "latent_concat"
        self.guidance_embed = False
        self.text_projection = "linear"
        self.use_attention_mask = True
        self.unpatchify_channels = out_ch
        self.heads_num = heads

        self.time_in = _MockTimeIn(hidden)
        self.vector_in = nn.Sequential(nn.Linear(text2_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.img_in = _MockPatchEmbed(in_ch, hidden, patch)
        self.txt_in = nn.Linear(text_dim, hidden)
        self.double_blocks = nn.ModuleList([MockDoubleBlock(hidden, heads) for _ in range(n_double)])
        self.single_blocks = nn.ModuleList([MockSingleBlock(hidden, heads) for _ in range(n_single)])
        self.final_layer = _MockFinal(hidden, patch, out_ch)

    def unpatchify(self, x, t, h, w):
        c = self.unpatchify_channels
        pt, ph, pw = self.patch_size
        x = x.reshape(x.shape[0], t, h, w, c, pt, ph, pw)
        x = torch.einsum("nthwcopq->nctohpwq", x)
        return x.reshape(x.shape[0], c, t * pt, h * ph, w * pw)


def test_model_forward_jvp(dtype=torch.float64):
    torch.manual_seed(0)
    hidden, heads, in_ch, out_ch = 32, 4, 8, 4
    patch = (1, 2, 2)
    mock = MockModel(hidden, heads, in_ch, out_ch, patch).to(dtype)

    B, T, H, W = 1, 2, 4, 4
    x = torch.randn(B, in_ch, T, H, W, dtype=dtype)
    t = torch.rand(B, dtype=dtype) * 1000
    t_x = torch.randn_like(x); t_x[:, out_ch:] = 0.0   # zero tangent on conditioning channels
    t_t = torch.ones_like(t)

    text_dim, L_txt = 16, 6
    text_states = torch.randn(B, L_txt, text_dim, dtype=dtype)
    text_mask = torch.zeros(B, L_txt); text_mask[:, :4] = 1.0   # true length 4
    text_states_2 = torch.randn(B, 12, dtype=dtype)

    head_dim = hidden // heads
    a = head_dim // 4
    cos, sin = get_nd_rotary_pos_embed([a, a, head_dim - 2 * a], (T, H // patch[1], W // patch[2]), use_real=True)
    cos, sin = cos.to(dtype), sin.to(dtype)
    freqs = (cos, sin)

    text_len = int(text_mask[0].sum().item())
    tt, th, tw = T // patch[0], H // patch[1], W // patch[2]

    # ---- reference: monolithic jvp w.r.t (x, t), text held constant ----
    txt_c = mock.txt_in(text_states)[:, :text_len]

    def ref(x, t):
        vec = mock.time_in(t) + mock.vector_in(text_states_2)
        img = mock.img_in(x)
        img_seq = img.shape[1]
        im, tx = img, txt_c
        for blk in mock.double_blocks:
            im, tx = blk.ref_forward(im, tx, vec, freqs)
        xx = torch.cat((im, tx), 1)
        for blk in mock.single_blocks:
            xx = blk.ref_forward(xx, vec, text_len, freqs)
        img_f = xx[:, :img_seq]
        return mock.unpatchify(mock.final_layer(img_f, vec), tt, th, tw)

    ref_out, ref_tout = torch.func.jvp(ref, (x, t), (t_x, t_t))

    # ---- under test ----
    out, tout = model_forward_jvp(
        mock, (x, t_x), (t, t_t),
        text_states=text_states, text_mask=text_mask, text_states_2=text_states_2,
        freqs_cos=cos, freqs_sin=sin, attn_op=naive_attention_withT,
    )

    torch.testing.assert_close(out, ref_out)
    torch.testing.assert_close(tout, ref_tout)

    # primal must stay differentiable to params; tangent must be detached
    assert out.requires_grad and out.grad_fn is not None, "primal lost its autograd graph"
    assert not tout.requires_grad, "tangent should be detached"
    loss = (out - tout.detach()) ** 2
    loss.mean().backward()
    g = mock.final_layer.linear.weight.grad
    assert g is not None and torch.isfinite(g).all(), "no/!finite grad to params"
    print("test_model_forward_jvp: PASSED")


if __name__ == "__main__":
    test_model_forward_jvp()
    print("MODEL-LEVEL JVP CPU TEST PASSED")
