# SPDX-License-Identifier: Apache-2.0
"""
CPU validation for the block-level JVP plumbing (``jvp_blocks.py``).

The real ``MMDoubleStreamBlock`` / ``MMSingleStreamBlock`` can't be imported on a
CPU-only box (``models.py`` pulls flash-attn / loguru), so we instead build *mock*
blocks that expose the exact submodule attribute names the JVP functions read, and
check the f1→attention→f2 split against a **monolithic** ``torch.func.jvp`` of the
same math (with naive attention). This pins down the error-prone parts — the
reshape/cat/split indexing and the attention boundary — independent of the kernel.

The value output is also checked to equal a plain (no-tangent) monolithic forward,
i.e. baseline parity of the primal.

Run:  python -m voyager.modules.jvp.test_jvp_blocks      (or via the standalone loader)
"""

import torch
import torch.nn as nn
from einops import rearrange

from ..modulate_layers import modulate, apply_gate
from ..posemb_layers import apply_rotary_emb, get_nd_rotary_pos_embed
from .jvp_attention import naive_attention_withT, _naive_sdpa_bshd
from .jvp_blocks import double_stream_block_jvp, single_stream_block_jvp


# --------------------------------------------------------------------------- #
# Mock blocks: same attribute surface as the real blocks, small + CPU-friendly. #
# --------------------------------------------------------------------------- #
class MockDoubleBlock(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.heads_num = heads
        mlp = hidden
        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))
        self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))
        self.img_norm1 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.img_norm2 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.txt_norm1 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.txt_norm2 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.img_attn_qkv = nn.Linear(hidden, 3 * hidden)
        self.txt_attn_qkv = nn.Linear(hidden, 3 * hidden)
        self.img_attn_q_norm = nn.Identity()
        self.img_attn_k_norm = nn.Identity()
        self.txt_attn_q_norm = nn.Identity()
        self.txt_attn_k_norm = nn.Identity()
        self.img_attn_proj = nn.Linear(hidden, hidden)
        self.txt_attn_proj = nn.Linear(hidden, hidden)
        self.img_mlp = nn.Sequential(nn.Linear(hidden, mlp), nn.GELU(), nn.Linear(mlp, hidden))
        self.txt_mlp = nn.Sequential(nn.Linear(hidden, mlp), nn.GELU(), nn.Linear(mlp, hidden))

    # monolithic reference (value path), naive attention
    def ref_forward(self, img, txt, vec, freqs_cis):
        H = self.heads_num
        img_len = img.shape[1]
        (s1, c1, g1, s2, c2, g2) = self.img_mod(vec).chunk(6, dim=-1)
        (ts1, tc1, tg1, ts2, tc2, tg2) = self.txt_mod(vec).chunk(6, dim=-1)
        im = modulate(self.img_norm1(img), shift=s1, scale=c1)
        iq, ik, iv = rearrange(self.img_attn_qkv(im), "B L (K H D) -> K B L H D", K=3, H=H)
        if freqs_cis is not None:
            iq, ik = apply_rotary_emb(iq, ik, freqs_cis, head_first=False)
        tm = modulate(self.txt_norm1(txt), shift=ts1, scale=tc1)
        tq, tk, tv = rearrange(self.txt_attn_qkv(tm), "B L (K H D) -> K B L H D", K=3, H=H)
        q = torch.cat((iq, tq), 1); k = torch.cat((ik, tk), 1); v = torch.cat((iv, tv), 1)
        attn = _naive_sdpa_bshd(q, k, v)
        B, S, _, _ = attn.shape
        attn = attn.reshape(B, S, -1)
        ia, ta = attn[:, :img_len], attn[:, img_len:]
        img_o = img + apply_gate(self.img_attn_proj(ia), gate=g1)
        img_o = img_o + apply_gate(self.img_mlp(modulate(self.img_norm2(img_o), shift=s2, scale=c2)), gate=g2)
        txt_o = txt + apply_gate(self.txt_attn_proj(ta), gate=tg1)
        txt_o = txt_o + apply_gate(self.txt_mlp(modulate(self.txt_norm2(txt_o), shift=ts2, scale=tc2)), gate=tg2)
        return img_o, txt_o


class MockSingleBlock(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.heads_num = heads
        self.hidden_size = hidden
        self.mlp_hidden_dim = hidden
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 3 * hidden))
        self.pre_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.linear1 = nn.Linear(hidden, 3 * hidden + self.mlp_hidden_dim)
        self.linear2 = nn.Linear(hidden + self.mlp_hidden_dim, hidden)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.mlp_act = nn.GELU()

    def ref_forward(self, x, vec, txt_len, freqs_cis):
        H = self.heads_num
        s, c, g = self.modulation(vec).chunk(3, dim=-1)
        x_mod = modulate(self.pre_norm(x), shift=s, scale=c)
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=H)
        if freqs_cis is not None:
            iq, txtq = q[:, :-txt_len], q[:, -txt_len:]
            ik, txtk = k[:, :-txt_len], k[:, -txt_len:]
            iq, ik = apply_rotary_emb(iq, ik, freqs_cis, head_first=False)
            q = torch.cat((iq, txtq), 1); k = torch.cat((ik, txtk), 1)
        attn = _naive_sdpa_bshd(q, k, v)
        B, S, _, _ = attn.shape
        attn = attn.reshape(B, S, -1)
        out = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + apply_gate(out, gate=g)


def _make_freqs(img_tokens, head_dim, grid):
    # split head_dim into 3 rope parts summing to head_dim (t, h, w)
    a = head_dim // 4
    rope_dim_list = [a, a, head_dim - 2 * a]
    cos, sin = get_nd_rotary_pos_embed(rope_dim_list, grid, use_real=True)
    return (cos, sin)


def test_double_block_jvp(dtype=torch.float64):
    torch.manual_seed(0)
    hidden, heads = 32, 4
    B, img_tokens, txt_tokens = 1, 24, 6  # img grid 2*3*4=24
    blk = MockDoubleBlock(hidden, heads).to(dtype)
    head_dim = hidden // heads
    freqs = _make_freqs(img_tokens, head_dim, (2, 3, 4))
    freqs = (freqs[0].to(dtype), freqs[1].to(dtype))

    img = torch.randn(B, img_tokens, hidden, dtype=dtype)
    txt = torch.randn(B, txt_tokens, hidden, dtype=dtype)
    vec = torch.randn(B, hidden, dtype=dtype)
    t_img = torch.randn_like(img); t_txt = torch.randn_like(txt); t_vec = torch.randn_like(vec)

    # reference: monolithic jvp + plain value
    def ref(img, txt, vec):
        return blk.ref_forward(img, txt, vec, freqs)
    (ref_io, ref_to), (ref_tio, ref_tto) = torch.func.jvp(ref, (img, txt, vec), (t_img, t_txt, t_vec))
    base_io, base_to = blk.ref_forward(img, txt, vec, freqs)

    # under test
    (io, tio), (to, tto) = double_stream_block_jvp(
        blk, (img, t_img), (txt, t_txt), (vec, t_vec), freqs_cis=freqs,
        attn_op=naive_attention_withT,
    )

    torch.testing.assert_close(io, base_io)   # primal == plain forward
    torch.testing.assert_close(to, base_to)
    torch.testing.assert_close(io, ref_io)
    torch.testing.assert_close(to, ref_to)
    torch.testing.assert_close(tio, ref_tio)  # tangent == reference jvp
    torch.testing.assert_close(tto, ref_tto)
    print("test_double_block_jvp: PASSED")


def test_single_block_jvp(dtype=torch.float64):
    torch.manual_seed(1)
    hidden, heads = 32, 4
    B, img_tokens, txt_tokens = 1, 24, 6
    blk = MockSingleBlock(hidden, heads).to(dtype)
    head_dim = hidden // heads
    freqs = _make_freqs(img_tokens, head_dim, (2, 3, 4))
    freqs = (freqs[0].to(dtype), freqs[1].to(dtype))

    x = torch.randn(B, img_tokens + txt_tokens, hidden, dtype=dtype)
    vec = torch.randn(B, hidden, dtype=dtype)
    t_x = torch.randn_like(x); t_vec = torch.randn_like(vec)

    def ref(x, vec):
        return blk.ref_forward(x, vec, txt_tokens, freqs)
    ref_o, ref_to = torch.func.jvp(ref, (x, vec), (t_x, t_vec))
    base_o = blk.ref_forward(x, vec, txt_tokens, freqs)

    o, to = single_stream_block_jvp(
        blk, (x, t_x), (vec, t_vec), txt_len=txt_tokens, freqs_cis=freqs,
        attn_op=naive_attention_withT,
    )
    torch.testing.assert_close(o, base_o)
    torch.testing.assert_close(o, ref_o)
    torch.testing.assert_close(to, ref_to)
    print("test_single_block_jvp: PASSED")


if __name__ == "__main__":
    test_double_block_jvp()
    test_single_block_jvp()
    print("ALL BLOCK-JVP CPU TESTS PASSED")
