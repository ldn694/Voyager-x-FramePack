# SPDX-License-Identifier: Apache-2.0
"""
Forward-mode (JVP) variants of the Hunyuan DiT blocks.

These are standalone functions that operate on the *existing* baseline block
instances (``MMDoubleStreamBlock`` / ``MMSingleStreamBlock``) by reading their
submodules — no subclassing, no new parameters, no edits to ``models.py``. The
baseline forward path is untouched; this is an additive parallel path used only
by the MeanFlow / rCM student.

Decomposition (the key idea)
----------------------------
Every op in a block is forward-mode differentiable via ``torch.func.jvp`` EXCEPT
FlashAttention. So each block forward is split into three stages::

    (q, k, v[, mlp]) = f1(img, txt, vec)          # cheap ops  -> torch.func.jvp
    attn             = attention_withT(q, k, v)    # the ONLY special op (kernel)
    out              = f2(attn, img, txt, vec)     # cheap ops  -> torch.func.jvp

``torch.func.jvp`` over the whole f1 / f2 segments handles all the product-rule
math (modulation, gating, RoPE) automatically; we only manually cross the
attention boundary. Tangents are detached at each block boundary (truncated /
first-order propagation — sCM and MeanFlow only need ``sg[dF/dt]``).

Only the standard conditioning path is implemented (``condition_type`` is NOT
``"token_replace"``). Voyager's RealEstate10K training defaults to
``i2v_condition_type="latent_concat"``, for which the blocks take exactly this
path. ``token_replace`` (per-first-frame modulation) is asserted out and left as
a follow-up.

``attn_op`` is injectable so tests can substitute the pure-PyTorch
``naive_attention_withT`` for the Triton kernel on CPU.
"""

from typing import Callable, Optional, Tuple

import torch
from einops import rearrange

from ..modulate_layers import modulate, apply_gate
from ..posemb_layers import apply_rotary_emb
from .jvp_attention import attention_withT, TensorWithT


def _detach_pair(p: TensorWithT) -> TensorWithT:
    v, t = p
    return v, t.detach()


def double_stream_block_jvp(
    block,
    img_withT: TensorWithT,
    txt_withT: TensorWithT,
    vec_withT: TensorWithT,
    freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    condition_type: Optional[str] = None,
    attn_op: Callable[[TensorWithT, TensorWithT, TensorWithT], TensorWithT] = attention_withT,
) -> Tuple[TensorWithT, TensorWithT]:
    """JVP of ``MMDoubleStreamBlock.forward`` (standard, non-token_replace path).

    Args/returns are ``(value, tangent)`` pairs. ``img``/``txt``/``vec`` each carry
    a tangent; returns updated ``(img, txt)`` pairs with detached tangents.
    """
    assert condition_type != "token_replace", "token_replace JVP path not implemented"
    img, t_img = img_withT
    txt, t_txt = txt_withT
    vec, t_vec = vec_withT
    H = block.heads_num
    img_len = img.shape[1]

    # ---- f1: cheap pre-attention ops -> q, k, v (post-RoPE, img+txt concatenated)
    def f1(img, txt, vec):
        (img_mod1_shift, img_mod1_scale, _, _, _, _) = block.img_mod(vec).chunk(6, dim=-1)
        (txt_mod1_shift, txt_mod1_scale, _, _, _, _) = block.txt_mod(vec).chunk(6, dim=-1)

        img_modulated = block.img_norm1(img)
        img_modulated = modulate(img_modulated, shift=img_mod1_shift, scale=img_mod1_scale)
        img_qkv = block.img_attn_qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B L H D", K=3, H=H)
        img_q = block.img_attn_q_norm(img_q).to(img_v)
        img_k = block.img_attn_k_norm(img_k).to(img_v)
        if freqs_cis is not None:
            img_q, img_k = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)

        txt_modulated = block.txt_norm1(txt)
        txt_modulated = modulate(txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale)
        txt_qkv = block.txt_attn_qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B L H D", K=3, H=H)
        txt_q = block.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = block.txt_attn_k_norm(txt_k).to(txt_v)

        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)
        v = torch.cat((img_v, txt_v), dim=1)
        return q, k, v

    (q, k, v), (tq, tk, tv) = torch.func.jvp(f1, (img, txt, vec), (t_img, t_txt, t_vec))

    # ---- attention (the only non-jvp op): [B,S,H,D] -> [B,S,H*D]
    attn, t_attn = attn_op((q, tq), (k, tk), (v, tv))
    B, S, _, _ = attn.shape
    attn = attn.reshape(B, S, -1)
    t_attn = t_attn.reshape(B, S, -1)

    # ---- f2: cheap post-attention ops (proj, gate, mlp, residuals)
    def f2(attn, img, txt, vec):
        (_, _, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate) = block.img_mod(vec).chunk(6, dim=-1)
        (_, _, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate) = block.txt_mod(vec).chunk(6, dim=-1)

        img_attn = attn[:, :img_len]
        txt_attn = attn[:, img_len:]

        img_out = img + apply_gate(block.img_attn_proj(img_attn), gate=img_mod1_gate)
        img_out = img_out + apply_gate(
            block.img_mlp(modulate(block.img_norm2(img_out), shift=img_mod2_shift, scale=img_mod2_scale)),
            gate=img_mod2_gate,
        )
        txt_out = txt + apply_gate(block.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt_out = txt_out + apply_gate(
            block.txt_mlp(modulate(block.txt_norm2(txt_out), shift=txt_mod2_shift, scale=txt_mod2_scale)),
            gate=txt_mod2_gate,
        )
        return img_out, txt_out

    (img_out, txt_out), (t_img_out, t_txt_out) = torch.func.jvp(
        f2, (attn, img, txt, vec), (t_attn, t_img, t_txt, t_vec)
    )
    return _detach_pair((img_out, t_img_out)), _detach_pair((txt_out, t_txt_out))


def single_stream_block_jvp(
    block,
    x_withT: TensorWithT,
    vec_withT: TensorWithT,
    txt_len: int,
    freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    condition_type: Optional[str] = None,
    attn_op: Callable[[TensorWithT, TensorWithT, TensorWithT], TensorWithT] = attention_withT,
) -> TensorWithT:
    """JVP of ``MMSingleStreamBlock.forward`` (standard path, no step_cfg debug).

    ``x`` is the concatenated ``[img, txt]`` stream. Returns updated ``(x, t_x)``
    with detached tangent.
    """
    assert condition_type != "token_replace", "token_replace JVP path not implemented"
    x, t_x = x_withT
    vec, t_vec = vec_withT
    H = block.heads_num

    # ---- f1: cheap pre-attention ops -> q, k, v, mlp
    def f1(x, vec):
        mod_shift, mod_scale, _ = block.modulation(vec).chunk(3, dim=-1)
        x_mod = modulate(block.pre_norm(x), shift=mod_shift, scale=mod_scale)
        qkv, mlp = torch.split(
            block.linear1(x_mod), [3 * block.hidden_size, block.mlp_hidden_dim], dim=-1
        )
        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=H)
        q = block.q_norm(q).to(v)
        k = block.k_norm(k).to(v)
        if freqs_cis is not None:
            if txt_len > 0:
                img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
                img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
            else:
                img_q, txt_q = q, None
                img_k, txt_k = k, None
            img_q, img_k = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            q = torch.cat((img_q, txt_q), dim=1) if txt_q is not None else img_q
            k = torch.cat((img_k, txt_k), dim=1) if txt_k is not None else img_k
        return q, k, v, mlp

    (q, k, v, mlp), (tq, tk, tv, t_mlp) = torch.func.jvp(f1, (x, vec), (t_x, t_vec))

    # ---- attention
    attn, t_attn = attn_op((q, tq), (k, tk), (v, tv))
    B, S, _, _ = attn.shape
    attn = attn.reshape(B, S, -1)
    t_attn = t_attn.reshape(B, S, -1)

    # ---- f2: cheap post-attention ops (mlp activation, linear2, gated residual)
    def f2(attn, mlp, x, vec):
        _, _, mod_gate = block.modulation(vec).chunk(3, dim=-1)
        output = block.linear2(torch.cat((attn, block.mlp_act(mlp)), dim=2))
        return x + apply_gate(output, gate=mod_gate)

    x_out, t_x_out = torch.func.jvp(f2, (attn, mlp, x, vec), (t_attn, t_mlp, t_x, t_vec))
    return _detach_pair((x_out, t_x_out))
