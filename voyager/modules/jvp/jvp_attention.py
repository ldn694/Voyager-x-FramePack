# SPDX-License-Identifier: Apache-2.0
"""
JVP (forward-mode tangent) primitives for Voyager's Hunyuan DiT.

This is the reusable "build-once" layer shared by the MeanFlow and rCM/sCM
distillation baselines. It provides:

  * ``JVP``                      — base nn.Module with the (value, tangent) dispatch
                                   convention (``forward(..., withT=True)`` →
                                   ``_forward_jvp``; otherwise ``_forward``).
  * ``attention_withT``         — dense full-attention with a fused JVP, wrapping the
                                   vendored Triton kernel. Handles the Hunyuan
                                   ``[B, S, H, D]`` ↔ kernel ``[B, H, S, D]`` transpose
                                   and detaches the output tangent (first-order /
                                   truncated tangent propagation).
  * ``apply_rotary_emb_with_tangent`` — RoPE that also rotates the tangent. RoPE is
                                   linear in q/k, so its JVP is the same rotation; we
                                   obtain it via ``torch.func.jvp`` over Voyager's
                                   existing ``apply_rotary_emb`` (the original function
                                   is left untouched).
  * ``naive_attention_withT``   — a pure-PyTorch reference (CPU-runnable) used to
                                   validate ``attention_withT`` and the block plumbing
                                   without the Triton kernel.

Mirrors the surface of ``rcm/rcm/utils/jvp_helper.py`` (``torch_attention_op_withT``,
``JVP``) minus the Ulysses context-parallel and Magi-mask machinery, which Voyager's
single-sequence non-causal i2v does not need.

NOTE: This module adds NEW code only. It does not import from or modify any of the
baseline attention / RoPE / model code paths.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..posemb_layers import apply_rotary_emb

# The vendored dense JVP attention kernel (Triton). ``attention`` is
# ``_attention.apply`` returning ``(o, to)`` for ``[B, H, S, D]`` inputs.
try:
    from .flash_attention_jvp_triton import attention as _kernel_attention
except Exception:  # pragma: no cover - kernel needs CUDA/triton/flash-attn at import time
    _kernel_attention = None


TensorWithT = Tuple[torch.Tensor, torch.Tensor]


class JVP(nn.Module):
    """Base class for modules that carry a forward-mode (JVP) rule.

    ``forward(..., withT=False)`` runs the ordinary path (``_forward``); with
    ``withT=True`` it runs the tangent-propagating path (``_forward_jvp``), which
    takes and returns ``(value, tangent)`` pairs. Subclasses override either or
    both. The baseline never passes ``withT=True``, so wrapping a module in this
    base class does not change its default behaviour.
    """

    def forward(self, *args, **kwargs):
        withT = kwargs.pop("withT", False)
        if withT:
            return self._forward_jvp(*args, **kwargs)
        return self._forward(*args, **kwargs)

    def _forward(self, *args, **kwargs):
        raise NotImplementedError

    def _forward_jvp(self, *args, **kwargs):
        raise NotImplementedError


def attention_withT(
    q_withT: TensorWithT,
    k_withT: TensorWithT,
    v_withT: TensorWithT,
    sm_scale: Optional[float] = None,
) -> TensorWithT:
    """Dense full attention with a fused JVP, via the vendored Triton kernel.

    Args:
        q_withT, k_withT, v_withT: ``(value, tangent)`` pairs, each ``[B, S, H, D]``
            (Hunyuan layout). Self- or cross-attention (Sq != Skv) and differing
            qk/v head dims are supported by the kernel.
        sm_scale: softmax scale; defaults to ``head_dim ** -0.5`` inside the kernel.

    Returns:
        ``(out, t_out)`` each ``[B, S, H, D]``. ``t_out`` is detached — the tangent
        is a forward-mode signal, never part of the autograd graph used for the
        parameter update (matches rCM's truncated tangent propagation).
    """
    if _kernel_attention is None:
        raise RuntimeError(
            "JVP attention kernel unavailable (needs CUDA + triton + flash-attn). "
            "Use naive_attention_withT for CPU validation."
        )
    q, tq = q_withT
    k, tk = k_withT
    v, tv = v_withT
    o, to = _kernel_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        tq.transpose(1, 2),
        tk.transpose(1, 2),
        tv.transpose(1, 2),
        sm_scale,
    )
    return o.transpose(1, 2), to.transpose(1, 2).detach()


def _naive_sdpa_bshd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     sm_scale: Optional[float] = None) -> torch.Tensor:
    """Plain (non-flash) full attention for ``[B, S, H, D]`` inputs, fp32 softmax."""
    scale = q.shape[-1] ** -0.5 if sm_scale is None else sm_scale
    qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # [B,H,S,D]
    p = (qt.float() @ kt.float().transpose(-2, -1)) * scale
    p = p.softmax(dim=-1)
    o = p @ vt.float()
    return o.transpose(1, 2).to(q.dtype)  # [B, S, H, D]


def naive_attention_withT(
    q_withT: TensorWithT,
    k_withT: TensorWithT,
    v_withT: TensorWithT,
    sm_scale: Optional[float] = None,
) -> TensorWithT:
    """Pure-PyTorch reference for :func:`attention_withT` (CPU-runnable).

    Uses ``torch.func.jvp`` over a naive attention so it can validate the kernel
    wrapper and the block-level plumbing without Triton/flash-attn. The output
    tangent is detached to match :func:`attention_withT`.
    """
    q, tq = q_withT
    k, tk = k_withT
    v, tv = v_withT

    def f(q_, k_, v_):
        return _naive_sdpa_bshd(q_, k_, v_, sm_scale)

    o, to = torch.func.jvp(f, (q, k, v), (tq, tk, tv))
    return o, to.detach()


def apply_rotary_emb_with_tangent(
    xq: torch.Tensor,
    txq: torch.Tensor,
    xk: torch.Tensor,
    txk: torch.Tensor,
    freqs_cis,
    head_first: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """RoPE that also carries the tangent through the rotation.

    RoPE is linear in ``xq``/``xk``, so its JVP is the *same* rotation applied to the
    tangent. We compute it with ``torch.func.jvp`` over the existing (unmodified)
    ``apply_rotary_emb``. Hunyuan calls RoPE with ``freqs_cis = (cos, sin)`` (the
    real/tuple branch), which is pointwise-linear and fully forward-mode
    differentiable. (The complex ``view_as_complex`` branch is not used by Voyager.)

    Returns ``(xq_out, txq_out, xk_out, txk_out)``. Tangents are NOT detached here;
    detaching happens at the attention (kernel) boundary in :func:`attention_withT`,
    matching the layer-boundary detach policy used for tangent truncation.
    """
    def f(q_, k_):
        return apply_rotary_emb(q_, k_, freqs_cis, head_first=head_first)

    (xq_out, xk_out), (txq_out, txk_out) = torch.func.jvp(f, (xq, xk), (txq, txk))
    return xq_out, txq_out, xk_out, txk_out
