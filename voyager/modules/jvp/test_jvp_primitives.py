# SPDX-License-Identifier: Apache-2.0
"""
Validation for the Voyager JVP primitives (``jvp_attention.py``).

Two tiers:
  * CPU tier (``test_rope_tangent``, ``test_naive_attention_withT``) — no CUDA /
    Triton / flash-attn needed. Run anywhere, including this dev laptop:
        python -m voyager.modules.jvp.test_jvp_primitives --cpu
  * GPU tier (``test_kernel_matches_naive``) — validates the vendored Triton kernel
    wrapper ``attention_withT`` against the naive reference. Run on the server:
        python -m voyager.modules.jvp.test_jvp_primitives

The CPU tier pins down the two Voyager-specific pieces (rope-with-tangent and the
[B,S,H,D] attention contract) so a kernel discrepancy on the server is isolated to
the kernel itself, whose own bundled tests (flash_attention_jvp_triton.test_jvp*)
are the other half of the validation.
"""

import torch

from .jvp_attention import (
    naive_attention_withT,
    apply_rotary_emb_with_tangent,
)
from ..posemb_layers import apply_rotary_emb, get_nd_rotary_pos_embed


def _rand_withT(shape, dtype, device):
    x = torch.randn(shape, dtype=dtype, device=device)
    tx = torch.randn(shape, dtype=dtype, device=device)
    return x, tx


def test_rope_tangent(device="cpu", dtype=torch.float32):
    """apply_rotary_emb_with_tangent matches torch.func.jvp over apply_rotary_emb,
    and its value output matches the unmodified apply_rotary_emb exactly."""
    torch.manual_seed(0)
    B, S, H, D = 1, 40, 4, 32  # head_dim 32 -> rope_dim_list sums to 32
    rope_dim_list = [8, 12, 12]
    # Build a (cos, sin) freqs tuple the way Hunyuan does (use_real=True).
    freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
        rope_dim_list, (2, 4, 5), use_real=True
    )
    freqs_cis = (freqs_cos.to(device), freqs_sin.to(device))

    xq, txq = _rand_withT((B, S, H, D), dtype, device)
    xk, txk = _rand_withT((B, S, H, D), dtype, device)

    # reference
    def f(q_, k_):
        return apply_rotary_emb(q_, k_, freqs_cis, head_first=False)

    (ref_q, ref_k), (ref_tq, ref_tk) = torch.func.jvp(f, (xq, xk), (txq, txk))

    # under test
    q_out, tq_out, k_out, tk_out = apply_rotary_emb_with_tangent(
        xq, txq, xk, txk, freqs_cis, head_first=False
    )

    # value output must equal the plain function (baseline parity)
    base_q, base_k = apply_rotary_emb(xq, xk, freqs_cis, head_first=False)
    torch.testing.assert_close(q_out, base_q)
    torch.testing.assert_close(k_out, base_k)
    # tangents must match the reference JVP
    torch.testing.assert_close(tq_out, ref_tq)
    torch.testing.assert_close(tk_out, ref_tk)
    print("test_rope_tangent: PASSED")


def test_naive_attention_withT(device="cpu", dtype=torch.float32):
    """naive_attention_withT is internally consistent: its value equals plain
    attention and its tangent equals an independent torch.func.jvp."""
    torch.manual_seed(0)
    B, S, H, D = 1, 64, 4, 32
    q, tq = _rand_withT((B, S, H, D), dtype, device)
    k, tk = _rand_withT((B, S, H, D), dtype, device)
    v, tv = _rand_withT((B, S, H, D), dtype, device)

    o, to = naive_attention_withT((q, tq), (k, tk), (v, tv))

    # independent reference
    from .jvp_attention import _naive_sdpa_bshd

    def f(q_, k_, v_):
        return _naive_sdpa_bshd(q_, k_, v_)

    ref_o, ref_to = torch.func.jvp(f, (q, k, v), (tq, tk, tv))
    torch.testing.assert_close(o, ref_o)
    torch.testing.assert_close(to, ref_to.detach())
    print("test_naive_attention_withT: PASSED")


def test_kernel_matches_naive(dtype=torch.float16):
    """GPU only: the Triton-kernel wrapper attention_withT matches the naive
    reference for both the output and its tangent."""
    from .jvp_attention import attention_withT

    device = "cuda"
    torch.manual_seed(0)
    for (B, S, H, D) in [(1, 1024, 4, 64), (1, 999, 8, 128)]:
        q, tq = _rand_withT((B, S, H, D), dtype, device)
        k, tk = _rand_withT((B, S, H, D), dtype, device)
        v, tv = _rand_withT((B, S, H, D), dtype, device)

        o_k, to_k = attention_withT((q, tq), (k, tk), (v, tv))
        o_n, to_n = naive_attention_withT((q, tq), (k, tk), (v, tv))

        atol = 2e-2 if dtype == torch.bfloat16 else 1e-2
        torch.testing.assert_close(o_k, o_n.to(o_k.dtype), atol=atol, rtol=1e-2)
        torch.testing.assert_close(to_k, to_n.to(to_k.dtype), atol=atol, rtol=1e-2)
        print(f"test_kernel_matches_naive [{B},{S},{H},{D}] {dtype}: PASSED")


if __name__ == "__main__":
    import sys

    cpu_only = "--cpu" in sys.argv
    test_rope_tangent()
    test_naive_attention_withT()
    if not cpu_only and torch.cuda.is_available():
        test_kernel_matches_naive(torch.float16)
        test_kernel_matches_naive(torch.bfloat16)
    else:
        print("(skipped GPU kernel test — run on the server without --cpu)")
