# JVP (forward-mode tangent) infrastructure for continuous-time consistency
# distillation (MeanFlow / sCM). The dense attention kernel is vendored from
# NVIDIA rCM (arxiv 2510.08431); the primitives layer is Voyager-specific.
#
# The kernel import is guarded: it requires CUDA + triton + flash-attn at import
# time, which are absent on CPU-only dev machines. The primitives in
# ``jvp_attention`` (JVP base, rope-with-tangent, naive reference) import fine
# without it.
try:
    from .flash_attention_jvp_triton import _attention, attention
except Exception:  # pragma: no cover
    _attention = None
    attention = None

from .jvp_attention import (
    JVP,
    TensorWithT,
    attention_withT,
    naive_attention_withT,
    apply_rotary_emb_with_tangent,
)

__all__ = [
    "_attention",
    "attention",
    "JVP",
    "TensorWithT",
    "attention_withT",
    "naive_attention_withT",
    "apply_rotary_emb_with_tangent",
]
