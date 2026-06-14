# JVP (forward-mode tangent) infrastructure for continuous-time consistency
# distillation (MeanFlow / sCM). Vendored from NVIDIA rCM (arxiv 2510.08431).
from .flash_attention_jvp_triton import _attention, attention

__all__ = ["_attention", "attention"]
