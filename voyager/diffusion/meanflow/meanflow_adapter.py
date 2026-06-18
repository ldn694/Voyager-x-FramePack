# SPDX-License-Identifier: Apache-2.0
"""
MeanFlow second-time (``r``) conditioning adapter for the Hunyuan DiT.

MeanFlow models the *average* velocity ``u_theta(z, r, t)`` over the interval
``[r, t]`` (two times), whereas the baseline Hunyuan DiT conditions on a single
time ``t`` through ``model.time_in``. The only architectural change MeanFlow needs
is a second time embedding for ``r`` that is summed into the modulation vector
``vec`` — everything else (the JVP forward, the loss) is handled outside the
model.

This module follows the four-function adapter surface used by the other Voyager
adapters (LoRA, patch-adapter, multi-kernel, double-branch):

    apply_meanflow_to_hunyuan_video(model, ...)   -> attaches model.r_in
    get_meanflow_parameters(model)                -> r_in params for the optimizer
    get_meanflow_state_dict(model)                -> r_in weights only
    load_meanflow_state_dict(model, state_dict)   -> restore them

``r_in`` mirrors ``time_in`` exactly (a ``TimestepEmbedder`` with the same hidden
size and SiLU activation) and is **zero-initialised on its output projection** so
that at the start of training ``r_in(input_r) == 0`` for all ``r``. This means the
freshly-applied model reproduces the pretrained single-time behaviour bit-for-bit
(``vec == time_in(t) + vector_in(text2)``), so a flow-matching / pretrained
initialisation is preserved and MeanFlow learns the ``r`` dependence from zero.

The r-embedding is consumed by ``model_forward_jvp(..., extra_vec=r_in(input_r))``
(see ``voyager/modules/jvp/jvp_model.py``); ``r`` carries a zero tangent in the
MeanFlow JVP direction, so this term never enters ``t_vec``.
"""

from typing import Dict, List

import torch
import torch.nn as nn

from ...modules.activation_layers import get_activation_layer
from ...modules.embed_layers import TimestepEmbedder

_R_IN_ATTR = "r_in"


def 
apply_meanflow_to_hunyuan_video(model, zero_init: bool = True):
    """Attach a second-time (``r``) embedder ``model.r_in`` mirroring ``time_in``.

    Args:
        model: a ``HYVideoDiffusionTransformer`` instance.
        zero_init: zero the output projection so ``r_in(r) == 0`` at init, keeping
            the model identical to the pretrained single-time baseline.

    Returns the (mutated) model. Idempotent: re-applying replaces ``r_in``.
    """
    ref = next(model.time_in.parameters())
    r_in = TimestepEmbedder(
        model.hidden_size,
        get_activation_layer("silu"),
        dtype=ref.dtype,
        device=ref.device,
    )
    if zero_init:
        # TimestepEmbedder.mlp = [Linear, SiLU, Linear]; zero the final linear so
        # the whole module outputs 0 regardless of r.
        nn.init.zeros_(r_in.mlp[2].weight)
        nn.init.zeros_(r_in.mlp[2].bias)

    setattr(model, _R_IN_ATTR, r_in)
    for p in r_in.parameters():
        p.requires_grad = True
    return model


def has_meanflow(model) -> bool:
    return getattr(model, _R_IN_ATTR, None) is not None


def get_meanflow_parameters(model) -> List[nn.Parameter]:
    r_in = getattr(model, _R_IN_ATTR, None)
    if r_in is None:
        return []
    return list(r_in.parameters())


def get_meanflow_state_dict(model) -> Dict[str, torch.Tensor]:
    r_in = getattr(model, _R_IN_ATTR, None)
    if r_in is None:
        return {}
    return {f"{_R_IN_ATTR}.{k}": v.detach().cpu() for k, v in r_in.state_dict().items()}


def load_meanflow_state_dict(model, state_dict: Dict[str, torch.Tensor], strict: bool = True):
    """Restore ``r_in`` weights saved by :func:`get_meanflow_state_dict`.

    Accepts keys either with or without the ``r_in.`` prefix.
    """
    r_in = getattr(model, _R_IN_ATTR, None)
    if r_in is None:
        raise RuntimeError("apply_meanflow_to_hunyuan_video must be called before loading r_in weights.")
    prefix = f"{_R_IN_ATTR}."
    cleaned = {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in state_dict.items()
    }
    ref = next(r_in.parameters())
    cleaned = {k: v.to(device=ref.device, dtype=ref.dtype) for k, v in cleaned.items()}
    return r_in.load_state_dict(cleaned, strict=strict)
