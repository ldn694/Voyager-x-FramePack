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
_R_IN_SECOND_ATTR = "r_in_second_branch"


def _make_r_embedder(hidden_size, ref, zero_init: bool):
    r_in = TimestepEmbedder(
        hidden_size,
        get_activation_layer("silu"),
        dtype=ref.dtype,
        device=ref.device,
    )
    if zero_init:
        # TimestepEmbedder.mlp = [Linear, SiLU, Linear]; zero the final linear so
        # the whole module outputs 0 regardless of r.
        nn.init.zeros_(r_in.mlp[2].weight)
        nn.init.zeros_(r_in.mlp[2].bias)
    for p in r_in.parameters():
        p.requires_grad = True
    return r_in


def apply_meanflow_to_hunyuan_video(model, zero_init: bool = True):
    """Attach a second-time (``r``) embedder ``model.r_in`` mirroring ``time_in``.

    For a dual-branch model (``use_second_branch``) a matching
    ``model.r_in_second_branch`` is also attached, mirroring
    ``time_in_second_branch`` (second-branch width), so the ``r`` conditioning
    reaches the second branch's modulation vector the same way ``t`` does. Both are
    zero-initialised, so the freshly-applied model reproduces the pretrained
    behaviour bit-for-bit.

    Args:
        model: a ``HYVideoDiffusionTransformer`` instance.
        zero_init: zero the output projection so ``r_in(r) == 0`` at init.

    Returns the (mutated) model. Idempotent: re-applying replaces the embedders.
    """
    ref = next(model.time_in.parameters())
    setattr(model, _R_IN_ATTR, _make_r_embedder(model.hidden_size, ref, zero_init))

    if getattr(model, "use_second_branch", False):
        # second-branch width = proj_to_second_branch output dim
        sb_hidden = model.proj_to_second_branch.out_features
        sb_ref = next(model.time_in_second_branch.parameters())
        setattr(model, _R_IN_SECOND_ATTR, _make_r_embedder(sb_hidden, sb_ref, zero_init))
    return model


def has_meanflow(model) -> bool:
    return getattr(model, _R_IN_ATTR, None) is not None


def has_meanflow_second_branch(model) -> bool:
    return getattr(model, _R_IN_SECOND_ATTR, None) is not None


def _meanflow_modules(model) -> Dict[str, nn.Module]:
    mods = {}
    for attr in (_R_IN_ATTR, _R_IN_SECOND_ATTR):
        m = getattr(model, attr, None)
        if m is not None:
            mods[attr] = m
    return mods


def get_meanflow_parameters(model) -> List[nn.Parameter]:
    params: List[nn.Parameter] = []
    for m in _meanflow_modules(model).values():
        params.extend(m.parameters())
    return params


def get_meanflow_state_dict(model) -> Dict[str, torch.Tensor]:
    sd = {}
    for attr, m in _meanflow_modules(model).items():
        for k, v in m.state_dict().items():
            sd[f"{attr}.{k}"] = v.detach().cpu()
    return sd


def load_meanflow_state_dict(model, state_dict: Dict[str, torch.Tensor], strict: bool = True):
    """Restore the ``r_in`` (and ``r_in_second_branch``) weights.

    Keys are grouped by their ``<attr>.`` prefix and loaded into the matching
    module. A checkpoint without the second-branch keys is fine (it simply isn't
    loaded); ``strict`` is applied per-module after grouping.
    """
    mods = _meanflow_modules(model)
    if not mods:
        raise RuntimeError("apply_meanflow_to_hunyuan_video must be called before loading r_in weights.")

    grouped: Dict[str, Dict[str, torch.Tensor]] = {attr: {} for attr in mods}
    # Longest-prefix match first so "r_in_second_branch." is not swallowed by "r_in.".
    for k, v in state_dict.items():
        for attr in sorted(mods, key=len, reverse=True):
            prefix = f"{attr}."
            if k.startswith(prefix):
                grouped[attr][k[len(prefix):]] = v
                break
        else:
            # un-prefixed keys default to the main r_in (back-compat with old ckpts)
            grouped[_R_IN_ATTR][k] = v

    for attr, m in mods.items():
        sub = grouped[attr]
        if not sub and attr == _R_IN_SECOND_ATTR:
            continue  # second-branch embedder not present in this checkpoint
        ref = next(m.parameters())
        sub = {k: v.to(device=ref.device, dtype=ref.dtype) for k, v in sub.items()}
        m.load_state_dict(sub, strict=strict)
