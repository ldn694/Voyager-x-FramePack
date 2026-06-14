# SPDX-License-Identifier: Apache-2.0
"""
Model-level forward-mode (JVP) pass over the Hunyuan DiT standard backbone.

``model_forward_jvp`` mirrors the standard forward path of
``HYVideoDiffusionTransformer`` (the ``else`` branch of ``models.py:forward`` —
patchify → time-embed → double×N → single×M → final → unpatchify) and threads a
``(value, tangent)`` pair through it, returning ``(out, t_out)`` where ``t_out`` is
the directional derivative ``dF`` along the supplied input tangents.

It is a STANDALONE function that reads the existing model's submodules
(``model.img_in``, ``model.time_in``, ``model.double_blocks``, …). It adds no
method to the model and edits nothing in ``models.py``; the baseline and
dual-branch forward paths are untouched and never invoke this code.

Tangent directions (set by the caller / loss):
  * ``x`` (latent input) carries tangent ``t_x``. In ``latent_concat`` mode ``x`` is
    ``[z_data, cond, mask, partial_cond, partial_mask]`` on the channel dim; the
    caller zeroes ``t_x`` on the conditioning channels (they are constant along the
    flow). For MeanFlow the data-channel tangent is the velocity ``v``.
  * ``t`` (timestep) carries tangent ``t_t`` (``1`` for MeanFlow's ``d/dt``;
    ``cos·sin`` for the sCM rearrangement, etc.).
  * The primal ``out`` keeps its autograd graph to the model parameters (so the
    consistency loss can update them); ``t_out`` and all intermediate tangents are
    detached (truncated / first-order forward-mode propagation).

DESIGN DECISION — text conditioning is treated as CONSTANT (zero tangent).
``txt_in`` (the SingleTokenRefiner) depends on ``t``, but propagating that tangent
would require a JVP path through the refiner's own flash-attention. We instead
compute ``txt`` once in normal mode and give it a zero tangent (it still acquires a
nonzero tangent *inside* the blocks via ``vec`` and via attention with the image
tokens — that part is exact). This matches the standard rCM/MeanFlow treatment of
conditioning. Flagged for later refinement (finite-diff on ``t`` for the refiner, or
routing the refiner through the kernel) if the text-time path proves to matter.

Assumptions (asserted): batch size 1 (distillation uses global-batch 1, so dense
``attention_withT`` over ``[img, cropped-txt]`` equals the baseline varlen attn with
no padding); standard backbone only (no MultiPatchEmbed, no context block, no
second/dual branch); ``i2v_condition_type != "token_replace"``.
"""

from typing import Callable, Optional, Tuple

import torch

from .jvp_attention import attention_withT, TensorWithT
from .jvp_blocks import double_stream_block_jvp, single_stream_block_jvp


def model_forward_jvp(
    model,
    x_withT: TensorWithT,
    t_withT: TensorWithT,
    text_states: torch.Tensor,
    text_mask: torch.Tensor,
    text_states_2: torch.Tensor,
    freqs_cos: Optional[torch.Tensor],
    freqs_sin: Optional[torch.Tensor],
    guidance: Optional[torch.Tensor] = None,
    extra_vec: Optional[torch.Tensor] = None,
    attn_op: Callable[[TensorWithT, TensorWithT, TensorWithT], TensorWithT] = attention_withT,
) -> TensorWithT:
    """JVP of the standard Hunyuan DiT forward. Returns ``(out, t_out)``.

    ``out`` has shape ``[B, C, T, H, W]`` like ``model.forward(..., return_dict=False)``.

    ``extra_vec`` (optional) is a CONSTANT-tangent term added to the modulation
    vector ``vec`` — used by MeanFlow to inject the second-time ``r`` embedding
    ``r_in(input_r)``. Because ``r`` is constant along the JVP direction
    (tangent ``0`` on ``r``), this term contributes nothing to ``t_vec``; it only
    shifts the primal modulation, exactly like ``vector_in``/``guidance_in``.
    """
    x, t_x = x_withT
    t, t_t = t_withT

    # --- scope guards: this path mirrors only the standard backbone ---
    assert x.shape[0] == 1, "model_forward_jvp currently supports batch size 1"
    assert getattr(model, "patch_sizes", None) is None, "MultiPatchEmbed not supported"
    assert not getattr(model, "use_context_block", False), "context block not supported"
    assert not getattr(model, "use_second_branch", False), "dual branch not supported"
    assert model.i2v_condition_type != "token_replace", "token_replace not supported"

    _, _, ot, oh, ow = x.shape
    pt, ph, pw = model.patch_size
    tt, th, tw = ot // pt, oh // ph, ow // pw

    # ---- time / vector / guidance modulation vector (tangent only from t) ----
    vec, t_vec = torch.func.jvp(model.time_in, (t,), (t_t,))
    vec = vec + model.vector_in(text_states_2)            # constant add, tangent unchanged
    if model.guidance_embed:
        if guidance is None:
            raise ValueError("guidance-distilled model requires `guidance`")
        vec = vec + model.guidance_in(guidance)           # constant add, tangent unchanged
    if extra_vec is not None:
        vec = vec + extra_vec                              # MeanFlow r-embedding (zero tangent)
    t_vec = t_vec.detach()

    # ---- patchify the latent (tangent from x) ----
    img, t_img = torch.func.jvp(model.img_in, (x,), (t_x,))
    t_img = t_img.detach()

    # ---- text (constant condition, zero tangent; cropped to true length) ----
    if model.text_projection == "linear":
        txt = model.txt_in(text_states)
    elif model.text_projection == "single_refiner":
        txt = model.txt_in(text_states, t, text_mask if model.use_attention_mask else None)
    else:
        raise NotImplementedError(model.text_projection)
    text_len = int(text_mask[0].sum().item()) if text_mask is not None else txt.shape[1]
    txt = txt[:, :text_len]
    t_txt = torch.zeros_like(txt)

    # ---- DiT blocks ----
    freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
    img_seq_len = img.shape[1]
    img_wT: TensorWithT = (img, t_img)
    txt_wT: TensorWithT = (txt, t_txt)
    vec_wT: TensorWithT = (vec, t_vec)

    for block in model.double_blocks:
        img_wT, txt_wT = double_stream_block_jvp(
            block, img_wT, txt_wT, vec_wT,
            freqs_cis=freqs_cis, condition_type=model.i2v_condition_type, attn_op=attn_op,
        )

    x_wT: TensorWithT = (
        torch.cat((img_wT[0], txt_wT[0]), dim=1),
        torch.cat((img_wT[1], txt_wT[1]), dim=1),
    )
    for block in model.single_blocks:
        x_wT = single_stream_block_jvp(
            block, x_wT, vec_wT, txt_len=text_len,
            freqs_cis=freqs_cis, condition_type=model.i2v_condition_type, attn_op=attn_op,
        )

    img_final = x_wT[0][:, :img_seq_len]
    t_img_final = x_wT[1][:, :img_seq_len].detach()

    # ---- final layer + unpatchify (tangent from img + vec) ----
    def _final(img_, vec_):
        out_ = model.final_layer(img_, vec_)
        return model.unpatchify(out_, tt, th, tw)

    out, t_out = torch.func.jvp(_final, (img_final, vec), (t_img_final, t_vec))
    return out, t_out.detach()
