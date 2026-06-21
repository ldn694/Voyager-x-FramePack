# SPDX-License-Identifier: Apache-2.0
"""
Model-level forward-mode (JVP) pass over the Hunyuan DiT *dual-branch* backbone.

``model_forward_jvp_double_branch`` mirrors the ``use_second_branch`` path of
``HYVideoDiffusionTransformer.forward`` (models.py: multi-kernel patchify → keyframe/
non-keyframe split → interleaved first/second-branch blocks with cross-attention per
the scheduler → merge_back → MultiFinalLayer → unpatchify_multi) and threads a
``(value, tangent)`` pair through it, returning ``(out, t_out)`` for MeanFlow's
``dudt``.

It is STANDALONE — reads the existing model submodules, edits nothing in
``models.py``/``double_branch.py``; the baseline forward is untouched.

Two tangent modes (``second_branch_tangent``), one shared primal path
---------------------------------------------------------------------
* ``"frozen"`` (Option B): the **first branch** is fully JVP'd (its self-attention
  carries the velocity tangent via ``attn_op``). The **second branch** runs in plain
  forward — exact primal, full grad to its params, but tangent treated as zero — and
  each **cross-attention** contributes only its residual passthrough to the first
  branch's tangent (the inter-branch coupling tangent is dropped). This needs no JVP
  through cross-attention; it is the same constant-tangent approximation already used
  for text/refiner conditioning in ``jvp_model.py``.
* ``"full"`` (Option A): second-branch blocks and cross-attention also carry exact
  tangents. Second-branch blocks are ``MMSingleStreamBlock`` (reusable via
  ``single_stream_block_jvp``); the only new piece is a JVP through
  ``MMCrossStreamBlock`` (``cross_stream_block_jvp``, WP1). Until that lands, ``"full"``
  raises ``NotImplementedError``.

In both modes the primal keeps its autograd graph to the parameters; all tangents are
detached (truncated / first-order forward-mode propagation).

Assumptions (asserted): batch size 1; ``use_second_branch`` and multi-kernel
(``patch_sizes is not None``); ``i2v_condition_type != "token_replace"``. Cross-attn /
second-branch primal uses the real flash-attention blocks, so this path runs on GPU
(env ``voyager``); validate it with ``check_primal_parity_double_branch.py``.
"""

import time
from typing import Callable, Optional, Tuple

import torch
import torch.utils.checkpoint
from loguru import logger

from ..attenion import get_cu_seqlens
from ..modulate_layers import ckpt_wrapper
from .jvp_attention import attention_withT, TensorWithT
from .jvp_blocks import double_stream_block_jvp, single_stream_block_jvp
from .jvp_model import _double_block_ckpt, _single_block_ckpt
from .jvp_multikernel import (
    multikernel_patchify_jvp,
    split_branches_jvp,
    merge_back_jvp,
    final_layer_multi_jvp,
    unpatchify_multi_jvp,
)
from .jvp_schedule import expand_double_branch_schedule


def model_forward_jvp_double_branch(
    model,
    x_withT: TensorWithT,
    t_withT: TensorWithT,
    text_states: torch.Tensor,
    text_mask: torch.Tensor,
    text_states_2: torch.Tensor,
    freqs_cos: Optional[torch.Tensor],
    freqs_sin: Optional[torch.Tensor],
    indices,
    guidance: Optional[torch.Tensor] = None,
    extra_vec: Optional[torch.Tensor] = None,
    extra_vec_second: Optional[torch.Tensor] = None,
    second_branch_tangent: str = "frozen",
    attn_op: Callable[[TensorWithT, TensorWithT, TensorWithT], TensorWithT] = attention_withT,
    verbose: bool = False,
) -> TensorWithT:
    """JVP of the dual-branch Hunyuan DiT forward. Returns ``(out, t_out)``.

    ``out`` has shape ``[B, C, T, H, W]`` like ``model.forward(..., return_dict=False)``.
    ``indices`` are the multi-kernel frame-index lists (as passed to ``model.forward``).
    ``extra_vec`` / ``extra_vec_second`` inject MeanFlow's ``r`` embedding (constant
    tangent) into the first / second branch modulation vectors.
    """
    x, t_x = x_withT
    t, t_t = t_withT

    # --- scope guards ---
    assert second_branch_tangent in ("frozen", "full"), second_branch_tangent
    if second_branch_tangent == "full":
        raise NotImplementedError(
            "second_branch_tangent='full' (Option A) needs cross_stream_block_jvp (WP1); "
            "use 'frozen' until that lands."
        )
    assert x.shape[0] == 1, "batch size 1 only"
    assert getattr(model, "use_second_branch", False), "model has no second branch"
    assert getattr(model, "patch_sizes", None) is not None, "dual branch requires multi-kernel"
    assert model.i2v_condition_type != "token_replace", "token_replace not supported"

    _t0 = [time.time()]
    _last = [time.time()]

    def _stamp(msg: str) -> None:
        if not verbose:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.time()
        logger.info(f"[jvp-db] {msg} (+{now - _t0[0]:.2f}s total, Δ{now - _last[0]:.2f}s)")
        _last[0] = now

    _stamp("entered")
    _, _, ot, oh, ow = x.shape
    cond_type = model.i2v_condition_type

    # ---- main-branch modulation vector (tangent only from t) ----
    vec, t_vec = torch.func.jvp(model.time_in, (t,), (t_t,))
    vec = vec + model.vector_in(text_states_2)
    if model.guidance_embed:
        if guidance is None:
            raise ValueError("guidance-distilled model requires `guidance`")
        vec = vec + model.guidance_in(guidance)
    if extra_vec is not None:
        vec = vec + extra_vec
    t_vec = t_vec.detach()

    # ---- second-branch modulation vector ----
    # In "frozen" mode the second branch is constant w.r.t. the JVP direction, so its
    # vec needs no tangent; it still gets the (constant) r term for the primal.
    vec_sb = model.time_in_second_branch(t)
    if extra_vec_second is not None:
        vec_sb = vec_sb + extra_vec_second

    # ---- multi-kernel patchify + keyframe/non-keyframe split ----
    img, t_img, patch_indices = multikernel_patchify_jvp(model.img_in, x, t_x, indices)
    (b1, freqs1), (b2, freqs2) = split_branches_jvp(
        img, t_img, freqs_cos, freqs_sin, patch_indices
    )
    img_len = b1[0].shape[1]
    _stamp(f"patchify+split done (key={img_len}, nonkey={b2[0].shape[1]})")

    # ---- text (constant condition, zero tangent; cropped to true length, batch=1) ----
    if model.text_projection == "linear":
        txt = model.txt_in(text_states)
    elif model.text_projection == "single_refiner":
        txt = model.txt_in(text_states, t, text_mask if model.use_attention_mask else None)
    else:
        raise NotImplementedError(model.text_projection)
    text_len = int(text_mask[0].sum().item()) if text_mask is not None else txt.shape[1]
    txt = txt[:, :text_len]
    t_txt = torch.zeros_like(txt)

    # ---- cu_seqlens for the plain second-branch / cross-attention calls (batch=1) ----
    no_text_mask = torch.zeros((1, 0), dtype=text_mask.dtype, device=text_mask.device)
    first_branch_cu_seqlens = get_cu_seqlens(no_text_mask, b1[0].shape[1])
    second_branch_cu_seqlens = get_cu_seqlens(no_text_mask, b2[0].shape[1])
    first_branch_max_seqlen = b1[0].shape[1]
    second_branch_max_seqlen = b2[0].shape[1]

    use_ckpt = bool(getattr(model, "training", False) and getattr(model, "gradient_checkpoint", False))
    ckpt_layers = getattr(model, "gradient_checkpoint_layers", -1)
    n_double = len(model.double_blocks)
    n_single = len(model.single_blocks)
    n_second = len(model.second_branch_blocks)

    def _ckpt_on(global_idx: int) -> bool:
        return use_ckpt and (ckpt_layers == -1 or global_idx < ckpt_layers)

    # ---- interleaved scheduler walk ----
    plan = expand_double_branch_schedule(
        model.double_branch_scheduler, n_double, n_single, n_second
    )

    last_img_wT: Optional[TensorWithT] = b1                 # (img1, t_img1)
    last_txt_wT: TensorWithT = (txt, t_txt)
    last_x_wT: Optional[TensorWithT] = None                 # cat(img,txt) once in single region
    # project non-keyframe tokens into the second-branch width (mirrors models.py:1262)
    last_sb: torch.Tensor = model.proj_to_second_branch(b2[0])   # primal (frozen tangent)

    for step in plan:
        # ---------------- first branch blocks ----------------
        for layer_num in step["first_range"]:
            if last_x_wT is None and layer_num >= n_double:
                last_x_wT = (
                    torch.cat((last_img_wT[0], last_txt_wT[0]), dim=1),
                    torch.cat((last_img_wT[1], last_txt_wT[1]), dim=1),
                )
            if layer_num < n_double:
                block = model.double_blocks[layer_num]
                if _ckpt_on(layer_num):
                    last_img_wT, last_txt_wT = _double_block_ckpt(
                        block, last_img_wT, last_txt_wT, (vec, t_vec), freqs1, cond_type, attn_op,
                    )
                else:
                    last_img_wT, last_txt_wT = double_stream_block_jvp(
                        block, last_img_wT, last_txt_wT, (vec, t_vec),
                        freqs_cis=freqs1, condition_type=cond_type, attn_op=attn_op,
                    )
            else:
                block = model.single_blocks[layer_num - n_double]
                if _ckpt_on(layer_num):
                    last_x_wT = _single_block_ckpt(
                        block, last_x_wT, (vec, t_vec), text_len, freqs1, cond_type, attn_op,
                    )
                else:
                    last_x_wT = single_stream_block_jvp(
                        block, last_x_wT, (vec, t_vec), txt_len=text_len,
                        freqs_cis=freqs1, condition_type=cond_type, attn_op=attn_op,
                    )
                last_img_wT = (last_x_wT[0][:, :img_len], last_x_wT[1][:, :img_len])

        # ---------------- second branch blocks (plain forward; frozen tangent) ----------------
        for layer_num in step["second_range"]:
            block = model.second_branch_blocks[layer_num]
            sb_args = [
                last_sb, vec_sb, 0,
                second_branch_cu_seqlens, second_branch_cu_seqlens,
                second_branch_max_seqlen, second_branch_max_seqlen,
                freqs2, cond_type, None, None,
            ]
            if _ckpt_on(layer_num):
                last_sb = torch.utils.checkpoint.checkpoint(
                    ckpt_wrapper(block), *sb_args, use_reentrant=False
                )
            else:
                last_sb = block(*sb_args)

        # ---------------- cross attention ----------------
        if step["cross"] == "first_q":
            # first branch (query) attends to second branch (kv); primal exact, tangent passes through
            ca = model.cross_attn_blocks[step["cross_attn_id"]]
            new_img = ca(
                last_img_wT[0], last_sb, vec, vec_sb,
                cu_seqlens_q=first_branch_cu_seqlens,
                cu_seqlens_kv=second_branch_cu_seqlens,
                max_seqlen_q=first_branch_max_seqlen,
                max_seqlen_kv=second_branch_max_seqlen,
                freqs_cis_q=freqs1, freqs_cis_kv=freqs2,
                condition_type=cond_type, token_replace_vec=None, frist_frame_token_num=None,
            )
            last_img_wT = (new_img, last_img_wT[1])          # frozen: passthrough query tangent
            if last_x_wT is not None:
                cur_txt = last_x_wT[0][:, img_len:]
                cur_txt_t = last_x_wT[1][:, img_len:]
                last_x_wT = (
                    torch.cat((last_img_wT[0], cur_txt), dim=1),
                    torch.cat((last_img_wT[1], cur_txt_t), dim=1),
                )
        elif step["cross"] == "second_q":
            # second branch (query) attends to first branch (kv); updates second-branch primal only
            ca = model.cross_attn_blocks[step["cross_attn_id"]]
            last_sb = ca(
                last_sb, last_img_wT[0], vec_sb, vec,
                cu_seqlens_q=second_branch_cu_seqlens,
                cu_seqlens_kv=first_branch_cu_seqlens,
                max_seqlen_q=second_branch_max_seqlen,
                max_seqlen_kv=first_branch_max_seqlen,
                freqs_cis_q=freqs2, freqs_cis_kv=freqs1,
                condition_type=cond_type, token_replace_vec=None, frist_frame_token_num=None,
            )

    _stamp("scheduler walk done")

    # ---- merge branches back (second-branch tangent is zero in frozen mode) ----
    last_sb_unproj = model.unproj_from_second_branch(last_sb)
    t_last_sb_unproj = torch.zeros_like(last_sb_unproj)
    img_merged, t_img_merged = merge_back_jvp(
        model, (last_img_wT[0], last_img_wT[1]), (last_sb_unproj, t_last_sb_unproj), patch_indices,
    )

    # ---- final layer + unpatchify (tangent from merged img + vec) ----
    outs, t_outs = final_layer_multi_jvp(
        model.final_layer, (img_merged, t_img_merged), (vec, t_vec), indices, patch_indices,
    )
    out, t_out = unpatchify_multi_jvp(
        model, list(zip(outs, t_outs)),
        T=ot, H=oh, W=ow, patch_sizes=model.patch_sizes, indices=indices,
    )
    _stamp("final layer + unpatchify done")
    return out, t_out.detach()
