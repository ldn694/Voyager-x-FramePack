# SPDX-License-Identifier: Apache-2.0
"""
Primal-parity check for the DUAL-BRANCH JVP forward, on the REAL Hunyuan DiT.

Proves that ``model_forward_jvp_double_branch(..., zero tangents)[0]`` reproduces the
ordinary dual-branch ``model.forward(...)['x']`` — i.e. the JVP path's reconstruction
of the multi-kernel + two-branch + cross-attention forward is faithful. This is the
gate to trust the frozen-mode ``dudt`` (the tangent itself is a deliberate
approximation in Option B, so parity is checked on the primal only; tangents are zero
here).

Run on the GPU server (env ``voyager``) with the SAME multi-kernel + double-branch
flags the training uses, e.g.::

    MODEL_BASE=ckpts python -m voyager.modules.jvp.check_primal_parity_double_branch \
        --model HYVideo-T/2 --vae 884-16c-hy \
        --i2v-condition-type latent_concat --embedded-cfg-scale 6.0 --flow-reverse \
        --train-multiple-kernels \
        --kernel-sizes 1 2 2 --kernel-sizes 1 4 4 \
        --kernel-indices 0 --kernel-indices 1 2 3 4 5 6 7 8 9 10 11 12 \
        --use-double-branch --model-with-double-branch HYVideo-T/2-2branch-cross_attn

The latent temporal length is derived from ``--kernel-indices`` (every frame must be
covered exactly once); spatial size from ``--video-size`` (÷8, rounded to the largest
patch). The double-branch / multi-kernel / final-layer weights are random/zero-init
(parity does not need trained weights — both forward paths use the *same* weights), but
``MultiFinalLayer`` is re-randomised so the output is non-trivial (its zero-init would
make parity trivially 0==0).
"""

import argparse
import os
from pathlib import Path

import torch
from loguru import logger

from voyager.config import (
    add_network_args, add_extra_models_args, add_denoise_schedule_args,
    add_i2v_args, add_lora_args, add_inference_args, add_parallel_args,
    add_patch_adapter_args, add_multiple_kernel_args, add_double_branch_args,
    add_step_sample_args, add_attn_map_args, add_dmd2_args, add_meanflow_args,
    sanity_check_args,
)
from voyager.utils.helpers import as_list_of_3tuple
from voyager.constants import PRECISION_TO_TYPE
from voyager.modules import load_model
from voyager.modules.models import HUNYUAN_VIDEO_CONFIG
from voyager.modules.multi_kernel import apply_multikernel_to_hunyuan_video
from voyager.modules.double_branch import apply_double_branch_to_hunyuan_video, TransformerBranchConfig
from voyager.utils.train_utils import load_state_dict, build_rope
from voyager.modules.jvp.jvp_attention import attention_withT
from voyager.modules.jvp.jvp_model_double_branch import model_forward_jvp_double_branch


def _parse_args():
    """Eval-mode parser (config.parse_args's builder list) PLUS the training-only
    multi-kernel flags, which live in deepspeed_train_render.py rather than any config
    builder. This harness builds a training-style (multi-kernel + double-branch) model,
    so it needs both surfaces."""
    parser = argparse.ArgumentParser(description="dual-branch JVP primal-parity check")
    for add in (
        add_network_args, add_extra_models_args, add_denoise_schedule_args,
        add_i2v_args, add_lora_args, add_inference_args, add_parallel_args,
        add_patch_adapter_args, add_multiple_kernel_args, add_double_branch_args,
        add_step_sample_args, add_attn_map_args, add_dmd2_args, add_meanflow_args,
    ):
        parser = add(parser)
    parser.add_argument('--train-multiple-kernels', action='store_true')
    parser.add_argument('--kernel-sizes', action='append', nargs='+', type=int)
    parser.add_argument('--kernel-indices', action='append', nargs='+', type=int)
    args = parser.parse_args()
    args.kernel_sizes = as_list_of_3tuple(args.kernel_sizes) if args.kernel_sizes is not None else None
    args = sanity_check_args(args)
    return args


def _resolve_double_branch_cfg(args):
    """Replicate deepspeed_train_render.py's double-branch config resolution."""
    cfg = HUNYUAN_VIDEO_CONFIG[args.model_with_double_branch]
    assert "second_branch_transformer_config" in cfg and "second_branch_mm_blocks_depth" in cfg, \
        f"{args.model_with_double_branch} is not a double-branch config"
    assert isinstance(cfg["second_branch_transformer_config"], TransformerBranchConfig)
    args.second_branch_transformer_config = cfg["second_branch_transformer_config"]
    args.second_branch_mm_blocks_depth = cfg["second_branch_mm_blocks_depth"]


def _build_double_branch_dit(args, device, dtype):
    """Build base DiT, load pretrained weights, then apply multi-kernel + double-branch
    exactly as deepspeed_train_render.py does (weights first, adapters after)."""
    factor_kwargs = {"device": device, "dtype": dtype}
    assert args.i2v_mode and args.i2v_condition_type == "latent_concat"
    assert args.train_multiple_kernels and args.kernel_sizes and args.kernel_indices
    assert args.use_double_branch

    in_channels = args.latent_channels * 3 + 2
    out_channels = args.latent_channels
    model = load_model(args, in_channels=in_channels, out_channels=out_channels, factor_kwargs=factor_kwargs)
    model_base = os.environ.get("MODEL_BASE", "ckpts")
    model = load_state_dict(args, model, logger, Path(model_base))

    patch_sizes = [tuple(ks) for ks in args.kernel_sizes]
    apply_multikernel_to_hunyuan_video(
        model, patch_sizes=patch_sizes, device=device, dtype=dtype,
        freeze_base=False, copy_old_weights=True,
    )
    _resolve_double_branch_cfg(args)
    apply_double_branch_to_hunyuan_video(
        model,
        second_branch_config=args.second_branch_transformer_config,
        second_branch_mm_blocks_depth=args.second_branch_mm_blocks_depth,
        device=device, dtype=dtype, freeze_base=False,
    )

    # MultiFinalLayer is zero-init -> output would be identically zero and parity would
    # pass trivially. Randomise it so both forward paths actually exercise the math.
    with torch.no_grad():
        for p in model.final_layer.parameters():
            p.normal_(std=0.02)

    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


@torch.no_grad()
def main():
    args = _parse_args()
    device = "cuda"
    dtype = PRECISION_TO_TYPE[args.precision]

    model = _build_double_branch_dit(args, device, dtype)
    logger.info(
        f"Built dual-branch DiT: hidden={model.hidden_size} second_branch_blocks="
        f"{len(model.second_branch_blocks)} cross_attn_blocks={len(model.cross_attn_blocks)} "
        f"scheduler_len={len(model.double_branch_scheduler)}"
    )

    indices = [list(g) for g in args.kernel_indices]
    # latent temporal length: every frame covered exactly once by the kernel indices
    all_frames = sorted(f for g in indices for f in g)
    Tlat = len(all_frames)
    assert all_frames == list(range(Tlat)), \
        f"kernel-indices must cover frames 0..N-1 exactly once, got {all_frames}"

    # spatial grid from --video-size, divisible by the largest patch size
    max_ph = max(ks[1] for ks in args.kernel_sizes)
    max_pw = max(ks[2] for ks in args.kernel_sizes)
    Hlat = (args.video_size[0] // 8) // max_ph * max_ph
    Wlat = (args.video_size[1] // 8) // max_pw * max_pw
    C = args.latent_channels
    B = 1
    logger.info(f"latent grid: T={Tlat} H={Hlat} W={Wlat} (max patch {max_ph}x{max_pw})")

    torch.manual_seed(0)
    z = torch.randn(B, C, Tlat, Hlat, Wlat, device=device, dtype=dtype)
    cond = torch.randn(B, C, Tlat, Hlat, Wlat, device=device, dtype=dtype)
    mask = torch.ones(B, 1, Tlat, Hlat, Wlat, device=device, dtype=dtype)
    partial_cond = torch.randn(B, C, Tlat, Hlat, Wlat, device=device, dtype=dtype)
    partial_mask = torch.ones(B, 1, Tlat, Hlat, Wlat, device=device, dtype=dtype)
    xt = torch.cat([z, cond, mask, partial_cond, partial_mask], dim=1)
    assert xt.shape[1] == model.in_channels, f"{xt.shape[1]} != in_channels {model.in_channels}"

    input_t = torch.rand(B, device=device, dtype=dtype) * 1000.0

    L = int(os.environ.get("PARITY_TXT_LEN", "64"))
    true_len = min(int(os.environ.get("PARITY_TXT_TRUE", "48")), L)
    text_states = torch.randn(B, L, model.text_states_dim, device=device, dtype=dtype)
    text_mask = torch.zeros(B, L, device=device, dtype=torch.long)
    text_mask[:, :true_len] = 1
    text_states_2 = torch.randn(B, model.text_states_dim_2, device=device, dtype=dtype)

    # RoPE via build_rope (multi-kernel CompressionScheduler path) — sized to the full
    # token sequence in MultiPatchEmbed order; both forwards slice it by patch_indices.
    freqs_cos, freqs_sin = build_rope(z, args, model)
    freqs_cos = freqs_cos.to(device)
    freqs_sin = freqs_sin.to(device)

    guidance = None
    if model.guidance_embed:
        scale = args.embedded_cfg_scale if args.embedded_cfg_scale is not None else 6.0
        guidance = torch.tensor([scale] * B, dtype=torch.float32, device=device).to(dtype) * 1000.0

    autocast_enabled = dtype != torch.float32

    # ---- baseline dual-branch forward ----
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=autocast_enabled):
        base = model(
            xt, input_t,
            text_states=text_states, text_mask=text_mask, text_states_2=text_states_2,
            freqs_cos=freqs_cos, freqs_sin=freqs_sin, guidance=guidance,
            indices=indices, return_dict=True,
        )["x"]

    # ---- JVP primal (zero tangents, frozen mode) ----
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=autocast_enabled):
        prim, t_out = model_forward_jvp_double_branch(
            model, (xt, torch.zeros_like(xt)), (input_t, torch.zeros_like(input_t)),
            text_states=text_states, text_mask=text_mask, text_states_2=text_states_2,
            freqs_cos=freqs_cos, freqs_sin=freqs_sin, indices=indices,
            guidance=guidance, second_branch_tangent="frozen", attn_op=attention_withT,
            verbose=True,
        )

    assert prim.shape == base.shape, f"shape mismatch {prim.shape} vs {base.shape}"
    diff = (prim.float() - base.float()).abs()
    denom = base.float().abs().mean().clamp_min(1e-6)
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    rel = (mean_abs / denom).item()

    logger.info(f"primal vs baseline: max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} "
                f"rel(mean/|base|)={rel:.3e}  base|mean|={denom.item():.3e}")
    logger.info(f"t_out (zero-tangent) max_abs={t_out.float().abs().max().item():.3e} (should be ~0)")

    tol = 5e-2 if dtype != torch.float32 else 1e-3
    if rel < tol:
        logger.success(f"DUAL-BRANCH PRIMAL PARITY PASSED (rel {rel:.3e} < tol {tol:.1e})")
    else:
        logger.error(f"DUAL-BRANCH PRIMAL PARITY FAILED (rel {rel:.3e} >= tol {tol:.1e})")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
