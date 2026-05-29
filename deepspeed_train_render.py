from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import time
from loguru import logger

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage 
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from dataset.RealEstate10K_render import RealEstate10K_render as RealEstate10K
from voyager.text_encoder import TextEncoder
from voyager.config import *
from voyager.diffusion import load_denoiser
from voyager.diffusion.flow import Transport
from voyager.utils.train_utils import set_reproducibility, prepare_model_inputs, get_training_output_dir, get_synchronized_training_output_dir
from voyager.utils.file_utils import save_videos_grid
from voyager.inference import load_models
from voyager.modules.lora_layers import (
    apply_lora_to_hunyuan_video,
    get_lora_parameters,
    get_lora_state_dict,
    load_lora_state_dict,
    set_active_lora_adapter,
    LoRALinear,
)
from voyager.diffusion.dmd2 import (
    DMD2Config,
    dmd2_config_from_args,
    uniform_timestep_grid,
    make_xt_linear_reverse,
    compute_x_hat_0_from_velocity,
    compute_dmd2_generator_loss,
    compute_fake_score_loss,
    attach_disc_head,
    get_disc_head,
    get_disc_parameters,
    get_disc_state_dict,
    load_disc_state_dict,
    hinge_d_loss,
    hinge_g_loss,
)
from voyager.diffusion.dmd2.dmd2_loss import sample_t_p as _sample_t_p
from voyager.modules.custom_patch_embed import apply_patch_adapter_to_hunyuan_video, get_patch_adapter_parameters, get_patch_adapter_state_dict, load_patch_adapter_state_dict
from voyager.modules.multi_kernel import apply_multikernel_to_hunyuan_video, get_multikernel_parameters, get_multikernel_state_dict, load_multikernel_state_dict
from voyager.modules.double_branch import apply_double_branch_to_hunyuan_video, get_double_branch_parameters, get_double_branch_state_dict, load_double_branch_state_dict, TransformerBranchConfig
from voyager.modules.models import HUNYUAN_VIDEO_CONFIG
from voyager.constants import PRECISION_TO_TYPE
from voyager.cache.text_cache import TextEncoderCache
from voyager.cache.model_input_cache import ModelInputCache
from gather_realestate import norm_partial_render_output
from utils.render import Camera, Frame
from utils.tensor import is_tensor_valid, check_nan_inf
from utils.misc import setup_logger, print_rank0, get_rank
from voyager.utils.helpers import as_list_of_3tuple

def build_optimizer(params, args, ds_config: dict):
    # Detect optimizer offload from ds_config
    offload = (
        ds_config.get("zero_optimization", {})
                 .get("offload_optimizer", {})
                 .get("device", "none")
    )
    offload_to_cpu = (str(offload).lower() == "cpu")

    if offload_to_cpu:
        # Optimizer states + step happen on CPU
        return DeepSpeedCPUAdam(
            params,
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )

    # No offload: use a GPU optimizer
    # Prefer FusedAdam if available; fall back to AdamW
    try:
        return FusedAdam(
            params,
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )
    except Exception:
        return torch.optim.AdamW(
            params,
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )

def parse_arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=str, required=True, help='Path to RealEstate10K dataset root')
    parser.add_argument('--width', type=int, default=256, help='Width of images to load')
    parser.add_argument('--height', type=int, default=384, help='Height of images to load')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--use-cache-text-encoder', action='store_true', help='Whether to cache text encoder outputs')
    parser.add_argument('--placeholder-row-length', type=int, default=64, help='Placeholder row length')
    parser.add_argument('--train-lora', action='store_true', help='Whether to train LoRA adapters')
    parser.add_argument('--use-model-input-cache', action='store_true', help='Whether to use cached model inputs')
    parser.add_argument(
        '--resume-lora',
        type=str,
        default=None,
        help='Path to a LoRA checkpoint (.pt) to resume from',
    )
    parser.add_argument(
        '--resume-multi-kernel',
        type=str,
        default=None,
        help='Path to a multi-kernel checkpoint (.pt) to resume from',
    )
    parser.add_argument(
        '--resume-double-branch',
        type=str,
        default=None,
        help='Path to a double-branch checkpoint (.pt) to resume from',
    )
    parser.add_argument(
        '--resume-patch-adapter',
        type=str,
        default=None,
        help='Path to a patch adapter checkpoint (.pt) to resume from'
    )
    parser.add_argument(
        '--save-every',
        type=int,
        default=1000,
        help='Save checkpoint every N effective steps (after grad accumulation).',
    )
    parser.add_argument(
        '--backup-every',
        type=int,
        default=2000,
        help='Backup checkpoint every N effective steps (after grad accumulation).'
    )
    parser.add_argument(
        "--patch_adapter_size",
        type=int,
        nargs="+",
        default=None,
        help="New patch size for PatchEmbed/FinalLayer (pt ph pw). "
            "If None, use pretrained patch size.",
    )
    parser.add_argument(
        '--num-workers-render',
        type=int,
        default=10,
        help='Number of worker threads for rendering',
    )
    parser.add_argument(
        '--train-multiple-kernels',
        action='store_true',
        help='Whether to train multiple patchify kernels',
    )
    parser.add_argument(
        "--kernel-sizes",
        action="append",      # collect multiple occurrences
        nargs="+",            # accept multiple ints per occurrence
        type=int
    )
    parser.add_argument(
        "--kernel-indices",
        action="append",      # collect multiple occurrences
        nargs="+",            # accept multiple ints per occurrence
        type=int
    )
    parser.add_argument(
        "--model-input-cache-name",
        type=str,
        default="model_input_debug",
    )
    parser.add_argument(
        "--resume-step-in-ckpt",
        action="store_true",
        help="Whether to resume step_in_epoch from checkpoint.",
    )
    
    parser = add_inference_args(parser)
    parser = add_training_args(parser)
    parser = add_optimizer_args(parser)
    parser = add_deepspeed_args(parser)
    parser = add_data_args(parser)
    parser = add_train_denoise_schedule_args(parser)
    parser = add_network_args(parser)
    parser = add_i2v_args(parser)
    parser = add_extra_models_args(parser)
    parser = add_denoise_schedule_args(parser)
    parser = add_lora_args(parser)
    parser = add_patch_adapter_args(parser)
    parser = add_multiple_kernel_args(parser)
    parser = add_double_branch_args(parser)
    parser = add_dmd2_args(parser)

    args = parser.parse_args()
    args.kernel_sizes = as_list_of_3tuple(args.kernel_sizes) if args.kernel_sizes is not None else None
    if args.use_double_branch:
        assert "second_branch_transformer_config" in HUNYUAN_VIDEO_CONFIG[args.model_with_double_branch], \
            f"Model {args.model_with_double_branch} does not support double-branch architecture, no second_branch_transformer_config found."
        assert "second_branch_mm_blocks_depth" in HUNYUAN_VIDEO_CONFIG[args.model_with_double_branch], \
            f"Model {args.model_with_double_branch} does not support double-branch architecture, no second_branch_mm_blocks_depth found."
        assert isinstance(HUNYUAN_VIDEO_CONFIG[args.model_with_double_branch]["second_branch_transformer_config"], TransformerBranchConfig), f"Model {args.model_with_double_branch} does not have valid second_branch_transformer_config."
        args.second_branch_transformer_config = HUNYUAN_VIDEO_CONFIG[args.model_with_double_branch]["second_branch_transformer_config"]
        args.second_branch_mm_blocks_depth = HUNYUAN_VIDEO_CONFIG[args.model_with_double_branch]["second_branch_mm_blocks_depth"]
        logger.info(f"Using double branch model {args.model_with_double_branch} with double branch config {args.second_branch_transformer_config}")
    return args

def training_step(
    x1,
    cond_latents,
    model_engine,
    denoiser: Transport,
    args,
    model_kwargs: Optional[dict] = None,
    partial_cond=None,
    partial_mask=None,
    return_all_terms: bool = False,
) -> float:
    x1 = x1.to(model_engine.device)

    model_engine.train()
    target_dtype = PRECISION_TO_TYPE[args.precision]
    autocast_enabled = (target_dtype != torch.float32) and not args.disable_autocast

    logger.info("Starting training step...")
    start_training_time = time.time()

    with torch.autocast(
        device_type="cuda",
        dtype=model_engine.module.dtype,
        enabled=autocast_enabled,
    ):
        model_output, terms = denoiser.training_losses(
            model=model_engine.module,
            x1=x1,
            model_kwargs=model_kwargs,
            timestep=None,
            n_tokens=None,
            i2v_mode=args.i2v_mode,
            cond_latents=cond_latents,
            args=args,
            partial_cond=partial_cond,
            partial_mask=partial_mask,
            t_range=args.sample_time_range if hasattr(args, "sample_time_range") else None,
        )
    

    loss = terms["loss"].mean()
    logger.info(f"Computed loss: {loss.item()} in {time.time() - start_training_time:.2f} seconds, starting backward pass...")

    backward_start_time = time.time()

    # DeepSpeed handles backward + optimizer
    model_engine.backward(loss)
    model_engine.step()

    logger.info(f"Completed backward pass and optimizer step in {time.time() - backward_start_time:.2f} seconds.")

    # with deepspeed.zero.GatheredParameters(model_engine.module.parameters(), modifier_rank=0):
    #     if dist.get_rank() == 0:
    #         # assert all params are not nan and inf
    #         for name, param in model_engine.module.named_parameters():
    #             if param.requires_grad:
    #                 if torch.isnan(param).any() or torch.isinf(param).any():
    #                     raise ValueError(f"Parameter {name} has NaN or Inf values!")

    if not return_all_terms:
        return loss.item()
    else:
        return loss.item(), terms["input_t"]


def load_rgbs_depths(rgbs, depths, placeholder_row_length):
    # rgbs should be (B, 3, T, H, W), values in [0, 1]
    # depths should be (B, 1, T, H, W), values in [~0, ~1]
    B, _, _, H, W = rgbs.shape
    # 1) Normalize rgbs to [-1, 1]
    rgbs = (rgbs - 0.5) / 0.5   # (B, 3, T, H, W)
    # 2) Broadcast depths to 3 channels and normalize
    depths = depths.repeat(1, 3, 1, 1, 1)   # (B, 3, T, H, W)
    depths = (depths - 0.5) / 0.5
    return torch.cat([rgbs, torch.ones_like(rgbs)[:, :, :, :placeholder_row_length, :], depths], dim=3)

def load_masks(masks, placeholder_row_length):
    # masks should be (B, 1, T, H, W), values in [0, 255]
    B, _, _, H, W = masks.shape
    masks = masks.repeat(1, 3, 1, 1, 1) / 255.0  # (B, 3, T, H, W), values in [0, 1]
    masks = (masks - 0.5) / 0.5  # masks now in [-1, 1]    
    return torch.cat([masks, torch.ones_like(masks)[:, :, :, :placeholder_row_length, :], masks], dim=3)

def _render_single(args):
    first_frame, frame, partial_rendering = args
    if not partial_rendering:
        rendered_image, mask, depth_buffer = frame.render(frame.camera)
    else:
        rendered_image, mask, depth_buffer = first_frame.render(frame.camera)

    rendered_image, mask, depth_buffer = norm_partial_render_output(
        rendered_image, mask, depth_buffer
    )
    return rendered_image, mask, depth_buffer


def render(frames, shape, args, partial_rendering=False):
    B, _, T, H, W = shape

    # Build job list: one job per frame
    jobs = []
    for b in range(B):
        first_frame = frames[b][0]
        for i in range(T):
            frame = frames[b][i]
            jobs.append((first_frame, frame, partial_rendering))

    # Run in parallel
    max_workers = min(args.num_workers_render, B * T)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_render_single, jobs))

    # Unpack back into tensors
    norm_rgbs = []
    norm_depths = []
    masks = []
    for rendered_image, mask, depth_buffer in results:
        # rendered_image: H x W x 3 (uint8 or float), depth_buffer: H x W, mask: H x W
        img_np = np.asarray(rendered_image, dtype=np.float32) / 255.0  # (H, W, 3)
        norm_rgbs.append(torch.from_numpy(img_np).permute(2, 0, 1))    # (3, H, W)

        depth_np = np.asarray(depth_buffer, dtype=np.float32)          # (H, W)
        norm_depths.append(torch.from_numpy(depth_np).unsqueeze(0))    # (1, H, W)

        mask_np = np.asarray(mask, dtype=np.float32)                   # (H, W)
        masks.append(torch.from_numpy(mask_np).unsqueeze(0))           # (1, H, W)

    # Now shape them back to (B, 3/1, T, H, W)
    norm_rgbs = torch.stack(norm_rgbs, dim=0).reshape(B, T, 3, H, W).permute(0, 2, 1, 3, 4)
    norm_depths = torch.stack(norm_depths, dim=0).reshape(B, T, 1, H, W).permute(0, 2, 1, 3, 4)
    masks = torch.stack(masks, dim=0).reshape(B, T, 1, H, W).permute(0, 2, 1, 3, 4)

    rgb_depths = load_rgbs_depths(norm_rgbs, norm_depths, args.placeholder_row_length).to(args.device)
    masks = load_masks(masks, args.placeholder_row_length).to(args.device)

    return rgb_depths, masks

def save_full_model_checkpoint(model_engine, args, epoch, step, training_output_dir):
    # Gather full (unsharded) model weights on rank 0
    # NOTE: modifier_rank=0 means rank 0 owns the gathered tensors
    with deepspeed.zero.GatheredParameters(model_engine.module.parameters(), modifier_rank=0):
        if dist.get_rank() == 0:
            # Now the underlying tensors are full-sized on rank 0,
            # so state_dict() will contain non-empty tensors.
            full_model_sd = model_engine.module.state_dict()

            ckpt = {
                "epoch": epoch,
                "step_in_epoch": step,
                "model": full_model_sd,
                "args": vars(args),
            }

            ckpt_path = Path(training_output_dir) / f"model_{args.model.replace('/', '_')}_last.pt"
            logger.info(f"Saving full model checkpoint to {ckpt_path}")
            torch.save(ckpt, ckpt_path)

def save_lora_checkpoint(model_engine, args, epoch, step, training_output_dir):
    # Collect the LoRA params list (same as you pass to optimizer)
    lora_params = get_lora_parameters(model_engine.module)

    # Gather full (unsharded) LoRA weights on rank 0
    # NOTE: modifier_rank=0 means rank 0 owns the gathered tensors
    with deepspeed.zero.GatheredParameters(lora_params, modifier_rank=0):
        if dist.get_rank() == 0:
            # Now the underlying tensors are full-sized on rank 0,
            # so state_dict() will contain non-empty tensors.
            lora_sd = get_lora_state_dict(model_engine.module)

            ckpt = {
                "epoch": epoch,
                "step_in_epoch": step,
                "lora": lora_sd,
                "args": vars(args),
            }

            ckpt_path = Path(training_output_dir) / "lora_last.pt"
            logger.info(f"Saving LoRA checkpoint to {ckpt_path}")
            torch.save(ckpt, ckpt_path)


def save_dmd2_lora_checkpoint(model_engine, args, epoch, step, training_output_dir, adapter_name: str, filename: str):
    """Save a single named LoRA adapter (e.g. 'gen' or 'fake' for DMD2)."""
    params = get_lora_parameters(model_engine.module, adapter_name=adapter_name)
    with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
        if dist.get_rank() == 0:
            sd = get_lora_state_dict(model_engine.module, adapter_name=adapter_name)
            ckpt = {
                "epoch": epoch,
                "step_in_epoch": step,
                "lora": sd,
                "adapter_name": adapter_name,
                "args": vars(args),
            }
            ckpt_path = Path(training_output_dir) / filename
            logger.info(f"Saving DMD2 {adapter_name} LoRA checkpoint to {ckpt_path}")
            torch.save(ckpt, ckpt_path)


def save_dmd2_disc_checkpoint(model_engine, args, epoch, step, training_output_dir, filename: str = "dmd2_disc_last.pt"):
    """Gather and save the DMD2 discriminator head (no-op if not attached)."""
    disc_params = get_disc_parameters(model_engine.module)
    if not disc_params:
        return
    with deepspeed.zero.GatheredParameters(disc_params, modifier_rank=0):
        if dist.get_rank() == 0:
            sd = get_disc_state_dict(model_engine.module)
            ckpt = {
                "epoch": epoch,
                "step_in_epoch": step,
                "disc": sd,
                "args": vars(args),
            }
            ckpt_path = Path(training_output_dir) / filename
            logger.info(f"Saving DMD2 discriminator checkpoint to {ckpt_path}")
            torch.save(ckpt, ckpt_path)


def save_patch_adapter_checkpoint(model_engine, args, epoch, step, training_output_dir):
    """
    Gather and save only patch-adapter parameters (PatchEmbed + FinalLayer)
    from a ZeRO-sharded DeepSpeed engine.
    """

    patch_params = get_patch_adapter_parameters(model_engine.module)

    # Gather unsharded weights on rank 0 only
    with deepspeed.zero.GatheredParameters(patch_params, modifier_rank=0):
        if dist.get_rank() == 0:
            patch_sd = get_patch_adapter_state_dict(model_engine.module)
            ckpt = {
                "epoch": epoch,
                "step_in_epoch": step,
                "patch_adapter": patch_sd,
                "args": vars(args),
            }
            ckpt_path = Path(training_output_dir) / "patch_adapter_last.pt"
            logger.info(f"Saving PatchAdapter checkpoint to {ckpt_path}")
            torch.save(ckpt, ckpt_path)

def save_multi_kernel_checkpoint(model_engine, args, epoch, step, training_output_dir):
    """
    Gather and save only multi-kernel parameters (MultiPatchEmbed + MultiFinalLayer)
    from a ZeRO-sharded DeepSpeed engine.
    """

    # Collect the multi-kernel params list (same as you pass to optimizer)
    multi_kernel_params = get_multikernel_parameters(model_engine.module)

    # Gather unsharded weights on rank 0 only
    with deepspeed.zero.GatheredParameters(multi_kernel_params, modifier_rank=0):
        if dist.get_rank() == 0:
            multi_kernel_sd = get_multikernel_state_dict(model_engine.module)
            ckpt = {
                "epoch": epoch,
                "step_in_epoch": step,
                "multi_kernel": multi_kernel_sd,
                "args": vars(args),
            }
            ckpt_path = Path(training_output_dir) / "multi_kernel_last.pt"
            logger.info(f"Saving Multi-Kernel checkpoint to {ckpt_path}")
            torch.save(ckpt, ckpt_path)

def save_double_branch_checkpoint(model_engine, args, epoch, step, training_output_dir):
    """
    Gather and save only 2-branch parameters
    from a ZeRO-sharded DeepSpeed engine.
    """

    # Collect the 2-branch params list (same as you pass to optimizer)
    branch_params = get_double_branch_parameters(model_engine.module)

    # Gather unsharded weights on rank 0 only
    with deepspeed.zero.GatheredParameters(branch_params, modifier_rank=0):
        if dist.get_rank() == 0:
            branch_sd = get_double_branch_state_dict(model_engine.module)
            ckpt = {
                "epoch": epoch,
                "step_in_epoch": step,
                "double_branch": branch_sd,
                "args": vars(args),
            }
            ckpt_path = Path(training_output_dir) / "double_branch_last.pt"
            logger.info(f"Saving Double Branch checkpoint to {ckpt_path}")
            torch.save(ckpt, ckpt_path)

def make_dmd2_forward_v(model_engine, denoiser, args, *, cond_latents, partial_cond, partial_mask, model_kwargs):
    """
    Build a closure used by the DMD2 loss functions to issue a single model
    forward, abstracting away the adapter routing, the i2v conditioning concat,
    and the timestep-to-model-input-scale conversion.

    Signature: forward_v(adapter_name_or_None, x_t_data, t) -> v_pred
        - adapter_name_or_None: pass None for the base (LoRA-off) "real" score.
        - x_t_data:             (B, latent_channels, T, H', W')
        - t:                    (B,) in [0, 1]
    Returns:
        v_pred over the *data* channels with the same shape as x_t_data.
    """

    def _forward_v(adapter, x_t_data, t):
        set_active_lora_adapter(model_engine.module, adapter)

        # i2v latent_concat: concat [x_t, first-frame-only cond, first-frame-only mask]
        # along the channel dim. Mirrors voyager/diffusion/flow/transport.py:training_losses.
        if args.i2v_mode and args.i2v_condition_type == "latent_concat":
            B_, _, T_, H_, W_ = x_t_data.shape
            x1_concat = cond_latents.repeat(1, 1, T_, 1, 1).clone()
            x1_concat[:, :, 1:, :, :] = 0.0
            mask_concat = torch.ones(
                B_, 1, T_, H_, W_,
                device=x_t_data.device, dtype=x_t_data.dtype,
            )
            mask_concat[:, :, 1:, ...] = 0.0
            xt = torch.cat([x_t_data, x1_concat, mask_concat], dim=1)
        elif args.i2v_mode and args.i2v_condition_type == "token_replace":
            xt = torch.cat([cond_latents, x_t_data[:, :, 1:, :, :]], dim=2)
        else:
            xt = x_t_data

        if partial_cond is not None and partial_mask is not None:
            xt = torch.cat([xt, partial_cond, partial_mask], dim=1)

        input_t = denoiser.get_model_t(t).to(x_t_data.device)
        xt = xt.to(model_engine.module.dtype)

        out = model_engine(xt, input_t, **model_kwargs)["x"]

        if args.i2v_mode and args.i2v_condition_type == "token_replace":
            out = out[:, :, 1:, :, :]

        return out

    return _forward_v


def _sample_t_i_from_grid(batch_size: int, t_grid: torch.Tensor, device) -> torch.Tensor:
    """One uniformly-sampled t_i per batch element from the DMD2 grid."""
    idx = torch.randint(0, t_grid.shape[0], (batch_size,), device=device)
    return t_grid[idx].to(torch.float32)


def _compute_lora_norms_per_adapter(model_engine, adapter_names=("gen", "fake")):
    """
    Diagnostic for hypothesis (A): if weight-decay on the DeepSpeed optimizer is
    leaking onto the inactive adapter every step, both adapters' A norms should
    drift downward monotonically (B starts at zero so it can only grow).

    Returns: {adapter_name: {"A_norm": float, "B_norm": float, "A_count": int,
              "B_count": int}}. With ZeRO stage 2 the full param tensors live on
    every rank, so no GatheredParameters wrapping is needed.
    """
    a_sq = {n: 0.0 for n in adapter_names}
    b_sq = {n: 0.0 for n in adapter_names}
    a_count = {n: 0 for n in adapter_names}
    b_count = {n: 0 for n in adapter_names}
    for module in model_engine.module.modules():
        if isinstance(module, LoRALinear):
            for name, branch in module.adapters.items():
                if name not in a_sq:
                    continue
                if branch.lora_A is not None:
                    a_sq[name] += branch.lora_A.detach().float().pow(2).sum().item()
                    a_count[name] += branch.lora_A.numel()
                if branch.lora_B is not None:
                    b_sq[name] += branch.lora_B.detach().float().pow(2).sum().item()
                    b_count[name] += branch.lora_B.numel()
    return {
        n: {
            "A_norm": float(a_sq[n] ** 0.5),
            "B_norm": float(b_sq[n] ** 0.5),
            "A_count": a_count[n],
            "B_count": b_count[n],
        }
        for n in adapter_names
    }


def _grad_cosine_fake_mse_vs_gan(fake_loss, gan_d_loss_weighted, fake_lora_params):
    """
    Diagnostic for hypothesis (B): inside the GAN-enabled fake step the fake
    LoRA receives gradients from BOTH (1) fake_loss = MSE(v_pred_fake, v_target)
    and (2) λ_D · d_loss. If those two pull in opposing directions on the SAME
    parameters, the cosine of the two gradient vectors will be negative and the
    fake-LoRA's MSE objective is being broken by the discriminator term.

    Uses torch.autograd.grad with retain_graph=True so the regular
    model_engine.backward(total_loss) below remains valid. Cost: roughly two
    extra backward recomputes through the gradient-checkpointed graph — gate
    this at the call site so it only runs for a short prefix of outer steps.

    Returns (cosine, fake_grad_norm, gan_grad_norm) as floats. Returns
    (nan, nan, nan) if either gradient is empty/None.
    """
    grads_mse = torch.autograd.grad(
        fake_loss, fake_lora_params,
        retain_graph=True, allow_unused=True, create_graph=False,
    )
    grads_gan = torch.autograd.grad(
        gan_d_loss_weighted, fake_lora_params,
        retain_graph=True, allow_unused=True, create_graph=False,
    )
    flat_mse = [g.detach().float().flatten() for g in grads_mse if g is not None]
    flat_gan = [g.detach().float().flatten() for g in grads_gan if g is not None]
    if not flat_mse or not flat_gan:
        return float("nan"), float("nan"), float("nan")
    flat_mse_t = torch.cat(flat_mse)
    flat_gan_t = torch.cat(flat_gan)
    n_mse = flat_mse_t.norm()
    n_gan = flat_gan_t.norm()
    denom = (n_mse * n_gan).clamp(min=1e-12)
    cos = (flat_mse_t * flat_gan_t).sum() / denom
    return float(cos.item()), float(n_mse.item()), float(n_gan.item())


def dmd2_fake_step(model_engine, denoiser, args, dmd2_cfg, t_grid, latents, forward_v, *, disc_head=None, fake_lora_params=None, log_grad_cosine=False):
    """
    One fake-score update. If `disc_head` is provided and `dmd2_cfg.use_gan`,
    the hinge-D loss against (clean real x_data vs. generator x_hat_0) is added
    to the same backward.

    Returns:
        (fake_loss_scalar, info_dict). `info_dict` contains the standard
        fake-score diagnostics plus, when GAN is on, `d_loss`, `d_logits_real`,
        `d_logits_fake`.
    """
    model_engine.train()
    B = latents.shape[0]
    t_i = _sample_t_i_from_grid(B, t_grid, latents.device)

    with torch.no_grad():
        noise_i = torch.randn_like(latents)
        x_t_in = make_xt_linear_reverse(latents, noise_i, t_i)
        v_pred_gen = forward_v(dmd2_cfg.gen_adapter_name, x_t_in, t_i)
        x_hat_0 = compute_x_hat_0_from_velocity(x_t_in, v_pred_gen, t_i).detach()

    do_gan = dmd2_cfg.use_gan and disc_head is not None and dmd2_cfg.gan_weight_d > 0

    target_dtype = PRECISION_TO_TYPE[args.precision]
    autocast_enabled = (target_dtype != torch.float32) and not args.disable_autocast

    if not do_gan:
        # Original code path, untouched.
        with torch.autocast(device_type="cuda", dtype=model_engine.module.dtype, enabled=autocast_enabled):
            fake_loss, fake_info = compute_fake_score_loss(
                x_hat_0,
                forward_v,
                min_tp=dmd2_cfg.min_tp,
                max_tp=dmd2_cfg.max_tp,
                fake_adapter=dmd2_cfg.fake_adapter_name,
            )
        model_engine.backward(fake_loss)
        model_engine.step()
        return float(fake_loss.detach().item()), fake_info

    # --- GAN-enabled fake step ----------------------------------------------
    # Share a single t_p across the fake and real branches (matches DMD2 paper).
    t_p = _sample_t_p(B, dmd2_cfg.min_tp, dmd2_cfg.max_tp, device=latents.device, dtype=torch.float32)

    # Fake branch: perturb the generator's clean prediction.
    noise_p_fake = torch.randn_like(x_hat_0)
    x_tp_fake = make_xt_linear_reverse(x_hat_0, noise_p_fake, t_p)
    v_target_fake = noise_p_fake - x_hat_0           # linear-reverse velocity target

    # Real branch: perturb the ground-truth clean data with an independent noise draw.
    noise_p_real = torch.randn_like(latents)
    x_tp_real = make_xt_linear_reverse(latents, noise_p_real, t_p)

    with torch.autocast(device_type="cuda", dtype=model_engine.module.dtype, enabled=autocast_enabled):
        # The fake-score velocity ALSO serves as the D feature map.
        v_pred_fake = forward_v(dmd2_cfg.fake_adapter_name, x_tp_fake, t_p)
        v_pred_real = forward_v(dmd2_cfg.fake_adapter_name, x_tp_real, t_p)

        fake_loss = (v_pred_fake - v_target_fake.to(v_pred_fake.dtype)).pow(2).mean()

        logits_fake = disc_head(v_pred_fake)
        logits_real = disc_head(v_pred_real)
        d_loss = hinge_d_loss(logits_real, logits_fake)

        total_loss = fake_loss + dmd2_cfg.gan_weight_d * d_loss

    # Both forwards used the fake adapter -> active stays "fake" for backward
    # recomputes under gradient checkpointing. (Defensive re-set in case some
    # outer code mutated it.)
    set_active_lora_adapter(model_engine.module, dmd2_cfg.fake_adapter_name)

    # ----- Hypothesis (B) diagnostic ----------------------------------------
    # Measure whether the MSE term and the GAN-D term pull the fake LoRA's
    # parameters in opposing directions BEFORE running the real backward.
    # Uses torch.autograd.grad so .grad on the params is untouched; the
    # subsequent model_engine.backward(total_loss) still produces the correct
    # accumulated grad for optimizer.step().
    grad_cosine_info = {}
    if log_grad_cosine and fake_lora_params is not None and len(fake_lora_params) > 0:
        cos, gnorm_mse, gnorm_gan = _grad_cosine_fake_mse_vs_gan(
            fake_loss,
            dmd2_cfg.gan_weight_d * d_loss,
            fake_lora_params,
        )
        grad_cosine_info = {
            "fake_grad_cosine_mse_vs_gan": cos,
            "fake_grad_norm_mse": gnorm_mse,
            "fake_grad_norm_gan": gnorm_gan,
        }
        # The two autograd.grad calls above re-entered the gradient-checkpoint
        # recompute path; defensively pin the active adapter back to "fake"
        # in case any forward inside that path mutated it.
        set_active_lora_adapter(model_engine.module, dmd2_cfg.fake_adapter_name)

    model_engine.backward(total_loss)
    model_engine.step()

    fake_info = {
        "t_p_mean": float(t_p.mean().item()),
        "v_target_norm": float(v_target_fake.flatten(1).norm(dim=1).mean().item()),
        "d_loss": float(d_loss.detach().item()),
        "d_logits_real": float(logits_real.detach().mean().item()),
        "d_logits_fake": float(logits_fake.detach().mean().item()),
    }
    fake_info.update(grad_cosine_info)
    return float(fake_loss.detach().item()), fake_info


def dmd2_gen_step(
    model_engine,
    denoiser,
    args,
    dmd2_cfg,
    t_grid,
    latents,
    forward_v,
    *,
    disc_head=None,
    do_gan_g: bool = False,
    fake_lora_params=None,
    disc_params=None,
):
    """
    One generator update. Returns the surrogate-loss scalar.

    GAN extension:
      When `do_gan_g` is True, an extra differentiable forward through the fake
      adapter on `x_hat_0` (perturbed at an independent t_p) is added; D's logits
      on that fake feature drive a hinge-G loss `-mean(D(fake))` that is summed
      into the surrogate before backward. To avoid the fake-LoRA / D-head params
      collecting gradients that the engine's `.step()` would mistakenly apply,
      we temporarily set `requires_grad=False` on them around this block —
      activations still flow, so the gradient still reaches x_hat_0 and on to
      the gen params, but the fake/D params do not accumulate grads.

      Gradient-checkpoint hazard:
        The gen forward runs with active="gen". `compute_dmd2_generator_loss`
        switches active to "fake" for its no_grad v_real/v_fake forwards. The
        GAN forward then re-routes through active="fake" (checkpointed). At
        backward, both the gen forward AND the GAN forward are recomputed, but
        they need DIFFERENT active_adapter values during their respective
        recomputes. We solve this by leaving active="fake" entering backward
        (so the GAN forward's recompute is correct) and registering a hook on
        x_hat_0 that flips active to "gen" the moment the autograd engine has
        aggregated all gradients into x_hat_0 — i.e. exactly at the topological
        boundary between the GAN sub-graph and the gen sub-graph.
    """
    model_engine.train()
    B = latents.shape[0]
    t_i = _sample_t_i_from_grid(B, t_grid, latents.device)

    target_dtype = PRECISION_TO_TYPE[args.precision]
    autocast_enabled = (target_dtype != torch.float32) and not args.disable_autocast

    # Freeze fake-LoRA + D-head params for the duration of the gen step's
    # differentiable forward(s). Done OUTSIDE autocast so the toggle is
    # observable when the autograd graph is constructed.
    frozen_params = []
    if do_gan_g:
        frozen_params = list(fake_lora_params or []) + list(disc_params or [])
        for p in frozen_params:
            p.requires_grad = False

    try:
        with torch.autocast(device_type="cuda", dtype=model_engine.module.dtype, enabled=autocast_enabled):
            noise = torch.randn_like(latents)
            x_t_in = make_xt_linear_reverse(latents, noise, t_i)
            v_pred = forward_v(dmd2_cfg.gen_adapter_name, x_t_in, t_i)
            x_hat_0 = compute_x_hat_0_from_velocity(x_t_in, v_pred, t_i)

            gen_loss, gen_info = compute_dmd2_generator_loss(
                x_hat_0,
                forward_v,
                min_tp=dmd2_cfg.min_tp,
                max_tp=dmd2_cfg.max_tp,
                weight_mode=dmd2_cfg.weight_mode,
                real_adapter=None,
                fake_adapter=dmd2_cfg.fake_adapter_name,
            )

            if do_gan_g:
                # Independent t_p draw for the GAN-G term. Reusing the surrogate's
                # t_p would conflate two different roles of the perturbation and
                # add correlation we don't want.
                t_p_g = _sample_t_p(B, dmd2_cfg.min_tp, dmd2_cfg.max_tp, device=latents.device, dtype=torch.float32)
                noise_g = torch.randn_like(x_hat_0)
                x_tp_g = make_xt_linear_reverse(x_hat_0, noise_g, t_p_g)
                v_pred_fake_g = forward_v(dmd2_cfg.fake_adapter_name, x_tp_g, t_p_g)
                logits_fake_g = disc_head(v_pred_fake_g)
                gan_g_loss = hinge_g_loss(logits_fake_g)
                gen_info["g_gan_loss"] = float(gan_g_loss.detach().item())
                gen_info["g_d_logits_fake_mean"] = float(logits_fake_g.detach().mean().item())
                total_loss = gen_loss + dmd2_cfg.gan_weight_g * gan_g_loss
            else:
                total_loss = gen_loss

        if do_gan_g:
            # Register the active-adapter flip hook on x_hat_0. Fires once the
            # autograd engine has aggregated both incoming gradient contributions
            # (from gen_loss and gan_g_loss) at x_hat_0, BEFORE backward proceeds
            # into x_hat_0's predecessors (the gen forward, which needs
            # active="gen" at recompute time).
            _adapter_name_gen = dmd2_cfg.gen_adapter_name
            _module_for_hook = model_engine.module
            def _flip_active_to_gen(_grad, _m=_module_for_hook, _n=_adapter_name_gen):
                set_active_lora_adapter(_m, _n)
                return None  # do not modify the gradient
            x_hat_0.register_hook(_flip_active_to_gen)

            # Leave active="fake" entering backward: the GAN forward's blocks
            # are the FIRST to be re-run (autograd is LIFO over the forward
            # graph), and they need active="fake". The hook then flips to "gen"
            # at the x_hat_0 boundary so the gen forward's recompute is correct.
            set_active_lora_adapter(model_engine.module, dmd2_cfg.fake_adapter_name)
        else:
            # Existing non-GAN path: compute_dmd2_generator_loss leaves active
            # on "fake" as a side effect of its no_grad real/fake forwards.
            # Reset to "gen" so the gen-forward recompute under gradient
            # checkpointing routes through the gen adapter.
            set_active_lora_adapter(model_engine.module, dmd2_cfg.gen_adapter_name)
            for _m in model_engine.module.modules():
                if isinstance(_m, LoRALinear) and _m.active_adapter != dmd2_cfg.gen_adapter_name:
                    raise RuntimeError(
                        f"LoRALinear active_adapter={_m.active_adapter!r}, expected "
                        f"{dmd2_cfg.gen_adapter_name!r} just before backward. A new "
                        "adapter mutation snuck in between the gen loss and the "
                        "backward call; gradient-checkpoint recomputation would "
                        "route through the wrong LoRA branch."
                    )

        model_engine.backward(total_loss)

        if do_gan_g:
            # Sanity: after backward, the frozen params must NOT have grads.
            # If they do, optimizer.step() would corrupt the fake-score head /
            # D head with a sign meant for the generator.
            bad = []
            for p in frozen_params:
                if p.grad is not None and p.grad.abs().sum().item() > 0:
                    bad.append(tuple(p.shape))
            if bad:
                raise RuntimeError(
                    f"DMD2 gen step: {len(bad)} frozen params accumulated grads "
                    f"({bad[:3]}...). requires_grad toggle did not prevent grad flow."
                )

        model_engine.step()
    finally:
        # Restore requires_grad. Note: model_engine.step() above clears grads,
        # so re-enabling here is safe; the next fake step expects True.
        for p in frozen_params:
            p.requires_grad = True

    return float(gen_loss.detach().item()), gen_info


def save_ckpt(args, model_engine, epoch, step, training_output_dir, total_skip):
    logger.warning(f"Skipped {total_skip} batches due to NaN/Inf so far.")
    if args.train_from_scratch:
        ckpt_path = Path(training_output_dir) / f"model_{args.model}_last.pt"
        logger.info(f"Saving full model checkpoint to {ckpt_path}")
        save_full_model_checkpoint(model_engine, args, epoch, step, training_output_dir)
    if getattr(args, "dmd2_steps", 0) > 0 and args.train_lora:
        # In DMD2 mode we save BOTH adapters separately and skip the legacy
        # single-adapter save (which would mix gen+fake into one file).
        save_dmd2_lora_checkpoint(model_engine, args, epoch, step, training_output_dir, "gen", "lora_gen_last.pt")
        save_dmd2_lora_checkpoint(model_engine, args, epoch, step, training_output_dir, "fake", "lora_fake_last.pt")
        save_dmd2_disc_checkpoint(model_engine, args, epoch, step, training_output_dir)
    elif args.train_lora:
        lora_ckpt_path = Path(training_output_dir) / f"lora_last.pt"
        logger.info(f"Saving LoRA checkpoint to {lora_ckpt_path}")
        save_lora_checkpoint(model_engine, args, epoch, step, training_output_dir)
    if args.patch_adapter_size is not None:
        patch_adapter_ckpt_path = Path(training_output_dir) / f"patch_adapter_last.pt"
        logger.info(f"Saving PatchAdapter checkpoint to {patch_adapter_ckpt_path}")
        save_patch_adapter_checkpoint(model_engine, args, epoch, step, training_output_dir)
    if args.train_multiple_kernels:
        multi_kernel_ckpt_path = Path(training_output_dir) / f"multi_kernel_last.pt"
        logger.info(f"Saving Multi-Kernel checkpoint to {multi_kernel_ckpt_path}")
        save_multi_kernel_checkpoint(model_engine, args, epoch, step, training_output_dir)
    if args.use_double_branch:
        double_branch_ckpt_path = Path(training_output_dir) / f"double_branch_last.pt"
        logger.info(f"Saving Double Branch checkpoint to {double_branch_ckpt_path}")
        save_double_branch_checkpoint(model_engine, args, epoch, step, training_output_dir)

if __name__ == "__main__":
    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    setup_logger(enabled=True)
    args = parse_arg()
    args.device = f"cuda:{local_rank}"
    logger.info(args)
    global_rank = dist.get_rank()

    #===================INITIALIZATION===================#

    denoiser = load_denoiser(args) 
    dtype = PRECISION_TO_TYPE[args.precision]
    set_reproducibility(True, args.global_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    training_output_dir = get_synchronized_training_output_dir(args.output_dir)
    os.makedirs(training_output_dir, exist_ok=True)
    deepspeed_config = args.deepspeed_config if hasattr(args, "deepspeed_config") else "ds_config.json"
    with open(deepspeed_config, 'r') as f:
        ds_config = json.load(f)


    if global_rank == 0:
        from copy import copy
        tmp_args = copy(args)
        if args.use_double_branch:
             tmp_args.second_branch_transformer_config = tmp_args.second_branch_transformer_config.to_dict()
        
        config = {
            'time': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            'output_dir': str(training_output_dir),
            'model_args': vars(tmp_args),
            'deepspeed_config': ds_config,
        }
        with open(Path(training_output_dir) / 'training_config.json', 'w') as f:
            json.dump(config, f, indent=4)
    logger.info(f"Training output dir: {training_output_dir}")

    #===================DATASET & DATALOADER===================#

    dataset_root = args.dataset_root
    dataset = RealEstate10K(dataset_root, set_name=args.task_flag, width=args.width, height=args.height, return_inverse_depth=True, gt_conditioning=args.use_gt_as_cond)
    logger.warning(f"USING GT AS CONDITION: {args.use_gt_as_cond}")
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=ds_config["train_micro_batch_size_per_gpu"],
        sampler=sampler,      # <--- Added
        shuffle=False,        # <--- Must be False when using sampler
        num_workers=args.num_workers,
        pin_memory=True       # Recommended for GPU training
    )

    #===================MODEL, OPTIMIZER & DEEPSPEED INIT===================#

    logger.info("Building model...")
    model, vae, text_encoder, text_encoder_2, vae_kwargs = \
        load_models(args, args.device, logger, Path(args.model_base) if args.model_base is not None else None)
    vae.enable_tiling()
    vae.eval()

    #===================CACHE LOADERS===================#

    if not args.use_cache_text_encoder:
        text_encoder.eval()
        text_encoder_2.eval()
        text_encoder_cache = None
    else:
        logger.info("Loading text encoder cache...")
        text_encoder_cache = TextEncoderCache(Path(args.dataset_root) / "cache" / "text_encoder")
    
    if not args.use_model_input_cache:
        model_input_cache = None
    else:
        logger.info("Loading model input cache...")
        model_input_cache = ModelInputCache(Path(args.dataset_root) / "cache" / args.model_input_cache_name)

    #===================MODEL MODIFICATIONS & OPTIMIZER===================#

    dmd2_active = getattr(args, "dmd2_steps", 0) > 0
    if dmd2_active and not args.train_lora:
        raise ValueError("--dmd2-steps requires --train-lora (DMD2 uses LoRA adapters for the generator).")

    if args.train_lora:
        logger.info("Applying LoRA adapters to model...")
        # In DMD2 mode we attach TWO named adapters on top of the same frozen base:
        # "gen"  = the few-step generator being distilled
        # "fake" = the fake-score critic
        if dmd2_active:
            apply_lora_to_hunyuan_video(
                model,
                r=args.lora_rank,
                lora_alpha=getattr(args, "lora_alpha", 16.0),
                lora_dropout=getattr(args, "lora_dropout", 0.0),
                freeze_base=not args.train_from_scratch,
                adapter_name="gen",
            )
            fake_r = args.dmd2_fake_lora_rank or args.lora_rank
            fake_alpha = args.dmd2_fake_lora_alpha if args.dmd2_fake_lora_alpha is not None else getattr(args, "lora_alpha", 16.0)
            apply_lora_to_hunyuan_video(
                model,
                r=fake_r,
                lora_alpha=fake_alpha,
                lora_dropout=getattr(args, "lora_dropout", 0.0),
                freeze_base=False,  # base is already frozen by the gen-adapter call
                adapter_name="fake",
            )
            # Default routing: generator active. The DMD2 inner loop flips this per call.
            set_active_lora_adapter(model, "gen")
            logger.info(f"DMD2 active: applied 'gen' (rank {args.lora_rank}) + 'fake' (rank {fake_r}) LoRA adapters.")

            # GAN extension: attach a per-token discriminator head on the fake-net's
            # final velocity output. in_channels matches the model's latent channel
            # count (so the head is a single small Linear). Trained alongside the
            # fake LoRA inside the fake step; used by both fake and gen steps.
            if getattr(args, "dmd2_use_gan", False):
                disc_in_channels = int(model.out_channels)
                attach_disc_head(
                    model,
                    in_channels=disc_in_channels,
                    dtype=dtype,
                    device=args.device,
                )
                logger.info(
                    f"DMD2 GAN active: attached disc head with in_channels={disc_in_channels}, "
                    f"λ_D={args.dmd2_gan_weight_d}, λ_G={args.dmd2_gan_weight_g}, "
                    f"gan_warmup={args.dmd2_gan_warmup_steps}."
                )
        else:
            apply_lora_to_hunyuan_video(
                model,
                r=args.lora_rank if hasattr(args, "lora_rank") else 8,
                lora_alpha=args.lora_alpha if hasattr(args, "lora_alpha") else 16.0,
                lora_dropout=getattr(args, "lora_dropout", 0.0),
                freeze_base=not args.train_from_scratch
            )
    if args.patch_adapter_size is not None:
        logger.warning(f"Applying Patch Size Adapters with new patch size: {args.patch_adapter_size}")
        apply_patch_adapter_to_hunyuan_video(
            model,
            new_patch_size=tuple(args.patch_adapter_size),
            freeze_base=not args.train_from_scratch,
        )
    if args.train_multiple_kernels:
        if args.kernel_sizes is None or args.kernel_indices is None:
            raise ValueError("When using --train-multiple-kernels, you must also provide --kernel-sizes and --kernel-indices")
        if len(args.kernel_sizes) != len(args.kernel_indices):
            raise ValueError("The number of --kernel-sizes entries must match the number of --kernel-indices entries")
        patch_sizes = [tuple(ks) for ks in args.kernel_sizes]
        kernel_indices = [list(ki) for ki in args.kernel_indices]
        logger.info(f"Applying Multi-Kernel with patch sizes: {patch_sizes} and indices: {kernel_indices}")
        apply_multikernel_to_hunyuan_video(
            model,
            patch_sizes=patch_sizes,
            device=args.device,
            dtype=dtype,
            freeze_base=not args.train_from_scratch,
            copy_old_weights=True,
        )
    if args.use_double_branch:
        logger.info("Applying 2-Branch architecture to model...")
        apply_double_branch_to_hunyuan_video(
            model,
            second_branch_config=args.second_branch_transformer_config,
            second_branch_mm_blocks_depth=args.second_branch_mm_blocks_depth,
            device=args.device,
            dtype=dtype,
            freeze_base=not args.train_from_scratch,
        )

    lora_params = get_lora_parameters(model) if args.train_lora else []
    patch_params = get_patch_adapter_parameters(model) if args.patch_adapter_size is not None else []
    multikernel_params = get_multikernel_parameters(model) if args.train_multiple_kernels else []
    double_branch_params = get_double_branch_parameters(model) if args.use_double_branch else []
    disc_head_params = get_disc_parameters(model) if dmd2_active and getattr(args, "dmd2_use_gan", False) else []
    logger.info(f"Number of LoRA parameters: {sum(p.numel() for p in lora_params)}")
    logger.info(f"Number of Patch Adapter parameters: {sum(p.numel() for p in patch_params)}")
    logger.info(f"Number of Multi-Kernel parameters: {sum(p.numel() for p in multikernel_params)}")
    logger.info(f"Number of Double Branch parameters: {sum(p.numel() for p in double_branch_params)}")
    logger.info(f"Number of DMD2 Discriminator Head parameters: {sum(p.numel() for p in disc_head_params)}")

    trainable_addons_params = lora_params + patch_params + multikernel_params + double_branch_params + disc_head_params
    for p in trainable_addons_params:
        p.requires_grad = True
    
    # get all trainable parameters in model, add into trainable_params
    trainable_params = []
    for p in model.parameters():
        if p.requires_grad:
            trainable_params.append(p)
    
    # log model's trainable params
    total_params_cnt = sum(p.numel() for p in model.parameters())
    trainable_params_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params_cnt}")
    logger.info(f"Trainable parameters: {trainable_params_cnt}")
    optimizer = build_optimizer(trainable_params, args, ds_config)

    runtime_seed = args.global_seed + global_rank
    set_reproducibility(True, runtime_seed)
    logger.warning(f"Rank {global_rank} re-seeded with {runtime_seed} for training loop.")

    #===================RESUME FROM CHECKPOINT IF ANY===================#

    start_epoch = 0
    start_step_in_epoch = -1

    if args.resume_lora is not None and os.path.isfile(args.resume_lora):
        logger.info(f"Resuming LoRA from {args.resume_lora}")
        ckpt = torch.load(args.resume_lora, map_location='cpu', weights_only=False)

        # In DMD2 mode this checkpoint targets the "gen" adapter; otherwise the legacy
        # single-adapter path (default name) is used.
        target_adapter = "gen" if dmd2_active else None
        load_lora_state_dict(model, ckpt["lora"], strict=False, adapter_name=target_adapter)

        # Optional: resume optimizer state if you saved it
        if "optimizer" in ckpt:
            logger.info("Loading optimizer state for LoRA...")
            optimizer.load_state_dict(ckpt["optimizer"])

        # Support both old and new ckpt format
        start_epoch = ckpt.get("epoch", 0)
        if args.resume_step_in_ckpt:
            start_step_in_epoch = ckpt.get("step_in_epoch", -1)

        logger.info(
            f"Resumed at epoch={start_epoch}, "
            f"step_in_epoch={start_step_in_epoch}, "
        )
    else:
        logger.info("No LoRA resume checkpoint provided.")

    if dmd2_active and args.resume_dmd2_fake_lora is not None and os.path.isfile(args.resume_dmd2_fake_lora):
        logger.info(f"Resuming DMD2 fake-score LoRA from {args.resume_dmd2_fake_lora}")
        ckpt = torch.load(args.resume_dmd2_fake_lora, map_location='cpu', weights_only=False)
        load_lora_state_dict(model, ckpt["lora"], strict=False, adapter_name="fake")
        logger.info("Resumed fake-score LoRA.")

    if (
        dmd2_active
        and getattr(args, "dmd2_use_gan", False)
        and getattr(args, "resume_dmd2_disc", None) is not None
        and os.path.isfile(args.resume_dmd2_disc)
    ):
        logger.info(f"Resuming DMD2 discriminator head from {args.resume_dmd2_disc}")
        ckpt = torch.load(args.resume_dmd2_disc, map_location='cpu', weights_only=False)
        load_disc_state_dict(model, ckpt["disc"])
        logger.info("Resumed DMD2 discriminator head.")
    
    if args.resume_multi_kernel is not None and os.path.isfile(args.resume_multi_kernel):
        logger.info(f"Resuming Multi-Kernel from {args.resume_multi_kernel}")
        ckpt = torch.load(args.resume_multi_kernel, map_location='cpu', weights_only=False)

        # Load multi-kernel weights into the wrapped model
        load_multikernel_state_dict(model, ckpt["multi_kernel"], strict=False)

        logger.info("Resumed Multi-Kernel checkpoint.")
    
    if args.resume_double_branch is not None and os.path.isfile(args.resume_double_branch):
        logger.info(f"Resuming Double Branch from {args.resume_double_branch}")
        ckpt = torch.load(args.resume_double_branch, map_location='cpu', weights_only=False)

        # Load double-branch weights into the wrapped model
        load_double_branch_state_dict(model, ckpt["double_branch"], strict=False)

        logger.info("Resumed Double Branch checkpoint.")

    if args.resume_patch_adapter is not None and os.path.isfile(args.resume_patch_adapter):
        logger.info(f"Resuming Patch Adapter from {args.resume_patch_adapter}")
        ckpt = torch.load(args.resume_patch_adapter, map_location='cpu', weights_only=False)

        # Load double-branch weights into the wrapped model
        load_patch_adapter_state_dict(model, ckpt["patch_adapter"], strict=False)

        logger.info("Resumed Patch Adapter checkpoint.")
    
    if args.resume is not None and os.path.isfile(args.resume):
        logger.info(f"Resuming full model from {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        logger.warning(f"Missing keys: {missing}")
        logger.warning(f"Unexpected keys: {unexpected}")
        # Support both old and new ckpt format
        start_epoch = ckpt.get("epoch", 0)
        if args.resume_step_in_ckpt:
            start_step_in_epoch = ckpt.get("step_in_epoch", -1)

        logger.info(
            f"Resumed at epoch={start_epoch}, "
            f"step_in_epoch={start_step_in_epoch}, "
        )
    
    #===================DEEPSPEED INITIALIZATION===================#


    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        model_parameters=trainable_params,  # the same params you passed to optimizer
        config=ds_config
    )
    steps_per_update = model_engine.gradient_accumulation_steps()
    logger.info(f"Grad accumulation steps: {steps_per_update}")

    #===================DMD2 RUNTIME SETUP===================#

    if dmd2_active:
        dmd2_cfg = dmd2_config_from_args(args)
        if steps_per_update != 1:
            raise ValueError(
                f"DMD2 requires gradient_accumulation_steps == 1 in the DeepSpeed config "
                f"(got {steps_per_update}). The two-timescale loop performs a full "
                f"optimizer step per fake/gen update; accumulation would entangle them."
            )
        t_grid = uniform_timestep_grid(dmd2_cfg.num_steps, device=args.device)
        dmd2_outer_step = 0
        logger.info(
            f"DMD2 enabled | num_steps={dmd2_cfg.num_steps} | "
            f"K_fake={dmd2_cfg.fake_updates_per_gen} | "
            f"warmup={dmd2_cfg.fake_warmup_steps} | "
            f"weight_mode={dmd2_cfg.weight_mode} | "
            f"t_p in [{dmd2_cfg.min_tp}, {dmd2_cfg.max_tp}]"
        )
        logger.info(f"DMD2 t_grid: {t_grid.tolist()}")
        if dmd2_cfg.use_gan:
            logger.info(
                f"DMD2 GAN enabled | λ_D={dmd2_cfg.gan_weight_d} | "
                f"λ_G={dmd2_cfg.gan_weight_g} | gan_warmup={dmd2_cfg.gan_warmup_steps}"
            )

        # Cached references used by dmd2_gen_step to freeze fake-LoRA + D-head params
        # during the gen step's differentiable GAN forward (see dmd2_gen_step docstring).
        dmd2_disc_head = get_disc_head(model_engine.module) if dmd2_cfg.use_gan else None
        dmd2_fake_lora_params_for_gen = get_lora_parameters(model_engine.module, adapter_name="fake") if dmd2_cfg.use_gan else []
        dmd2_disc_params_for_gen = get_disc_parameters(model_engine.module) if dmd2_cfg.use_gan else []

        # ------- Hypothesis-confirmation diagnostics (read-only logging) ----
        # (A) Per-adapter LoRA norms: logged once per outer step.
        # (B) Cosine of (∂fake_loss/∂fake_LoRA) vs (∂λ_D·d_loss/∂fake_LoRA):
        #     only meaningful when GAN is on. Each measurement adds two extra
        #     backward recomputes through the gradient-checkpointed graph, so
        #     gate to a prefix of outer steps and only on the FIRST fake update
        #     of each gated outer step.
        DMD2_GRAD_COSINE_LOG_OUTER_STEPS = 100
        dmd2_fake_lora_params_for_gradcos = (
            get_lora_parameters(model_engine.module, adapter_name="fake")
            if dmd2_cfg.use_gan else []
        )

    #===================TRAINING LOOP===================#
    loss_values = []
    avg_loss_values = []
    start_time = time.time()
    running_loss = 0.0
    running_fake_loss = 0.0
    running_gen_loss = 0.0
    running_gen_count = 0
    running_fake_tp = 0.0
    running_fake_vtgt = 0.0
    running_gen_grad_signal = 0.0
    running_gen_x_hat_abs = 0.0
    running_gen_weight = 0.0
    running_gen_tp = 0.0
    # GAN running counters (populated only when --dmd2-use-gan).
    running_d_loss = 0.0
    running_d_logits_real = 0.0
    running_d_logits_fake = 0.0
    running_gan_g_loss = 0.0
    running_gan_g_count = 0
    # Hypothesis-(B) running counters: cosine of fake-LoRA gradient from
    # fake_loss vs from λ_D·d_loss. Only populated on the first fake step of
    # the first DMD2_GRAD_COSINE_LOG_OUTER_STEPS outer steps.
    running_grad_cosine = 0.0
    running_grad_norm_mse = 0.0
    running_grad_norm_gan = 0.0
    running_grad_cosine_count = 0
    total_skip = 0
    should_stop_training = False

    for epoch in range(start_epoch, args.epochs):
        dataloader.sampler.set_epoch(epoch)
        if get_rank() == 0:
            pbar = tqdm(enumerate(dataloader), desc=f"Epoch {epoch + 1}")
        else:
            pbar = enumerate(dataloader)
        if should_stop_training:
            break
        for step, data in pbar:
            # update desc, print hour, minute, second
            elasped_time = time.time() - start_time
            if get_rank() == 0:
                pbar.set_description_str(f"Epoch {epoch + 1} | Time {int(elasped_time // 3600):02d}:{int((elasped_time % 3600) // 60):02d}:{int(elasped_time % 60):02d}")
            if epoch == start_epoch and step <= start_step_in_epoch:
                # We already processed these batches before checkpoint; skip them.
                continue
            rgbs = data['rgb']  # (B, 3, T, H, W)
            inverse_depths = data['depth']  # (B, 1, T, H, W)
            intrinsics = data['intrinsic']  # (B, T, 3, 3)
            w2cs = data['w2c']  # (B, T, 4, 4)
            sample_id = data['sample_id'] # [str] of length B
            prompt = data['prompt']  # [str] of length B
            logger.info(prompt)
            logger.info(sample_id)

            check_nan_inf({
                "rgbs": rgbs,
                "inverse_depths": inverse_depths,
                "intrinsics": intrinsics,
                "w2cs": w2cs,
            })

            # Prepare parital cond
            B, _, T, H, W = rgbs.shape
        
            logger.info("Computing model inputs...")

            if text_encoder_cache is None:
                text_inputs = text_encoder.text2tokens(prompt)
                text_ids_1 = text_inputs['input_ids'].to(args.device)
                text_mask_1 = text_inputs['attention_mask'].to(args.device)
                text_inputs = text_encoder_2.text2tokens(prompt)
                text_ids_2 = text_inputs['input_ids'].to(args.device)
                text_mask_2 = text_inputs['attention_mask'].to(args.device)
                text_batch = [text_ids_1, text_mask_1, text_ids_2, text_mask_2, {"type": ["video"]}]
            else:
                llm_i2v_text_states, llm_i2v_text_masks = text_encoder_cache.get_llm_i2v_text_state_and_mask(sample_id)
                llm_i2v_text_states = llm_i2v_text_states.to(args.device).to(model_engine.module.dtype)
                llm_i2v_text_masks = llm_i2v_text_masks.to(args.device)
                clipl_text_states = text_encoder_cache.get_clipl_text_state(sample_id).to(args.device).to(model_engine.module.dtype)
                text_batch = [llm_i2v_text_states, llm_i2v_text_masks, clipl_text_states, {"type": ["video"]}]

            need_compute_from_scratch = True
            if model_input_cache is not None:
                cached_inputs = model_input_cache.get_model_input(sample_id)
                valid_tensor = torch.tensor([1.0], device=args.device) if cached_inputs is not None else torch.tensor([0.0], device=args.device)
                dist.all_reduce(valid_tensor, op=dist.ReduceOp.MIN)
                if valid_tensor.item() < 1.0:
                    logger.warning(f"Some ranks do not have cached model inputs for sample IDs: {sample_id}. Computing from scratch.")
                    need_compute_from_scratch = True
                else:
                    logger.info(f"Using cached model inputs for sample IDs: {sample_id}")
                    assert len(cached_inputs) == 4
                    latents, cond_latents, partial_cond, partial_mask = cached_inputs
                    batch = [torch.tensor([0]), latents]
                    batch.extend(text_batch)
                    batch = tuple(batch)
                    latents, model_kwargs, n_tokens = prepare_model_inputs \
                                                                    (args, batch, args.device, \
                                                                    model_engine.module, vae, text_encoder, text_encoder_2,\
                                                                    args.rope_theta_rescale_factor, args.rope_interpolation_factor,
                                                                    skip_cond_latent=True)
                    cond_latents = cond_latents.to(model_engine.module.dtype).to(args.device)
                    partial_cond = partial_cond.to(model_engine.module.dtype).to(args.device)
                    partial_mask = partial_mask.to(model_engine.module.dtype).to(args.device)
                    logger.info(f"Loaded cached model inputs with latents shape {latents.shape}, cond_latents shape {cond_latents.shape}")
                    need_compute_from_scratch = False
            if need_compute_from_scratch:
                logger.info("Computing partial cond from scratch...")
                # Compute partial cond
                logger.info("Rendering partial cond rgb and depth...")
                frames = []
            
                logger.info("Computing model inputs...")
                for b in range(B):
                    frames.append([Frame(
                        rgb=ToPILImage()(rgbs[b, :, i]),
                        depth=inverse_depths[b, 0, i].numpy(),
                        camera=Camera(intrinsics[b, i].numpy(), w2cs[b, i].numpy()),
                        is_reverse_depth=True,
                    ) for i in range(T)
                    ])
                ground_truth_rgb_depths, _ = render(frames, rgbs.shape, args, partial_rendering=False)
                ground_truth_rgb_depths = ground_truth_rgb_depths.to(model_engine.module.dtype)
                partial_rgb_depths, partial_mask = render(frames, rgbs.shape, args, partial_rendering=True)
                # Encode partial cond to latent space
                partial_rgb_depths = partial_rgb_depths.to(vae.dtype).to(args.device)
                logger.info(f"Encoding partial cond with shape {partial_rgb_depths.shape} to latent space...")
                start_encode_time = time.time()
                partial_cond = vae.encode(
                        partial_rgb_depths).latent_dist.sample()
                logger.info(f"Encoded partial cond to latent space with shape {partial_cond.shape} in {time.time() - start_encode_time:.2f} seconds.")
                valid_tensor = is_tensor_valid({
                    "partial_cond": partial_cond,
                    "partial_mask": partial_mask,
                })
                valid_tensor = torch.tensor([1.0], device=args.device) if valid_tensor else torch.tensor([0.0], device=args.device)
                dist.all_reduce(valid_tensor, op=dist.ReduceOp.MIN)
                if valid_tensor.item() < 1.0:
                    total_skip += 1
                    logger.warning(f"Detected NaN or Inf in partial cond or partial mask for sample IDs: {sample_id}. Skipping this batch. Total skipped: {total_skip}")
                    continue

                partial_cond.mul_(vae.config.scaling_factor)
                partial_cond = partial_cond.to(model_engine.module.dtype)
                # Invert the mask
                partial_mask = 1 - partial_mask
                first_mask = partial_mask[:, :, 0:1, :, :]  # (B, 3, 1, H*2 + placeholder_row_length, W)
                partial_mask = torch.cat([first_mask, first_mask, first_mask, partial_mask], dim=2)  # (B, 3, T, H*2 + placeholder_row_length, W)
                partial_mask = torch.nn.functional.max_pool3d(
                    partial_mask, kernel_size=(4, 8, 8), stride=(4, 8, 8)
                )
                # Invert the mask again
                partial_mask = 1 - partial_mask
                partial_mask = partial_mask[: , 0:1].to(model_engine.module.dtype)

                start_prepare_time = time.time()
                logger.info("Preparing model inputs...")
                batch = [ground_truth_rgb_depths, torch.tensor([0])]
                batch.extend(text_batch)
                batch = tuple(batch)
                latents, model_kwargs, n_tokens, cond_latents = prepare_model_inputs \
                                                                        (args, batch, args.device, \
                                                                        model_engine.module, vae, text_encoder, text_encoder_2,\
                                                                        args.rope_theta_rescale_factor, args.rope_interpolation_factor)
                logger.info(f"Computed model inputs with latents shape {latents.shape}, cond_latents shape {cond_latents.shape} in {time.time() - start_prepare_time:.2f} seconds.")
            logger.info(f"Done preparing model inputs with latents shape {latents.shape}, cond_latents shape {cond_latents.shape}")
        
            # # DEBUG
            # with torch.autocast(
            #     device_type="cuda", dtype=vae.dtype, enabled=True
            # ):
            #     image = vae.decode(latents / vae.config.scaling_factor)[0]
            # if image.shape[2] == 1:
            #     image = image.squeeze(2)
            # logger.info(f"Reconstructed rgb_depths from latents with shape {image.shape}")
            # image = (image / 2 + 0.5).clamp(0, 1)
            # image = image.cpu().float()
            # half_height = args.height
            # rgb = image[..., :half_height, :]
            # depth = image[..., -half_height:, :]
            # depth = depth[:, 0] * 0.299 + depth[:, 1] * 0.587 + depth[:, 2] * 0.114
            # depth = depth.unsqueeze(1).repeat(1, 3, 1, 1, 1)
            # image = torch.cat([rgb, depth], dim=-2)
            # logger.info(f"Reconstructed rgb shape: {rgb.shape}, depth shape: {depth.shape}")
            # save_videos_grid(image, "/raid/hvtham/ldnhuan/HunyuanWorld-Voyager/debug.mp4", fps=24)
            # # END DEBUG

            avg_fake = None
            gen_loss_val = None
            fake_loss_vals = None

            if dmd2_active:
                # ----- DMD2 two-timescale step -----
                forward_v = make_dmd2_forward_v(
                    model_engine, denoiser, args,
                    cond_latents=cond_latents,
                    partial_cond=partial_cond,
                    partial_mask=partial_mask,
                    model_kwargs=model_kwargs,
                )

                fake_loss_vals = []
                fake_infos = []
                for fake_iter in range(dmd2_cfg.fake_updates_per_gen):
                    # Hypothesis (B) gate: only on the first fake update of each
                    # outer step, and only for the first
                    # DMD2_GRAD_COSINE_LOG_OUTER_STEPS outer steps, because each
                    # measurement adds ~2 extra grad-ckpt recomputes.
                    log_grad_cosine_now = (
                        dmd2_cfg.use_gan
                        and fake_iter == 0
                        and dmd2_outer_step < DMD2_GRAD_COSINE_LOG_OUTER_STEPS
                    )
                    fval, finfo = dmd2_fake_step(
                        model_engine, denoiser, args, dmd2_cfg, t_grid, latents, forward_v,
                        disc_head=dmd2_disc_head,
                        fake_lora_params=dmd2_fake_lora_params_for_gradcos,
                        log_grad_cosine=log_grad_cosine_now,
                    )
                    fake_loss_vals.append(fval)
                    fake_infos.append(finfo)

                do_gen_update = dmd2_outer_step >= dmd2_cfg.fake_warmup_steps
                # GAN G term is held out of the gen loss until its own warmup.
                do_gan_g = (
                    dmd2_cfg.use_gan
                    and dmd2_disc_head is not None
                    and dmd2_cfg.gan_weight_g > 0
                    and dmd2_outer_step >= dmd2_cfg.gan_warmup_steps
                )
                gen_loss_val = None
                gen_info_val = None
                if do_gen_update:
                    gen_loss_val, gen_info_val = dmd2_gen_step(
                        model_engine, denoiser, args, dmd2_cfg, t_grid, latents, forward_v,
                        disc_head=dmd2_disc_head,
                        do_gan_g=do_gan_g,
                        fake_lora_params=dmd2_fake_lora_params_for_gen,
                        disc_params=dmd2_disc_params_for_gen,
                    )

                dmd2_outer_step += 1

                # ---- Hypothesis (A) diagnostic: per-adapter LoRA norms -----
                # Cheap (just reads param tensors) so we compute every outer step.
                lora_norms_now = _compute_lora_norms_per_adapter(model_engine)
                # ---- Hypothesis (B) diagnostic: pull cosine off the first ----
                # fake update's info dict (only present when log_grad_cosine_now
                # was True, i.e. GAN-on and outer_step in the gated prefix).
                grad_cosine_now = fake_infos[0].get("fake_grad_cosine_mse_vs_gan")
                grad_norm_mse_now = fake_infos[0].get("fake_grad_norm_mse")
                grad_norm_gan_now = fake_infos[0].get("fake_grad_norm_gan")

                avg_fake = sum(fake_loss_vals) / max(1, len(fake_loss_vals))
                avg_fake_t_p = sum(i["t_p_mean"] for i in fake_infos) / len(fake_infos)
                avg_fake_v_target_norm = sum(i["v_target_norm"] for i in fake_infos) / len(fake_infos)
                # GAN diagnostics: only present when use_gan and disc_head is not None.
                avg_d_loss = None
                avg_d_logits_real = None
                avg_d_logits_fake = None
                if dmd2_cfg.use_gan and fake_infos and "d_loss" in fake_infos[0]:
                    avg_d_loss = sum(i["d_loss"] for i in fake_infos) / len(fake_infos)
                    avg_d_logits_real = sum(i["d_logits_real"] for i in fake_infos) / len(fake_infos)
                    avg_d_logits_fake = sum(i["d_logits_fake"] for i in fake_infos) / len(fake_infos)
                if gen_loss_val is not None:
                    gan_part = ""
                    if do_gan_g and "g_gan_loss" in gen_info_val:
                        gan_part = (
                            f" g_gan_loss={gen_info_val['g_gan_loss']:.4f}"
                            f" g_d_logits_fake={gen_info_val['g_d_logits_fake_mean']:.4f}"
                        )
                    d_part = ""
                    if avg_d_loss is not None:
                        d_part = (
                            f" d_loss={avg_d_loss:.4f}"
                            f" d_logits_real={avg_d_logits_real:.4f}"
                            f" d_logits_fake={avg_d_logits_fake:.4f}"
                        )
                    # Diagnostic suffix: (A) adapter norms + (B) grad cosine.
                    diag_part = (
                        f" | A_gen={lora_norms_now['gen']['A_norm']:.4e}"
                        f" B_gen={lora_norms_now['gen']['B_norm']:.4e}"
                        f" A_fake={lora_norms_now['fake']['A_norm']:.4e}"
                        f" B_fake={lora_norms_now['fake']['B_norm']:.4e}"
                    )
                    if grad_cosine_now is not None:
                        diag_part += (
                            f" | grad_cos(MSE,GAN-D)={grad_cosine_now:.4f}"
                            f" gnorm_MSE={grad_norm_mse_now:.3e}"
                            f" gnorm_GAN-D={grad_norm_gan_now:.3e}"
                        )
                    logger.info(
                        f"[DMD2 outer={dmd2_outer_step}] "
                        f"fake_loss_avg={avg_fake:.4f} fake_tp={avg_fake_t_p:.3f} fake_vtgt_norm={avg_fake_v_target_norm:.3f}{d_part} | "
                        f"gen_loss={gen_loss_val:.4f} "
                        f"x_hat0_abs={gen_info_val['x_hat_0_abs_mean']:.4f} "
                        f"grad_sig_norm={gen_info_val['grad_signal_norm']:.4f} "
                        f"weight={gen_info_val['weight_mean']:.4f} "
                        f"gen_tp={gen_info_val['t_p_mean']:.3f}{gan_part}{diag_part}"
                    )
                else:
                    d_part = ""
                    if avg_d_loss is not None:
                        d_part = (
                            f" d_loss={avg_d_loss:.4f}"
                            f" d_logits_real={avg_d_logits_real:.4f}"
                            f" d_logits_fake={avg_d_logits_fake:.4f}"
                        )
                    diag_part = (
                        f" | A_gen={lora_norms_now['gen']['A_norm']:.4e}"
                        f" B_gen={lora_norms_now['gen']['B_norm']:.4e}"
                        f" A_fake={lora_norms_now['fake']['A_norm']:.4e}"
                        f" B_fake={lora_norms_now['fake']['B_norm']:.4e}"
                    )
                    if grad_cosine_now is not None:
                        diag_part += (
                            f" | grad_cos(MSE,GAN-D)={grad_cosine_now:.4f}"
                            f" gnorm_MSE={grad_norm_mse_now:.3e}"
                            f" gnorm_GAN-D={grad_norm_gan_now:.3e}"
                        )
                    logger.info(
                        f"[DMD2 outer={dmd2_outer_step}] "
                        f"fake_loss_avg={avg_fake:.4f} fake_tp={avg_fake_t_p:.3f} fake_vtgt_norm={avg_fake_v_target_norm:.3f}{d_part} | "
                        f"gen=warmup({dmd2_outer_step}/{dmd2_cfg.fake_warmup_steps}){diag_part}"
                    )

                # Surface a scalar for the rest of the loop's bookkeeping (loss logs, checkpoints).
                loss = gen_loss_val if gen_loss_val is not None else avg_fake
                input_t = []
            else:
                loss, input_t = training_step(
                    latents,
                    cond_latents,
                    model_engine,
                    denoiser,
                    args,
                    model_kwargs=model_kwargs,
                    partial_cond=partial_cond,
                    partial_mask=partial_mask,
                    return_all_terms=True,
                )

            global_step = epoch * len(dataloader) + step
            effective_update = (global_step + 1) // steps_per_update
            is_update_step = (global_step + 1) % steps_per_update == 0

            # When one effective update is done
            if model_engine.global_rank == 0:
                running_loss += loss
                if dmd2_active:
                    running_fake_loss += avg_fake
                    running_fake_tp += avg_fake_t_p
                    running_fake_vtgt += avg_fake_v_target_norm
                    if avg_d_loss is not None:
                        running_d_loss += avg_d_loss
                        running_d_logits_real += avg_d_logits_real
                        running_d_logits_fake += avg_d_logits_fake
                    if gen_loss_val is not None:
                        running_gen_loss += gen_loss_val
                        running_gen_grad_signal += gen_info_val['grad_signal_norm']
                        running_gen_x_hat_abs += gen_info_val['x_hat_0_abs_mean']
                        running_gen_weight += gen_info_val['weight_mean']
                        running_gen_tp += gen_info_val['t_p_mean']
                        running_gen_count += 1
                        if "g_gan_loss" in gen_info_val:
                            running_gan_g_loss += gen_info_val['g_gan_loss']
                            running_gan_g_count += 1
                    # Hypothesis (B) accumulator. NaN guard so a degenerate
                    # cosine (all-zero grad) doesn't poison the running mean.
                    if (
                        grad_cosine_now is not None
                        and isinstance(grad_cosine_now, float)
                        and grad_cosine_now == grad_cosine_now  # not NaN
                    ):
                        running_grad_cosine += grad_cosine_now
                        running_grad_norm_mse += grad_norm_mse_now
                        running_grad_norm_gan += grad_norm_gan_now
                        running_grad_cosine_count += 1

                loss_entry = {
                    'step': global_step,
                    'loss': loss,
                    'sample_id': sample_id,
                    'input_t': input_t,
                }
                if dmd2_active:
                    loss_entry['fake_loss_avg'] = avg_fake
                    loss_entry['fake_loss_vals'] = fake_loss_vals
                    loss_entry['fake_tp_mean'] = avg_fake_t_p
                    loss_entry['fake_v_target_norm'] = avg_fake_v_target_norm
                    loss_entry['fake_tp_vals'] = [i["t_p_mean"] for i in fake_infos]
                    loss_entry['fake_v_target_norm_vals'] = [i["v_target_norm"] for i in fake_infos]
                    if avg_d_loss is not None:
                        loss_entry['d_loss_avg'] = avg_d_loss
                        loss_entry['d_logits_real_avg'] = avg_d_logits_real
                        loss_entry['d_logits_fake_avg'] = avg_d_logits_fake
                    if gen_loss_val is not None:
                        loss_entry['gen_loss'] = gen_loss_val
                        loss_entry['gen_grad_signal_norm'] = gen_info_val['grad_signal_norm']
                        loss_entry['gen_x_hat_0_abs_mean'] = gen_info_val['x_hat_0_abs_mean']
                        loss_entry['gen_weight_mean'] = gen_info_val['weight_mean']
                        loss_entry['gen_tp_mean'] = gen_info_val['t_p_mean']
                        if 'g_gan_loss' in gen_info_val:
                            loss_entry['g_gan_loss'] = gen_info_val['g_gan_loss']
                            loss_entry['g_d_logits_fake_mean'] = gen_info_val['g_d_logits_fake_mean']
                    # Hypothesis (A) per-step adapter norms.
                    loss_entry['lora_norm_A_gen'] = lora_norms_now['gen']['A_norm']
                    loss_entry['lora_norm_B_gen'] = lora_norms_now['gen']['B_norm']
                    loss_entry['lora_norm_A_fake'] = lora_norms_now['fake']['A_norm']
                    loss_entry['lora_norm_B_fake'] = lora_norms_now['fake']['B_norm']
                    loss_entry['lora_count_A_gen'] = lora_norms_now['gen']['A_count']
                    loss_entry['lora_count_B_gen'] = lora_norms_now['gen']['B_count']
                    # Hypothesis (B) per-step cosine + raw grad norms (only on
                    # the gated outer-step prefix; otherwise these keys are
                    # absent).
                    if grad_cosine_now is not None:
                        loss_entry['fake_grad_cosine_mse_vs_gan'] = grad_cosine_now
                        loss_entry['fake_grad_norm_mse'] = grad_norm_mse_now
                        loss_entry['fake_grad_norm_gan'] = grad_norm_gan_now
                loss_values.append(loss_entry)

                if is_update_step:
                    avg_loss = running_loss / steps_per_update
                    avg_entry = {
                        'step': global_step + 1,
                        'avg_loss': avg_loss,
                        'time': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                    }
                    if dmd2_active:
                        avg_entry['avg_fake_loss'] = running_fake_loss / steps_per_update
                        avg_entry['avg_fake_tp_mean'] = running_fake_tp / steps_per_update
                        avg_entry['avg_fake_v_target_norm'] = running_fake_vtgt / steps_per_update
                        if dmd2_cfg.use_gan:
                            avg_entry['avg_d_loss'] = running_d_loss / steps_per_update
                            avg_entry['avg_d_logits_real'] = running_d_logits_real / steps_per_update
                            avg_entry['avg_d_logits_fake'] = running_d_logits_fake / steps_per_update
                        if running_gen_count > 0:
                            avg_entry['avg_gen_loss'] = running_gen_loss / running_gen_count
                            avg_entry['avg_gen_grad_signal_norm'] = running_gen_grad_signal / running_gen_count
                            avg_entry['avg_gen_x_hat_0_abs_mean'] = running_gen_x_hat_abs / running_gen_count
                            avg_entry['avg_gen_weight_mean'] = running_gen_weight / running_gen_count
                            avg_entry['avg_gen_tp_mean'] = running_gen_tp / running_gen_count
                            if running_gan_g_count > 0:
                                avg_entry['avg_g_gan_loss'] = running_gan_g_loss / running_gan_g_count
                        # Hypothesis (A) snapshot at the END of this averaging
                        # window — adapter norms are a state (not a per-step
                        # quantity), so we just record the most recent value.
                        avg_entry['lora_norm_A_gen_last'] = lora_norms_now['gen']['A_norm']
                        avg_entry['lora_norm_B_gen_last'] = lora_norms_now['gen']['B_norm']
                        avg_entry['lora_norm_A_fake_last'] = lora_norms_now['fake']['A_norm']
                        avg_entry['lora_norm_B_fake_last'] = lora_norms_now['fake']['B_norm']
                        # Hypothesis (B) averaged cosine over the window.
                        if running_grad_cosine_count > 0:
                            avg_entry['avg_fake_grad_cosine_mse_vs_gan'] = (
                                running_grad_cosine / running_grad_cosine_count
                            )
                            avg_entry['avg_fake_grad_norm_mse'] = (
                                running_grad_norm_mse / running_grad_cosine_count
                            )
                            avg_entry['avg_fake_grad_norm_gan'] = (
                                running_grad_norm_gan / running_grad_cosine_count
                            )
                            avg_entry['fake_grad_cosine_n_samples'] = running_grad_cosine_count
                    avg_loss_values.append(avg_entry)

                    log_msg = f"[Update {(global_step + 1) // steps_per_update}] avg_loss = {avg_loss:.4f}"
                    if dmd2_active:
                        log_msg += f" | avg_fake_loss = {avg_entry['avg_fake_loss']:.4f}"
                        if 'avg_gen_loss' in avg_entry:
                            log_msg += f" | avg_gen_loss = {avg_entry['avg_gen_loss']:.4f}"
                    logger.info(log_msg)

                    running_loss = 0.0
                    running_fake_loss = 0.0
                    running_gen_loss = 0.0
                    running_gen_count = 0
                    running_fake_tp = 0.0
                    running_fake_vtgt = 0.0
                    running_gen_grad_signal = 0.0
                    running_gen_x_hat_abs = 0.0
                    running_gen_weight = 0.0
                    running_gen_tp = 0.0
                    running_d_loss = 0.0
                    running_d_logits_real = 0.0
                    running_d_logits_fake = 0.0
                    running_gan_g_loss = 0.0
                    running_gan_g_count = 0
                    running_grad_cosine = 0.0
                    running_grad_norm_mse = 0.0
                    running_grad_norm_gan = 0.0
                    running_grad_cosine_count = 0

                    with open(Path(training_output_dir) / 'loss_log.json', 'w') as f:
                        json.dump({'loss_values': loss_values, 'avg_loss_values': avg_loss_values}, f, indent=4)

                    plt.figure()
                    steps = [v['step'] for v in avg_loss_values]
                    plt.plot(steps, [v['avg_loss'] for v in avg_loss_values], label='loss')
                    if dmd2_active:
                        fake_steps = [v['step'] for v in avg_loss_values if 'avg_fake_loss' in v]
                        fake_vals  = [v['avg_fake_loss'] for v in avg_loss_values if 'avg_fake_loss' in v]
                        if fake_steps:
                            plt.plot(fake_steps, fake_vals, label='fake_loss')
                        gen_pairs = [(v['step'], v['avg_gen_loss']) for v in avg_loss_values if 'avg_gen_loss' in v]
                        if gen_pairs:
                            plt.plot([s for s, _ in gen_pairs], [l for _, l in gen_pairs], label='gen_loss')
                        plt.legend()
                    plt.xlabel('Step')
                    plt.ylabel('Avg Loss')
                    plt.title('Average Training Loss')
                    plt.savefig(Path(training_output_dir) / 'avg_loss_curve.png')
                    plt.close()

                    if dmd2_active:
                        diag_keys = [
                            'avg_gen_grad_signal_norm',
                            'avg_gen_x_hat_0_abs_mean',
                            'avg_gen_weight_mean',
                            'avg_fake_v_target_norm',
                        ]
                        any_present = any(any(k in v for v in avg_loss_values) for k in diag_keys)
                        if any_present:
                            plt.figure()
                            for k in diag_keys:
                                pairs = [(v['step'], v[k]) for v in avg_loss_values if k in v]
                                if pairs:
                                    plt.plot([s for s, _ in pairs], [x for _, x in pairs], label=k)
                            plt.xlabel('Step')
                            plt.ylabel('Value')
                            plt.yscale('log')
                            plt.title('DMD2 diagnostics')
                            plt.legend()
                            plt.savefig(Path(training_output_dir) / 'dmd2_diagnostics_curve.png')
                            plt.close()

                        # ---- Hypothesis (A) plot: per-adapter LoRA norms ----
                        norm_keys = [
                            'lora_norm_A_gen_last',
                            'lora_norm_B_gen_last',
                            'lora_norm_A_fake_last',
                            'lora_norm_B_fake_last',
                        ]
                        norms_present = any(
                            any(k in v for v in avg_loss_values) for k in norm_keys
                        )
                        if norms_present:
                            plt.figure()
                            for k in norm_keys:
                                pairs = [(v['step'], v[k]) for v in avg_loss_values if k in v]
                                if pairs:
                                    plt.plot([s for s, _ in pairs], [x for _, x in pairs], label=k)
                            plt.xlabel('Step')
                            plt.ylabel('||param||_2')
                            plt.title('LoRA adapter norms (Hypothesis A)')
                            plt.legend()
                            plt.savefig(Path(training_output_dir) / 'lora_norms_curve.png')
                            plt.close()

                        # ---- Hypothesis (B) plot: MSE vs GAN-D grad cosine ---
                        cos_pairs = [
                            (v['step'], v['avg_fake_grad_cosine_mse_vs_gan'])
                            for v in avg_loss_values
                            if 'avg_fake_grad_cosine_mse_vs_gan' in v
                        ]
                        if cos_pairs:
                            plt.figure()
                            xs = [s for s, _ in cos_pairs]
                            ys = [x for _, x in cos_pairs]
                            plt.plot(xs, ys, marker='o', label='cos(grad_MSE, grad_GAN-D)')
                            plt.axhline(0.0, color='k', linewidth=0.5)
                            plt.ylim(-1.05, 1.05)
                            plt.xlabel('Step')
                            plt.ylabel('cosine')
                            plt.title('Fake-LoRA grad cosine: MSE vs λ_D·d_loss (Hypothesis B)')
                            plt.legend()
                            plt.savefig(Path(training_output_dir) / 'fake_lora_grad_cosine_curve.png')
                            plt.close()
            
            if is_update_step and args.save_every > 0 and (effective_update % args.save_every == 0):
                save_ckpt(args, model_engine, epoch, step, training_output_dir, total_skip)
            if is_update_step and args.backup_every > 0 and (effective_update % args.backup_every == 0):
                backup_path = os.path.join(training_output_dir, "backup", f"Step-{effective_update:05d}")
                os.makedirs(backup_path, exist_ok=True)
                save_ckpt(args, model_engine, epoch, step, backup_path, total_skip)

            if args.early_stop_training_loss is not None:
                # 1. Create a tensor for the current local loss
                # We clone it to avoid modifying the computation graph
                loss_tensor = torch.tensor(loss, device=args.device)

                # 2. Sync: Average the loss across ALL GPUs
                # This ensures every rank sees the exact same "avg_loss" value
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                global_avg_loss = loss_tensor.item()

                # 3. Decision: Check threshold
                if global_avg_loss < args.early_stop_training_loss:
                    # Only log on Rank 0 to avoid spam
                    if dist.get_rank() == 0:
                        logger.info(f"Early stopping triggered! Global Loss {global_avg_loss:.4f} < {args.early_stop_training_loss}")
                        
                    # Optional: Save a final checkpoint before quitting
                    save_ckpt(args, model_engine, epoch, step, training_output_dir, total_skip)

                    # 4. Set the flag to True on ALL ranks
                    should_stop_training = True
                    
                    # 5. Break the INNER (Step) loop
                    break

            logger.info(f"{latents.shape=}")
            logger.info(f"{cond_latents.shape=}")
            logger.info(f"{n_tokens=}")
            
            