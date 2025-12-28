from loguru import logger
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

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage 
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
import torch.distributed as dist

from dataset.RealEstate10K_render import RealEstate10K_render as RealEstate10K
from voyager.text_encoder import TextEncoder
from voyager.config import *
from voyager.diffusion import load_denoiser
from voyager.diffusion.flow import Transport
from voyager.utils.train_utils import set_reproducibility, prepare_model_inputs, get_training_output_dir
from voyager.utils.file_utils import save_videos_grid
from voyager.inference import load_models
from voyager.modules.lora_layers import apply_lora_to_hunyuan_video, get_lora_parameters, get_lora_state_dict, load_lora_state_dict
from voyager.modules.custom_patch_embed import apply_patch_adapter_to_hunyuan_video, get_patch_adapter_parameters, get_patch_adapter_state_dict
from voyager.modules.multi_kernel import apply_multikernel_to_hunyuan_video, get_multikernel_parameters, get_multikernel_state_dict, load_multikernel_state_dict
from voyager.modules.double_branch import apply_double_branch_to_hunyuan_video, get_double_branch_parameters, get_double_branch_state_dict, load_double_branch_state_dict, TransformerBranchConfig
from voyager.modules.models import HUNYUAN_VIDEO_CONFIG
from voyager.constants import PRECISION_TO_TYPE
from voyager.cache.text_cache import TextEncoderCache
from voyager.cache.model_input_cache import ModelInputCache
from gather_realestate import norm_partial_render_output
from utils.render import Camera, Frame
from utils.tensor import is_tensor_valid, check_nan_inf
from voyager.utils.helpers import as_list_of_3tuple

def build_optimizer(param, args):
    optimizer = DeepSpeedCPUAdam(
        param,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    return optimizer

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
        '--save-every',
        type=int,
        default=1000,
        help='Save checkpoint every N effective steps (after grad accumulation).',
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
    return args

def training_step(
    x1,
    cond_latents,
    model_engine,
    denoiser: Transport,
    args,
    model_kwargs: Optional[dict] = None,
    partial_cond=None,
    partial_mask=None
) -> float:
    x1 = x1.to(model_engine.device)

    model_engine.train()
    target_dtype = PRECISION_TO_TYPE[args.precision]
    autocast_enabled = (target_dtype != torch.float32) and not args.disable_autocast

    logger.info("Starting training step...")
    start_training_time = time.time()

    with torch.no_grad():
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
            )
    

    loss = terms["loss"].mean()
    logger.info(f"Computed loss: {loss.item()} in {time.time() - start_training_time:.2f} seconds, starting backward pass...")

    backward_start_time = time.time()

    # DeepSpeed handles backward + optimizer
    # model_engine.backward(loss)
    # model_engine.step()

    logger.info(f"Completed backward pass and optimizer step in {time.time() - backward_start_time:.2f} seconds.")

    # with deepspeed.zero.GatheredParameters(model_engine.module.parameters(), modifier_rank=0):
    #     if dist.get_rank() == 0:
    #         # assert all params are not nan and inf
    #         for name, param in model_engine.module.named_parameters():
    #             if param.requires_grad:
    #                 if torch.isnan(param).any() or torch.isinf(param).any():
    #                     raise ValueError(f"Parameter {name} has NaN or Inf values!")

    return loss.item()


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
            print(f"Saving LoRA checkpoint to {ckpt_path}")
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
            print(f"Saving PatchAdapter checkpoint to {ckpt_path}")
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
            print(f"Saving Multi-Kernel checkpoint to {ckpt_path}")
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
            print(f"Saving Double Branch checkpoint to {ckpt_path}")
            torch.save(ckpt, ckpt_path)

def save_ckpt(args, model_engine, epoch, step, training_output_dir, total_skip):
    print(torch.cuda.memory_summary())
    logger.warning(f"Skipped {total_skip} batches due to NaN/Inf so far.")
    if args.train_from_scratch:
        ckpt_path = Path(training_output_dir) / f"model_{args.model}_last.pt"
        logger.info(f"Saving full model checkpoint to {ckpt_path}")
        save_full_model_checkpoint(model_engine, args, epoch, step, training_output_dir)
    if args.train_lora:
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
    args = parse_arg()
    print(args)

    #===================INITIALIZATION===================#

    denoiser = load_denoiser(args) 
    dtype = PRECISION_TO_TYPE[args.precision]
    set_reproducibility(True, args.global_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    training_output_dir = get_training_output_dir(args.output_dir)
    os.makedirs(training_output_dir, exist_ok=True)
    deepspeed_config = args.deepspeed_config if hasattr(args, "deepspeed_config") else "ds_config.json"
    with open(deepspeed_config, 'r') as f:
        ds_config = json.load(f)

    from copy import copy
    if args.use_double_branch:
        tmp_args = copy(args)
        tmp_args.second_branch_transformer_config = tmp_args.second_branch_transformer_config.to_dict()
    else:
        tmp_args = copy(args)
    config = {
        'time': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        'output_dir': training_output_dir.__str__(),
        'model_args': vars(tmp_args),
        'deepspeed_config': ds_config,
    }
    with open(Path(training_output_dir) / 'training_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    logger.info(f"Training output dir: {training_output_dir}")

    #===================DATASET & DATALOADER===================#

    dataset_root = args.dataset_root
    dataset = RealEstate10K(dataset_root, set_name=args.task_flag, width=args.width, height=args.height, return_inverse_depth=True)
    dataloader = DataLoader(dataset, batch_size=ds_config["train_micro_batch_size_per_gpu"], shuffle=True, num_workers=args.num_workers)

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

    if args.train_lora:
        logger.info("Applying LoRA adapters to model...")
        apply_lora_to_hunyuan_video(
            model,
            r=args.lora_rank if hasattr(args, "lora_rank") else 8,
            lora_alpha=args.lora_alpha if hasattr(args, "lora_alpha") else 16.0,
            lora_dropout=getattr(args, "lora_dropout", 0.0),
            freeze_base=not args.train_from_scratch
        )
    if args.patch_adapter_size is not None:
        logger.info(f"Applying Patch Size Adapters with new patch size: {args.patch_adapter_size}")
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
    logger.info(f"Number of LoRA parameters: {sum(p.numel() for p in lora_params)}")
    logger.info(f"Number of Patch Adapter parameters: {sum(p.numel() for p in patch_params)}")
    logger.info(f"Number of Multi-Kernel parameters: {sum(p.numel() for p in multikernel_params)}")
    logger.info(f"Number of Double Branch parameters: {sum(p.numel() for p in double_branch_params)}")
    
    trainable_addons_params = lora_params + patch_params + multikernel_params + double_branch_params
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
    optimizer = build_optimizer(trainable_params, args)

    #===================RESUME FROM CHECKPOINT IF ANY===================#

    start_epoch = 0
    start_step_in_epoch = -1

    if args.resume_lora is not None and os.path.isfile(args.resume_lora):
        logger.info(f"Resuming LoRA from {args.resume_lora}")
        ckpt = torch.load(args.resume_lora, map_location='cpu', weights_only=False)

        # Load LoRA weights into the wrapped model
        load_lora_state_dict(model, ckpt["lora"], strict=False)

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
        model_parameters=lora_params,  # the same params you passed to optimizer
        config=ds_config
    )
    steps_per_update = model_engine.gradient_accumulation_steps()
    logger.info(f"Grad accumulation steps: {steps_per_update}")
    
    #===================TRAINING LOOP===================#
    loss_values = []
    avg_loss_values = []
    start_time = time.time()
    running_loss = 0.0
    total_skip = 0
    for epoch in range(start_epoch, args.epochs):
        pbar = tqdm(enumerate(dataloader), desc=f"Epoch {epoch + 1}")
        for step, data in pbar:
            # update desc, print hour, minute, second
            elasped_time = time.time() - start_time
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
            print(prompt)
            print(sample_id)

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
                    cond_latents = cond_latents.to(model_engine.module.dtype).to(model_engine.module.device)
                    partial_cond = partial_cond.to(model_engine.module.dtype).to(model_engine.module.device)
                    partial_mask = partial_mask.to(model_engine.module.dtype).to(model_engine.module.device)
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

            loss = training_step(
                latents,
                cond_latents,
                model_engine,
                denoiser,
                args,
                model_kwargs=model_kwargs,
                partial_cond=partial_cond,
                partial_mask=partial_mask,
            )

            # When one effective update is done
            if model_engine.global_rank == 0:
                running_loss += loss
                global_step = epoch * len(dataloader) + step
                loss_values.append({
                    'step': global_step,
                    'loss': loss,
                    'sample_id': sample_id,
                })
                
                if (global_step + 1) % steps_per_update == 0:
                    avg_loss = running_loss / steps_per_update
                    avg_loss_values.append({
                        'step': global_step + 1,
                        'avg_loss': avg_loss,
                        'time': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                    })
                    logger.info(f"[Update {(global_step + 1) // steps_per_update}] avg_loss = {avg_loss:.4f}")
                    running_loss = 0.0                                     
                    with open(Path(training_output_dir) / 'loss_log.json', 'w') as f:
                        json.dump({'loss_values': loss_values, 'avg_loss_values': avg_loss_values}, f, indent=4)
                    plt.figure()
                    plt.plot([v['step'] for v in avg_loss_values], [v['avg_loss'] for v in avg_loss_values])
                    plt.xlabel('Step')
                    plt.ylabel('Avg Loss')
                    plt.title('Average Training Loss')
                    plt.savefig(Path(training_output_dir) / 'avg_loss_curve.png')
                    plt.close()

                    effective_update = (global_step + 1) // steps_per_update

                    if args.save_every > 0 and (effective_update % args.save_every == 0):
                        save_ckpt(args, model_engine, epoch, step, training_output_dir, total_skip)

                    ####DEBUG ONLY, EARLY STOP WHEN LOSS < 0.03
                    if args.early_stop_training_loss is not None and loss < args.early_stop_training_loss:
                        save_ckpt(args, model_engine, epoch, step, training_output_dir, total_skip)
                        logger.info(f"Early stopping at epoch {epoch}, step {step} due to low loss {loss}, smaller than {args.early_stop_training_loss}")
                        exit(0)
                    ####END DEBUG ONLY
                print(f"{latents.shape=}")
                print(f"{cond_latents.shape=}")
                print(f"{n_tokens=}")
            
            