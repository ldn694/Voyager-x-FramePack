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

from dataset.RealEstate10K import RealEstate10K
from voyager.text_encoder import TextEncoder
from voyager.config import *
from voyager.diffusion import load_denoiser
from voyager.diffusion.flow import Transport
from voyager.utils.train_utils import set_reproducibility, prepare_model_inputs, get_training_output_dir
from voyager.utils.file_utils import save_videos_grid
from voyager.inference import load_models
from voyager.modules.lora_layers import apply_lora_to_hunyuan_video, get_lora_parameters, get_lora_state_dict, load_lora_state_dict
from voyager.modules.custom_patch_embed import apply_patch_adapter_to_hunyuan_video, get_patch_adapter_parameters, get_patch_adapter_state_dict
from voyager.constants import PRECISION_TO_TYPE
from voyager.cache.text_cache import TextEncoderCache
from gather_realestate import norm_partial_render_output
from utils.render import Camera, Frame

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
    parser.add_argument(
        '--resume-lora',
        type=str,
        default=None,
        help='Path to a LoRA checkpoint (.pt) to resume from',
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

    args = parser.parse_args()
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
    logger.info(f"Computed loss: {loss.item()}, starting backward pass...")

    # DeepSpeed handles backward + optimizer
    model_engine.backward(loss)
    model_engine.step()

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
    config = {
        'time': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        'output_dir': training_output_dir.__str__(),
        'model_args': vars(args),
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
        load_models(args, args.device, logger, Path(args.model_base))
    vae.enable_tiling()
    vae.eval()
    if not args.use_cache_text_encoder:
        text_encoder.eval()
        text_encoder_2.eval()
        text_encoder_cache = None
    else:
        logger.info("Loading text encoder cache...")
        text_encoder_cache = TextEncoderCache(Path(args.dataset_root) / "cache" / "text_encoder")
    if args.train_lora:
        logger.info("Applying LoRA adapters to model...")
        apply_lora_to_hunyuan_video(
            model,
            r=args.lora_rank if hasattr(args, "lora_rank") else 8,
            lora_alpha=args.lora_alpha if hasattr(args, "lora_alpha") else 16.0,
            lora_dropout=getattr(args, "lora_dropout", 0.0),
            freeze_base=True,
        )
    if args.patch_adapter_size is not None:
        logger.info(f"Applying Patch Size Adapters with new patch size: {args.patch_adapter_size}")
        apply_patch_adapter_to_hunyuan_video(
            model,
            new_patch_size=tuple(args.patch_adapter_size),
            freeze_base=True,
        )

    lora_params = get_lora_parameters(model) if args.train_lora else []
    patch_params = get_patch_adapter_parameters(model) if args.patch_adapter_size is not None else []
    trainable_params = lora_params + patch_params
    for p in trainable_params:
        p.requires_grad = True
    
    # log model's trainable params
    total_params_cnt = sum(p.numel() for p in model.parameters())
    trainable_params_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params_cnt}")
    logger.info(f"Trainable parameters: {trainable_params_cnt}")
    optimizer = build_optimizer(trainable_params, args)

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
        start_step_in_epoch = ckpt.get("step_in_epoch", -1)

        logger.info(
            f"Resumed at epoch={start_epoch}, "
            f"step_in_epoch={start_step_in_epoch}, "
        )
    else:
        logger.info("No LoRA resume checkpoint provided.")

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
            # Prepare parital cond
            frames = []
            B, _, T, H, W = rgbs.shape
            for b in range(B):
                frames.append([Frame(
                    rgb=ToPILImage()(rgbs[b, :, i]),
                    depth=inverse_depths[b, 0, i].numpy(),
                    camera=Camera(intrinsics[b, i].numpy(), w2cs[b, i].numpy()),
                    is_reverse_depth=True,
                ) for i in range(T)
                ])
            # Compute ground truth media
            logger.info("Rendering ground truth rgb and depth...")
            ground_truth_rgb_depths, _ = render(frames, rgbs.shape, args, partial_rendering=False)
            ground_truth_rgb_depths = ground_truth_rgb_depths.to(model_engine.module.dtype)
            # Compute partial cond
            logger.info("Rendering partial cond rgb and depth...")
            partial_rgb_depths, partial_mask = render(frames, rgbs.shape, args, partial_rendering=True)
            # Encode partial cond to latent space
            partial_rgb_depths = partial_rgb_depths.to(vae.dtype).to(args.device)
            logger.info(f"Encoding partial cond with shape {partial_rgb_depths.shape} to latent space...")
            partial_cond = vae.encode(
                    partial_rgb_depths).latent_dist.sample()
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


            if text_encoder_cache is None:
                text_inputs = text_encoder.text2tokens(prompt)
                text_ids_1 = text_inputs['input_ids'].to(args.device)
                text_mask_1 = text_inputs['attention_mask'].to(args.device)
                text_inputs = text_encoder_2.text2tokens(prompt)
                text_ids_2 = text_inputs['input_ids'].to(args.device)
                text_mask_2 = text_inputs['attention_mask'].to(args.device)
                batch = (ground_truth_rgb_depths, torch.tensor([0]), text_ids_1, text_mask_1, text_ids_2, text_mask_2, {"type": ["video"]})
            else:
                llm_i2v_text_states, llm_i2v_text_masks = text_encoder_cache.get_llm_i2v_text_state_and_mask(sample_id)
                llm_i2v_text_states = llm_i2v_text_states.to(args.device).to(model_engine.module.dtype)
                llm_i2v_text_masks = llm_i2v_text_masks.to(args.device)
                clipl_text_states = text_encoder_cache.get_clipl_text_state(sample_id).to(args.device).to(model_engine.module.dtype)
                batch = (ground_truth_rgb_depths, torch.tensor([0]), llm_i2v_text_states, llm_i2v_text_masks, clipl_text_states, {"type": ["video"]})
            latents, model_kwargs, n_tokens, cond_latents = prepare_model_inputs \
                                                                    (args, batch, args.device, \
                                                                    model_engine.module, vae, text_encoder, text_encoder_2,\
                                                                    args.rope_theta_rescale_factor, args.rope_interpolation_factor)
            
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
                        if args.train_lora:
                            lora_ckpt_path = Path(training_output_dir) / f"lora_last.pt"
                            logger.info(f"Saving LoRA checkpoint to {lora_ckpt_path}")
                            save_lora_checkpoint(model_engine, args, epoch, step, training_output_dir)
                        if args.patch_adapter_size is not None:
                            patch_adapter_ckpt_path = Path(training_output_dir) / f"patch_adapter_last.pt"
                            logger.info(f"Saving PatchAdapter checkpoint to {patch_adapter_ckpt_path}")
                            save_patch_adapter_checkpoint(model_engine, args, epoch, step, training_output_dir)

                print(f"{latents.shape=}")
                print(f"{cond_latents.shape=}")
                print(f"{n_tokens=}")
            