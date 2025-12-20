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
from types import SimpleNamespace

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
from voyager.inference import load_vae_only
from voyager.modules.lora_layers import apply_lora_to_hunyuan_video, get_lora_parameters, get_lora_state_dict, load_lora_state_dict
from voyager.modules.custom_patch_embed import apply_patch_adapter_to_hunyuan_video, get_patch_adapter_parameters, get_patch_adapter_state_dict
from voyager.modules.multi_kernel import apply_multikernel_to_hunyuan_video, get_multikernel_parameters, get_multikernel_state_dict
from voyager.constants import PRECISION_TO_TYPE
from voyager.cache.text_cache import TextEncoderCache
from gather_realestate import norm_partial_render_output
from utils.render import Camera, Frame
from utils.tensor import is_tensor_valid, check_nan_inf
from voyager.utils.helpers import as_list_of_3tuple

def parse_arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=str, required=True, help='Path to RealEstate10K dataset root')
    parser.add_argument('--width', type=int, default=256, help='Width of images to load')
    parser.add_argument('--height', type=int, default=384, help='Height of images to load')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--use-cache-text-encoder', action='store_true', help='Whether to cache text encoder outputs')
    parser.add_argument('--placeholder-row-length', type=int, default=64, help='Placeholder row length')
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
    
    parser = add_training_args(parser)
    parser = add_network_args(parser)
    parser = add_extra_models_args(parser)
    parser = add_data_args(parser)

    args = parser.parse_args()
    return args


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

if __name__ == "__main__":
    args = parse_arg()
    print(args)

    #===================INITIALIZATION===================#

    dtype = PRECISION_TO_TYPE[args.precision]
    set_reproducibility(True, args.global_seed)

    #===================DATASET & DATALOADER===================#

    dataset_root = args.dataset_root
    dataset = RealEstate10K(dataset_root, set_name=args.task_flag, width=args.width, height=args.height, return_inverse_depth=True)
    dataloader = DataLoader(dataset, batch_size=args.global_batch_size[0], shuffle=False, num_workers=args.num_workers)

    os.makedirs(args.output_dir, exist_ok=True)
    shards_dir = Path(args.output_dir) / "shards"
    os.makedirs(shards_dir, exist_ok=True)
    latents_path = Path(args.output_dir) / "latents.pt"
    cond_latents_path = Path(args.output_dir) / "cond_latents.pt"
    partial_cond_path = Path(args.output_dir) / "partial_cond.pt"
    partial_mask_path = Path(args.output_dir) / "partial_mask.pt"
    json_path = Path(args.output_dir) / "sample_id_to_index.json"

    # Load existing progress if it exists
    if json_path.exists():
        with open(json_path, 'r') as f:
            map_sample_id_to_index = json.load(f)
        logger.info(f"Resuming: {len(map_sample_id_to_index)} samples already cached.")
    else:
        map_sample_id_to_index = {}

    #===================MODEL, OPTIMIZER & DEEPSPEED INIT===================#

    logger.info("Building model...")
    vae, vae_kwargs = load_vae_only(args, args.device)
    vae.enable_tiling()
    vae.eval()
    if not args.use_cache_text_encoder:
        assert False, "Cache text encoder must be used for caching model input!"
    else:
        logger.info("Loading text encoder cache...")
        text_encoder_cache = TextEncoderCache(Path(args.dataset_root) / "cache" / "text_encoder")
    
    #===================TRAINING LOOP===================#
    start_time = time.time()
    pbar = tqdm(enumerate(dataloader), desc=f"Caching model input")
    total_skip = 0
    for step, data in pbar:
        # update desc, print hour, minute, second
        elasped_time = time.time() - start_time
        pbar.set_description_str(f"Time {int(elasped_time // 3600):02d}:{int((elasped_time % 3600) // 60):02d}:{int(elasped_time % 60):02d}")
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

        # Check if ALL samples in this batch are already processed
        if all(sid in map_sample_id_to_index for sid in sample_id):
            continue

        # Prepare parital cond
        frames = []
        B, _, T, H, W = rgbs.shape
    
        logger.info("Computing model inputs...")
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
        ground_truth_rgb_depths = ground_truth_rgb_depths.to(vae.dtype).to(args.device)
        # Compute partial cond
        logger.info("Rendering partial cond rgb and depth...")
        partial_rgb_depths, partial_mask = render(frames, rgbs.shape, args, partial_rendering=True)
        # Encode partial cond to latent space
        partial_rgb_depths = partial_rgb_depths.to(vae.dtype).to(args.device)
        logger.info(f"Encoding partial cond with shape {partial_rgb_depths.shape} to latent space...")
        start_encode_time = time.time()
        partial_cond = vae.encode(
                partial_rgb_depths).latent_dist.sample()
        logger.info(f"Encoded partial cond to latent space with shape {partial_cond.shape} in {time.time() - start_encode_time:.2f} seconds.")

        partial_cond.mul_(vae.config.scaling_factor)
        partial_cond = partial_cond.to(vae.dtype)
        # Invert the mask
        partial_mask = 1 - partial_mask
        first_mask = partial_mask[:, :, 0:1, :, :]  # (B, 3, 1, H*2 + placeholder_row_length, W)
        partial_mask = torch.cat([first_mask, first_mask, first_mask, partial_mask], dim=2)  # (B, 3, T, H*2 + placeholder_row_length, W)
        partial_mask = torch.nn.functional.max_pool3d(
            partial_mask, kernel_size=(4, 8, 8), stride=(4, 8, 8)
        )
        # Invert the mask again
        partial_mask = 1 - partial_mask
        partial_mask = partial_mask[: , 0:1].to(vae.dtype)


        llm_i2v_text_states, llm_i2v_text_masks = text_encoder_cache.get_llm_i2v_text_state_and_mask(sample_id)
        llm_i2v_text_states = llm_i2v_text_states.to(args.device).to(vae.dtype)
        llm_i2v_text_masks = llm_i2v_text_masks.to(args.device)
        clipl_text_states = text_encoder_cache.get_clipl_text_state(sample_id).to(args.device).to(vae.dtype)
        batch = (ground_truth_rgb_depths, torch.tensor([0]), llm_i2v_text_states, llm_i2v_text_masks, clipl_text_states, {"type": ["video"]})
        start_prepare_time = time.time()
        fake_model = SimpleNamespace(
            dtype=vae.dtype,
            hidden_size=480,
            heads_num=5,
            rope_dim_list=[12, 42, 42],
            patch_size = [1, 2, 2]
        )
        logger.info("Preparing model inputs...")
        latents, model_kwargs, n_tokens, cond_latents = prepare_model_inputs \
                                                                (args, batch, args.device, \
                                                                fake_model, vae, None, None,\
                                                                args.rope_theta_rescale_factor, args.rope_interpolation_factor)
        logger.info(f"Computed model inputs with latents shape {latents.shape}, cond_latents shape {cond_latents.shape} in {time.time() - start_prepare_time:.2f} seconds.")

        # save latents, cond latents, model_kwargs, partial_cond, partial_mask to disk
        B = latents.shape[0]
        assert B == cond_latents.shape[0] == partial_cond.shape[0] == partial_mask.shape[0]

        for i in range(B):
            sid = sample_id[i]
            if not is_tensor_valid({
                "media": ground_truth_rgb_depths[i],
                "latents": latents[i],
                "cond_latents": cond_latents[i],
                "partial_cond": partial_cond[i],
                "partial_mask": partial_mask[i],
            }):
                total_skip += 1
                logger.warning(f"Skipping sample {sample_id} due to invalid partial cond tensor.")
                logger.warning(f"Total skipped samples: {total_skip}")
                continue
            # Save this specific sample to a shard file
            shard_data = {
                "latent": latents[i].cpu(),
                "cond_latent": cond_latents[i].cpu(),
                "partial_cond": partial_cond[i].cpu(),
                "partial_mask": partial_mask[i].cpu(),
            }
            torch.save(shard_data, shards_dir / f"{sid}.pt")
            map_sample_id_to_index[sid] = len(map_sample_id_to_index)
        with open(json_path, 'w') as f:
            json.dump(map_sample_id_to_index, f, indent=4)
        logger.info(f"Saved progress at step {step}, total cached samples: {len(map_sample_id_to_index)}")
        logger.warning(f"Total skipped samples so far: {total_skip}")
    # At the end of the script, after the loop:
    logger.info("All samples processed. Consolidating shards...")
    
    all_latents, all_cond, all_p_cond, all_p_mask = [], [], [], []
    # Sort keys to maintain consistent indexing
    sorted_ids = sorted(map_sample_id_to_index.keys(), key=lambda x: map_sample_id_to_index[x])
    
    for sid in tqdm(sorted_ids, desc="Staking tensors"):
        shard = torch.load(shards_dir / f"{sid}.pt", map_location='cpu')
        all_latents.append(shard["latent"])
        all_cond.append(shard["cond_latent"])
        all_p_cond.append(shard["partial_cond"])
        all_p_mask.append(shard["partial_mask"])

    torch.save(torch.stack(all_latents), Path(args.output_dir) / "latents.pt")
    torch.save(torch.stack(all_cond), Path(args.output_dir) / "cond_latents.pt")
    torch.save(torch.stack(all_p_cond), Path(args.output_dir) / "partial_cond.pt")
    torch.save(torch.stack(all_p_mask), Path(args.output_dir) / "partial_mask.pt")
    logger.info("Consolidation complete.")
