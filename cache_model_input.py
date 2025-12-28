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

from dataset.RealEstate10K_render import RealEstate10K_render as RealEstate10K
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

if __name__ == "__main__":
    args = parse_arg()
    print(args)

    #===================INITIALIZATION===================#

    dtype = PRECISION_TO_TYPE[args.precision]
    set_reproducibility(True, args.global_seed)

    #===================DATASET & DATALOADER===================#

    dataset_root = args.dataset_root
    dataset = RealEstate10K(dataset_root, set_name=args.task_flag, width=args.width, height=args.height, return_inverse_depth=True, return_partial_render=True)
    dataloader = DataLoader(dataset, batch_size=args.global_batch_size[0], shuffle=True, num_workers=args.num_workers)

    os.makedirs(args.output_dir, exist_ok=True)
    shards_dir = Path(args.output_dir) / "shards"
    os.makedirs(shards_dir, exist_ok=True)

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
    with torch.no_grad():
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
            gt_rgb_raw, gt_depth_raw = data['gt_render']
            p_rgb_raw, p_depth_raw = data['partial_render']
            p_mask_raw = data['partial_mask']
            print(prompt)
            print(sample_id)
            
            # Prepare parital cond
            B, _, T, H, W = rgbs.shape

            assert B == 1, "Batch size greater than 1 not supported in caching script."

            cache_hit = False
            for i in range(B):
                sid = sample_id[i]
                shard_path = shards_dir / f"{sid}.pt"
                if os.path.exists(shard_path):
                    logger.warning(f"Sample {sid} already cached, skipping...")
                    cache_hit = True
                    break
            if cache_hit:
                continue

            check_nan_inf({
                "rgbs": rgbs,
                "inverse_depths": inverse_depths,
                "intrinsics": intrinsics,
                "w2cs": w2cs,
            })

        
            logger.info("Computing model inputs...")
            # Compute ground truth media
            logger.info("Rendering ground truth rgb and depth...")
            ground_truth_rgb_depths = load_rgbs_depths(gt_rgb_raw, gt_depth_raw, args.placeholder_row_length)
            ground_truth_rgb_depths = ground_truth_rgb_depths.to(vae.dtype).to(args.device)
            # Compute partial cond
            logger.info("Rendering partial cond rgb and depth...")
            partial_rgb_depths= load_rgbs_depths(p_rgb_raw, p_depth_raw, args.placeholder_row_length)
            partial_mask = load_masks(p_mask_raw, args.placeholder_row_length)
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
                    "partial_cond": partial_cond[i].cpu(),
                    "partial_mask": partial_mask[i].cpu(),
                }):
                    total_skip += 1
                    logger.warning(f"Skipping sample {sid} due to invalid tensor.")
                    logger.warning(f"Total skipped samples: {total_skip}")
                    continue
                # Save this specific sample to a shard file
                shard_data = {
                    "latent": latents[i].cpu(),
                    "cond_latent": cond_latents[i].cpu(),
                    "partial_cond": partial_cond[i].cpu(),
                    "partial_mask": partial_mask[i].cpu(),
                }
                shard_path = shards_dir / f"{sid}.pt"
                if os.path.exists(shard_path):
                    total_skip += 1
                    logger.warning(f"Sample {sid} already cached, skipping...")
                    continue
                torch.save(shard_data, shard_path)
                logger.info(f"Saved cached model input for sample {sid} to {shard_path}.")
            logger.warning(f"Total skipped samples so far: {total_skip}")
        # At the end of the script, after the loop:
        logger.info("All samples processed.")
        
        # all_latents, all_cond, all_p_cond, all_p_mask = [], [], [], []
        # # Sort keys to maintain consistent indexing
        # sorted_ids = sorted(map_sample_id_to_index.keys(), key=lambda x: map_sample_id_to_index[x])
        
        # for sid in tqdm(sorted_ids, desc="Staking tensors"):
        #     shard = torch.load(shards_dir / f"{sid}.pt", map_location='cpu')
        #     all_latents.append(shard["latent"])
        #     all_cond.append(shard["cond_latent"])

        # torch.save(torch.stack(all_latents), Path(args.output_dir) / "latents.pt")
        # torch.save(torch.stack(all_cond), Path(args.output_dir) / "cond_latents.pt")
        # logger.info("Consolidation complete.")
