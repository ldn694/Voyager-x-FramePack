from loguru import logger
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Optional, List

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage 
import torchvision.transforms

from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.models.lora import adjust_lora_scale_text_encoder

from dataset.RealEstate10K import RealEstate10K
from voyager.text_encoder import TextEncoder
from voyager.config import *
from voyager.utils.data_utils import black_image
from voyager.diffusion import load_denoiser
from voyager.diffusion.flow import Transport
from voyager.utils.train_utils import set_reproducibility, load_state_dict, prepare_model_inputs
from voyager.inference import load_models
from voyager.modules.lora_layers import apply_lora_to_hunyuan_video, get_lora_parameters
from voyager.constants import PRECISION_TO_TYPE
from voyager.cache.text_cache import TextEncoderCache
from gather_realestate import norm_partial_render_output
from utils.render import Camera, Frame

def build_optimizer(param, args):
    optimizer = torch.optim.AdamW(
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

    args = parser.parse_args()
    return args

def training_step(x1, cond_latents, optimizer: torch.optim, denoiser: Transport, args, model_kwargs: Optional[dict] = None, partial_cond=None, partial_mask=None) -> float:
    x1 = x1.to(args.device)

    model.train()
    optimizer.zero_grad()
    target_dtype = PRECISION_TO_TYPE[args.precision]
    autocast_enabled = (
        target_dtype != torch.float32
    ) and not args.disable_autocast

    # i2v_mode=False if you’re not doing image-to-video stuff
    with torch.autocast(
        device_type="cuda", dtype=model.dtype, enabled=autocast_enabled
    ):
        model_output, terms = denoiser.training_losses(
            model=model,
            x1=x1,
            model_kwargs=model_kwargs,   # extra conditioning, if any
            timestep=None,     # random t; or set fixed timestep
            n_tokens=None,
            i2v_mode=args.i2v_mode,
            cond_latents=cond_latents,
            args=args,
            partial_cond=partial_cond,
            partial_mask=partial_mask,
        )

    loss = terms["loss"].mean()
    loss.backward()
    optimizer.step()

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

def render(frames, shape, args, partial_rendering=False):
    B, _, T, H, W = shape
    norm_rgbs = []
    norm_depths = []
    masks = []
    for b in range(B):
        for i in range(T):
            first_frame = frames[b][0]
            frame = frames[b][i]
            if not partial_rendering:
                rendered_image, mask, depth_buffer = frame.render(frame.camera)
            else:
                rendered_image, mask, depth_buffer = first_frame.render(frame.camera)
            rendered_image, mask, depth_buffer = norm_partial_render_output(rendered_image, mask, depth_buffer)
            norm_rgbs.append(torch.tensor(np.array(rendered_image).astype(np.float32) / 255.0).permute(2,0,1))  # (3, H, W)
            norm_depths.append(torch.tensor(depth_buffer).unsqueeze(0))  # (1, H, W)
            masks.append(torch.tensor(mask).unsqueeze(0))  # (1, H, W), range [0, 255]
    norm_rgbs = torch.stack(norm_rgbs, dim=0).reshape(B, T, 3, H, W).permute(0,2,1,3,4)  # (B, 3, T, H, W)
    norm_depths = torch.stack(norm_depths, dim=0).reshape(B, T, 1, H, W).permute(0,2,1,3,4)  # (B, 1, T, H, W)
    masks = torch.stack(masks, dim=0).reshape(B, T, 1, H, W).permute(0,2,1,3,4)  # (B, 1, T, H, W)
    rgb_depths = load_rgbs_depths(norm_rgbs, norm_depths, args.placeholder_row_length).to(args.device)  # (B, 3, T, H*2 + placeholder_row_length, W)
    masks = load_masks(masks, args.placeholder_row_length).to(args.device)  # (B, 3, T, H*2 + placeholder_row_length, W)
    return rgb_depths, masks

if __name__ == "__main__":
    args = parse_arg()
    print(args)
    denoiser = load_denoiser(args) 
    dtype = PRECISION_TO_TYPE[args.precision]
    set_reproducibility(True, args.global_seed)
    dataset_root = args.dataset_root
    dataset = RealEstate10K(dataset_root, set_name=args.task_flag, width=args.width, height=args.height, return_inverse_depth=True)
    dataloader = DataLoader(dataset, batch_size=args.global_batch_size[0], shuffle=True, num_workers=args.num_workers)
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
    apply_lora_to_hunyuan_video(
        model,
        r=args.lora_rank if hasattr(args, "lora_rank") else 8,
        lora_alpha=args.lora_alpha if hasattr(args, "lora_alpha") else 16.0,
        lora_dropout=getattr(args, "lora_dropout", 0.0),
        freeze_base=True,
    )

    # log model's trainable params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params}")
    logger.info(f"Trainable parameters: {trainable_params}")

    lora_params = get_lora_parameters(model)
    optimizer = build_optimizer(lora_params, args)

    # while (True):
    for batch_idx, data in tqdm(enumerate(dataloader), desc="Iterating over dataset"):
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
        ground_truth_rgb_depths = ground_truth_rgb_depths.to(model.dtype)
        # Compute partial cond
        logger.info("Rendering partial cond rgb and depth...")
        partial_rgb_depths, partial_mask = render(frames, rgbs.shape, args, partial_rendering=True)
        # Encode partial cond to latent space
        partial_rgb_depths = partial_rgb_depths.to(vae.dtype).to(args.device)
        logger.info(f"Encoding partial cond with shape {partial_rgb_depths.shape} to latent space...")
        partial_cond = vae.encode(
                partial_rgb_depths).latent_dist.sample()
        partial_cond.mul_(vae.config.scaling_factor)
        partial_cond = partial_cond.to(model.dtype)
        # Invert the mask
        partial_mask = 1 - partial_mask
        first_mask = partial_mask[:, :, 0:1, :, :]  # (B, 3, 1, H*2 + placeholder_row_length, W)
        partial_mask = torch.cat([first_mask, first_mask, first_mask, partial_mask], dim=2)  # (B, 3, T, H*2 + placeholder_row_length, W)
        partial_mask = torch.nn.functional.max_pool3d(
            partial_mask, kernel_size=(4, 8, 8), stride=(4, 8, 8)
        )
        # Invert the mask again
        partial_mask = 1 - partial_mask
        partial_mask = partial_mask[: , 0:1].to(model.dtype)


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
            llm_i2v_text_states = llm_i2v_text_states.to(args.device).to(model.dtype)
            llm_i2v_text_masks = llm_i2v_text_masks.to(args.device)
            clipl_text_states = text_encoder_cache.get_clipl_text_state(sample_id).to(args.device).to(model.dtype)
            batch = (ground_truth_rgb_depths, torch.tensor([0]), llm_i2v_text_states, llm_i2v_text_masks, clipl_text_states, {"type": ["video"]})
        latents, model_kwargs, n_tokens, cond_latents = prepare_model_inputs \
                                                                (args, batch, args.device, \
                                                                model, vae, text_encoder, text_encoder_2,\
                                                                args.rope_theta_rescale_factor, args.rope_interpolation_factor)
        import pdb; pdb.set_trace()
        training_step(latents, cond_latents, optimizer, denoiser, args, model_kwargs=model_kwargs, partial_cond=partial_cond, partial_mask=partial_mask)                                                
        print(f"{latents.shape=}")
        print(f"{cond_latents.shape=}")
        print(f"{n_tokens=}")
        # batch = 
        # prepare_model_inputs()