from typing import Union, Optional
from diffusers.models.embeddings import get_1d_rotary_pos_embed
import os
import time
import random
import functools
import json
from typing import List, Optional, Tuple, Union

from pathlib import Path
from loguru import logger

import torch
import torch.distributed as dist
from voyager.constants import PROMPT_TEMPLATE, NEGATIVE_PROMPT, PRECISION_TO_TYPE, NEGATIVE_PROMPT_I2V
from voyager.vae import load_vae
from voyager.modules import load_model
from voyager.text_encoder import TextEncoder
from voyager.utils.data_utils import align_to, get_closest_ratio, generate_crop_size_list
from voyager.utils.lora_utils import load_lora_for_pipeline
from voyager.utils.geometry import get_plucker_coordinates
from voyager.utils.train_utils import load_state_dict
from voyager.modules.compression_scheduler import CompressionScheduler
from voyager.modules.posemb_layers import get_nd_rotary_pos_embed
from voyager.modules.fp8_optimization import convert_fp8_linear
from voyager.diffusion.schedulers import FlowMatchDiscreteScheduler
from voyager.diffusion.pipelines import HunyuanVideoPipeline
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from safetensors.torch import load_file
import cv2
import pyexr
import torchvision.transforms as T

from voyager.modules.lora_layers import apply_lora_to_hunyuan_video, load_lora_state_dict
from voyager.modules.custom_patch_embed import apply_patch_adapter_to_hunyuan_video, load_patch_adapter_state_dict
from voyager.modules.multi_kernel import apply_multikernel_to_hunyuan_video, load_multikernel_state_dict
from voyager.modules.double_branch import apply_double_branch_to_hunyuan_video, load_double_branch_state_dict
from voyager.modules.transformer_branch_config import get_transformer_branch_config_from_args

try:
    import xfuser
    from xfuser.core.distributed import (
        get_sequence_parallel_world_size,
        get_sequence_parallel_rank,
        get_sp_group,
        initialize_model_parallel,
        init_distributed_environment
    )
except:
    xfuser = None
    get_sequence_parallel_world_size = None
    get_sequence_parallel_rank = None
    get_sp_group = None
    initialize_model_parallel = None
    init_distributed_environment = None


def load_init_camera_params(camera_path, Height, Width):
    if not os.path.exists(camera_path):
        raise FileNotFoundError(f"Camera data not found: {camera_path}")

    with open(camera_path, 'r') as f:
        data = json.load(f)

    cameras = data.get("cameras_interp", [])
    extrinsics = np.array(cameras).reshape(-1, 4, 4)
    fx = data.get("focal_length", 500) / Width
    fy = data.get("focal_length", 500) / Height

    cx = 0.5
    cy = 0.5
    intrinsics = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    intrinsics = np.stack([intrinsics] * extrinsics.shape[0], axis=0)

    c2w = torch.from_numpy(np.linalg.inv(extrinsics)).float()
    intrinsics = torch.from_numpy(intrinsics).float()

    # camera centering
    camera_dist_2med = torch.norm(
        c2w[:, :3, 3] - c2w[:, :3, 3].median(0, keepdim=True).values,
        dim=-1,
    )
    valid_mask = camera_dist_2med <= torch.clamp(
        torch.quantile(camera_dist_2med, 0.97) * 10,
        max=1e6,
    )
    c2w[:, :3, 3] -= c2w[valid_mask, :3, 3].mean(0, keepdim=True)
    w2c = torch.from_numpy(np.linalg.inv(np.array(c2w))).float()

    # camera normalization
    camera_dists = c2w[:, :3, 3].clone()
    translation_scaling_factor = (
        2
        if torch.isclose(
            torch.norm(camera_dists[0]),
            torch.zeros(1),
            atol=1e-5,
        ).any()
        else (2 / torch.norm(camera_dists[0]))
    )
    w2c[:, :3, 3] *= translation_scaling_factor
    c2w[:, :3, 3] *= translation_scaling_factor

    # get plucker coordinates
    plucker_coordinate = get_plucker_coordinates(
        extrinsics_src=w2c[0],
        extrinsics=w2c,
        intrinsics=intrinsics.float().clone(),
        target_size=(Height//8, Width//8),
    )
    return plucker_coordinate


###############################################
# 20250308 pftq: Riflex workaround to fix 192-frame-limit bug, credit to Kijai for finding it in ComfyUI
# and thu-ml for making it
# https://github.com/thu-ml/RIFLEx/blob/main/riflex_utils.py


def get_1d_rotary_pos_embed_riflex(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    k: Optional[int] = None,
    L_test: Optional[int] = None,
):
    """
    RIFLEx: Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        k (`int`, *optional*, defaults to None): the index for the intrinsic frequency in RoPE
        L_test (`int`, *optional*, defaults to None): the number of frames for inference
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]

    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=pos.device)
                  [: (dim // 2)].float() / dim)
    )  # [D/2]

    # === Riflex modification start ===
    # Reduce the intrinsic frequency to stay within a single period after extrapolation (see Eq. (8)).
    # Empirical observations show that a few videos may exhibit repetition in the tail frames.
    # To be conservative, we multiply by 0.9 to keep the extrapolated length below 90% of a single period.
    if k is not None:
        freqs[k-1] = 0.9 * 2 * torch.pi / L_test
    # === Riflex modification end ===

    freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # lumina
        freqs_cis = torch.polar(torch.ones_like(
            freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis


###############################################

def parallelize_transformer(pipe):
    transformer = pipe.transformer
    original_forward = transformer.forward

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,  # Should be in range(0, 1000).
        text_states: torch.Tensor = None,
        text_mask: torch.Tensor = None,  # Now we don't use it.
        # Text embedding for modulation.
        text_states_2: Optional[torch.Tensor] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        freqs_cos_cond: Optional[torch.Tensor] = None,
        freqs_sin_cond: Optional[torch.Tensor] = None,
        # Guidance for modulation, should be cfg_scale x 1000.
        guidance: torch.Tensor = None,
        return_dict: bool = True,
    ):
        if x.shape[-2] // 2 % get_sequence_parallel_world_size() == 0:
            # try to split x by height
            split_dim = -2
        elif x.shape[-1] // 2 % get_sequence_parallel_world_size() == 0:
            # try to split x by width
            split_dim = -1
        else:
            raise ValueError(
            f"Cannot split video sequence into ulysses_degree x ring_degree \
            ({get_sequence_parallel_world_size()}) parts evenly")

        # patch sizes for the temporal, height, and width dimensions are 1, 2, and 2.
        temporal_size, h, w = x.shape[2], x.shape[3] // 2, x.shape[4] // 2
        h_cond = h // 2

        x = torch.chunk(x, get_sequence_parallel_world_size(), dim=split_dim)[
            get_sequence_parallel_rank()]

        dim_thw = freqs_cos.shape[-1]
        freqs_cos = freqs_cos.reshape(temporal_size, h, w, dim_thw)
        freqs_cos = torch.chunk(freqs_cos, get_sequence_parallel_world_size(
        ), dim=split_dim - 1)[get_sequence_parallel_rank()]
        freqs_cos = freqs_cos.reshape(-1, dim_thw)
        dim_thw = freqs_sin.shape[-1]
        freqs_sin = freqs_sin.reshape(temporal_size, h, w, dim_thw)
        freqs_sin = torch.chunk(freqs_sin, get_sequence_parallel_world_size(
        ), dim=split_dim - 1)[get_sequence_parallel_rank()]
        freqs_sin = freqs_sin.reshape(-1, dim_thw)

        dim_thw_cond = freqs_cos_cond.shape[-1]
        freqs_cos_cond = freqs_cos_cond.reshape(temporal_size, h_cond, w, dim_thw_cond)
        freqs_cos_cond = torch.chunk(freqs_cos_cond, get_sequence_parallel_world_size(
        ), dim=split_dim - 1)[get_sequence_parallel_rank()]
        freqs_cos_cond = freqs_cos_cond.reshape(-1, dim_thw_cond)
        dim_thw_cond = freqs_sin_cond.shape[-1]
        freqs_sin_cond = freqs_sin_cond.reshape(temporal_size, h_cond, w, dim_thw_cond)
        freqs_sin_cond = torch.chunk(freqs_sin_cond, get_sequence_parallel_world_size(
        ), dim=split_dim - 1)[get_sequence_parallel_rank()]
        freqs_sin_cond = freqs_sin_cond.reshape(-1, dim_thw_cond)

        from xfuser.core.long_ctx_attention import xFuserLongContextAttention

        for block in transformer.double_blocks + transformer.single_blocks:
            block.hybrid_seq_parallel_attn = xFuserLongContextAttention()

        output = original_forward(
            x,
            t,
            text_states,
            text_mask,
            text_states_2,
            freqs_cos,
            freqs_sin,
            freqs_cos_cond,
            freqs_sin_cond,
            guidance,
            return_dict,
        )

        return_dict = not isinstance(output, tuple)
        sample = output["x"]
        sample = get_sp_group().all_gather(sample, dim=split_dim)
        output["x"] = sample
        return output

    new_forward = new_forward.__get__(transformer)
    transformer.forward = new_forward

def load_vae_only(args, device):
    vae, _, s_ratio, t_ratio = load_vae(
        args.vae,
        args.vae_precision,
        logger=logger,
        device=device
    )
    vae_kwargs = {"s_ratio": s_ratio, "t_ratio": t_ratio}
    vae = vae.to(device)
    return vae, vae_kwargs

def load_models(args, device, logger, pretrained_model_path):
    factor_kwargs = {"device": device, "dtype": PRECISION_TO_TYPE[args.precision]}
    if args.i2v_mode and args.i2v_condition_type == "latent_concat":
        in_channels = args.latent_channels * 3 + 2
        image_embed_interleave = 2
    elif args.i2v_mode and args.i2v_condition_type == "token_replace":
        in_channels = args.latent_channels
        image_embed_interleave = 4
    else:
        in_channels = args.latent_channels
        image_embed_interleave = 1
    out_channels = args.latent_channels

    model = load_model(
        args,
        in_channels=in_channels,
        out_channels=out_channels,
        factor_kwargs=factor_kwargs,
    )
    
    # model = model.to(device)
    if pretrained_model_path is not None:
        if not args.load_all:
            model = load_state_dict(args, model, logger, pretrained_model_path)

    if args.use_lora and args.lora_path is not None:
        logger.info(f"Loading LoRA from {args.lora_path}...")
        lora_ckpt = torch.load(args.lora_path, map_location="cpu", weights_only=False)
        lora_args = lora_ckpt["args"]
        logger.info(lora_args)
        # If running the DMD2 inference path, the saved adapter is "gen" with
        # adapter-agnostic keys; route it explicitly so multi-adapter routing works.
        dmd2_target = "gen" if getattr(args, "dmd2_steps", 0) > 0 else None
        apply_lora_to_hunyuan_video(
            model,
            r=lora_args["lora_rank"] if "lora_rank" in lora_args else 4,
            lora_alpha=lora_args["lora_alpha"] if "lora_alpha" in lora_args else 16.0,
            lora_dropout=lora_args["lora_dropout"] if "lora_dropout" in lora_args else 0.0,
            freeze_base=True,
            adapter_name=dmd2_target if dmd2_target is not None else "default",
        )
        load_lora_state_dict(model, lora_ckpt["lora"], strict=False, adapter_name=dmd2_target)
    if args.use_patch_adapter and args.patch_adapter_path is not None:
        logger.info(f"Loading Patch Adapter from {args.patch_adapter_path}...")
        patch_adapter_ckpt = torch.load(args.patch_adapter_path, map_location="cpu", weights_only=False)
        patch_adapter_args = patch_adapter_ckpt["args"]
        logger.info(patch_adapter_args)
        apply_patch_adapter_to_hunyuan_video(
            model,
            new_patch_size=tuple(patch_adapter_args["patch_adapter_size"]) if "patch_adapter_size" in patch_adapter_args else (1,2,2),
            freeze_base=True,
        )
        load_patch_adapter_state_dict(model, patch_adapter_ckpt["patch_adapter"], strict=False)
    if args.use_multiple_kernels and args.multiple_kernels_path is not None:
        logger.info(f"Loading Multiple Kernels from {args.multiple_kernels_path}...")
        multiple_kernels_ckpt = torch.load(args.multiple_kernels_path, map_location="cpu", weights_only=False)
        multiple_kernels_args = multiple_kernels_ckpt["args"]
        logger.info(multiple_kernels_args)
        apply_multikernel_to_hunyuan_video(
            model,
            patch_sizes=multiple_kernels_args["kernel_sizes"] if "kernel_sizes" in multiple_kernels_args else [[1, 2, 2]],
            copy_old_weights=True
        )
        load_multikernel_state_dict(model, multiple_kernels_ckpt["multi_kernel"], strict=False)
    if args.use_double_branch and args.double_branch_path is not None:
        logger.info(f"Loading Double Branch from {args.double_branch_path}...")
        double_branch_ckpt = torch.load(args.double_branch_path, map_location="cpu", weights_only=False)
        double_branch_args = double_branch_ckpt["args"]
        logger.info(double_branch_args)
        apply_double_branch_to_hunyuan_video(
            model, 
            second_branch_config=get_transformer_branch_config_from_args(double_branch_args["second_branch_transformer_config"]),
            second_branch_mm_blocks_depth=double_branch_args["second_branch_mm_blocks_depth"],
            freeze_base=True,
        )
        load_double_branch_state_dict(model, double_branch_ckpt["double_branch"], strict=False)
        
    if pretrained_model_path is not None:
        if args.load_all:
            logger.info(f"Loading all model states from {pretrained_model_path}...")
            checkpoint = torch.load(pretrained_model_path, map_location="cpu", weights_only=False)
            load_output = model.load_state_dict(checkpoint["model"], strict=False)
            logger.warning("Missing keys:", load_output.missing_keys)
            logger.warning("Unexpected keys:", load_output.unexpected_keys)

    if hasattr(args, "train_from_scratch") and args.train_from_scratch:
        logger.info("Training from scratch, not loading any pre-trained weights.")
        model.train()
    else:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # VAE
    vae, _, s_ratio, t_ratio = load_vae(
        args.vae,
        args.vae_precision,
        logger=logger,
        device=device if not args.use_cpu_offload else "cpu",
    )
    vae_kwargs = {"s_ratio": s_ratio, "t_ratio": t_ratio}
    vae = vae.to(device)

    if args.i2v_mode:
        args.text_encoder = "llm-i2v"
        args.tokenizer = "llm-i2v"
        args.prompt_template = "dit-llm-encode-i2v"
        args.prompt_template_video = "dit-llm-encode-video-i2v"

    if args.prompt_template_video is not None:
        crop_start = PROMPT_TEMPLATE[args.prompt_template_video].get(
            "crop_start", 0)
    elif args.prompt_template is not None:
        crop_start = PROMPT_TEMPLATE[args.prompt_template].get(
            "crop_start", 0)
    else:
        crop_start = 0
    max_length = args.text_len + crop_start

    prompt_template = PROMPT_TEMPLATE[args.prompt_template] if args.prompt_template is not None else None
    prompt_template_video = PROMPT_TEMPLATE[
        args.prompt_template_video] if args.prompt_template_video is not None else None

    if hasattr(args, "use_cache_text_encoder") and args.use_cache_text_encoder:
        logger.info("Text encoder caching is enabled.")
        text_encoder = None
        text_encoder_2 = None
    else:
        # Text encoder
        text_encoder = TextEncoder(
            text_encoder_type=args.text_encoder,
            max_length=max_length,
            text_encoder_precision=args.text_encoder_precision,
            tokenizer_type=args.tokenizer,
            i2v_mode=args.i2v_mode,
            prompt_template=prompt_template,
            prompt_template_video=prompt_template_video,
            hidden_state_skip_layer=args.hidden_state_skip_layer,
            apply_final_norm=args.apply_final_norm,
            reproduce=args.reproduce,
            logger=logger,
            device=device if not args.use_cpu_offload else "cpu",
            image_embed_interleave=image_embed_interleave
        ).to(device)

        text_encoder_2 = None
        if args.text_encoder_2 is not None:
            text_encoder_2 = TextEncoder(
                text_encoder_type=args.text_encoder_2,
                max_length=args.text_len_2,
                text_encoder_precision=args.text_encoder_precision_2,
                tokenizer_type=args.tokenizer_2,
                reproduce=args.reproduce,
                logger=logger,
                device=device if not args.use_cpu_offload else "cpu",
            ).to(device)

    return model, vae, text_encoder, text_encoder_2, vae_kwargs


class Inference(object):
    def __init__(
        self,
        args,
        vae,
        vae_kwargs,
        text_encoder,
        model,
        text_encoder_2=None,
        pipeline=None,
        use_cpu_offload=False,
        device=None,
        logger=None,
        parallel_args=None,
    ):
        self.vae = vae
        self.vae_kwargs = vae_kwargs
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.model = model
        self.pipeline = pipeline
        self.use_cpu_offload = use_cpu_offload
        self.args = args
        self.device = (
            device
            if device is not None
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.logger = logger
        self.parallel_args = parallel_args

    # 20250316 pftq: Fixed multi-GPU loading times going up to 20 min due to loading contention
    # by loading models only to one GPU and braodcasting to the rest.
    @classmethod
    def from_pretrained(cls, pretrained_model_path, args, device=None, **kwargs):
        """
        Initialize the Inference pipeline.

        Args:
            pretrained_model_path (str or pathlib.Path): The model path, including t2v, \
                text encoder and vae checkpoints.
            args (argparse.Namespace): The arguments for the pipeline.
            device (int): The device for inference. Default is None.
        """
        logger.info(
            f"Got text-to-video model root path: {pretrained_model_path}")

        # ========================================================================
        # Initialize Distributed Environment
        # ========================================================================
        # 20250316 pftq: Modified to extract rank and world_size early for sequential loading
        if args.ulysses_degree > 1 or args.ring_degree > 1:
            assert xfuser is not None, "Ulysses Attention and Ring Attention requires xfuser package."
            assert args.use_cpu_offload is False, "Cannot enable use_cpu_offload in the distributed environment."
            # 20250316 pftq: Set local rank and device explicitly for NCCL
            local_rank = int(os.environ['LOCAL_RANK'])
            device = torch.device(f"cuda:{local_rank}")
            # 20250316 pftq: Set CUDA device explicitly
            torch.cuda.set_device(local_rank)
            # 20250316 pftq: Removed device_id, rely on set_device
            dist.init_process_group("nccl")
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            assert world_size == args.ring_degree * args.ulysses_degree, \
                "number of GPUs should be equal to ring_degree * ulysses_degree."
            init_distributed_environment(rank=rank, world_size=world_size)
            initialize_model_parallel(
                sequence_parallel_degree=world_size,
                ring_degree=args.ring_degree,
                ulysses_degree=args.ulysses_degree,
            )
        else:
            rank = 0  # 20250316 pftq: Default rank for single GPU
            world_size = 1  # 20250316 pftq: Default world_size for single GPU
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"

        parallel_args = {"ulysses_degree": args.ulysses_degree,
                         "ring_degree": args.ring_degree}
        torch.set_grad_enabled(False)

        # ========================================================================
        # Build main model, VAE, and text encoder sequentially on rank 0
        # ========================================================================
        # 20250316 pftq: Load models only on rank 0, then broadcast
        if rank == 0:
            logger.info("Building model...")
            model, vae, text_encoder, text_encoder_2, vae_kwargs = \
                load_models(args, device, logger, pretrained_model_path)
        else:
            # 20250316 pftq: Initialize as None on non-zero ranks
            model = None
            vae = None
            vae_kwargs = None
            text_encoder = None
            text_encoder_2 = None

        # 20250316 pftq: Broadcast models to all ranks
        if world_size > 1:
            logger.info(f"Rank {rank}: Starting broadcast synchronization")
            dist.barrier()  # Ensure rank 0 finishes loading before broadcasting
            if rank != 0:
                # Reconstruct model skeleton on non-zero ranks
                model, vae, text_encoder, text_encoder_2, vae_kwargs = \
                    load_models(args, device, logger, pretrained_model_path)

            # Broadcast model parameters with logging
            logger.info(f"Rank {rank}: Broadcasting model parameters")
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
            model.eval()
            logger.info(f"Rank {rank}: Broadcasting VAE parameters")
            for param in vae.parameters():
                dist.broadcast(param.data, src=0)
            # 20250316 pftq: Use broadcast_object_list for vae_kwargs
            logger.info(f"Rank {rank}: Broadcasting vae_kwargs")
            vae_kwargs_list = [vae_kwargs] if rank == 0 else [None]
            dist.broadcast_object_list(vae_kwargs_list, src=0)
            vae_kwargs = vae_kwargs_list[0]
            logger.info(f"Rank {rank}: Broadcasting text_encoder parameters")
            for param in text_encoder.parameters():
                dist.broadcast(param.data, src=0)
            if text_encoder_2 is not None:
                logger.info(
                    f"Rank {rank}: Broadcasting text_encoder_2 parameters")
                for param in text_encoder_2.parameters():
                    dist.broadcast(param.data, src=0)

        return cls(
            args=args,
            vae=vae,
            vae_kwargs=vae_kwargs,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            model=model,
            use_cpu_offload=args.use_cpu_offload,
            device=device,
            logger=logger,
            parallel_args=parallel_args
        )

    @staticmethod
    def parse_size(size):
        if isinstance(size, int):
            size = [size]
        if not isinstance(size, (list, tuple)):
            raise ValueError(
                f"Size must be an integer or (height, width), got {size}.")
        if len(size) == 1:
            size = [size[0], size[0]]
        if len(size) != 2:
            raise ValueError(
                f"Size must be an integer or (height, width), got {size}.")
        return size


class HunyuanVideoSampler(Inference):
    def __init__(
        self,
        args,
        vae,
        vae_kwargs,
        text_encoder,
        model,
        text_encoder_2=None,
        pipeline=None,
        use_cpu_offload=False,
        device=0,
        logger=None,
        parallel_args=None
    ):
        super().__init__(
            args,
            vae,
            vae_kwargs,
            text_encoder,
            model,
            text_encoder_2=text_encoder_2,
            pipeline=pipeline,
            use_cpu_offload=use_cpu_offload,
            device=device,
            logger=logger,
            parallel_args=parallel_args
        )

        self.pipeline = self.load_diffusion_pipeline(
            args=args,
            vae=self.vae,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            model=self.model,
            device=self.device,
        )

        if args.i2v_mode:
            self.default_negative_prompt = NEGATIVE_PROMPT_I2V
            # if args.use_lora:
            #     self.pipeline = load_lora_for_pipeline(
            #         self.pipeline, args.lora_path, LORA_PREFIX_TRANSFORMER="Hunyuan_video_I2V_lora",
            #         alpha=args.lora_scale, device=self.device,
            #         is_parallel=(self.parallel_args['ulysses_degree'] > 1 or self.parallel_args['ring_degree'] > 1))
            #     logger.info(
            #     f"load lora {args.lora_path} into pipeline, lora scale is {args.lora_scale}.")
        else:
            self.default_negative_prompt = NEGATIVE_PROMPT

        if self.parallel_args['ulysses_degree'] > 1 or self.parallel_args['ring_degree'] > 1:
            parallelize_transformer(self.pipeline)

    def load_diffusion_pipeline(
        self,
        args,
        vae,
        text_encoder,
        text_encoder_2,
        model,
        scheduler=None,
        device=None,
        progress_bar_config=None,
    ):
        if scheduler is None:
            if args.denoise_type == "flow":
                scheduler = FlowMatchDiscreteScheduler(
                    shift=args.flow_shift,
                    reverse=args.flow_reverse,
                    solver=args.flow_solver,
                )
            else:
                raise ValueError(f"Invalid denoise type {args.denoise_type}")

        pipeline = HunyuanVideoPipeline(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            transformer=model,
            scheduler=scheduler,
            progress_bar_config=progress_bar_config,
            args=args,
        )
        if self.use_cpu_offload:
            pipeline.enable_sequential_cpu_offload()
        else:
            pipeline = pipeline.to(device)

        return pipeline

    # 20250317 pftq: Modified to use Riflex when >192 frames
    def get_rotary_pos_embed(self, video_length, height, width, use_old_patch_size=False):
        target_ndim = 3
        ndim = 5 - 2  # B, C, F, H, W -> F, H, W

        # Compute latent sizes based on VAE type
        if "884" in self.args.vae:
            latents_size = [(video_length - 1) // 4 +
                            1, height // 8, width // 8]
        elif "888" in self.args.vae:
            latents_size = [(video_length - 1) // 8 +
                            1, height // 8, width // 8]
        else:
            latents_size = [video_length, height // 8, width // 8]

        # Compute rope sizes
        if hasattr(self.model, "old_patch_size") and use_old_patch_size:
            ps = self.model.old_patch_size
        else:
            ps = self.model.patch_size
        if isinstance(ps, int):
            assert all(s % ps == 0 for s in latents_size), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({ps}), "
                f"but got {latents_size}."
            )
            rope_sizes = [s // ps for s in latents_size]
        elif isinstance(ps, list):
            assert all(
                s % ps[idx] == 0
                for idx, s in enumerate(latents_size)
            ), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({ps}), "
                f"but got {latents_size}."
            )
            rope_sizes = [s // ps[idx]
                        for idx, s in enumerate(latents_size)]

        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)
                                ) + rope_sizes  # Pad time axis

        # 20250316 pftq: Add RIFLEx logic for > 192 frames
        L_test = rope_sizes[0]  # Latent frames
        L_train = 25  # Training length from HunyuanVideo
        actual_num_frames = video_length  # Use input video_length directly

        head_dim = self.model.hidden_size // self.model.heads_num
        rope_dim_list = self.model.rope_dim_list or [
            head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(
            rope_dim_list) == head_dim, "sum(rope_dim_list) must equal head_dim"

        if actual_num_frames > 192:
            k = 2+((actual_num_frames + 3) // (4 * L_train))
            k = max(4, min(8, k))
            logger.debug(f"actual_num_frames = {actual_num_frames} > 192, RIFLEx applied with k = {k}")

            # Compute positional grids for RIFLEx
            axes_grids = [torch.arange(
                size, device=self.device, dtype=torch.float32) for size in rope_sizes]
            grid = torch.meshgrid(*axes_grids, indexing="ij")
            grid = torch.stack(grid, dim=0)  # [3, t, h, w]
            pos = grid.reshape(3, -1).t()  # [t * h * w, 3]

            # Apply RIFLEx to temporal dimension
            freqs = []
            for i in range(3):
                if i == 0:  # Temporal with RIFLEx
                    freqs_cos, freqs_sin = get_1d_rotary_pos_embed_riflex(
                        rope_dim_list[i],
                        pos[:, i],
                        theta=self.args.rope_theta,
                        use_real=True,
                        k=k,
                        L_test=L_test
                    )
                else:  # Spatial with default RoPE
                    freqs_cos, freqs_sin = get_1d_rotary_pos_embed_riflex(
                        rope_dim_list[i],
                        pos[:, i],
                        theta=self.args.rope_theta,
                        use_real=True,
                        k=None,
                        L_test=None
                    )
                freqs.append((freqs_cos, freqs_sin))
                logger.debug(f"freq[{i}] shape: {freqs_cos.shape}, device: {freqs_cos.device}")

            freqs_cos = torch.cat([f[0] for f in freqs], dim=1)
            freqs_sin = torch.cat([f[1] for f in freqs], dim=1)
            logger.debug(f"freqs_cos shape: {freqs_cos.shape}, device: {freqs_cos.device}")
        else:
            # 20250316 pftq: Original code for <= 192 frames
            logger.debug(f"actual_num_frames = {actual_num_frames} <= 192, using original RoPE")
            freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
                rope_dim_list,
                rope_sizes,
                theta=self.args.rope_theta,
                use_real=True,
                theta_rescale_factor=1,
            )
            logger.debug(f"freqs_cos shape: {freqs_cos.shape}, device: {freqs_cos.device}")

        return freqs_cos, freqs_sin

    def process(self, pil_img):
        if pil_img.mode == 'L':
            pil_img = pil_img.convert('RGB')
        image = np.asarray(pil_img, dtype=np.float32) / 255.
        image = image[:, :, :3]
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        image = T.Normalize(mean=[0.5, 0.5, 0.5], std=[
                            0.5, 0.5, 0.5], inplace=True)(image)
        return image

    def load_image(self, path, image_size=(512, 512)):
        if isinstance(path, tuple):
            ref_rgb = self.load_image(path[0], image_size)
            ref_depth = self.load_image(path[1], image_size)
            return torch.cat([ref_rgb, torch.ones_like(ref_rgb)[..., :64, :], ref_depth], dim=1)

        if path.endswith('.exr'):
            depth = torch.from_numpy(cv2.resize(pyexr.read(path).squeeze(
            ), (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)).float()
            image = depth.unsqueeze(0).repeat(3, 1, 1)
            image = T.Normalize(mean=[0.5, 0.5, 0.5], std=[
                                0.5, 0.5, 0.5], inplace=True)(image)
        else:
            pil_img = Image.open(path) if isinstance(
                path, str) else Image.fromarray(path)
            pil_img = pil_img.resize((image_size[1], image_size[0]))
            image = self.process(pil_img)

        return image

    @torch.no_grad()
    def predict(
        self,
        prompt,
        height=192,
        width=336,
        video_length=129,
        seed=None,
        negative_prompt=None,
        infer_steps=50,
        guidance_scale=6.0,
        flow_shift=5.0,
        embedded_guidance_scale=None,
        batch_size=1,
        num_videos_per_prompt=1,
        i2v_mode=False,
        i2v_resolution="720p",
        i2v_image_path=None,
        i2v_condition_type=None,
        i2v_stability=True,
        ulysses_degree=1,
        ring_degree=1,
        ref_images=None,
        partial_cond=None,
        partial_mask=None,
        use_kernel_indices=None,
        step_sample=0,
        attn_map=0,
        dmd2_steps=0,
        **kwargs,
    ):
        out_dict = dict()

        if isinstance(seed, torch.Tensor):
            seed = seed.tolist()
        if seed is None:
            seeds = [
                random.randint(0, 1_000_000)
                for _ in range(batch_size * num_videos_per_prompt)
            ]
        elif isinstance(seed, int):
            seeds = [
                seed + i
                for _ in range(batch_size)
                for i in range(num_videos_per_prompt)
            ]
        elif isinstance(seed, (list, tuple)):
            if len(seed) == batch_size:
                seeds = [
                    int(seed[i]) + j
                    for i in range(batch_size)
                    for j in range(num_videos_per_prompt)
                ]
            elif len(seed) == batch_size * num_videos_per_prompt:
                seeds = [int(s) for s in seed]
            else:
                raise ValueError(
                    f"Length of seed must be equal to number of prompt(batch_size) or "
                    f"batch_size * num_videos_per_prompt ({batch_size} * {num_videos_per_prompt}), got {seed}."
                )
        else:
            raise ValueError(
                f"Seed must be an integer, a list of integers, or None, got {seed}."
            )
        generator = [torch.Generator(
            self.device).manual_seed(seed) for seed in seeds]
        out_dict["seeds"] = seeds

        if width <= 0 or height <= 0 or video_length <= 0:
            raise ValueError(
                f"`height` and `width` and `video_length` must be positive integers, \
                    got height={height}, width={width}, video_length={video_length}"
            )
        if (video_length - 1) % 4 != 0:
            raise ValueError(
                f"`video_length-1` must be a multiple of 4, got {video_length}"
            )

        logger.info(
            f"Input (height, width, video_length) = ({height}, {width}, {video_length})"
        )

        target_height = height  # align_to(height, 16)
        target_height = target_height * 2 + 64
        target_width = width  # align_to(width, 16)
        target_video_length = video_length

        out_dict["size"] = (target_height, target_width, target_video_length)

        if not isinstance(prompt, str):
            raise TypeError(
                f"`prompt` must be a string, but got {type(prompt)}")
        prompt = [prompt.strip()]

        if negative_prompt is None or negative_prompt == "":
            negative_prompt = self.default_negative_prompt
        if guidance_scale == 1.0:
            negative_prompt = ""
        if not isinstance(negative_prompt, str):
            raise TypeError(
                f"`negative_prompt` must be a string, but got {type(negative_prompt)}"
            )
        negative_prompt = [negative_prompt.strip()]

        scheduler = FlowMatchDiscreteScheduler(
            shift=flow_shift,
            reverse=self.args.flow_reverse,
            solver=self.args.flow_solver
        )
        self.pipeline.scheduler = scheduler

        # Set the target image size for processing reference images and partial conditions
        # This size should match the model's expected input dimensions
        closest_size = (height, width)
        
        # Load and preprocess reference images for the video generation
        # Convert image paths to pixel values and stack them into a batch
        ref_images_pixel_values = [self.load_image(
            image_path, image_size=closest_size) for image_path in ref_images]
        ref_images_pixel_values = torch.cat(
            ref_images_pixel_values).unsqueeze(0).unsqueeze(2).to(self.device)
        
        # Convert pixel values back to PIL Image format for visualization/debugging
        # Normalize from [-1, 1] range to [0, 255] range and save as PNG
        ref_images = [Image.fromarray(((torch.clamp(ref_images_pixel_values[0, :, 0].permute(
            1, 2, 0), min=-1, max=1).cpu().numpy() + 1) * 0.5 * 255).astype(np.uint8))]

        # Load partial condition images (frames that will guide the video generation)
        # These images provide temporal guidance for the video sequence
        partial_cond = [self.load_image(
            image_path, image_size=closest_size) for image_path in partial_cond]
        partial_cond = torch.stack(
            partial_cond, dim=1).unsqueeze(0).to(self.device)
        
        # Load partial mask images (indicate which regions should be preserved/modified)
        # Masks control which parts of the video should be generated vs. kept from conditions
        partial_mask = [self.load_image(
            image_path, image_size=closest_size) for image_path in partial_mask]
        partial_mask = torch.stack(
            partial_mask, dim=1).unsqueeze(0).to(self.device)
        
        logger.info("Before encoding condition:")
        logger.info(f"{partial_mask.shape=} {partial_mask.min()=} {partial_mask.max()=}")
        logger.info(f"{partial_cond.shape=} {partial_cond.min()=} {partial_cond.max()=}")

        # Use automatic mixed precision for memory efficiency during encoding
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            # Enable VAE tiling for processing large images/videos efficiently
            self.pipeline.vae.enable_tiling()

            # Encode reference images to latent space representation
            # This creates a compressed representation that guides the video generation
            ref_latents = self.pipeline.vae.encode(
                ref_images_pixel_values).latent_dist.sample()  # B, C, F, H, W
            ref_latents.mul_(self.pipeline.vae.config.scaling_factor)

            # Encode partial condition images to latent space
            # These latents provide temporal guidance for the video sequence
            partial_cond = self.pipeline.vae.encode(
                partial_cond).latent_dist.sample()
            partial_cond.mul_(self.pipeline.vae.config.scaling_factor)

            # Process mask frames for controlling video generation
            # Invert the mask so that 1 indicates regions to be generated
            mask_frames = 1 - partial_mask
            first_mask = mask_frames[:, :, 0:1]  # Extract the first mask frame
            
            # Prepend 3 copies of the first mask to create temporal consistency
            # This ensures the initial frames have consistent masking
            mask_frames = torch.cat(
                [first_mask, first_mask, first_mask, mask_frames], dim=2)
            
            # Apply 3D max pooling to downsample masks to match latent space dimensions
            # Reduces temporal dimension by 4, spatial dimensions by 8
            mask_frames = torch.nn.functional.max_pool3d(
                mask_frames,  # Input: [1, C, F, H, W]
                kernel_size=(4, 8, 8),  # Reduce F by 4, H and W by 8
                stride=(4, 8, 8)
            )  # Output: [C, F//4, H//8, W//8]
            
            # Invert the mask again so that 1 indicates regions to preserve
            mask_frames = 1 - mask_frames
            partial_mask = mask_frames[:, 0:1]

            # # Load camera parameters (Plücker coordinates) for 3D scene understanding
            # # These parameters define the camera poses for each frame
            # plucker_features = load_init_camera_params(
            #     camera_path, closest_size[0], closest_size[1])
            # plucker_features = plucker_features.transpose(0, 1).to(self.device)
            
            # # Extract the first camera parameter and repeat it 3 times
            # # This ensures consistent camera parameters for the initial frames
            # first_plucker_feature = plucker_features[:, 0:1]
            # plucker_features = torch.cat(
            #     [first_plucker_feature, first_plucker_feature, first_plucker_feature, plucker_features], dim=1)
            
            # # Apply 3D average pooling to downsample camera parameters
            # # Reduces temporal dimension by 4 while preserving spatial information
            # plucker_features = torch.nn.functional.avg_pool3d(
            #     plucker_features.unsqueeze(0),
            #     kernel_size=(4, 1, 1),
            #     stride=(4, 1, 1)
            # )
            
            # # Pad camera parameters with ones to match expected dimensions
            # # This ensures the feature tensor has the correct shape for processing
            # plucker_features = torch.cat([plucker_features,
            #     torch.ones(1, 6, plucker_features.shape[2], 2, plucker_features.shape[-1]).to(self.device),
            #     plucker_features], dim=-2)
        
        logger.info("After encoding condition:")
        logger.info(f"{partial_mask.shape=} {partial_mask.min()=} {partial_mask.max()=}")
        logger.info(f"{partial_cond.shape=} {partial_cond.min()=} {partial_cond.max()=}")
        logger.info(f"{ref_latents.shape=} {ref_latents.min()=} {ref_latents.max()=}")
        logger.info(f"{len(ref_images)} {type(ref_images[0])} {ref_images[0].size}")

        # Generate rotary position embeddings for the target video dimensions
        # These embeddings provide positional information to the transformer model
        if self.args.use_multiple_kernels:
            logger.info("Using multiple kernels, calculating rotary embeddings.")
            # Compute latent size
            if "884" in self.args.vae:
                latents_size = [(target_video_length - 1) // 4 +
                                1, target_height // 8, target_width // 8]
            elif "888" in self.args.vae:
                latents_size = [(target_video_length - 1) // 8 +
                                1, target_height // 8, target_width // 8]
            else:
                latents_size = [target_video_length,
                                target_height // 8, target_width // 8]
            target_ndim = 3
            ndim = 5 - 2  # B, C, F, H, W -> F, H, W
            scheduler = CompressionScheduler(
                schedule_config={
                    "patch_sizes": self.args.use_kernel_sizes if self.args.use_kernel_sizes is not None else self.model.patch_size,
                }
            )
            indices = self.args.use_kernel_indices if self.args.use_kernel_indices is not None else [range(latents_size[0])]
            freqs_cos, freqs_sin = scheduler.get_rope_freq(
                indices,
                self.args.rope_theta,
                self.model.hidden_size // self.model.heads_num,
                self.model.rope_dim_list,
                latents_size,
                ndim,
                target_ndim,
                rope_theta_rescale_factor=1.0,
                rope_interpolation_factor=1.0,
            )
        else:
            freqs_cos, freqs_sin = self.get_rotary_pos_embed(
            target_video_length, target_height, target_width
        )   
        freqs_cos_full, freqs_sin_full = self.get_rotary_pos_embed(
            target_video_length, target_height, target_width, use_old_patch_size=True
        )   
        
        # Generate rotary position embeddings for conditional frames
        # Adjusted dimensions account for the conditional frame structure
        freqs_cos_cond, freqs_sin_cond = self.get_rotary_pos_embed(
            target_video_length, (target_height - 64) // 2, target_width
        )
        
        # Calculate the total number of tokens for the transformer model
        # This determines the sequence length for attention mechanisms
        n_tokens = freqs_cos.shape[0] if freqs_cos is not None else 0

        debug_str = f"""
                        height: {target_height}
                         width: {target_width}
                  video_length: {target_video_length}
                        prompt: {prompt}
                    neg_prompt: {negative_prompt}
                          seed: {seed}
                   infer_steps: {infer_steps}
         num_videos_per_prompt: {num_videos_per_prompt}
                guidance_scale: {guidance_scale}
                      n_tokens: {n_tokens}
            use_kernel_indices: {use_kernel_indices}
                    flow_shift: {flow_shift}
       embedded_guidance_scale: {embedded_guidance_scale}
                 i2v_stability: {i2v_stability}"""
        if ulysses_degree != 1 or ring_degree != 1:
            debug_str += f"""
                ulysses_degree: {ulysses_degree}
                   ring_degree: {ring_degree}"""
        logger.debug(debug_str)

        start_time = time.time()
        samples = self.pipeline(
            prompt=prompt,
            height=target_height,
            width=target_width,
            video_length=target_video_length,
            num_inference_steps=infer_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=generator,
            output_type="pil",
            freqs_cis=(freqs_cos, freqs_sin),
            freqs_cis_cond=(freqs_cos_cond, freqs_sin_cond),
            n_tokens=n_tokens,
            embedded_guidance_scale=embedded_guidance_scale,
            # data_type="video" if target_video_length > 1 else "image",
            data_type="video",
            is_progress_bar=True,
            vae_ver=self.args.vae,
            enable_tiling=self.args.vae_tiling,
            i2v_mode=i2v_mode,
            i2v_condition_type=i2v_condition_type,
            i2v_stability=i2v_stability,
            img_latents=ref_latents,
            semantic_images=ref_images,
            partial_cond=partial_cond,
            partial_mask=partial_mask,
            use_kernel_indices=use_kernel_indices,
            freqs_cis_full=(freqs_cos_full, freqs_sin_full),
            logger=logger,
            mode_scheduler_name=self.args.mode_scheduler_name,
            step_sample = step_sample,
            attn_map = attn_map,
            dmd2_steps = dmd2_steps,
        )[0]


        # if step_sample > 0 and isinstance(samples, tuple) and len(samples) > 1:
        #     out_dict["samples"] = samples[0]       # The final video
        #     out_dict["sample_list"] = samples[1]   # The list of intermediate (step, video) tuples
        # else:
        #     out_dict["samples"] = samples
        #     out_dict["prompts"] = prompt

        # gen_time = time.time() - start_time
        # logger.info(f"Success, time: {gen_time}")

        # return out_dict

        # 1. Handle the samples logic
        if step_sample > 0 and isinstance(samples, tuple) and len(samples) > 1:
            out_dict["samples"] = samples[0]      # The final video
            out_dict["sample_list"] = samples[1]  # The list of intermediate steps
        else:
            out_dict["samples"] = samples

        # 2. ALWAYS add metadata (Move these OUTSIDE the if/else)
        out_dict["prompts"] = prompt
        out_dict["seeds"] = [seed] * len(out_dict["samples"]) # Or however your seeds are tracked

        gen_time = time.time() - start_time
        logger.info(f"Success, time: {gen_time}")

        return out_dict

    @torch.no_grad()
    def predict_meanflow(
        self,
        prompt,
        height=192,
        width=336,
        video_length=49,
        seed=0,
        negative_prompt=None,
        flow_shift=7.0,
        embedded_guidance_scale=None,
        num_steps=1,
        i2v_condition_type="latent_concat",
        ref_images=None,
        partial_cond=None,
        partial_mask=None,
        **kwargs,
    ):
        """Few-step MeanFlow inference (standalone of the DMD2/teacher pipeline loop).

        Reuses this sampler's conditioning prep (ref/partial VAE-encode, RoPE, prompt
        encode) but replaces the scheduler denoise loop with ``meanflow_sample``: the
        student conditions on two times ``u_theta(z, r, t)`` and steps
        ``z_r = z_t - (t-r)*u`` from ``t=1`` (noise) to ``0`` (data). The model must
        already carry the trained ``r_in`` (see ``apply_meanflow_to_hunyuan_video`` +
        ``load_meanflow_state_dict`` in the calling script). No CFG (1-NFE student).
        """
        from voyager.diffusion.meanflow import make_meanflow_forward_u, meanflow_sample, has_meanflow

        assert has_meanflow(self.model), \
            "model has no r_in — apply_meanflow + load_meanflow_state_dict before calling predict_meanflow"
        assert i2v_condition_type == "latent_concat", "MeanFlow inference supports latent_concat only"

        out_dict = {}
        if seed is None:
            seed = random.randint(0, 1_000_000)
        generator = torch.Generator(self.device).manual_seed(int(seed))

        if (video_length - 1) % 4 != 0:
            raise ValueError(f"`video_length-1` must be a multiple of 4, got {video_length}")

        # Voyager stacks rgb over depth (+64 placeholder rows); target image is 2x+64 tall.
        target_height = height * 2 + 64
        target_width = width
        target_video_length = video_length
        closest_size = (height, width)

        if negative_prompt is None or negative_prompt == "":
            negative_prompt = self.default_negative_prompt

        # ---- reference frame (first frame) -> latent ----
        ref_images_pixel_values = [self.load_image(p, image_size=closest_size) for p in ref_images]
        ref_images_pixel_values = torch.cat(ref_images_pixel_values).unsqueeze(0).unsqueeze(2).to(self.device)
        ref_pil = [Image.fromarray(((torch.clamp(ref_images_pixel_values[0, :, 0].permute(1, 2, 0), min=-1, max=1).cpu().numpy() + 1) * 0.5 * 255).astype(np.uint8))]

        # ---- partial conditions (rendered partial RGB-D + mask) ----
        partial_cond = torch.stack([self.load_image(p, image_size=closest_size) for p in partial_cond], dim=1).unsqueeze(0).to(self.device)
        partial_mask = torch.stack([self.load_image(p, image_size=closest_size) for p in partial_mask], dim=1).unsqueeze(0).to(self.device)

        vae = self.pipeline.vae
        logger.info("MeanFlow: VAE-encoding ref + partial conditions (slow under --use-cpu-offload)...")
        _t = time.time()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            vae.enable_tiling()
            ref_latents = vae.encode(ref_images_pixel_values).latent_dist.sample()
            ref_latents.mul_(vae.config.scaling_factor)
            partial_cond = vae.encode(partial_cond).latent_dist.sample()
            partial_cond.mul_(vae.config.scaling_factor)

            # mask: invert, prepend 3 first-frames, maxpool to latent grid, invert back
            mask_frames = 1 - partial_mask
            first_mask = mask_frames[:, :, 0:1]
            mask_frames = torch.cat([first_mask, first_mask, first_mask, mask_frames], dim=2)
            mask_frames = torch.nn.functional.max_pool3d(mask_frames, kernel_size=(4, 8, 8), stride=(4, 8, 8))
            mask_frames = 1 - mask_frames
            partial_mask = mask_frames[:, 0:1]
        logger.info(f"MeanFlow: VAE-encode done in {time.time() - _t:.2f}s")

        target_dtype = PRECISION_TO_TYPE[self.args.precision]
        ref_latents = ref_latents.to(target_dtype)
        partial_cond = partial_cond.to(target_dtype)
        partial_mask = partial_mask.to(target_dtype)

        # ---- RoPE (standard backbone) ----
        freqs_cos, freqs_sin = self.get_rotary_pos_embed(target_video_length, target_height, target_width)

        # ---- text (no CFG for the 1-NFE student) ----
        logger.info("MeanFlow: encoding prompt (LLaVA LLM + CLIP-L; slow under --use-cpu-offload)...")
        _t = time.time()
        prompt_embeds, _, prompt_mask, _ = self.pipeline.encode_prompt(
            [prompt.strip()], self.device, 1, False, [negative_prompt.strip()],
            data_type="video", semantic_images=ref_pil,
        )
        prompt_embeds_2 = None
        if self.pipeline.text_encoder_2 is not None:
            prompt_embeds_2, _, _, _ = self.pipeline.encode_prompt(
                [prompt.strip()], self.device, 1, False, [negative_prompt.strip()],
                text_encoder=self.pipeline.text_encoder_2, data_type="video",
            )
        logger.info(f"MeanFlow: prompt encode done in {time.time() - _t:.2f}s")
        model_kwargs = dict(
            text_states=prompt_embeds.to(target_dtype),
            text_mask=prompt_mask,
            text_states_2=prompt_embeds_2.to(target_dtype) if prompt_embeds_2 is not None else None,
            freqs_cos=freqs_cos.to(self.device),
            freqs_sin=freqs_sin.to(self.device),
        )

        # ---- init noise z at t=1 (z_1 = eps under rectified-flow reverse) ----
        if "884" in self.args.vae:
            t_lat = (target_video_length - 1) // 4 + 1
        elif "888" in self.args.vae:
            t_lat = (target_video_length - 1) // 8 + 1
        else:
            t_lat = target_video_length
        latent_ch = int(self.model.out_channels)
        z = torch.randn(
            1, latent_ch, t_lat, target_height // 8, target_width // 8,
            generator=generator, device=self.device, dtype=target_dtype,
        )

        guidance = None
        if getattr(self.model, "guidance_embed", False) and embedded_guidance_scale is not None:
            guidance = torch.tensor([embedded_guidance_scale], dtype=torch.float32, device=self.device).to(target_dtype) * 1000.0

        forward_u = make_meanflow_forward_u(
            self.model,
            get_model_t=lambda tt: tt * 1000.0,
            model_kwargs=model_kwargs,
            cond_latents=ref_latents,
            partial_cond=partial_cond,
            partial_mask=partial_mask,
            guidance=guidance,
            verbose=True,
        )

        logger.info(f"MeanFlow: running {num_steps}-step sampling (first model forward absorbs flash-attn/cuDNN autotune)...")
        start_time = time.time()
        target_dtype_cast = (target_dtype != torch.float32) and not self.args.disable_autocast
        with torch.autocast(device_type="cuda", dtype=target_dtype, enabled=target_dtype_cast):
            x0 = meanflow_sample(z, forward_u, num_steps=num_steps, verbose=True)
        logger.info(f"MeanFlow {num_steps}-step sampling done in {time.time() - start_time:.2f}s")

        # ---- decode latents -> RGB-D video (mirrors pipeline tail) ----
        logger.info("MeanFlow: VAE-decoding latents -> RGB-D video...")
        _t = time.time()
        vae_dtype = PRECISION_TO_TYPE[self.args.vae_precision]
        if hasattr(vae.config, "shift_factor") and vae.config.shift_factor:
            x0 = x0 / vae.config.scaling_factor + vae.config.shift_factor
        else:
            x0 = x0 / vae.config.scaling_factor
        with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_dtype != torch.float32):
            vae.enable_tiling()
            image = vae.decode(x0, return_dict=False, generator=generator)[0]
        logger.info(f"MeanFlow: VAE-decode done in {time.time() - _t:.2f}s")
        if image.shape[2] == 1:
            image = image.squeeze(2)
        image = (image / 2 + 0.5).clamp(0, 1).cpu().float()

        half_height = (target_height - 64) // 2
        rgb = image[..., :half_height, :]
        depth = image[..., -half_height:, :]
        depth = depth[:, 0] * 0.299 + depth[:, 1] * 0.587 + depth[:, 2] * 0.114
        depth = depth.unsqueeze(1).repeat(1, 3, 1, 1, 1)
        if len(rgb.shape) == 4:
            rgb = rgb.unsqueeze(2)
        image = torch.cat([rgb, depth], dim=-2)

        out_dict["samples"] = image
        out_dict["prompts"] = prompt
        out_dict["seeds"] = [seed] * image.shape[0]
        return out_dict
