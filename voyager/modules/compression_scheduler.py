from loguru import logger
from typing import List
import torch

from utils.tensor import center_down_sample_3d
from voyager.utils.train_utils import get_nd_rotary_pos_embed

def get_rope_freq_from_single_patch_size(
    rope_theta,
    patch_size,
    head_dim,
    rope_dim_list,
    latents_size,
    ndim,
    target_ndim,
    rope_theta_rescale_factor=1.0,
    rope_interpolation_factor=1.0,
):
    # COPY AND MODIFIED FROM voyager/utils/train_utils.py get_rope_freq_from_size

    if isinstance(patch_size, int):
        assert all(s % patch_size == 0 for s in latents_size), (
            f"Latent size(last {ndim} dimensions) should be divisible by patch size({patch_size}), "
            f"but got {latents_size}."
        )
        rope_sizes = [s // patch_size for s in latents_size]
    elif isinstance(patch_size, list) or isinstance(patch_size, tuple):
        assert all(
            s % patch_size[idx] == 0 for idx, s in enumerate(latents_size)
        ), (
            f"Latent size(last {ndim} dimensions) should be divisible by patch size({patch_size}), "
            f"but got {latents_size}."
        )
        rope_sizes = [s // patch_size[idx]
                      for idx, s in enumerate(latents_size)]

    if len(rope_sizes) != target_ndim:
        rope_sizes = [1] * (target_ndim - len(rope_sizes)
                            ) + rope_sizes  # time axis

    if rope_dim_list is None:
        rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
    assert (
        sum(rope_dim_list) == head_dim
    ), "sum(rope_dim_list) should equal to head_dim of attention layer"

    freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
        rope_dim_list,
        rope_sizes,
        theta=rope_theta,
        use_real=True,
        theta_rescale_factor=rope_theta_rescale_factor,
        interpolation_factor=rope_interpolation_factor,
    )

    return freqs_cos, freqs_sin

class CompressionScheduler:
    def __init__(self, schedule_config):
        patch_sizes = schedule_config.get("patch_sizes", None)
        assert patch_sizes is not None, "patch_sizes must be provided in schedule_config"
        assert isinstance(patch_sizes, list), "patch_sizes must be a list of lists"
        assert len(patch_sizes) >= 1, "At least one patch_size must be provided"
        if not isinstance(patch_sizes[0], list) and not isinstance(patch_sizes[0], tuple):
            logger.warning("Only one patch_size provided, converting to list of lists")
            patch_sizes = [patch_sizes]
        # Patch sizes must be provided in ascending order, the current one must divide the next one
        for i in range(len(patch_sizes)):
            assert len(patch_sizes[i]) == len(patch_sizes[0]), "All patch_sizes must have the same number of dimensions"
        for i in range(len(patch_sizes) - 1):
            assert all(
                patch_sizes[i + 1][j] % patch_sizes[i][j] == 0 for j in range(len(patch_sizes[i]))
            ), f"patch_sizes {patch_sizes[i]} must divide {patch_sizes[i + 1]}"
        self.patch_sizes = patch_sizes
    
    def get_rope_freq(
        self, 
        indices: List,
        rope_theta,
        head_dim,
        rope_dim_list,
        latents_size,
        ndim, 
        target_ndim, 
        rope_theta_rescale_factor=1.0,
        rope_interpolation_factor=1.0,
    ):
        assert len(indices) == len(self.patch_sizes), "Length of indices must match length of patch_sizes"
        assert len(latents_size) == ndim, "Length of latents_size must match ndim"
        merged_indices = set()
        for p in range(len(indices)):
            for i in indices[p]:
                assert i >= 0 and i < latents_size[0], f"Index {i} out of bounds for latents_size {latents_size}"
                merged_indices.add(i)
        assert len(merged_indices) == latents_size[0], f"Indices must cover all frames, but got {len(merged_indices)} unique indices while latents have {latents_size[0]} frames"

        logger.info(f"Generating RoPE frequencies for patch sizes: {self.patch_sizes} and indices: {indices}")

        freqs_cos, freqs_sin = get_rope_freq_from_single_patch_size(
            rope_theta,
            self.patch_sizes[0],
            head_dim,
            rope_dim_list,
            latents_size,
            ndim,
            target_ndim,
            rope_theta_rescale_factor,
            rope_interpolation_factor,
        ) # [seq_len, head_dim]

        first_rope_size = [s // self.patch_sizes[0][idx] for idx, s in enumerate(latents_size)]

        # reshape to first rope size
        freqs_cos = freqs_cos.reshape(*first_rope_size, head_dim)
        freqs_sin = freqs_sin.reshape(*first_rope_size, head_dim)

        T = latents_size[0]
        refined_freqs_cos = [[] for _ in range(latents_size[0])]
        refined_freqs_sin = [[] for _ in range(latents_size[0])]
        for p in range(len(self.patch_sizes)):
            downsample_kernel = [self.patch_sizes[p][i] // self.patch_sizes[0][i] for i in range(len(self.patch_sizes[0]))]
            current_freqs_cos = freqs_cos[indices[p]] # [len(indices[p]), first_rope_size[1], first_rope_size[2], head_dim]
            current_freqs_sin = freqs_sin[indices[p]]
            if p > 0:
                current_freqs_cos = current_freqs_cos.unsqueeze(0).permute(0, 4, 1, 2, 3) # (1, C, T, H, W)
                current_freqs_sin = current_freqs_sin.unsqueeze(0).permute(0, 4, 1, 2, 3)
                current_freqs_cos = center_down_sample_3d(current_freqs_cos, downsample_kernel) # (1, C, T', H', W')
                current_freqs_sin = center_down_sample_3d(current_freqs_sin, downsample_kernel)
                current_freqs_cos = current_freqs_cos.squeeze(0).permute(1, 2, 3, 0) # (T', H', W', C)
                current_freqs_sin = current_freqs_sin.squeeze(0).permute(1, 2, 3, 0)
            for local_i, t_idx in enumerate(indices[p]):
                refined_freqs_cos[t_idx].append(current_freqs_cos[local_i].reshape(-1, head_dim)) 
                refined_freqs_sin[t_idx].append(current_freqs_sin[local_i].reshape(-1, head_dim))
            logger.info(f"Patch size {self.patch_sizes[p]}: selected {len(indices[p])} frames")
        
        for t_idx in range(T):
            refined_freqs_cos[t_idx] = torch.cat(refined_freqs_cos[t_idx], dim=0)  # [num_patches, head_dim]
            refined_freqs_sin[t_idx] = torch.cat(refined_freqs_sin[t_idx], dim=0)  # [num_patches, head_dim]
        
        refined_freqs_cos = torch.cat(refined_freqs_cos, dim=0)  # [total_num_patches, head_dim]
        refined_freqs_sin = torch.cat(refined_freqs_sin, dim=0)  # [total_num_patches, head_dim]
        return refined_freqs_cos, refined_freqs_sin