import math
import torch
import torch.nn as nn
from einops import rearrange, repeat

from ..utils.helpers import to_3tuple


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding

    Image to Patch Embedding using Conv2d

    A convolution based approach to patchifying a 2D image w/ embedding projection.

    Based on the impl in https://github.com/google-research/vision_transformer

    Hacked together by / Copyright 2020 Ross Wightman

    Remove the _assert function in forward function to be compatible with multi-resolution images.
    """

    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        patch_size = to_3tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
            **factory_kwargs
        )
        nn.init.xavier_uniform_(
            self.proj.weight.view(self.proj.weight.size(0), -1))
        if bias:
            nn.init.zeros_(self.proj.bias)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class MultiPatchEmbed(nn.Module):
    """Multi Patch Embedding for multiple patch sizes

    Adapted from PatchEmbed
    """

    def __init__(
        self,
        patch_sizes,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        bias=True,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.patch_sizes = patch_sizes

        self.projs = nn.ModuleList()
        for i, patch_size in enumerate(patch_sizes):
            patch_size = to_3tuple(patch_size)
            assert patch_size[0] == 1, "Only support patch size 1 in temporal dimension for MultiPatchEmbed"
            if i == 0:
                proj = nn.Conv3d(
                    in_chans,
                    embed_dim,
                    kernel_size=patch_size,
                    stride=patch_size,
                    bias=bias,
                    **factory_kwargs
                )
            else:
                # assert all dim of patch size is divisible by previous patch size
                prev_patch_size = to_3tuple(patch_sizes[i-1])
                assert all(
                    patch_size[d] % prev_patch_size[d] == 0 for d in range(3)
                ), "Patch sizes must be multiples of each other"
                kernel_size = tuple([patch_size[d] // prev_patch_size[d] for d in range(3)])
                proj = nn.Conv3d(
                    embed_dim,
                    embed_dim,
                    kernel_size=kernel_size,
                    stride=kernel_size,
                    bias=bias,
                    **factory_kwargs
                )

            nn.init.xavier_uniform_(
                proj.weight.view(proj.weight.size(0), -1))
            if bias:
                nn.init.zeros_(proj.bias)
            self.projs.append(proj)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor, indices) -> torch.Tensor:
        # x: B, C, T, H, W
        B, C, T, H, W = x.shape

        # For each frame t in [0, T) we collect a list of tensors of shape B, embed_dim, P_tk
        per_frame_feats = [[] for _ in range(T)]
        per_frame_kernel_idx = [[] for _ in range(T)]

        tmp_x = x
        for i, (idxs, proj) in enumerate(zip(indices, self.projs)):
            # pick frames for this kernel: B, C, T', H, W
            tmp_x = proj(tmp_x)  # B, embed_dim, T', H', W'
            x_patch = tmp_x[:, :, idxs, :, :]
            x_patch = x_patch.flatten(3)  # B, embed_dim, T', P

            # scatter features back to global frame indices
            for local_i, t_idx in enumerate(idxs):
                feat = x_patch[:, :, local_i, :]   # B, embed_dim, P
                per_frame_feats[t_idx].append(feat)
                per_frame_kernel_idx[t_idx].append(torch.full((feat.size(0), feat.size(2)), i, dtype=torch.long, device=feat.device))  # record which kernel this patch comes from

        # sanity check: every frame must have at least one feature
        for t_idx, feats in enumerate(per_frame_feats):
            if len(feats) == 0:
                raise ValueError(f"Frame {t_idx} is missing in MultiPatchEmbed")

        # For each frame, concatenate patches from all kernels along patch dimension
        frame_tokens = []
        frame_patch_indices = []
        for t_idx, feats in enumerate(per_frame_feats):
            if len(feats) == 1:
                frame_feat = feats[0]                # B, embed_dim, P_t
                frame_patch_index = per_frame_kernel_idx[t_idx][0]  # B, P_t
            else:
                frame_feat = torch.cat(feats, dim=-1) # B, embed_dim, sum_k P_tk
                frame_patch_index = torch.cat(per_frame_kernel_idx[t_idx], dim=-1)  # B, sum_k P_tk
            frame_tokens.append(frame_feat)
            frame_patch_indices.append(frame_patch_index)

        # Now keep temporal order: frame 0, then 1, ..., then T-1
        # Concatenate along patch dimension across frames:
        # list length T, each B, embed_dim, P_t  ->  B, embed_dim, N_total
        x_out = torch.cat(frame_tokens, dim=-1)
        frame_patch_indices = torch.cat(frame_patch_indices, dim=-1)  # B, N_total

        # Match PatchEmbed convention: B, N, C
        x_out = x_out.transpose(1, 2)   # B, N_total, embed_dim

        x_out = self.norm(x_out)
        return x_out, frame_patch_indices


class TextProjection(nn.Module):
    """
    Projects text embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_channels, hidden_size, act_layer, dtype=None, device=None):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.linear_1 = nn.Linear(
            in_features=in_channels,
            out_features=hidden_size,
            bias=True,
            **factory_kwargs
        )
        self.act_1 = act_layer()
        self.linear_2 = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=True,
            **factory_kwargs
        )

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    Args:
        t (torch.Tensor): a 1-D Tensor of N indices, one per batch element. These may be fractional.
        dim (int): the dimension of the output.
        max_period (int): controls the minimum frequency of the embeddings.

    Returns:
        embedding (torch.Tensor): An (N, D) Tensor of positional embeddings.

    .. ref_link: https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(
        self,
        hidden_size,
        act_layer,
        frequency_embedding_size=256,
        max_period=10000,
        out_size=None,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        if out_size is None:
            out_size = hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(
                frequency_embedding_size, hidden_size, bias=True, **factory_kwargs
            ),
            act_layer(),
            nn.Linear(hidden_size, out_size, bias=True, **factory_kwargs),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def forward(self, t):
        t_freq = timestep_embedding(
            t, self.frequency_embedding_size, self.max_period
        ).type(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb
