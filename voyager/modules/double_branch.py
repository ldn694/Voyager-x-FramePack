from dataclasses import dataclass

from typing import List, Optional, Dict, Sequence, Union
import torch
import torch.nn as nn

from .models import (
    HYVideoDiffusionTransformer, 
    MMSingleStreamBlock, 
    MMCrossStreamBlock
)
from .transformer_branch_config import TransformerBranchConfig
from .embed_layers import MultiPatchEmbed, TimestepEmbedder
from .mlp_layers import MultiFinalLayer
from .activation_layers import get_activation_layer

def apply_double_branch_to_hunyuan_video(
    model: HYVideoDiffusionTransformer,
    second_branch_config: TransformerBranchConfig,
    second_branch_mm_blocks_depth: int,
    device=None,
    dtype=None,
    freeze_base: bool = False,
) -> None:
    """
    Upgrades a vanilla HYVideo model to a 2-branch architecture with cross-attention.
    This also applies the multi-kernel upgrade (PatchEmbed -> MultiPatchEmbed) 
    required for token splitting.
    """
    if not isinstance(model, HYVideoDiffusionTransformer):
        raise TypeError("Expected HYVideoDiffusionTransformer model.")
    
    assert isinstance(model.img_in, MultiPatchEmbed), \
        "Model must already have MultiPatchEmbed. Please apply multi-kernel upgrade first."
    
    assert isinstance(model.final_layer, MultiFinalLayer), \
        "Model must already have MultiFinalLayer. Please apply multi-kernel upgrade first."

    factory_kwargs = {"device": device, "dtype": dtype}
    model.use_second_branch = True
    
    # 2. Initialize Projections
    model.proj_to_second_branch = nn.Linear(
        model.hidden_size,
        second_branch_config.hidden_size,
        **factory_kwargs,
    )
    model.unproj_from_second_branch = nn.Linear(
        second_branch_config.hidden_size,
        model.hidden_size,
        **factory_kwargs,
    )
    model.time_in_second_branch = TimestepEmbedder(
        second_branch_config.hidden_size, get_activation_layer("silu"), **factory_kwargs
    )

    # 3. Initialize Second Branch Blocks (Single Stream)
    model.second_branch_blocks = nn.ModuleList([
        MMSingleStreamBlock(
            second_branch_config.hidden_size,
            second_branch_config.heads_num,
            mlp_width_ratio=second_branch_config.mlp_width_ratio,
            mlp_act_type=second_branch_config.mlp_act_type,
            qk_norm=second_branch_config.qk_norm,
            qk_norm_type=second_branch_config.qk_norm_type,
            **factory_kwargs,
        )
        for _ in range(second_branch_mm_blocks_depth)
    ])

    # 4. Initialize Cross-Attention Blocks and Scheduler
    model.cross_attn_blocks = nn.ModuleList()
    model.double_branch_scheduler = list(second_branch_config.scheduler)
    
    # We replicate the init logic for cross-attention from the model class
    for edge in model.double_branch_scheduler:
        u, v = edge
        if u > 0: # First branch (query) -> Second branch (KV)
            model.cross_attn_blocks.append(
                MMCrossStreamBlock(
                    q_hidden_size=model.hidden_size,
                    kv_hidden_size=second_branch_config.hidden_size,
                    heads_num=model.heads_num,
                    mlp_width_ratio=model.config.mlp_width_ratio,
                    mlp_act_type=model.config.mlp_act_type,
                    qk_norm=model.config.qk_norm,
                    qk_norm_type=model.config.qk_norm_type,
                    qkv_bias=model.config.qkv_bias,
                    **factory_kwargs,
                )
            )
        else: # Second branch (query) -> First branch (KV)
            model.cross_attn_blocks.append(
                MMCrossStreamBlock(
                    q_hidden_size=second_branch_config.hidden_size,
                    kv_hidden_size=model.hidden_size,
                    heads_num=second_branch_config.heads_num,
                    mlp_width_ratio=second_branch_config.mlp_width_ratio,
                    mlp_act_type=second_branch_config.mlp_act_type,
                    qk_norm=second_branch_config.qk_norm,
                    qk_norm_type=second_branch_config.qk_norm_type,
                    qkv_bias=second_branch_config.qkv_bias,
                    **factory_kwargs,
                )
            )
    
    # Add the termination signal for the forward loop
    model.double_branch_scheduler.append((-1, -1))

    # 5. Handle Parameter Freezing
    if freeze_base:
        for p in model.parameters():
            p.requires_grad = False
        
        # Enable grad for the new components
        for p in get_double_branch_parameters(model):
            p.requires_grad = True

def get_double_branch_parameters(model: nn.Module):
    """Returns a list of unique parameters belonging to the second branch."""
    params = []
    new_module_names = [
        "proj_to_second_branch", 
        "unproj_from_second_branch", 
        "time_in_second_branch",
        "second_branch_blocks", 
        "cross_attn_blocks"
    ]
    
    # Iterate through individual parameters, not modules
    for name, param in model.named_parameters():
        if any(m in name for m in new_module_names):
            params.append(param)
    return params

def get_double_branch_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Returns a state_dict containing only the 2-branch specific weights."""
    sd = model.state_dict()
    new_keys = [
        "proj_to_second_branch", 
        "unproj_from_second_branch", 
        "time_in_second_branch",
        "second_branch_blocks", 
        "cross_attn_blocks"
    ]
    return {k: v.detach().clone() for k, v in sd.items() if any(k.startswith(nk) for nk in new_keys)}

def load_double_branch_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor], strict: bool = False):
    """Loads weights into the 2-branch components."""
    return model.load_state_dict(state_dict, strict=strict)