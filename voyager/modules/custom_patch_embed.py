from __future__ import annotations
from typing import Optional, Dict, Any, Iterable, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from loguru import logger

from .embed_layers import PatchEmbed
from .activation_layers import get_activation_layer
from .mlp_layers import FinalLayer


def get_flexidit_projection_matrix(
    old_shape: Tuple[int, int, int], 
    new_shape: Tuple[int, int, int], 
    dtype=torch.float32, 
    device='cpu'
) -> torch.Tensor:
    """
    Constructs the interpolation projection matrix P by interpolating an identity matrix.
    P has shape (N_new, N_old). 
    
    As described in FlexiDiT, this matrix (which acts as Q^dagger) is used 
    to initialize the flexible patch embedding and de-embedding weights.
    """
    N_old = math.prod(old_shape)
    N_new = math.prod(new_shape)

    # Identity matrix representing flattened basis vectors of the old patch
    I = torch.eye(N_old, dtype=dtype, device=device)
    
    # Reshape to (Batch=N_old, Channels=1, kt, kh, kw) for F.interpolate
    I_reshaped = I.view(N_old, 1, *old_shape)

    # Interpolate the identity matrix to the new patch shape (equivalent to cv2.resize/interpolate)
    P_T = F.interpolate(I_reshaped, size=new_shape, mode='trilinear', align_corners=False)

    # Flatten spatial dimensions: (N_old, N_new)
    P_T = P_T.view(N_old, N_new)

    # Transpose to get P: (N_new, N_old)
    P = P_T.t()
    return P


def _init_conv3d_flexidit(new_conv: nn.Conv3d, old_conv: nn.Conv3d) -> None:
    """
    Initialize new_conv weights from old_conv using the FlexiDiT projection matrix.
    """
    logger.info(
        f"Initializing new Conv3d of shape {new_conv.weight.shape} "
        f"from old Conv3d of shape {old_conv.weight.shape} via FlexiDiT projection."
    )
    old_w = old_conv.weight.data
    new_w = new_conv.weight.data

    # Same shape -> direct copy
    if old_w.shape == new_w.shape:
        new_conv.weight.data.copy_(old_w)
        if old_conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.data.copy_(old_conv.bias.data)
        return

    O, I, kt, kh, kw = old_w.shape
    _, _, nkt, nkh, nkw = new_w.shape

    # Get projection matrix
    P = get_flexidit_projection_matrix(
        old_shape=(kt, kh, kw), 
        new_shape=(nkt, nkh, nkw), 
        dtype=old_w.dtype, 
        device=old_w.device
    )

    # Flatten old_w spatial dimensions: (O, I, N_old)
    old_w_flat = old_w.view(O, I, -1)

    # Apply projection: P is (N_new, N_old), we multiply along the spatial dimension.
    # W_new = P @ W_old along the spatial axis -> shape (O, I, N_new)
    new_w_flat = torch.einsum('no, dio -> din', P, old_w_flat)

    # Reshape back to Conv3d weight shape
    new_conv.weight.data.copy_(new_w_flat.view(O, I, nkt, nkh, nkw))

    # Copy bias if present
    if old_conv.bias is not None and new_conv.bias is not None:
        new_conv.bias.data.copy_(old_conv.bias.data)


def _init_linear_flexidit(
    new_linear: nn.Linear, 
    old_linear: nn.Linear, 
    old_patch_size: Tuple[int, int, int], 
    new_patch_size: Tuple[int, int, int], 
    out_channels: int
) -> None:
    """
    Initialize new FinalLayer linear weights using the FlexiDiT projection matrix.
    """
    logger.info(
        f"Initializing new FinalLayer Linear of shape {new_linear.weight.shape} "
        f"from old of shape {old_linear.weight.shape} via FlexiDiT projection."
    )
    old_w = old_linear.weight.data
    new_w = new_linear.weight.data

    if old_w.shape == new_w.shape:
        new_linear.weight.data.copy_(old_w)
        if old_linear.bias is not None and new_linear.bias is not None:
            new_linear.bias.data.copy_(old_linear.bias.data)
        return

    N_old = math.prod(old_patch_size)
    N_new = math.prod(new_patch_size)
    hidden_size = old_w.shape[1]

    # Get projection matrix
    P = get_flexidit_projection_matrix(
        old_shape=old_patch_size, 
        new_shape=new_patch_size, 
        dtype=old_w.dtype, 
        device=old_w.device
    )

    # old_w shape: (N_old * out_channels, hidden_size)
    # Reshape to isolate N_old: (N_old, out_channels, hidden_size)
    old_w_reshaped = old_w.view(N_old, out_channels, hidden_size)

    # Apply projection: P is (N_new, N_old)
    # W_new = P @ W_old -> shape (N_new, out_channels, hidden_size)
    new_w_reshaped = torch.einsum('no, och -> nch', P, old_w_reshaped)

    # Flatten back to linear shape
    new_linear.weight.data.copy_(new_w_reshaped.view(N_new * out_channels, hidden_size))

    # Apply the same projection to the bias
    if old_linear.bias is not None and new_linear.bias is not None:
        old_b = old_linear.bias.data
        old_b_reshaped = old_b.view(N_old, out_channels)
        new_b_reshaped = torch.einsum('no, oc -> nc', P, old_b_reshaped)
        new_linear.bias.data.copy_(new_b_reshaped.view(N_new * out_channels))


def apply_patch_adapter_to_hunyuan_video(
    model: nn.Module,
    new_patch_size: Tuple[int, int, int],
    freeze_base: bool = True,
) -> None:
    # --------------- normalize new_patch_size ---------------
    if isinstance(new_patch_size, int):
        new_patch_size = (1, new_patch_size, new_patch_size)
    elif len(new_patch_size) == 2:
        new_patch_size = (1, new_patch_size[0], new_patch_size[1])
    else:
        new_patch_size = tuple(new_patch_size)
    
    logger.info(f"Applying Patch Adapter with new_patch_size={new_patch_size}...")

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    in_chans = getattr(model, "in_channels", None)
    hidden_size = getattr(model, "hidden_size", None)
    out_channels = getattr(model, "out_channels", None)
    old_patch_size = getattr(model, "patch_size", None)

    if in_chans is None or hidden_size is None or out_channels is None or old_patch_size is None:
        raise ValueError(
            "apply_patch_adapter_to_hunyuan_video expects a HYVideoDiffusionTransformer-"
            "like model with in_channels, hidden_size, out_channels, and patch_size attributes."
        )

    old_patch_size = tuple(old_patch_size)

    # ------------------------------------------------------------------
    # 0) Keep references
    # ------------------------------------------------------------------
    old_final_layer = getattr(model, "final_layer", None)
    old_img_in = getattr(model, "img_in", None)
    old_condition_in = getattr(model, "condition_in", None) if getattr(
        model, "use_context_block", False
    ) and hasattr(model, "condition_in") else None

    # ------------------------------------------------------------------
    # 1) Replace img_in PatchEmbed, init using FlexiDiT projection
    # ------------------------------------------------------------------
    new_img_in = PatchEmbed(
        patch_size=new_patch_size,
        in_chans=in_chans,
        embed_dim=hidden_size,
        norm_layer=None,
        flatten=True,
        bias=True,
        device=device,
        dtype=dtype,
    )

    if isinstance(old_img_in, PatchEmbed):
        try:
            # _init_conv3d_flexidit(new_img_in.proj, old_img_in.proj)
            pass
        except Exception as e:
            logger.warning(f"Could not init img_in Conv3d via FlexiDiT projection: {e}")

    model.img_in = new_img_in
    model.old_img_in = old_img_in  

    if getattr(model, "use_context_block", False):
        new_condition_in = PatchEmbed(
            patch_size=new_patch_size,
            in_chans=in_chans,
            embed_dim=hidden_size,
            norm_layer=None,
            flatten=True,
            bias=True,
            device=device,
            dtype=dtype,
        )

        if isinstance(old_condition_in, PatchEmbed):
            try:
                # _init_conv3d_flexidit(new_condition_in.proj, old_condition_in.proj)
                pass
            except Exception as e:
                logger.warning(f"Could not init condition_in Conv3d via FlexiDiT projection: {e}")

        model.condition_in = new_condition_in

    # ------------------------------------------------------------------
    # 2) Replace FinalLayer & init linear using FlexiDiT projection
    # ------------------------------------------------------------------
    act_layer = get_activation_layer("silu")
    new_final_layer = FinalLayer(
        hidden_size=hidden_size,
        patch_size=list(new_patch_size),
        out_channels=out_channels,
        act_layer=act_layer,
        device=device,
        dtype=dtype,
    )

    if isinstance(old_final_layer, FinalLayer):
        try:
            _init_linear_flexidit(
                new_final_layer.linear, 
                old_final_layer.linear, 
                old_patch_size, 
                new_patch_size, 
                out_channels
            )
            logger.info("Copying adaLN_modulation weights from old FinalLayer to new FinalLayer...")
            new_final_layer.adaLN_modulation.load_state_dict(
                old_final_layer.adaLN_modulation.state_dict()
            )
        except RuntimeError as e:
            logger.warning(
                f"Could not load/init weights from old FinalLayer: {e}"
            )

    model.final_layer = new_final_layer
    model.old_final_layer = old_final_layer
    model.old_patch_size = list(old_patch_size)
    model.patch_size = list(new_patch_size)
    model.used_patch_adapter = True

    # ------------------------------------------------------------------
    # 3) Freeze base, then selectively unfreeze only adapter params
    # ------------------------------------------------------------------
    if freeze_base:
        for p in model.parameters():
            p.requires_grad = False

    for p in get_patch_adapter_parameters(model):
        p.requires_grad = True


def get_patch_adapter_parameters(model: nn.Module) -> List[nn.Parameter]:
    params: List[nn.Parameter] = []
    for name, p in model.named_parameters():
        if name.startswith("img_in."):
            params.append(p)
        # elif name.startswith("condition_in.") and getattr(model, "use_context_block", False):
        #     params.append(p)
        elif name.startswith("final_layer."):
            params.append(p)
    return params

def get_patch_adapter_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    full_sd = model.state_dict()
    patch_sd: Dict[str, torch.Tensor] = {}
    for k, v in full_sd.items():
        if k.startswith("img_in.") or k.startswith("final_layer."):
            patch_sd[k] = v.detach().clone()
    return patch_sd

def load_patch_adapter_state_dict(
    model: nn.Module,
    patch_state_dict: Dict[str, torch.Tensor],
    strict: bool = False,
):
    missing, unexpected = model.load_state_dict(patch_state_dict, strict=strict)
    return missing, unexpected