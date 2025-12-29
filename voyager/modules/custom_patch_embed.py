from __future__ import annotations
from typing import Optional, Dict, Any, Iterable, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from .embed_layers import PatchEmbed
from .activation_layers import get_activation_layer
from .mlp_layers import FinalLayer

def _init_conv3d_from_pretrained(
    new_conv: nn.Conv3d,
    old_conv: nn.Conv3d,
) -> None:
    logger.info(f"Initializing new Conv3d of shape {new_conv.weight.shape} from old Conv3d of shape {old_conv.weight.shape}")
    """
    Initialize new_conv weights (and bias) from old_conv.
    If kernel sizes differ, resize the kernel with trilinear interpolation.
    """
    old_w = old_conv.weight.data
    new_w = new_conv.weight.data

    # Same shape -> direct copy
    if old_w.shape == new_w.shape:
        new_conv.weight.data.copy_(old_w)
    else:
        # Interpolate kernel to new size (kt, kh, kw)
        O, I, kt, kh, kw = old_w.shape
        nkt, nkh, nkw = new_w.shape[2:]

        # [O, I, kt, kh, kw] -> [O*I, 1, kt, kh, kw]
        w = old_w.view(O * I, 1, kt, kh, kw)
        w = F.interpolate(
            w,
            size=(nkt, nkh, nkw),
            mode="trilinear",
            align_corners=False,
        )
        # Back to [O, I, nkt, nkh, nkw]
        w = w.view(O, I, nkt, nkh, nkw)
        new_conv.weight.data.copy_(w)

    # Copy bias if present and same shape
    if old_conv.bias is not None and new_conv.bias is not None:
        if old_conv.bias.data.shape == new_conv.bias.data.shape:
            new_conv.bias.data.copy_(old_conv.bias.data)

def _init_conv3d_from_pretrained_top_left(
    new_conv: nn.Conv3d,
    old_conv: nn.Conv3d,
) -> None:
    """
    Initialize new_conv weights (and bias) from old_conv by copying the old
    kernel into the top-left corner of the new kernel and zeroing the rest.

    - If kernel shapes are identical, do a direct copy.
    - Otherwise, assumes:
        * out_channels match
        * in_channels match
        * new kernel size is >= old kernel size in each dim
      and copies:
        new_w[:, :, :kt, :kh, :kw] = old_w
      with everything else set to 0.
    """
    logger.info(
        f"Initializing new Conv3d (top-left copy) of shape {new_conv.weight.shape} "
        f"from old Conv3d of shape {old_conv.weight.shape}"
    )

    old_w = old_conv.weight.data
    new_w = new_conv.weight.data

    # Same full shape -> direct copy
    if old_w.shape == new_w.shape:
        new_w.copy_(old_w)
    else:
        # Check channel dims match
        if old_w.shape[0] != new_w.shape[0] or old_w.shape[1] != new_w.shape[1]:
            raise ValueError(
                "Channel mismatch when initializing Conv3d:\n"
                f"  old weight shape: {old_w.shape}\n"
                f"  new weight shape: {new_w.shape}\n"
                "Expected same out_channels and in_channels."
            )

        O, I, kt, kh, kw = old_w.shape
        _, _, nkt, nkh, nkw = new_w.shape

        # Ensure new kernel is not smaller than old kernel
        if nkt < kt or nkh < kh or nkw < kw:
            raise ValueError(
                "New Conv3d kernel must be >= old kernel in each dimension for "
                "top-left copy init.\n"
                f"  old kernel: (kt={kt}, kh={kh}, kw={kw})\n"
                f"  new kernel: (kt={nkt}, kh={nkh}, kw={nkw})"
            )

        # Zero everything, then copy old kernel into top-left corner
        new_w.zero_()
        new_w[:, :, 0:kt, 0:kh, 0:kw] = old_w

    # Copy bias if present and same shape
    if old_conv.bias is not None and new_conv.bias is not None:
        if old_conv.bias.data.shape == new_conv.bias.data.shape:
            new_conv.bias.data.copy_(old_conv.bias.data)

def apply_patch_adapter_to_hunyuan_video(
    model: nn.Module,
    new_patch_size: Tuple[int, int, int],
    freeze_base: bool = True,
) -> None:
    # --------------- normalize new_patch_size (same as before) ---------------
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

    if in_chans is None or hidden_size is None or out_channels is None:
        raise ValueError(
            "apply_patch_adapter_to_hunyuan_video expects a HYVideoDiffusionTransformer-"
            "like model with in_channels, hidden_size, out_channels attributes."
        )

    # ------------------------------------------------------------------
    # 0) Keep reference to old final_layer for AdaLN reuse
    # ------------------------------------------------------------------
    old_final_layer = getattr(model, "final_layer", None)
    old_img_in = getattr(model, "img_in", None)
    old_condition_in = getattr(model, "condition_in", None) if getattr(
        model, "use_context_block", False
    ) and hasattr(model, "condition_in") else None

    # ------------------------------------------------------------------
    # 1) Replace img_in PatchEmbed, init from old conv3d if available
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

    # If we have a previous PatchEmbed, initialize new Conv3d from it
    if isinstance(old_img_in, PatchEmbed):
        try:
            _init_conv3d_from_pretrained_top_left(
                new_img_in.proj,
                old_img_in.proj,
            )
        except Exception as e:
            logger.warning(f"Could not init img_in Conv3d from pretrained weights: {e}")

    model.img_in = new_img_in

    # If context-block is used, replace its condition_in as well
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
                _init_conv3d_from_pretrained_top_left(
                    new_condition_in.proj,
                    old_condition_in.proj,
                )
            except Exception as e:
                logger.warning(
                    f"Could not init condition_in Conv3d from pretrained weights: {e}"
                )

        model.condition_in = new_condition_in

    # ------------------------------------------------------------------
    # 2) Replace FinalLayer: copy AdaLN, but treat it as frozen / non-adapter
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

    # # Copy pretrained adaLN_modulation weights if possible
    if isinstance(old_final_layer, FinalLayer):
        try:
            logger.info("Copying adaLN_modulation weights from old FinalLayer to new FinalLayer...")
            new_final_layer.adaLN_modulation.load_state_dict(
                old_final_layer.adaLN_modulation.state_dict()
            )
        except RuntimeError as e:
            logger.warning(
                f"Could not load adaLN_modulation weights from old FinalLayer: {e}"
            )

    # # Freeze AdaLN params so they are NOT updated during adapter training
    # for p in new_final_layer.adaLN_modulation.parameters():
    #     p.requires_grad = False

    model.final_layer = new_final_layer
    model.patch_size = list(new_patch_size)

    # ------------------------------------------------------------------
    # 3) Freeze base, then selectively unfreeze only adapter params
    # ------------------------------------------------------------------
    if freeze_base:
        for p in model.parameters():
            p.requires_grad = False

    # unfreeze img_in (adapter)
    # for p in model.img_in.parameters():
    #     p.requires_grad = True

    # unfreeze condition_in (adapter) if present
    # if getattr(model, "use_context_block", False):
    #     for p in model.condition_in.parameters():
    #         p.requires_grad = True

    # unfreeze ONLY the patch-size dependent head inside FinalLayer (linear)
    for name, p in model.final_layer.named_parameters():
        # keep adaLN_modulation frozen
        if name.startswith("linear"):
            p.requires_grad = True


def get_patch_adapter_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Collect parameters belonging to the patch adapter:
    - model.img_in.*
    - model.condition_in.* (if present)
    - model.final_layer.linear.*   (exclude adaLN_modulation)
    """
    params: List[nn.Parameter] = []

    for name, p in model.named_parameters():
        if name.startswith("img_in."):
            params.append(p)
        elif name.startswith("condition_in.") and getattr(model, "use_context_block", False):
            params.append(p)
        elif name.startswith("final_layer."):
            params.append(p)

    return params

def get_patch_adapter_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Return a state_dict containing ONLY the patch adapter parameters:
    - img_in.*
    - condition_in.* (if present)
    - final_layer.linear.* (exclude final_layer.adaLN_modulation.*)
    """
    full_sd = model.state_dict()
    patch_sd: Dict[str, torch.Tensor] = {}

    for k, v in full_sd.items():
        # if k.startswith("img_in."):
        #     patch_sd[k] = v
        # elif k.startswith("condition_in.") and getattr(model, "use_context_block", False):
        #     patch_sd[k] = v
        # elif k.startswith("final_layer."):
        #     patch_sd[k] = v
        if k.startswith("final_layer.linear."):
            patch_sd[k] = v.detach().clone()

    return patch_sd


def load_patch_adapter_state_dict(
    model: nn.Module,
    patch_state_dict: Dict[str, torch.Tensor],
    strict: bool = False,
):
    """
    Load patch adapter weights into a model that already had
    apply_patch_adapter_to_hunyuan_video() called with the same new_patch_size.

    NOTE: This only loads img_in, condition_in, and final_layer.linear.
    AdaLN modulation is assumed to come from the base model and stay frozen.
    """
    missing, unexpected = model.load_state_dict(patch_state_dict, strict=strict)
    return missing, unexpected
