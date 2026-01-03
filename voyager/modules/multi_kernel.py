from typing import Sequence, Union, Dict
import torch
import torch.nn as nn
from loguru import logger

from .embed_layers import PatchEmbed, MultiPatchEmbed
from .mlp_layers import FinalLayer, MultiFinalLayer
from ..utils.helpers import to_3tuple
from .models import HYVideoDiffusionTransformer


def _normalize_patch_sizes(
    patch_sizes: Sequence[Union[int, Sequence[int]]],
) -> Sequence[tuple]:
    """Normalize patch sizes to 3-tuples."""
    return [to_3tuple(ps) for ps in patch_sizes]


def apply_multikernel_to_hunyuan_video(
    model: HYVideoDiffusionTransformer,
    patch_sizes: Sequence[Union[int, Sequence[int]]],
    device=None,
    dtype=None,
    freeze_base: bool = False,
    copy_old_weights: bool = False,
) -> None:
    """
    Upgrade a vanilla HYVideoDiffusionTransformer to multi-kernel version.

    Steps:
      1) Replace model.img_in (PatchEmbed) with MultiPatchEmbed.
         - First conv in MultiPatchEmbed is initialized from the original PatchEmbed.
         - Additional convs remain randomly initialized.
      2) Replace model.final_layer (FinalLayer) with MultiFinalLayer.
         - First Linear in MultiFinalLayer is initialized from the original FinalLayer.
         - Additional linears remain zero-initialized.
      3) Set model.patch_sizes so that the forward path uses the multi-kernel code.

    This should be called AFTER loading the vanilla pretrained weights.
    """

    if not isinstance(model, HYVideoDiffusionTransformer):
        raise TypeError(
            "apply_multikernel_to_hunyuan_video expects a HYVideoDiffusionTransformer model"
        )

    # -----------------------------------------------------------
    # 1) Normalize patch sizes and check first equals original
    # -----------------------------------------------------------
    orig_patch = to_3tuple(model.patch_size)
    norm_patch_sizes = _normalize_patch_sizes(patch_sizes)

    if norm_patch_sizes[0] != orig_patch:
        raise ValueError(
            f"First patch size {norm_patch_sizes[0]} must match original "
            f"model.patch_size {orig_patch}."
        )

    model.patch_sizes = list(norm_patch_sizes)

    # -----------------------------------------------------------
    # 2) Upgrade img_in: PatchEmbed -> MultiPatchEmbed
    # -----------------------------------------------------------
    old_img_in = model.img_in
    if not isinstance(old_img_in, PatchEmbed):
        raise TypeError(
            "Expected model.img_in to be a PatchEmbed before applying multi-kernel upgrade."
        )

    old_conv: nn.Conv3d = old_img_in.proj
    in_chans = old_conv.in_channels
    embed_dim = old_conv.out_channels
    bias = old_conv.bias is not None

    norm_layer_cls = (
        type(old_img_in.norm)
        if not isinstance(old_img_in.norm, nn.Identity)
        else None
    )

    conv_device = device if device is not None else old_conv.weight.device
    conv_dtype = dtype if dtype is not None else old_conv.weight.dtype
    factory_kwargs = {"device": conv_device, "dtype": conv_dtype}

    new_img_in = MultiPatchEmbed(
        patch_sizes=[list(ps) for ps in norm_patch_sizes],
        in_chans=in_chans,
        embed_dim=embed_dim,
        norm_layer=norm_layer_cls,
        bias=bias,
        **factory_kwargs,
    )

    if copy_old_weights:
        logger.warning("Copying old img_in weights to multi-kernel img_in.")
        with torch.no_grad():
            new_img_in.projs[0].weight.copy_(old_conv.weight)
            if bias and old_conv.bias is not None:
                new_img_in.projs[0].bias.copy_(old_conv.bias)

    model.old_img_in = old_img_in
    model.img_in = new_img_in

    # -----------------------------------------------------------
    # 3) Upgrade final_layer: FinalLayer -> MultiFinalLayer
    # -----------------------------------------------------------
    old_final = model.final_layer
    if not isinstance(old_final, FinalLayer):
        raise TypeError(
            "Expected model.final_layer to be a FinalLayer before applying multi-kernel upgrade."
        )

    act_layer_cls = type(old_final.adaLN_modulation[0])

    final_device = device if device is not None else old_final.linear.weight.device
    final_dtype = dtype if dtype is not None else old_final.linear.weight.dtype
    factory_kwargs = {"device": final_device, "dtype": final_dtype}

    new_final = MultiFinalLayer(
        hidden_size=model.hidden_size,
        patch_sizes=[list(ps) for ps in norm_patch_sizes],
        out_channels=model.out_channels,
        act_layer=act_layer_cls,
        **factory_kwargs,
    )

    if copy_old_weights:
        logger.warning("Copying old final_layer weights to multi-kernel final_layer.")
        with torch.no_grad():
            new_final.norm_final.load_state_dict(
                old_final.norm_final.state_dict()
            )

            new_final.adaLN_modulation[1].weight.copy_(
                old_final.adaLN_modulation[1].weight
            )
            new_final.adaLN_modulation[1].bias.copy_(
                old_final.adaLN_modulation[1].bias
            )

            new_final.linears[0].weight.copy_(old_final.linear.weight)
            new_final.linears[0].bias.copy_(old_final.linear.bias)

    model.old_final_layer = old_final
    model.final_layer = new_final

    if freeze_base:
        for p in model.parameters():
            p.requires_grad = False

    for p in get_multikernel_parameters(model):
        p.requires_grad = True


def get_multikernel_parameters(model: nn.Module):
    """
    Return a list of trainable parameters introduced by multi-kernel upgrade:
      - img_in.projs.*
      - final_layer.*
    """
    params = []
    for name, module in model.named_modules():
        if name == "img_in" and isinstance(module, MultiPatchEmbed):
            for conv in module.projs:
                params.extend(conv.parameters())
        elif name == "final_layer" and isinstance(module, MultiFinalLayer):
            for linear in module.linears:
                params.extend(linear.parameters())
            params.extend(module.adaLN_modulation.parameters())
    return params


def get_multikernel_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Return a state_dict containing only multi-kernel parameters:
      - img_in.projs.*
      - final_layer.*
    """
    sd = model.state_dict()
    mk_sd = {}
    for k, v in sd.items():
        if (
            k.startswith("img_in.projs.")
            or k.startswith("final_layer.")
        ):
            mk_sd[k] = v.detach().clone()
    return mk_sd


def load_multikernel_state_dict(
    model: nn.Module,
    multikernel_state_dict: Dict[str, torch.Tensor],
    strict: bool = False,
) -> None:
    """
    Load multi-kernel parameters into a model that has already been upgraded
    by apply_multikernel_to_hunyuan_video.
    """
    missing, unexpected = model.load_state_dict(
        multikernel_state_dict, strict=strict
    )
    # Optional debug:
    # logger.warning("Missing keys:", missing)
    # logger.warning("Unexpected keys:", unexpected)