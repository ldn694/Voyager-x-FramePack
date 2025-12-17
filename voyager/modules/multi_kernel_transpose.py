from typing import Sequence, Union, Dict
import torch
import torch.nn as nn

from .embed_layers import PatchEmbed, MultiPatchEmbed
from .mlp_layers import FinalLayer, MultiFinalLayer
from .activation_layers import get_activation_layer
from ..utils.helpers import to_3tuple
from .models import HYVideoDiffusionTransformer  # adjust import path


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
) -> None:
    """
    Upgrade a vanilla HYVideoDiffusionTransformer to multi-kernel version.

    Steps:
      1) Replace `model.img_in` (PatchEmbed) with MultiPatchEmbed.
         - First conv in MultiPatchEmbed is initialized from the original PatchEmbed.
         - Additional convs remain randomly initialized.
      2) Replace `model.final_layer` (FinalLayer) with MultiFinalLayer.
         - The shared Linear `unproj` in MultiFinalLayer is initialized from the original FinalLayer.linear
           for the smallest patch size.
         - ConvTranspose upsamplers are left with their default init.
      3) Set `model.patch_sizes` so that the forward path uses the multi-kernel code.

    This should be called AFTER loading the vanilla pretrained weights.

    Args:
        model: HYVideoDiffusionTransformer instance with single PatchEmbed / FinalLayer.
        patch_sizes: list of patch sizes (ints or (t, h, w)-like) for MultiPatchEmbed.
                     The FIRST patch size MUST match the original `model.patch_size`.
        device, dtype: optional overrides; defaults to model parameters' device/dtype.
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

    # Save normalized patch sizes into model (so forward() uses multi-kernel path)
    model.patch_sizes = list(norm_patch_sizes)

    # -----------------------------------------------------------
    # 2) Upgrade img_in: PatchEmbed -> MultiPatchEmbed
    # -----------------------------------------------------------
    old_img_in = model.img_in
    if not isinstance(old_img_in, PatchEmbed):
        raise TypeError(
            "Expected model.img_in to be a PatchEmbed before applying multi-kernel upgrade."
        )

    # Infer in/out channels & bias from the existing conv
    old_conv: nn.Conv3d = old_img_in.proj
    in_chans = old_conv.in_channels
    embed_dim = old_conv.out_channels
    bias = old_conv.bias is not None

    # Try to preserve existing norm layer type (if not Identity)
    norm_layer_cls = (
        type(old_img_in.norm)
        if not isinstance(old_img_in.norm, nn.Identity)
        else None
    )

    conv_device = device if device is not None else old_conv.weight.device
    conv_dtype = dtype if dtype is not None else old_conv.weight.dtype
    factory_kwargs = {"device": conv_device, "dtype": conv_dtype}

    # Create new MultiPatchEmbed
    new_img_in = MultiPatchEmbed(
        patch_sizes=[list(ps) for ps in norm_patch_sizes],
        in_chans=in_chans,
        embed_dim=embed_dim,
        norm_layer=norm_layer_cls,
        bias=bias,
        **factory_kwargs,
    )

    # Copy the original PatchEmbed weights/bias into the first kernel
    with torch.no_grad():
        new_img_in.projs[0].weight.copy_(old_conv.weight)
        if bias and old_conv.bias is not None:
            new_img_in.projs[0].bias.copy_(old_conv.bias)

    model.img_in = new_img_in

    # -----------------------------------------------------------
    # 3) Upgrade final_layer: FinalLayer -> MultiFinalLayer
    # -----------------------------------------------------------
    old_final = model.final_layer
    if not isinstance(old_final, FinalLayer):
        raise TypeError(
            "Expected model.final_layer to be a FinalLayer before applying multi-kernel upgrade."
        )

    # Recover act layer class from existing adaLN_modulation (Sequential[act, Linear])
    act_layer_cls = type(old_final.adaLN_modulation[0])

    final_device = device if device is not None else old_final.linear.weight.device
    final_dtype = dtype if dtype is not None else old_final.linear.weight.dtype
    factory_kwargs = {"device": final_device, "dtype": final_dtype}

    # NOTE: MultiFinalLayer now uses:
    #   - one shared `unproj` linear for the smallest patch size
    #   - a list of ConvTranspose3d upsamplers between scales
    new_final = MultiFinalLayer(
        hidden_size=model.hidden_size,
        patch_sizes=norm_patch_sizes,  # pass tuples; MultiFinalLayer will _to_3d internally
        out_channels=model.out_channels,
        act_layer=act_layer_cls,
        **factory_kwargs,
    )

    # Copy norm_final and AdaLN modulation weights, and seed unproj from old FinalLayer.linear
    with torch.no_grad():
        # LayerNorm (no affine, but copy for robustness)
        new_final.norm_final.load_state_dict(old_final.norm_final.state_dict())

        # adaLN_modulation is Sequential[act, Linear]; copy Linear (index 1)
        new_final.adaLN_modulation[1].weight.copy_(
            old_final.adaLN_modulation[1].weight
        )
        new_final.adaLN_modulation[1].bias.copy_(
            old_final.adaLN_modulation[1].bias
        )

        # Initialize the shared unproj from the original FinalLayer.linear.
        # Shapes match because both use the *original* patch size as output spatial size.
        new_final.unproj.weight.copy_(old_final.linear.weight)
        new_final.unproj.bias.copy_(old_final.linear.bias)

        # ConvTranspose upsamplers keep their default initialization.

    model.final_layer = new_final

    # -----------------------------------------------------------
    # 4) Freeze base params if requested; unfreeze only multikernel params
    # -----------------------------------------------------------
    if freeze_base:
        for p in model.parameters():
            p.requires_grad = False
    for p in get_multikernel_parameters(model):
        p.requires_grad = True

    # From this point:
    #   - model.img_in is MultiPatchEmbed
    #   - model.final_layer is MultiFinalLayer
    #   - model.patch_sizes is set
    # so `HYVideoDiffusionTransformer.forward` will follow the multi-kernel path
    # (img_in(..., indices), final_layer(..., indices), unpatchify_multi, etc.).
    # Existing pretrained weights are preserved in the "base scale" of each module.


def get_multikernel_parameters(model: nn.Module):
    """
    Return a list of trainable parameters introduced by multi-kernel upgrade:
      - img_in.projs.*             (all convs in MultiPatchEmbed)
      - final_layer.unproj.*       (shared unprojection linear)
      - final_layer.upsamplers.*   (ConvTranspose upsamplers)

    This can be used to set `requires_grad=True` only for multi-kernel params.
    """
    params = []
    for name, module in model.named_modules():
        if name == "img_in" and isinstance(module, MultiPatchEmbed):
            for conv in module.projs:
                params.extend([p for p in conv.parameters()])
        elif name == "final_layer" and isinstance(module, MultiFinalLayer):
            # shared linear
            params.extend([p for p in module.unproj.parameters()])
            # progressive upsamplers
            for up in module.upsamplers:
                params.extend([p for p in up.parameters()])
    return params


def get_multikernel_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Return a state_dict containing only multi-kernel parameters:
      - img_in.projs.*
      - final_layer.unproj.*
      - final_layer.upsamplers.*

    This can be saved as a small checkpoint on top of a vanilla base model
    (after upgrading that base model with `apply_multikernel_to_hunyuan_video`).
    """
    sd = model.state_dict()
    mk_sd = {}
    for k, v in sd.items():
        if (
            k.startswith("img_in.projs.")
            or k.startswith("final_layer.unproj.")
            or k.startswith("final_layer.upsamplers.")
        ):
            mk_sd[k] = v
    return mk_sd


def load_multikernel_state_dict(
    model: nn.Module,
    multikernel_state_dict: Dict[str, torch.Tensor],
    strict: bool = False,
) -> None:
    """
    Load multi-kernel parameters into a model that has already been
    upgraded by `apply_multikernel_to_hunyuan_video`.

    Base (vanilla) weights remain unchanged.
    """
    missing, unexpected = model.load_state_dict(multikernel_state_dict, strict=strict)
    # Optional: log these if you want to debug
    # print("Missing multi-kernel keys:", missing)
    # print("Unexpected multi-kernel keys:", unexpected)
