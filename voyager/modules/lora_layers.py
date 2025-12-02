from __future__ import annotations
from typing import Optional, Dict, Any
import torch
import torch.nn as nn

from .models import MMDoubleStreamBlock, MMSingleStreamBlock


class LoRALinear(nn.Linear):
    """
    Standard Linear layer + LoRA low-rank adapter.

    y = (W + scale * B @ A) x + b
    where A ∈ R^{r x in}, B ∈ R^{out x r}, r << min(in, out).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 4,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(in_features, out_features, bias=bias, **factory_kwargs)

        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

        if r > 0:
            # A: r x in, B: out x r
            self.lora_A = nn.Parameter(torch.zeros(r, in_features, **factory_kwargs))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r, **factory_kwargs))

            # init: A random, B zero (common LoRA pattern)
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            # nn.init.kaiming_uniform_(self.lora_B, a=5**0.5) # sanity check
            nn.init.zeros_(self.lora_B)

            # scaling factor
            self.scaling = lora_alpha / r
        else:
            # r = 0 disables LoRA
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.scaling = 0.0

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        r: int = 4,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
    ) -> "LoRALinear":
        """
        Construct a LoRALinear from an existing nn.Linear (copy weight/bias).
        """
        device = linear.weight.device
        dtype = linear.weight.dtype

        new = cls(
            linear.in_features,
            linear.out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=linear.bias is not None,
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            new.weight.copy_(linear.weight)
            if linear.bias is not None:
                new.bias.copy_(linear.bias)
        return new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base linear
        result = nn.functional.linear(x, self.weight, self.bias)

        # LoRA update
        if self.r > 0 and self.lora_A is not None and self.lora_B is not None:
            # x @ A^T => (..., r)
            lora_in = self.lora_dropout(x)
            lora_update = (lora_in @ self.lora_A.t()) @ self.lora_B.t()
            result = result + self.scaling * lora_update

        return result


def _replace_linear_with_lora(
    module: nn.Module,
    attr_name: str,
    r: int,
    lora_alpha: float,
    lora_dropout: float,
):
    """
    Replace module.<attr_name> (must be nn.Linear) with LoRALinear wrapper.
    """
    old = getattr(module, attr_name, None)
    if old is None or not isinstance(old, nn.Linear):
        return

    new = LoRALinear.from_linear(
        old,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    setattr(module, attr_name, new)


def apply_lora_to_hunyuan_video(
    model: nn.Module,
    r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    freeze_base: bool = True,
) -> None:
    """
    Inject LoRA into HunyuanVideo transformer:

    - MMDoubleStreamBlock.img_attn_qkv
    - MMDoubleStreamBlock.img_attn_proj
    - MMSingleStreamBlock.linear1
    - MMSingleStreamBlock.linear2

    Call this *after* loading the pretrained weights.
    """
    # 1) Replace the desired Linear layers with LoRALinear
    for module in model.modules():
        if isinstance(module, MMDoubleStreamBlock):
            _replace_linear_with_lora(module, "img_attn_qkv", r, lora_alpha, lora_dropout)
            _replace_linear_with_lora(module, "img_attn_proj", r, lora_alpha, lora_dropout)

        if isinstance(module, MMSingleStreamBlock):
            _replace_linear_with_lora(module, "linear1", r, lora_alpha, lora_dropout)
            _replace_linear_with_lora(module, "linear2", r, lora_alpha, lora_dropout)

    # 2) Optionally freeze base weights and only train LoRA params
    if freeze_base:
        for p in model.parameters():
            p.requires_grad = False

    for module in model.modules():
        if isinstance(module, LoRALinear):
            # base weight/bias frozen
            module.weight.requires_grad = False
            if module.bias is not None:
                module.bias.requires_grad = False
            # LoRA matrices trainable
            if module.lora_A is not None:
                module.lora_A.requires_grad = True
            if module.lora_B is not None:
                module.lora_B.requires_grad = True


def get_lora_parameters(model: nn.Module):
    """
    Convenience helper: collect just LoRA parameters for optimizer.
    """
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            if module.lora_A is not None:
                lora_params.append(module.lora_A)
            if module.lora_B is not None:
                lora_params.append(module.lora_B)
    return lora_params

def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Return a state_dict containing *only* LoRA parameters (lora_A / lora_B).
    This is small and can be saved as a separate checkpoint.
    """
    full_sd = model.state_dict()
    lora_sd = {k: v for k, v in full_sd.items() if "lora_" in k}
    return lora_sd


def load_lora_state_dict(
    model: nn.Module,
    lora_state_dict: Dict[str, torch.Tensor],
    strict: bool = False,
) -> None:
    """
    Load LoRA parameters into an already-LoRA-ified model.
    Base weights stay as in the pretrained checkpoint.
    """
    # We ONLY pass the LoRA keys, so strict=False is recommended.
    missing, unexpected = model.load_state_dict(lora_state_dict, strict=strict)
    # You can log these if you want:
    # print("Missing keys:", missing)
    # print("Unexpected keys:", unexpected)
