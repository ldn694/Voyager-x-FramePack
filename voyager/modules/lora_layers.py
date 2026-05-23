from __future__ import annotations
from typing import Optional, Dict, Any, Iterable, List
import torch
import torch.nn as nn

from .models import MMDoubleStreamBlock, MMSingleStreamBlock


DEFAULT_ADAPTER_NAME = "default"


class _LoRABranch(nn.Module):
    """A single LoRA (A, B) pair. Kept as an nn.Module so it lives inside a ModuleDict."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        lora_alpha: float,
        lora_dropout: float,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features, **factory_kwargs))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r, **factory_kwargs))
            nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
            nn.init.zeros_(self.lora_B)
            self.scaling = lora_alpha / r
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.scaling = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Only called by LoRALinear when r > 0, so A/B are not None.
        lora_in = self.lora_dropout(x)
        update = (lora_in @ self.lora_A.t()) @ self.lora_B.t()
        return self.scaling * update


class LoRALinear(nn.Linear):
    """
    nn.Linear with one or more named LoRA branches.

    Exactly one branch (or none) is "active" at any time and contributes to the
    forward pass. Routing is controlled by self.active_adapter:
        - str  -> name of the active branch (must exist in self.adapters)
        - None -> base linear only (no LoRA update)

    Use set_active_lora_adapter(model, name) to flip every LoRALinear in the
    module tree at once.
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
        adapter_name: str = DEFAULT_ADAPTER_NAME,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(in_features, out_features, bias=bias, **factory_kwargs)

        # Back-compat: `lora_enabled` toggles the whole module on/off, regardless of routing.
        self.lora_enabled = True
        self.adapters: nn.ModuleDict = nn.ModuleDict()
        self.active_adapter: Optional[str] = None

        if r > 0:
            self.add_adapter(
                adapter_name,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            self.active_adapter = adapter_name

    # ------------------------------------------------------------------ adapters

    def add_adapter(
        self,
        name: str,
        r: int,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
    ) -> None:
        if name in self.adapters:
            return  # idempotent
        branch = _LoRABranch(
            in_features=self.in_features,
            out_features=self.out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        self.adapters[name] = branch
        # First adapter added becomes the active one if none is set.
        if self.active_adapter is None:
            self.active_adapter = name

    def set_active_adapter(self, name: Optional[str]) -> None:
        if name is not None and name not in self.adapters:
            raise KeyError(
                f"LoRALinear has no adapter named {name!r}. "
                f"Known: {list(self.adapters.keys())}"
            )
        self.active_adapter = name

    def adapter_parameters(self, name: str) -> List[nn.Parameter]:
        if name not in self.adapters:
            return []
        branch = self.adapters[name]
        out = []
        if branch.lora_A is not None:
            out.append(branch.lora_A)
        if branch.lora_B is not None:
            out.append(branch.lora_B)
        return out

    # --------------------------------------------------------------- construction

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        r: int = 4,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        adapter_name: str = DEFAULT_ADAPTER_NAME,
    ) -> "LoRALinear":
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
            adapter_name=adapter_name,
        )
        with torch.no_grad():
            new.weight.copy_(linear.weight)
            if linear.bias is not None:
                new.bias.copy_(linear.bias)
        return new

    # ------------------------------------------------------------------- forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = nn.functional.linear(x, self.weight, self.bias)

        if not self.lora_enabled:
            return result
        if self.active_adapter is None:
            return result
        if self.active_adapter not in self.adapters:
            return result
        branch = self.adapters[self.active_adapter]
        if branch.r <= 0:
            return result

        return result + branch(x)


# ----------------------------------------------------------------- module helpers


def set_active_lora_adapter(model: nn.Module, name: Optional[str]) -> None:
    """Flip the active LoRA adapter for every LoRALinear in the model tree."""
    count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            if name is None or name in module.adapters:
                module.set_active_adapter(name)
                count += 1
    # Quiet by default; callers print if they want.


def toggle_lora(model: nn.Module, enable: bool = True) -> None:
    """
    Enable or disable LoRA computation across the entire model.
    Preserves the active-adapter routing; only flips the master switch.
    """
    toggled_count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_enabled = enable
            toggled_count += 1
    print(f"Toggled LoRA {'ON' if enable else 'OFF'} for {toggled_count} layers.")


def _replace_linear_with_lora(
    module: nn.Module,
    attr_name: str,
    r: int,
    lora_alpha: float,
    lora_dropout: float,
    adapter_name: str = DEFAULT_ADAPTER_NAME,
):
    """
    Replace module.<attr_name> (must be nn.Linear) with LoRALinear wrapper,
    or add a new adapter to an existing LoRALinear at that slot.
    """
    old = getattr(module, attr_name, None)
    if old is None:
        return

    if isinstance(old, LoRALinear):
        # Already wrapped — just add another named adapter.
        old.add_adapter(adapter_name, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        return

    if not isinstance(old, nn.Linear):
        return

    new = LoRALinear.from_linear(
        old,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        adapter_name=adapter_name,
    )
    setattr(module, attr_name, new)


def apply_lora_to_hunyuan_video(
    model: nn.Module,
    r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    freeze_base: bool = True,
    adapter_name: str = DEFAULT_ADAPTER_NAME,
) -> None:
    """
    Inject LoRA into HunyuanVideo transformer at:

      - MMDoubleStreamBlock.img_attn_qkv
      - MMDoubleStreamBlock.img_attn_proj
      - MMSingleStreamBlock.linear1
      - MMSingleStreamBlock.linear2

    Calling this multiple times with different `adapter_name`s adds additional
    named LoRA adapters to the same layers; only one is active at a time (see
    `set_active_lora_adapter`).

    Call this *after* loading the pretrained weights.
    """
    for module in model.modules():
        if isinstance(module, MMDoubleStreamBlock):
            _replace_linear_with_lora(module, "img_attn_qkv", r, lora_alpha, lora_dropout, adapter_name)
            _replace_linear_with_lora(module, "img_attn_proj", r, lora_alpha, lora_dropout, adapter_name)

        if isinstance(module, MMSingleStreamBlock):
            _replace_linear_with_lora(module, "linear1", r, lora_alpha, lora_dropout, adapter_name)
            _replace_linear_with_lora(module, "linear2", r, lora_alpha, lora_dropout, adapter_name)

    if freeze_base:
        for p in model.parameters():
            p.requires_grad = False

    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.weight.requires_grad = False
            if module.bias is not None:
                module.bias.requires_grad = False
            for branch in module.adapters.values():
                if branch.lora_A is not None:
                    branch.lora_A.requires_grad = True
                if branch.lora_B is not None:
                    branch.lora_B.requires_grad = True


def get_lora_parameters(
    model: nn.Module,
    adapter_name: Optional[str] = None,
) -> List[nn.Parameter]:
    """
    Collect LoRA parameters from the model.

    If `adapter_name` is given, returns parameters only for that adapter.
    If `None` (default), returns parameters for *all* adapters across all layers
    (back-compat with the single-adapter callers).
    """
    out: List[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            if adapter_name is None:
                for branch in module.adapters.values():
                    if branch.lora_A is not None:
                        out.append(branch.lora_A)
                    if branch.lora_B is not None:
                        out.append(branch.lora_B)
            else:
                out.extend(module.adapter_parameters(adapter_name))
    return out


def _lora_key_belongs_to(key: str, adapter_name: str) -> bool:
    """Match keys produced by nn.ModuleDict for a given adapter (e.g. '...adapters.foo.lora_A')."""
    return f"adapters.{adapter_name}.lora_" in key


def get_lora_state_dict(
    model: nn.Module,
    adapter_name: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Return a state_dict containing only LoRA parameters.

    If `adapter_name` is given, only keys belonging to that adapter are included
    (and the adapter name is stripped from each key so the dict can be loaded
    back into either the same adapter or a different one — see
    `load_lora_state_dict`).

    If `None`, returns keys for all adapters with their full paths (legacy path
    used by older callers that round-trip the full module's LoRA tensors).
    """
    full_sd = model.state_dict()
    if adapter_name is None:
        return {k: v.detach().clone() for k, v in full_sd.items() if "lora_" in k}

    prefix_token = f"adapters.{adapter_name}."
    out: Dict[str, torch.Tensor] = {}
    for k, v in full_sd.items():
        if prefix_token not in k:
            continue
        # Strip the adapter-name segment so the saved keys are adapter-agnostic.
        stripped = k.replace(prefix_token, "adapters.__active__.")
        out[stripped] = v.detach().clone()
    return out


def _remap_legacy_lora_keys(state_dict: Dict[str, torch.Tensor], target_adapter: str) -> Dict[str, torch.Tensor]:
    """
    Map old single-adapter LoRA keys ("<...>.lora_A") into the new layout
    ("<...>.adapters.<target_adapter>.lora_A"). Pass-through for any key that
    already contains ".adapters.".
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if ".adapters." in k:
            out[k] = v
            continue
        # Match the trailing ".lora_A" / ".lora_B" patterns.
        for tag in (".lora_A", ".lora_B"):
            idx = k.rfind(tag)
            if idx != -1:
                head = k[:idx]
                tail = k[idx:]  # ".lora_A" or ".lora_A.<...>" — preserve everything after
                out[f"{head}.adapters.{target_adapter}{tail}"] = v
                break
        else:
            out[k] = v
    return out


def load_lora_state_dict(
    model: nn.Module,
    lora_state_dict: Dict[str, torch.Tensor],
    strict: bool = False,
    adapter_name: Optional[str] = None,
) -> None:
    """
    Load LoRA parameters into an already-LoRA-ified model.

    Behavior by `adapter_name`:
      - None  -> Legacy path. If the dict uses old-style keys (no ".adapters."
                 segment), they are routed into the DEFAULT_ADAPTER_NAME slot
                 so older `lora_last.pt` files keep working unchanged.
      - str   -> Keys with adapter namespaces are remapped into the named slot.
                 Both adapter-agnostic ("adapters.__active__.") and old-style
                 ("...lora_A") formats are accepted.
    """
    if adapter_name is None:
        # Old saves use ".lora_A"/".lora_B" directly; route into default adapter.
        remapped = _remap_legacy_lora_keys(lora_state_dict, DEFAULT_ADAPTER_NAME)
        model.load_state_dict(remapped, strict=strict)
        return

    remapped: Dict[str, torch.Tensor] = {}
    target_token = f"adapters.{adapter_name}."
    for k, v in lora_state_dict.items():
        if "adapters.__active__." in k:
            remapped[k.replace("adapters.__active__.", target_token)] = v
        elif ".adapters." in k:
            # Strip whatever adapter name was saved and re-prefix with target.
            head, sep, tail = k.partition(".adapters.")
            _, _, tail_rest = tail.partition(".")
            remapped[head + "." + target_token + tail_rest] = v
        else:
            remapped[k] = v
    # Old-style keys (no .adapters. segment) still need legacy remap.
    remapped = _remap_legacy_lora_keys(remapped, adapter_name)
    model.load_state_dict(remapped, strict=strict)
