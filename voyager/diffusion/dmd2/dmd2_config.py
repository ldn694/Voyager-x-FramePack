from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class DMD2Config:
    """Hyperparameters for DMD2 + LoRA distillation."""

    num_steps: int                              # generator timesteps (e.g. 4)
    fake_updates_per_gen: int = 5
    fake_warmup_steps: int = 200
    fake_lora_rank: Optional[int] = None        # defaults to args.lora_rank
    fake_lora_alpha: Optional[float] = None     # defaults to args.lora_alpha
    weight_mode: str = "normalized"             # "normalized" or "uniform"
    min_tp: float = 0.02
    max_tp: float = 0.98
    fake_lr_mult: float = 1.0
    cfg_scale_real: float = 1.0                 # 1.0 disables CFG on the real branch

    # adapter names — fixed for now, but routed through the config so the rest
    # of the codebase has a single source of truth.
    gen_adapter_name: str = "gen"
    fake_adapter_name: str = "fake"

    # ------ GAN extension (DMD2 §3.3). Disabled when use_gan=False. -----------
    use_gan: bool = False
    gan_weight_g: float = 1e-3                  # λ on -mean(D(fake)) in the gen loss
    gan_weight_d: float = 1.0                   # λ on hinge D loss in the fake step
    gan_warmup_steps: int = 200                 # outer-steps before GAN term joins gen loss

    def __post_init__(self) -> None:
        if self.num_steps <= 0:
            raise ValueError(f"DMD2Config.num_steps must be > 0, got {self.num_steps}")
        if self.weight_mode not in ("normalized", "uniform"):
            raise ValueError(f"Unknown weight_mode {self.weight_mode!r}")
        if not (0.0 <= self.min_tp < self.max_tp <= 1.0):
            raise ValueError(
                f"Bad t_p range: min={self.min_tp}, max={self.max_tp}"
            )
        if self.use_gan:
            if self.gan_weight_g < 0 or self.gan_weight_d < 0:
                raise ValueError("DMD2 GAN weights must be non-negative.")
            if self.gan_warmup_steps < 0:
                raise ValueError("dmd2_gan_warmup_steps must be >= 0.")


def dmd2_config_from_args(args) -> DMD2Config:
    """Build a DMD2Config from argparse Namespace. See voyager/config.py:add_dmd2_args."""
    fake_rank = getattr(args, "dmd2_fake_lora_rank", None) or getattr(args, "lora_rank", 64)
    fake_alpha = getattr(args, "dmd2_fake_lora_alpha", None)
    if fake_alpha is None:
        fake_alpha = getattr(args, "lora_alpha", 16.0)
    return DMD2Config(
        num_steps=args.dmd2_steps,
        fake_updates_per_gen=args.dmd2_fake_updates_per_gen,
        fake_warmup_steps=args.dmd2_fake_warmup_steps,
        fake_lora_rank=fake_rank,
        fake_lora_alpha=fake_alpha,
        weight_mode=args.dmd2_weight_mode,
        min_tp=args.dmd2_min_tp,
        max_tp=args.dmd2_max_tp,
        fake_lr_mult=args.dmd2_fake_lr_mult,
        cfg_scale_real=args.dmd2_cfg_scale_real,
        use_gan=getattr(args, "dmd2_use_gan", False),
        gan_weight_g=getattr(args, "dmd2_gan_weight_g", 1e-3),
        gan_weight_d=getattr(args, "dmd2_gan_weight_d", 1.0),
        gan_warmup_steps=getattr(args, "dmd2_gan_warmup_steps", 200),
    )
