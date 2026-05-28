"""
GAN extension for DMD2 (Yin et al. 2024, §3.3).

The fake-score DiT doubles as the discriminator's backbone: we attach a small
per-token Linear head on top of the fake-net's velocity output, train it with a
hinge loss against (real x_data vs. generator x_hat_0) inside the fake step,
and feed `-D(fake)` back into the generator-step loss.

Design notes
------------
* The D head only sees the fake-net's *final* output (the velocity feature map,
  shape (B, C, T, H, W)). Tapping a mid-block hidden state instead is a one-line
  change: have `forward_features` return that hidden state (with the appropriate
  channel dim) and re-create the head with matching `in_channels`. The hinge
  losses below are tap-agnostic.
* D head is initialised to zeros so logits start at exactly 0 — hinge_d starts
  at a constant 2 (real and fake equally indistinguishable), hinge_g at 0. This
  prevents the GAN term from kicking the generator with noise before D has any
  signal, *independently* of the explicit gan-warmup schedule.
* The hinge formulas use the conventional sign:
      L_D = mean(relu(1 - D(real))) + mean(relu(1 + D(fake)))
      L_G = -mean(D(fake))
  D descends L_D -> push D(real) up, D(fake) down. G descends L_G -> push
  D(fake) up. No double-negatives, no sign-flips downstream.
"""

from __future__ import annotations
from typing import Dict, List, Optional

import torch
import torch.nn as nn


DISC_ATTR_NAME = "dmd2_disc_head"


class DMD2DiscriminatorHead(nn.Module):
    """
    Per-spatio-temporal-cell linear classifier.

    Input:  (B, C, T, H, W) feature map (default tap: fake-net velocity output).
    Output: (B,) scalar logits = mean over (T, H, W) of per-cell logits.
    """

    def __init__(self, in_channels: int, dtype=None, device=None) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.linear = nn.Linear(in_channels, 1, dtype=dtype, device=device)
        # Zero init: D(x) == 0 at start. See module docstring for rationale.
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (B, C, T, H, W) -> (B, T, H, W, C) for Linear over the channel dim
        x = feat.permute(0, 2, 3, 4, 1).contiguous()
        x = x.to(self.linear.weight.dtype)
        logits = self.linear(x).squeeze(-1)              # (B, T, H, W)
        return logits.mean(dim=(1, 2, 3))                # (B,)


# ------------------------------------------------------------ module plumbing


def attach_disc_head(
    model: nn.Module,
    in_channels: int,
    dtype=None,
    device=None,
) -> DMD2DiscriminatorHead:
    """Attach a discriminator head onto `model` as `model.dmd2_disc_head`.

    Idempotent: re-attaching with a matching `in_channels` is a no-op.
    """
    if hasattr(model, DISC_ATTR_NAME):
        head = getattr(model, DISC_ATTR_NAME)
        if head.in_channels != in_channels:
            raise ValueError(
                f"Existing {DISC_ATTR_NAME} has in_channels={head.in_channels}, "
                f"requested {in_channels}."
            )
        return head
    head = DMD2DiscriminatorHead(in_channels=in_channels, dtype=dtype, device=device)
    setattr(model, DISC_ATTR_NAME, head)
    return head


def get_disc_head(model: nn.Module) -> Optional[DMD2DiscriminatorHead]:
    return getattr(model, DISC_ATTR_NAME, None)


def get_disc_parameters(model: nn.Module) -> List[nn.Parameter]:
    head = get_disc_head(model)
    return list(head.parameters()) if head is not None else []


def get_disc_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    head = get_disc_head(model)
    if head is None:
        return {}
    return {k: v.detach().clone() for k, v in head.state_dict().items()}


def load_disc_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    head = get_disc_head(model)
    if head is None:
        raise RuntimeError(
            f"Cannot load disc state dict: model has no {DISC_ATTR_NAME!r}. "
            f"Call attach_disc_head() first."
        )
    head.load_state_dict(state_dict, strict=False)


# ------------------------------------------------------------------- losses


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """L_D = mean(relu(1 - D(real))) + mean(relu(1 + D(fake)))."""
    return torch.relu(1.0 - logits_real).mean() + torch.relu(1.0 + logits_fake).mean()


def hinge_g_loss(logits_fake: torch.Tensor) -> torch.Tensor:
    """L_G = -mean(D(fake))."""
    return -logits_fake.mean()
