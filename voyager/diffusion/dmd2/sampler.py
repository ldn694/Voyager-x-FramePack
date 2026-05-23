"""
N-step DMD2 inference loop for the linear-reverse flow path.

Decoupled from any specific pipeline: takes a `forward_v` closure that handles
adapter routing, model-input concatenation, and timestep scaling. The denoising
math itself lives here.
"""

from __future__ import annotations
from typing import Callable, Optional

import torch

from .timestep_grid import uniform_timestep_grid
from .dmd2_loss import compute_x_hat_0_from_velocity, make_xt_linear_reverse


@torch.no_grad()
def dmd2_sample(
    z: torch.Tensor,
    forward_v: Callable[[Optional[str], torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    num_steps: int,
    gen_adapter: str = "gen",
) -> torch.Tensor:
    """
    Run the N-step DMD2 generator starting from `z` (pure noise at t=1).

    At each step the generator predicts v at the current t_i, recovers
    x_hat_0 = x_t - t_i * v, and re-noises to the next uniform timestep.
    The final iterate returns the predicted clean latent (no re-noising).
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be > 0, got {num_steps}")

    t_grid = uniform_timestep_grid(num_steps, device=z.device, dtype=torch.float32)
    x_t = z

    for i in range(num_steps):
        t_i = t_grid[i].expand(z.shape[0])
        v = forward_v(gen_adapter, x_t, t_i)
        x_hat_0 = compute_x_hat_0_from_velocity(x_t, v, t_i)

        if i < num_steps - 1:
            t_next = t_grid[i + 1].expand(z.shape[0])
            eps = torch.randn_like(x_hat_0)
            x_t = make_xt_linear_reverse(x_hat_0, eps, t_next)
        else:
            x_t = x_hat_0

    return x_t
