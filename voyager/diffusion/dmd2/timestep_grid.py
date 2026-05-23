from __future__ import annotations
import torch


def uniform_timestep_grid(num_steps: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Uniformly spaced generator timesteps in (0, 1] for an N-step DMD2 generator.

    With `--flow-reverse` (the only mode supported), t=1 is noise and t=0 is
    data. The N-step generator is invoked at the start of each uniform interval:

        N=4  ->  [1.00, 0.75, 0.50, 0.25]

    The grid is intentionally *not* warped by `--flow-shift`. The teacher's
    shift schedule only enters via v_real on the perturbation timestep t_p.
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be > 0, got {num_steps}")
    return torch.tensor(
        [(num_steps - i) / num_steps for i in range(num_steps)],
        device=device,
        dtype=dtype,
    )
