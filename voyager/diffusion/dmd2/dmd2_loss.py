"""
Pure DMD2 loss functions for flow-matching distillation with `--flow-reverse`.

Convention (matches voyager/diffusion/flow/path.py:ICPlan(reverse=True)):
    alpha_t = 1 - t        (coefficient of clean data)
    sigma_t = t            (coefficient of noise)
    x_t     = (1 - t) * x_data + t * noise
    v       = noise - x_data

With these:
    x_hat_data = x_t - t * v_pred
    x_t        = (1 - t) * x_data + t * noise
"""

from __future__ import annotations
from typing import Callable, Optional, Tuple

import torch


def _expand_t(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return t.view(t.shape[0], *[1] * (ref.dim() - 1))


def make_xt_linear_reverse(x_data: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """x_t = (1 - t) * x_data + t * noise   (ICPlan linear, reverse=True)."""
    t_b = _expand_t(t, x_data)
    return (1.0 - t_b) * x_data + t_b * noise


def compute_x_hat_0_from_velocity(x_t: torch.Tensor, v_pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Recover the clean-data estimate from x_t and v_pred under linear-reverse:
        x_hat_data = x_t - t * v_pred
    """
    t_b = _expand_t(t, x_t)
    return x_t - t_b * v_pred


def dmd_weight(
    x_hat_0: torch.Tensor,
    mode: str = "normalized",
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Per-sample weight for the DMD2 generator surrogate loss.

    - "normalized": 1 / mean(|x_hat_0|) over spatial/temporal dims (the
      common stabilization trick from DMD2).
    - "uniform":   constant 1.0 (ablation).
    """
    if mode == "uniform":
        return torch.ones(x_hat_0.shape[0], device=x_hat_0.device, dtype=x_hat_0.dtype)

    if mode == "normalized":
        # reduce over everything except batch dim
        reduce_dims = tuple(range(1, x_hat_0.dim()))
        mag = x_hat_0.detach().abs().mean(dim=reduce_dims)
        return 1.0 / (mag + eps)

    raise ValueError(f"Unknown DMD weight mode {mode!r}")


def sample_t_p(
    batch_size: int,
    min_tp: float,
    max_tp: float,
    device,
    dtype=torch.float32,
) -> torch.Tensor:
    """Uniform t_p ∈ [min_tp, max_tp]."""
    u = torch.rand((batch_size,), device=device, dtype=dtype)
    return min_tp + (max_tp - min_tp) * u


# ---------------------------------------------------------------------- losses


def compute_dmd2_generator_loss(
    x_hat_0: torch.Tensor,
    forward_v: Callable[[str, torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    min_tp: float,
    max_tp: float,
    weight_mode: str = "normalized",
    real_adapter: Optional[str] = None,
    fake_adapter: str = "fake",
) -> Tuple[torch.Tensor, dict]:
    """
    DMD2 generator surrogate loss for the linear-reverse flow path.

    Args:
        x_hat_0:        Generator's clean-data prediction, *with grad* through θ.
        forward_v:      Closure forward_v(adapter_name_or_None, x_t, t) -> v_pred
                        that handles all of: setting the active LoRA adapter,
                        building the concatenated model input, mapping `t` to
                        the model's expected scale, and slicing the data
                        channels out of the model output.
        min_tp, max_tp: Sampling range for the perturbation timestep t_p.
        weight_mode:    "normalized" | "uniform".
        real_adapter:   Adapter to use for v_real. None -> base model (LoRA off).
        fake_adapter:   Adapter name for v_fake.

    Returns:
        (loss_scalar, info_dict). The loss is the standard DMD2 surrogate:
            L = mean( w(x_hat_0) * stop_grad(v_fake - v_real) * x_hat_0 )
        whose gradient w.r.t. θ matches ∇θ KL up to the constant absorbed in w.
    """
    B = x_hat_0.shape[0]
    device = x_hat_0.device

    t_p = sample_t_p(B, min_tp, max_tp, device=device, dtype=torch.float32)
    eps = torch.randn_like(x_hat_0)
    x_tp = make_xt_linear_reverse(x_hat_0.detach(), eps, t_p)

    with torch.no_grad():
        v_real = forward_v(real_adapter, x_tp, t_p)
        v_fake = forward_v(fake_adapter, x_tp, t_p)

    grad_signal = (v_fake - v_real).detach()
    w = dmd_weight(x_hat_0, mode=weight_mode).detach()
    w_b = _expand_t(w, x_hat_0)

    # Surrogate loss. Gradient flows through x_hat_0 only.
    loss = (w_b * grad_signal * x_hat_0).mean()

    info = {
        "t_p_mean": float(t_p.mean().item()),
        "grad_signal_norm": float(grad_signal.flatten(1).norm(dim=1).mean().item()),
        "x_hat_0_abs_mean": float(x_hat_0.detach().abs().mean().item()),
        "weight_mean": float(w.mean().item()),
    }
    return loss, info


def compute_fake_score_loss(
    x_hat_0_detached: torch.Tensor,
    forward_v: Callable[[str, torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    min_tp: float,
    max_tp: float,
    fake_adapter: str = "fake",
) -> Tuple[torch.Tensor, dict]:
    """
    Standard flow-matching MSE on generator outputs, used to train the fake
    score (a separate LoRA adapter).

    Target velocity for the linear-reverse path: v_target = noise - x_data,
    where x_data = x_hat_0_detached.
    """
    assert not x_hat_0_detached.requires_grad, "Pass a detached generator output."

    B = x_hat_0_detached.shape[0]
    device = x_hat_0_detached.device

    t_p = sample_t_p(B, min_tp, max_tp, device=device, dtype=torch.float32)
    noise = torch.randn_like(x_hat_0_detached)
    x_tp = make_xt_linear_reverse(x_hat_0_detached, noise, t_p)
    v_target = noise - x_hat_0_detached     # = noise - data (linear-reverse target)

    v_pred = forward_v(fake_adapter, x_tp, t_p)

    loss = (v_pred - v_target.to(v_pred.dtype)).pow(2).mean()
    info = {
        "t_p_mean": float(t_p.mean().item()),
        "v_target_norm": float(v_target.flatten(1).norm(dim=1).mean().item()),
    }
    return loss, info
