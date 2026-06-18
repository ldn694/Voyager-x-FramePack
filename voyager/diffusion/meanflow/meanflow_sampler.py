# SPDX-License-Identifier: Apache-2.0
"""
Few-step MeanFlow sampler for the Hunyuan-Voyager i2v model.

MeanFlow learns the *average* velocity ``u_theta(z, r, t)`` over ``[r, t]``, so the
displacement along the flow is exact::

    z_r = z_t - (t - r) * u_theta(z_t, r, t)

Sampling therefore needs **no ODE solver** — just walk a decreasing schedule of
times from ``t = 1`` (pure noise) down to ``0`` (data). The 1-step special case is
``x = eps - u_theta(eps, r=0, t=1)``.

This module is decoupled from any specific pipeline (like ``dmd2_sample``): the
denoising math lives in :func:`meanflow_sample`, which takes a ``forward_u`` closure
that maps ``(z_data, t, r) -> u`` (handling latent_concat assembly, the ``r``
embedding, and model-time scaling). :func:`make_meanflow_forward_u` builds that
closure on top of the validated JVP path (``model_forward_jvp`` with zero tangents
gives exactly the primal ``u_theta``; the constant ``r`` embedding is injected via
``extra_vec``).

Convention matches training (``ICPlan(reverse=True)``): ``z_t = (1-t)x + t*eps``,
``t=1`` → noise, ``t=0`` → data, velocity ``v = eps - x``.
"""

import time
from typing import Callable, List, Optional, Tuple

import torch
from loguru import logger

from ...modules.jvp.jvp_attention import attention_withT, TensorWithT
from ...modules.jvp.jvp_model import model_forward_jvp


def _sync_now() -> float:
    """Wall-clock time after draining the CUDA queue, so timing logs reflect
    actual device work rather than async launch latency."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def meanflow_timesteps(num_steps: int) -> List[Tuple[float, float]]:
    """Uniform ``(t, r)`` interval schedule from ``t=1`` down to ``0``.

    ``num_steps`` intervals → boundaries ``[1, (N-1)/N, ..., 1/N, 0]`` and pairs
    ``(t_i, r_i) = (boundary[i], boundary[i+1])``. ``num_steps=1`` → ``[(1.0, 0.0)]``.
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be > 0, got {num_steps}")
    boundaries = [1.0 - i / num_steps for i in range(num_steps + 1)]
    boundaries[-1] = 0.0
    return [(boundaries[i], boundaries[i + 1]) for i in range(num_steps)]


def make_meanflow_forward_u(
    model,
    get_model_t: Callable[[torch.Tensor], torch.Tensor],
    *,
    model_kwargs: dict,
    cond_latents: Optional[torch.Tensor] = None,
    partial_cond: Optional[torch.Tensor] = None,
    partial_mask: Optional[torch.Tensor] = None,
    guidance: Optional[torch.Tensor] = None,
    i2v_condition_type: str = "latent_concat",
    attn_op: Callable[[TensorWithT, TensorWithT, TensorWithT], TensorWithT] = attention_withT,
    verbose: bool = False,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build a ``forward_u(z_data, t, r) -> u`` closure for :func:`meanflow_sample`.

    ``u`` is the predicted average velocity over ``[r, t]`` in data-channel space.
    Uses ``model_forward_jvp`` with zero tangents (its primal == ``u_theta``) and
    injects the second-time embedding ``r_in(input_r)`` via ``extra_vec``.

    ``get_model_t`` maps a flow-time tensor in ``[0,1]`` to the model's time input
    (e.g. ``denoiser.get_model_t`` at train time, or ``lambda t: t * 1000`` at
    inference) — keeps this decoupled from any specific scheduler/``Transport``.
    """
    assert i2v_condition_type == "latent_concat", "MeanFlow sampler supports latent_concat only"
    text_states = model_kwargs["text_states"]
    text_mask = model_kwargs["text_mask"]
    text_states_2 = model_kwargs["text_states_2"]
    freqs_cos = model_kwargs["freqs_cos"]
    freqs_sin = model_kwargs["freqs_sin"]

    def _forward_u(z_data: torch.Tensor, t: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if verbose:
            logger.info("[meanflow] _forward_u: assembling latent_concat input...")
        B, _, T, H, W = z_data.shape
        if cond_latents is not None:
            x1_concat = cond_latents.repeat(1, 1, T, 1, 1).clone()
        else:
            x1_concat = z_data.clone()
        x1_concat[:, :, 1:, :, :] = 0.0
        mask_concat = torch.ones(B, 1, T, H, W, device=z_data.device, dtype=z_data.dtype)
        mask_concat[:, :, 1:, ...] = 0.0
        xt = torch.cat([z_data, x1_concat, mask_concat], dim=1)
        if partial_cond is not None and partial_mask is not None:
            xt = torch.cat([xt, partial_cond, partial_mask], dim=1)
        xt = xt.to(model.dtype)

        input_t = get_model_t(t).to(z_data.device)
        input_r = get_model_t(r).to(z_data.device)
        if verbose:
            logger.info("[meanflow] _forward_u: computing r-embedding (model.r_in)...")
        extra_vec = model.r_in(input_r)

        if verbose:
            logger.info("[meanflow] _forward_u: entering model_forward_jvp (DiT forward)...")
            _t0 = _sync_now()
        u, _ = model_forward_jvp(
            model,
            (xt, torch.zeros_like(xt)),
            (input_t, torch.zeros_like(input_t)),
            text_states=text_states, text_mask=text_mask, text_states_2=text_states_2,
            freqs_cos=freqs_cos, freqs_sin=freqs_sin,
            guidance=guidance, extra_vec=extra_vec, attn_op=attn_op,
            verbose=verbose,
        )
        if verbose:
            logger.info(f"[meanflow] _forward_u: model_forward_jvp done in {_sync_now() - _t0:.2f}s")
        return u

    return _forward_u


@torch.no_grad()
def meanflow_sample(
    z: torch.Tensor,
    forward_u: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    num_steps: int = 1,
    schedule: Optional[List[Tuple[float, float]]] = None,
    verbose: bool = False,
) -> torch.Tensor:
    """Run few-step MeanFlow sampling from ``z`` (pure noise at ``t=1``).

    Args:
        z: initial noise ``[B, C_latent, T, H, W]`` (the data-channel latent).
        forward_u: closure mapping ``(z_data, t, r) -> u`` (see
            :func:`make_meanflow_forward_u`).
        num_steps: number of intervals (uniform schedule) if ``schedule`` is None.
        schedule: explicit list of ``(t, r)`` pairs (decreasing, ending at 0).

    Returns the predicted clean latent ``x`` (at ``t = 0``).
    """
    pairs = schedule if schedule is not None else meanflow_timesteps(num_steps)
    if verbose:
        logger.info(f"[meanflow] meanflow_sample: {len(pairs)} step(s), schedule={pairs}")
    x_t = z
    for i, (t, r) in enumerate(pairs):
        if verbose:
            logger.info(f"[meanflow] step {i + 1}/{len(pairs)}: (t={t:.4f}, r={r:.4f}) -> forward_u")
            _t0 = _sync_now()
        t_b = torch.full((z.shape[0],), float(t), device=z.device, dtype=torch.float32)
        r_b = torch.full((z.shape[0],), float(r), device=z.device, dtype=torch.float32)
        u = forward_u(x_t, t_b, r_b)
        # z_r = z_t - (t - r) * u  (displacement = average velocity * interval)
        x_t = x_t - (float(t) - float(r)) * u.to(x_t.dtype)
        if verbose:
            logger.info(f"[meanflow] step {i + 1}/{len(pairs)} done in {_sync_now() - _t0:.2f}s")
    return x_t
