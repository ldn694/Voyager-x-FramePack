# SPDX-License-Identifier: Apache-2.0
"""
MeanFlow training loss for the Hunyuan-Voyager RGB-D i2v model.

MeanFlow (*Mean Flows for One-step Generative Modeling*, Geng et al.) learns the
**average velocity** ``u_theta(z, r, t)`` over an interval ``[r, t]`` instead of the
instantaneous velocity ``v(z, t)``. Its sole training signal is the *MeanFlow
identity*::

    u(z_t, r, t) = v(z_t, t) - (t - r) * d/dt u(z_t, r, t)

where the total time-derivative expands (chain rule, ``dz/dt = v``, ``dr/dt = 0``,
``dt/dt = 1``) to a Jacobian-vector product of ``u`` along the tangent direction
``(v, 0, 1)`` on ``(z, r, t)``::

    (u, dudt) = jvp(u_theta, (z, r, t), (v, 0, 1))
    u_tgt     = v - (t - r) * dudt
    loss      = adaptive_weight * || u - stopgrad(u_tgt) ||^2

No teacher, no DMD, no fake-score — this is the simplest of the two distillation
baselines and the end-to-end validation of the JVP infrastructure.

Implementation notes
--------------------
* **Flow time.** Voyager's rectified-flow path (``ICPlan(reverse=True)``) gives
  ``z_t = (1-t) x + t eps`` with conditional velocity ``ut = eps - x = v`` — exactly
  MeanFlow's convention. We sample ``t`` via ``denoiser.sample`` (its shift / SNR
  schedule), draw a second time ``r <= t`` (with a configurable fraction ``r == t``
  that reduces the loss to plain flow-matching), and obtain ``(z_t, v)`` from
  ``path_sampler.plan``.
* **Model time + tangent.** The DiT conditions on ``input_t = get_model_t(t)``
  (``t * training_timesteps``), so the JVP time-tangent that realises ``d/d(flow t)``
  is the chain-rule factor ``d(input_t)/dt = +/- training_timesteps``. The MeanFlow
  factor ``(t - r)`` stays in flow-time.
* **Conditioning channels.** In ``latent_concat`` mode the model input is
  ``[z_t, cond, mask, partial_cond, partial_mask]`` on the channel dim; only the
  data channels carry the velocity tangent ``v`` — the rest are constant (zero
  tangent), matching the rCM/MeanFlow treatment of conditioning.
* **Second time r.** Injected through ``model.r_in`` (see ``meanflow_adapter.py``)
  as a constant additive term to the modulation vector (``extra_vec``); ``r`` has
  zero tangent so it never enters ``dudt``.
"""

from typing import Callable, Optional

import torch

from ..flow.transport import SNRType
from ...modules.jvp.jvp_attention import attention_withT, TensorWithT
from ...modules.jvp.jvp_model import model_forward_jvp
from ...modules.jvp.jvp_model_double_branch import model_forward_jvp_double_branch
from .meanflow_adapter import has_meanflow, has_meanflow_second_branch


def _mean_flat(x: torch.Tensor) -> torch.Tensor:
    """Mean over all non-batch dims -> [B]."""
    return x.flatten(1).mean(dim=1)


def _sample_flow_times(denoiser, n: int, device, dtype) -> torch.Tensor:
    """Sample ``n`` flow-times in ``[t0, t1]`` mirroring ``Transport.sample``.

    Uses the denoiser's current ``shift`` (set to ``video_shift`` by the caller),
    ``snr_type`` and ``reverse`` so the second time ``r`` is drawn from the same
    distribution as the primary time ``t``.
    """
    t0, t1 = denoiser.check_interval(denoiser.train_eps, denoiser.sample_eps)
    if denoiser.snr_type == SNRType.UNIFORM:
        t = torch.rand((n,)) * (t1 - t0) + t0
    elif denoiser.snr_type == SNRType.LOGNORM:
        u = torch.normal(mean=0.0, std=1.0, size=(n,))
        t = 1.0 / (1.0 + torch.exp(-u)) * (t1 - t0) + t0
    else:
        raise ValueError(f"Unknown snr type: {denoiser.snr_type}")

    if denoiser.shift != 1.0:
        if denoiser.reverse:
            t = (denoiser.shift * t) / (1 + (denoiser.shift - 1) * t)
        else:
            t = t / (denoiser.shift - (denoiser.shift - 1) * t)
    return t.to(device=device, dtype=dtype)


def meanflow_training_losses(
    model,
    denoiser,
    x1: torch.Tensor,
    *,
    args,
    model_kwargs: dict,
    cond_latents: Optional[torch.Tensor] = None,
    partial_cond: Optional[torch.Tensor] = None,
    partial_mask: Optional[torch.Tensor] = None,
    attn_op: Callable[[TensorWithT, TensorWithT, TensorWithT], TensorWithT] = attention_withT,
):
    """Compute the MeanFlow loss for one batch.

    Args:
        model: the (adapter-wrapped) ``HYVideoDiffusionTransformer`` with ``r_in``.
        denoiser: the ``Transport`` (provides the flow path, time schedule, shift).
        x1: data latents ``[B, C_latent, T, H', W']`` (``x1`` = data in flow terms).
        args: training args (``i2v_mode``, ``i2v_condition_type``, ``embedded_cfg_scale``,
            ``meanflow_flow_ratio``, ``meanflow_loss_c``, ``meanflow_loss_p``).
        model_kwargs: the dict from ``prepare_model_inputs`` (text_states / mask /
            text_states_2 / freqs_cos / freqs_sin).
        cond_latents / partial_cond / partial_mask: i2v conditioning, as in
            ``transport.training_losses``.

    Returns ``(u, terms)`` where ``u`` is the predicted average velocity (primal,
    differentiable to params) and ``terms['loss']`` is the per-sample loss ``[B]``.
    """
    assert args.i2v_mode and args.i2v_condition_type == "latent_concat", \
        "MeanFlow currently supports i2v latent_concat mode only."

    device = x1.device
    B = x1.shape[0]

    # ---- sample flow-times t (primary) and r <= t ----
    denoiser.shift = denoiser.video_shift  # mirror Transport.training_losses
    t, x0, x1 = denoiser.sample(x1)        # t flow-time (shifted), x0 noise, x1 data
    r_raw = _sample_flow_times(denoiser, B, device, t.dtype)
    r = torch.minimum(r_raw, t)
    # fraction with r == t -> pure flow-matching (the identity's 2nd term vanishes)
    flow_mask = torch.rand(B, device=device) < float(args.meanflow_flow_ratio)
    r = torch.where(flow_mask, t, r)

    # ---- interpolate: z_t and conditional velocity v = ut ----
    t, xt, ut = denoiser.path_sampler.plan(t, x0, x1)
    v = ut  # [B, C_latent, T, H', W']
    latent_channels = xt.shape[1]

    # ---- build the latent_concat model input (mirrors transport.training_losses) ----
    if cond_latents is not None:
        x1_concat = cond_latents.repeat(1, 1, x1.shape[2], 1, 1).clone()
    else:
        x1_concat = x1.clone()
    x1_concat[:, :, 1:, :, :] = 0.0
    mask_concat = torch.ones(
        x1.shape[0], 1, x1.shape[2], x1.shape[3], x1.shape[4], device=device, dtype=x1.dtype
    )
    mask_concat[:, :, 1:, ...] = 0.0
    xt_full = torch.cat([xt, x1_concat, mask_concat], dim=1)
    if partial_cond is not None and partial_mask is not None:
        xt_full = torch.cat([xt_full, partial_cond, partial_mask], dim=1)
    xt_full = xt_full.to(model.dtype)

    # ---- tangent direction (v, 0, 1): v on data channels, 0 on conditioning ----
    t_x = torch.zeros_like(xt_full)
    t_x[:, :latent_channels] = v.to(model.dtype)

    # model time + chain-rule tangent d(input_t)/d(flow t)
    input_t = denoiser.get_model_t(t).to(device)
    input_r = denoiser.get_model_t(r).to(device)
    dmodel_dt = -denoiser.training_timesteps if denoiser.reverse_time_schedule else denoiser.training_timesteps
    t_t = torch.full_like(input_t, float(dmodel_dt))

    # r embedding (constant additive modulation term; zero tangent)
    extra_vec = model.r_in(input_r) if has_meanflow(model) else None

    # dual-branch: r also conditions the second branch (separate width); the JVP
    # forward and the real forward both take it as a constant additive modulation.
    use_db = bool(getattr(model, "use_second_branch", False))
    extra_vec_second = model.r_in_second_branch(input_r) if has_meanflow_second_branch(model) else None
    indices = model_kwargs.get("indices") if use_db else None
    db_tangent = getattr(args, "meanflow_db_tangent", "frozen")
    if use_db:
        assert indices is not None, \
            "dual-branch MeanFlow needs multi-kernel `indices` in model_kwargs (--train-multiple-kernels)."

    # guidance modulation (cfg-distilled backbone), as in transport.training_losses
    guidance = None
    if getattr(model, "guidance_embed", False) and args.embedded_cfg_scale is not None:
        guidance = (
            torch.tensor([args.embedded_cfg_scale] * B, dtype=torch.float32, device=device)
            .to(model.dtype)
            * 1000.0
        )

    # ---- forward: u = u_theta(z, r, t), dudt = d/d(flow t) u (detached) ----
    # When the whole batch has r == t the MeanFlow identity collapses to plain
    # flow-matching (the `(t-r)*dudt` term vanishes), so the expensive JVP tangent
    # pass would be computed and multiplied by zero. Take a plain forward instead
    # (~3-4x cheaper). `extra_vec` still injects the r-embedding so r_in keeps its
    # gradient and the primal `u` is identical to the JVP primal. Mixed batches
    # (some r==t, some r!=t) still go through the JVP path.
    used_jvp = not bool(flow_mask.all())
    if used_jvp:
        if use_db:
            u, dudt = model_forward_jvp_double_branch(
                model,
                (xt_full, t_x),
                (input_t, t_t),
                text_states=model_kwargs["text_states"],
                text_mask=model_kwargs["text_mask"],
                text_states_2=model_kwargs["text_states_2"],
                freqs_cos=model_kwargs["freqs_cos"],
                freqs_sin=model_kwargs["freqs_sin"],
                indices=indices,
                guidance=guidance,
                extra_vec=extra_vec,
                extra_vec_second=extra_vec_second,
                second_branch_tangent=db_tangent,
                attn_op=attn_op,
            )
        else:
            u, dudt = model_forward_jvp(
                model,
                (xt_full, t_x),
                (input_t, t_t),
                text_states=model_kwargs["text_states"],
                text_mask=model_kwargs["text_mask"],
                text_states_2=model_kwargs["text_states_2"],
                freqs_cos=model_kwargs["freqs_cos"],
                freqs_sin=model_kwargs["freqs_sin"],
                guidance=guidance,
                extra_vec=extra_vec,
                attn_op=attn_op,
            )
    else:
        u = model(
            xt_full,
            input_t,
            text_states=model_kwargs["text_states"],
            text_mask=model_kwargs["text_mask"],
            text_states_2=model_kwargs["text_states_2"],
            freqs_cos=model_kwargs["freqs_cos"],
            freqs_sin=model_kwargs["freqs_sin"],
            guidance=guidance,
            extra_vec=extra_vec,
            extra_vec_second=extra_vec_second,
            indices=indices,
            return_dict=False,
        )
        dudt = torch.zeros_like(u)  # t_minus_r == 0 below, so this is never read
    assert u.shape == v.shape, f"MeanFlow output {u.shape} must match velocity {v.shape}"

    # ---- MeanFlow identity target (flow-time t, r) ----
    t_minus_r = (t - r).view(B, *([1] * (u.dim() - 1))).to(u.dtype)
    u_tgt = v.to(u.dtype) - t_minus_r * dudt   # dudt already detached in model_forward_jvp
    error = u - u_tgt.detach()                 # sg ensures no double-backward through the JVP

    # ---- adaptive loss weight w = 1/(||Δ||^2 + c)^p  (stop-grad denominator) ----
    loss_per = _mean_flat(error.float() ** 2)  # [B]
    c = float(args.meanflow_loss_c)
    p = float(args.meanflow_loss_p)
    w = 1.0 / (loss_per.detach() + c) ** p
    loss = w * loss_per

    terms = {
        "loss": loss,
        "mse": loss_per.detach(),  # raw (unweighted) per-sample MSE -> overfit signal
        "used_jvp": used_jvp,      # False -> r==t fast path (plain fwd, no JVP)
        "input_t": input_t.detach().cpu().tolist(),
        "input_r": input_r.detach().cpu().tolist(),
        "r_eq_t_frac": float(flow_mask.float().mean().item()),
    }
    return u, terms
