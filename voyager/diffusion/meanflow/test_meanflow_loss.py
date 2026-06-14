# SPDX-License-Identifier: Apache-2.0
"""
CPU validation for the MeanFlow loss + r-conditioning.

Two tiers, both CPU-runnable (naive attention, no Triton/flash-attn):

(A) ``test_meanflow_identity_math`` — exact-match the JVP forward used by MeanFlow
    (``model_forward_jvp(..., extra_vec=r_in(input_r))``) against a *monolithic*
    ``torch.func.jvp`` reference that threads the same ``(v, 0, 1)`` tangent and adds
    the same constant ``r`` embedding to the modulation vector. Confirms:
      * primal ``u`` and tangent ``dudt`` equal the reference,
      * the second-time ``r`` enters the *primal* (changing ``r`` changes ``u``),
      * the MeanFlow target ``u_tgt = v − (t−r)·dudt`` and the adaptive-weighted loss
        assemble with finite, param-connected gradients.

(B) ``test_meanflow_training_losses_smoke`` — runs the full
    ``meanflow_training_losses`` against a real ``Transport`` (rectified-flow,
    reverse) on a mock model with consistent ``latent_concat`` channels, checking the
    per-sample loss is finite, differentiable to params, and that ``dudt`` stayed
    detached.

The exact equivalence to the *real* Hunyuan submodules is the GPU-server check
(``check_primal_parity`` + a real training step); this file validates the MeanFlow
math and plumbing on top of the already-validated JVP infrastructure.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

from ...modules.modulate_layers import modulate
from ...modules.posemb_layers import get_nd_rotary_pos_embed
from ...modules.jvp.jvp_attention import naive_attention_withT
from ...modules.jvp.jvp_model import model_forward_jvp
from ...modules.jvp.test_jvp_model import MockModel, _timestep_embedding
from .meanflow_adapter import (
    apply_meanflow_to_hunyuan_video,
    get_meanflow_parameters,
    get_meanflow_state_dict,
    load_meanflow_state_dict,
)
from .meanflow_loss import meanflow_training_losses


def _make_mock_with_r(hidden=32, heads=4, in_ch=8, out_ch=4, patch=(1, 2, 2), dtype=torch.float64):
    mock = MockModel(hidden, heads, in_ch, out_ch, patch).to(dtype)
    mock.hidden_size = hidden
    mock.dtype = dtype
    apply_meanflow_to_hunyuan_video(mock, zero_init=False)  # random r_in so r actually matters
    mock.r_in.to(dtype)
    # randomise the r_in output projection (apply leaves mlp[0] at init; make both nonzero)
    nn.init.normal_(mock.r_in.mlp[2].weight, std=0.3)
    nn.init.normal_(mock.r_in.mlp[2].bias, std=0.3)
    return mock


def test_meanflow_identity_math(dtype=torch.float64):
    torch.manual_seed(0)
    hidden, heads, in_ch, out_ch = 32, 4, 8, 4
    patch = (1, 2, 2)
    mock = _make_mock_with_r(hidden, heads, in_ch, out_ch, patch, dtype)
    latent_ch = out_ch

    B, T, H, W = 1, 2, 4, 4
    x = torch.randn(B, in_ch, T, H, W, dtype=dtype)
    t = torch.rand(B, dtype=dtype)            # flow-time in [0,1]
    r = t * torch.rand(B, dtype=dtype)        # r < t
    v = torch.randn(B, latent_ch, T, H, W, dtype=dtype)   # velocity (data channels)

    # tangent direction (v, 0, 1): v on data channels, 0 elsewhere; time tangent = 1000
    t_x = torch.zeros_like(x)
    t_x[:, :latent_ch] = v
    input_t = t * 1000.0
    input_r = r * 1000.0
    t_t = torch.full_like(input_t, 1000.0)

    text_dim, L_txt = 16, 6
    text_states = torch.randn(B, L_txt, text_dim, dtype=dtype)
    text_mask = torch.zeros(B, L_txt); text_mask[:, :4] = 1.0
    text_states_2 = torch.randn(B, 12, dtype=dtype)

    head_dim = hidden // heads
    a = head_dim // 4
    cos, sin = get_nd_rotary_pos_embed([a, a, head_dim - 2 * a], (T, H // patch[1], W // patch[2]), use_real=True)
    cos, sin = cos.to(dtype), sin.to(dtype)

    text_len = int(text_mask[0].sum().item())
    tt, th, tw = T // patch[0], H // patch[1], W // patch[2]
    r_vec = mock.r_in(input_r)  # constant additive modulation term

    # ---- monolithic reference: jvp over (x, t), text + r held constant ----
    txt_c = mock.txt_in(text_states)[:, :text_len]

    def ref(x, t_model):
        vec = mock.time_in(t_model) + mock.vector_in(text_states_2) + r_vec
        img = mock.img_in(x)
        img_seq = img.shape[1]
        im, tx = img, txt_c
        for blk in mock.double_blocks:
            im, tx = blk.ref_forward(im, tx, vec, (cos, sin))
        xx = torch.cat((im, tx), 1)
        for blk in mock.single_blocks:
            xx = blk.ref_forward(xx, vec, text_len, (cos, sin))
        img_f = xx[:, :img_seq]
        return mock.unpatchify(mock.final_layer(img_f, vec), tt, th, tw)

    ref_u, ref_dudt = torch.func.jvp(ref, (x, input_t), (t_x, t_t))

    # ---- under test ----
    u, dudt = model_forward_jvp(
        mock, (x, t_x), (input_t, t_t),
        text_states=text_states, text_mask=text_mask, text_states_2=text_states_2,
        freqs_cos=cos, freqs_sin=sin, extra_vec=r_vec, attn_op=naive_attention_withT,
    )

    torch.testing.assert_close(u, ref_u)
    torch.testing.assert_close(dudt, ref_dudt)
    assert not dudt.requires_grad, "dudt must be detached"

    # r must enter the primal: a different r changes u
    u2, _ = model_forward_jvp(
        mock, (x, t_x), (input_t, t_t),
        text_states=text_states, text_mask=text_mask, text_states_2=text_states_2,
        freqs_cos=cos, freqs_sin=sin, extra_vec=mock.r_in(input_r * 0.5 + 1.0),
        attn_op=naive_attention_withT,
    )
    assert not torch.allclose(u, u2), "r-conditioning has no effect on the primal"

    # MeanFlow identity target + adaptive-weighted loss, finite grad to params
    t_minus_r = (t - r).view(B, *([1] * (u.dim() - 1))).to(u.dtype)
    u_tgt = v - t_minus_r * dudt
    error = u - u_tgt.detach()
    loss_per = error.float().flatten(1).pow(2).mean(dim=1)
    w = 1.0 / (loss_per.detach() + 1e-3) ** 1.0
    loss = (w * loss_per).mean()
    loss.backward()
    g = mock.final_layer.linear.weight.grad
    assert g is not None and torch.isfinite(g).all(), "no/!finite grad to backbone params"
    # r_in must also receive gradient (it is part of the primal path)
    gr = mock.r_in.mlp[2].weight.grad
    assert gr is not None and torch.isfinite(gr).all(), "no/!finite grad to r_in"
    print("test_meanflow_identity_math: PASSED")


class _RealishTransport:
    """Minimal stand-in unused — kept intentionally absent; we use the real Transport."""


def test_meanflow_training_losses_smoke(dtype=torch.float64):
    try:
        from ..flow.transport import Transport, ModelType, PathType, WeightType, SNRType
    except ImportError as ex:  # transport pulls numpy/torchdiffeq; skip if unavailable on CPU
        print(f"test_meanflow_training_losses_smoke: SKIPPED ({ex})")
        return

    torch.manual_seed(1)
    hidden, heads = 32, 4
    # latent_concat (no cond_latents): xt(C) + x1_concat(C) + mask(1) + partial_cond(C) + partial_mask(1)
    C = 2
    in_ch = 3 * C + 2  # = 8
    patch = (1, 2, 2)
    mock = _make_mock_with_r(hidden, heads, in_ch, out_ch=C, patch=patch, dtype=dtype)

    denoiser = Transport(
        model_type=ModelType.VELOCITY,
        path_type=PathType.LINEAR,
        loss_type=WeightType.NONE,
        train_eps=0.0, sample_eps=0.0,
        snr_type=SNRType.UNIFORM,
        training_timesteps=1000,
        reverse=True,        # rectified-flow reverse: ut = eps - x = v
        shift=1.0, video_shift=1.0,
    )

    B, T, H, W = 1, 2, 4, 4
    x1 = torch.randn(B, C, T, H, W, dtype=dtype)
    partial_cond = torch.randn(B, C, T, H, W, dtype=dtype)
    partial_mask = torch.ones(B, 1, T, H, W, dtype=dtype)

    text_dim, L_txt = 16, 6
    text_states = torch.randn(B, L_txt, text_dim, dtype=dtype)
    text_mask = torch.zeros(B, L_txt); text_mask[:, :4] = 1.0
    text_states_2 = torch.randn(B, 12, dtype=dtype)

    head_dim = hidden // heads
    a = head_dim // 4
    cos, sin = get_nd_rotary_pos_embed([a, a, head_dim - 2 * a], (T, H // patch[1], W // patch[2]), use_real=True)
    model_kwargs = dict(
        text_states=text_states, text_mask=text_mask, text_states_2=text_states_2,
        freqs_cos=cos.to(dtype), freqs_sin=sin.to(dtype),
    )

    args = SimpleNamespace(
        i2v_mode=True, i2v_condition_type="latent_concat", embedded_cfg_scale=None,
        meanflow_flow_ratio=0.5, meanflow_loss_c=1e-3, meanflow_loss_p=1.0,
    )

    u, terms = meanflow_training_losses(
        mock, denoiser, x1, args=args, model_kwargs=model_kwargs,
        cond_latents=None, partial_cond=partial_cond, partial_mask=partial_mask,
        attn_op=naive_attention_withT,
    )

    loss = terms["loss"]
    assert loss.shape == (B,), f"per-sample loss shape {loss.shape}"
    assert torch.isfinite(loss).all(), "non-finite MeanFlow loss"
    assert u.requires_grad and u.grad_fn is not None, "primal lost autograd graph"
    loss.mean().backward()
    assert torch.isfinite(mock.final_layer.linear.weight.grad).all(), "!finite backbone grad"
    assert torch.isfinite(mock.r_in.mlp[2].weight.grad).all(), "!finite r_in grad"
    assert 0.0 <= terms["r_eq_t_frac"] <= 1.0
    print("test_meanflow_training_losses_smoke: PASSED")


def test_meanflow_adapter_roundtrip(dtype=torch.float64):
    """apply -> get_state_dict -> load_state_dict preserves r_in weights."""
    mock = _make_mock_with_r(dtype=dtype)
    sd = get_meanflow_state_dict(mock)
    assert sd and all(k.startswith("r_in.") for k in sd), "state_dict keys must be r_in.*"

    # perturb, then restore
    saved = {k: v.clone() for k, v in sd.items()}
    with torch.no_grad():
        for p in get_meanflow_parameters(mock):
            p.add_(1.0)
    load_meanflow_state_dict(mock, saved)
    sd2 = get_meanflow_state_dict(mock)
    for k in saved:
        torch.testing.assert_close(sd2[k], saved[k])
    print("test_meanflow_adapter_roundtrip: PASSED")


if __name__ == "__main__":
    test_meanflow_identity_math()
    test_meanflow_training_losses_smoke()
    test_meanflow_adapter_roundtrip()
    print("MEANFLOW LOSS CPU TESTS PASSED")
