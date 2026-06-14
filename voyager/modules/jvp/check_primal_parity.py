# SPDX-License-Identifier: Apache-2.0
"""
Primal-parity check for the model-level JVP forward, on the REAL Hunyuan DiT.

Run on the GPU server (env ``voyager``) — needs CUDA + triton + flash-attn so the
vendored JVP attention kernel (``attention_withT``) runs, and the real pretrained
weights so the comparison is meaningful::

    MODEL_BASE=ckpts python -m voyager.modules.jvp.check_primal_parity \
        --model HYVideo-T/2 --vae 884-16c-hy \
        --i2v-condition-type latent_concat \
        --embedded-cfg-scale 6.0 --flow-reverse

(``--i2v-mode`` defaults to True and *takes a value* — omit it, or pass
``--i2v-mode True``; do not pass it as a bare flag.)

It builds ONLY the DiT (``load_model`` + ``load_state_dict`` — no VAE / text
encoders), synthesises one correctly-shaped ``latent_concat`` batch (batch size 1),
and asserts:

    model_forward_jvp(model, (xt, 0), (t, 0), ...)[0]  ≈  model(xt, t, ...)['x']

i.e. the **primal** of the JVP path equals the ordinary forward. The only nontrivial
thing this proves on real data is that the dense ``attention_withT`` over
``[img, cropped-txt]`` reproduces the baseline's varlen / masked attention over
``[img, full-txt]`` — the one place the JVP decomposition departs from the baseline.
Tangents are zero here (the primal is invariant to them), so a mismatch isolates to
the forward path, not the tangent math (that is covered by the CPU/GPU JVP tests).

Latent grid size is taken from ``--video-size H W`` (÷8 spatial) and the env vars
``PARITY_LAT_T`` (default 13), ``PARITY_TXT_LEN`` (default 64), ``PARITY_TXT_TRUE``
(default 48) so no dataset / renderer is needed.

This harness doubles as the scaffold for a real MeanFlow training step: the
model + latent_concat + freqs + text construction mirror
``deepspeed_train_render.py`` / ``transport.training_losses``.
"""

import os
from pathlib import Path

import torch
from loguru import logger

from voyager.config import parse_args
from voyager.constants import PRECISION_TO_TYPE
from voyager.modules import load_model
from voyager.utils.train_utils import load_state_dict, get_rope_freq_from_size
from voyager.modules.jvp.jvp_model import model_forward_jvp
from voyager.modules.jvp.jvp_attention import attention_withT


def _build_dit_only(args, device):
    """Construct just the Hunyuan DiT with pretrained weights (no VAE/text enc)."""
    factor_kwargs = {"device": device, "dtype": PRECISION_TO_TYPE[args.precision]}
    assert args.i2v_mode and args.i2v_condition_type == "latent_concat", \
        "parity check targets the i2v latent_concat backbone"
    in_channels = args.latent_channels * 3 + 2   # [z, cond, mask, partial_cond, partial_mask]
    out_channels = args.latent_channels
    model = load_model(args, in_channels=in_channels, out_channels=out_channels, factor_kwargs=factor_kwargs)
    model_base = os.environ.get("MODEL_BASE", "ckpts")
    model = load_state_dict(args, model, logger, Path(model_base))
    model.eval()
    return model


@torch.no_grad()
def main():
    args = parse_args(mode="eval")
    device = "cuda"
    dtype = PRECISION_TO_TYPE[args.precision]

    # scope guard: standard backbone only (model_forward_jvp asserts these too)
    assert not getattr(args, "train_multiple_kernels", False) and getattr(args, "patch_adapter_size", None) is None
    assert not getattr(args, "use_double_branch", False)

    model = _build_dit_only(args, device)
    logger.info(f"Built DiT: hidden={model.hidden_size} in={model.in_channels} out={model.out_channels} "
                f"guidance_embed={model.guidance_embed} text_proj={model.text_projection}")

    # ---- synthesise one latent_concat batch (B=1) ----
    B = 1
    Tlat = int(os.environ.get("PARITY_LAT_T", "13"))
    Hlat = args.video_size[0] // 8
    Wlat = args.video_size[1] // 8
    pt, ph, pw = model.patch_size
    # make spatial dims divisible by the patch size
    Hlat -= Hlat % ph
    Wlat -= Wlat % pw
    Tlat -= (Tlat - 1) % pt if pt > 1 else 0
    C = args.latent_channels

    torch.manual_seed(0)
    z = torch.randn(B, C, Tlat, Hlat, Wlat, device=device, dtype=dtype)
    cond = torch.randn(B, C, Tlat, Hlat, Wlat, device=device, dtype=dtype)
    mask = torch.ones(B, 1, Tlat, Hlat, Wlat, device=device, dtype=dtype)
    partial_cond = torch.randn(B, C, Tlat, Hlat, Wlat, device=device, dtype=dtype)
    partial_mask = torch.ones(B, 1, Tlat, Hlat, Wlat, device=device, dtype=dtype)
    xt = torch.cat([z, cond, mask, partial_cond, partial_mask], dim=1)
    assert xt.shape[1] == model.in_channels, f"{xt.shape[1]} != in_channels {model.in_channels}"

    input_t = torch.rand(B, device=device, dtype=dtype) * 1000.0

    # ---- text conditioning (prefix mask exercises the cropping path) ----
    L = int(os.environ.get("PARITY_TXT_LEN", "64"))
    true_len = min(int(os.environ.get("PARITY_TXT_TRUE", "48")), L)
    text_states = torch.randn(B, L, model.text_states_dim, device=device, dtype=dtype)
    text_mask = torch.zeros(B, L, device=device, dtype=torch.long)
    text_mask[:, :true_len] = 1
    text_states_2 = torch.randn(B, model.text_states_dim_2, device=device, dtype=dtype)

    # ---- RoPE freqs (standard backbone) ----
    # Call get_rope_freq_from_size directly rather than build_rope: build_rope's
    # wrapper reads the training-only arg `train_multiple_kernels`, absent from the
    # eval-mode namespace. For the standard backbone this is exactly the inference path.
    latents_size = list(z.shape[-3:])  # [T, H, W] latent grid
    freqs_cos, freqs_sin = get_rope_freq_from_size(
        args, model, latents_size, ndim=3, target_ndim=3,
    )
    freqs_cos = freqs_cos.to(device)
    freqs_sin = freqs_sin.to(device)

    guidance = None
    if model.guidance_embed:
        scale = args.embedded_cfg_scale if args.embedded_cfg_scale is not None else 6.0
        guidance = torch.tensor([scale] * B, dtype=torch.float32, device=device).to(dtype) * 1000.0

    autocast_enabled = dtype != torch.float32

    # ---- baseline forward ----
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=autocast_enabled):
        base = model(
            xt, input_t,
            text_states=text_states, text_mask=text_mask, text_states_2=text_states_2,
            freqs_cos=freqs_cos, freqs_sin=freqs_sin, guidance=guidance, return_dict=True,
        )["x"]

    # ---- JVP primal (zero tangents) ----
    t_x = torch.zeros_like(xt)
    t_t = torch.zeros_like(input_t)
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=autocast_enabled):
        prim, t_out = model_forward_jvp(
            model, (xt, t_x), (input_t, t_t),
            text_states=text_states, text_mask=text_mask, text_states_2=text_states_2,
            freqs_cos=freqs_cos, freqs_sin=freqs_sin, guidance=guidance,
            attn_op=attention_withT,
        )

    assert prim.shape == base.shape, f"shape mismatch {prim.shape} vs {base.shape}"
    diff = (prim.float() - base.float()).abs()
    denom = base.float().abs().mean().clamp_min(1e-6)
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    rel = (mean_abs / denom).item()

    logger.info(f"primal vs baseline: max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} "
                f"rel(mean/|base|)={rel:.3e}  base|mean|={denom.item():.3e}")
    logger.info(f"t_out (zero-tangent) max_abs={t_out.float().abs().max().item():.3e} (should be ~0)")

    # bf16/fp16 kernel vs varlen: ~1e-2 relative is expected; fp32 should be ~1e-4
    tol = 5e-2 if dtype != torch.float32 else 1e-3
    if rel < tol:
        logger.success(f"PRIMAL PARITY PASSED (rel {rel:.3e} < tol {tol:.1e})")
    else:
        logger.error(f"PRIMAL PARITY FAILED (rel {rel:.3e} >= tol {tol:.1e})")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
