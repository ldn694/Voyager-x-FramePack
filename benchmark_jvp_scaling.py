#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Scaling benchmark for the MeanFlow / rCM JVP forward+backward path.

Goal: understand how forward-JVP time, backward time, and peak GPU memory scale
with problem size, so the ~49-minute forward "hang" seen at the real 49-frame
config (latent [1,16,13,136,96] -> 42,432 image tokens, 20+40 blocks) can be
attributed to a concrete cause:

  * If `jvp_fwd / primal_fwd` ratio is ~constant and small (say 2-4x) and total
    time scales ~linearly with (#blocks x #tokens), the cost is *legitimate
    compute* -> reduce tokens / use checkpointing, the hang is just size.
  * If the ratio blows up with size (functorch eager has per-op Python overhead
    and no fusion), the JVP *machinery* is the bottleneck -> the fix is a
    different JVP implementation (manual dual numbers / fused kernel), not memory.
  * The attention term is the only O(S^2) piece; the seqlen sweep vs the depth
    sweep separates `a * N * S` (linear) from `b * N * S^2` (attention).

For every config the SAME real `HYVideoDiffusionTransformer` blocks are used
(random init -- no weights needed), measuring:
    primal   : plain `model.forward`            (reference, no functorch)
    jvp      : `model_forward_jvp` forward only
    jvp+bwd  : `model_forward_jvp` fwd + backward on a dummy scalar loss
each at gradient-checkpoint off and/or on, with peak allocated memory.

Run on the GPU server (needs the `voyager` env: torch + flash-attn + triton):

    CUDA_VISIBLE_DEVICES=5 python benchmark_jvp_scaling.py --sweep seqlen
    CUDA_VISIBLE_DEVICES=5 python benchmark_jvp_scaling.py --sweep frames
    CUDA_VISIBLE_DEVICES=5 python benchmark_jvp_scaling.py --sweep depth
    CUDA_VISIBLE_DEVICES=5 python benchmark_jvp_scaling.py --sweep width
    CUDA_VISIBLE_DEVICES=5 python benchmark_jvp_scaling.py --sweep real   # single real-size point

OOM rows are expected for the large no-checkpoint configs (that *is* the bug);
the script catches them and keeps going.
"""

import argparse
import gc
import sys
import time
from types import SimpleNamespace
from typing import Optional

import torch

# Quiet the backbone's per-forward DEBUG logging (keeps the table readable).
try:
    from loguru import logger as _loguru_logger
    _loguru_logger.remove()
    _loguru_logger.add(sys.stderr, level="INFO")
except Exception:
    pass

from voyager.modules.models import HYVideoDiffusionTransformer
from voyager.modules.posemb_layers import get_nd_rotary_pos_embed
from voyager.modules.jvp.jvp_model import model_forward_jvp
from voyager.modules.jvp.jvp_attention import attention_withT
from voyager.modules.lora_layers import apply_lora_to_hunyuan_video


_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def make_rope_dim_list(head_dim: int) -> list:
    """Split head_dim into [t, h, w] rotary dims, mirroring the 16:56:56 ratio.

    Each part is even and the three sum to head_dim (required by get_nd_rotary_pos_embed).
    """
    t = max(2, int(round(head_dim * 16 / 128)))
    t -= t % 2
    rem = head_dim - t
    h = (rem // 2)
    h -= h % 2
    w = head_dim - t - h
    return [t, h, w]


def build_model(hidden, heads, n_double, n_single, in_ch, out_ch, patch, device, dtype):
    """Instantiate the real backbone at the requested size (random init)."""
    args = SimpleNamespace(
        i2v_condition_type="latent_concat",
        text_states_dim=4096,     # LLaVA-style; only the (tiny) txt path uses it
        text_states_dim_2=768,    # CLIP-L pooled, feeds vector_in
        gradient_checkpoint=False,
        gradient_checkpoint_layers=-1,
        use_context_block=False,
    )
    model = HYVideoDiffusionTransformer(
        args,
        patch_size=list(patch),
        in_channels=in_ch,
        out_channels=out_ch,
        hidden_size=hidden,
        heads_num=heads,
        mlp_width_ratio=4.0,
        mm_double_blocks_depth=n_double,
        mm_single_blocks_depth=n_single,
        rope_dim_list=make_rope_dim_list(hidden // heads),
        guidance_embed=False,
        text_projection="linear",   # avoids the heavy refiner; DiT-block scaling is identical
        use_attention_mask=True,
        dtype=dtype,
        device=torch.device(device),
    )
    # The constructor's factory_kwargs don't reach every submodule (e.g. final_layer
    # stays on CPU/fp32), so place the whole model explicitly.
    model.to(device=torch.device(device), dtype=dtype)
    model.eval()
    return model


def make_inputs(model, B, T, H, W, L_txt, device, dtype):
    """Synthesize a latent_concat batch + JVP tangent directions (v on data channels)."""
    in_ch = model.in_channels
    x = torch.randn(B, in_ch, T, H, W, device=device, dtype=dtype)
    t = torch.rand(B, device=device, dtype=dtype) * 1000.0
    text_states = torch.randn(B, L_txt, model.text_states_dim, device=device, dtype=dtype)
    text_mask = torch.ones(B, L_txt, device=device)
    text_states_2 = torch.randn(B, model.text_states_dim_2, device=device, dtype=dtype)

    pt, ph, pw = model.patch_size
    cos, sin = get_nd_rotary_pos_embed(
        model.rope_dim_list, (T // pt, H // ph, W // pw), use_real=True
    )
    cos = cos.to(device=device, dtype=dtype)
    sin = sin.to(device=device, dtype=dtype)

    # tangent: velocity on the first 16 (data) channels, zero on conditioning
    t_x = torch.zeros_like(x)
    n_data = min(16, in_ch)
    t_x[:, :n_data] = torch.randn_like(x[:, :n_data])
    t_t = torch.full_like(t, 1.0)

    tokens = (T // pt) * (H // ph) * (W // pw)
    return dict(
        x=x, t=t, t_x=t_x, t_t=t_t,
        text_states=text_states, text_mask=text_mask, text_states_2=text_states_2,
        cos=cos, sin=sin, tokens=tokens,
    )


def _sync():
    torch.cuda.synchronize()


def _cleanup(model):
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()


def time_primal(model, inp, ckpt, dtype):
    """Plain model.forward (builds the autograd graph, no functorch)."""
    model.train()
    model.gradient_checkpoint = ckpt
    torch.cuda.reset_peak_memory_stats()

    def fwd():
        with torch.autocast("cuda", dtype=dtype, enabled=(dtype != torch.float32)):
            out = model(
                inp["x"], inp["t"],
                text_states=inp["text_states"], text_mask=inp["text_mask"],
                text_states_2=inp["text_states_2"],
                freqs_cos=inp["cos"], freqs_sin=inp["sin"],
                return_dict=False,
            )
        return out[0] if isinstance(out, (tuple, list)) else out

    fwd()  # warmup (triton/cudnn autotune for this shape)
    _sync(); t0 = time.perf_counter()
    out = fwd()
    _sync(); fwd_ms = (time.perf_counter() - t0) * 1e3
    peak = torch.cuda.max_memory_allocated() / 1e9
    del out
    _cleanup(model)
    return fwd_ms, peak


def time_jvp(model, inp, ckpt, dtype):
    """model_forward_jvp forward, then backward on a dummy loss."""
    model.train()
    model.gradient_checkpoint = ckpt

    def fwd():
        with torch.autocast("cuda", dtype=dtype, enabled=(dtype != torch.float32)):
            u, dudt = model_forward_jvp(
                model,
                (inp["x"], inp["t_x"]),
                (inp["t"], inp["t_t"]),
                text_states=inp["text_states"], text_mask=inp["text_mask"],
                text_states_2=inp["text_states_2"],
                freqs_cos=inp["cos"], freqs_sin=inp["sin"],
                attn_op=attention_withT,
            )
        return u, dudt

    # warmup (autotune + any first-call functorch tracing)
    u, _ = fwd()
    if u.requires_grad:
        ((u.float() ** 2).mean()).backward()
    _cleanup(model)

    torch.cuda.reset_peak_memory_stats()
    _sync(); t0 = time.perf_counter()
    u, _ = fwd()
    _sync(); fwd_ms = (time.perf_counter() - t0) * 1e3

    bwd_ms = float("nan")
    if u.requires_grad:  # frozen backbone -> no graph -> forward-only lower bound
        loss = (u.float() ** 2).mean()
        _sync(); t1 = time.perf_counter()
        loss.backward()
        _sync(); bwd_ms = (time.perf_counter() - t1) * 1e3
        del loss
    peak = torch.cuda.max_memory_allocated() / 1e9

    del u
    _cleanup(model)
    return fwd_ms, bwd_ms, peak


# ---- sweep definitions: (label, hidden, heads, n_double, n_single, T, H, W) ----
# H, W are LATENT dims (multiples of patch=2). Real config: T=13, H=136, W=96.
def sweep_configs(name):
    REAL = dict(hidden=3072, heads=24, nd=20, ns=40)
    if name == "seqlen":            # fix real width/depth + T=13, grow spatial
        for H, W in [(34, 24), (68, 48), (102, 72), (136, 96)]:
            yield dict(**REAL, T=13, H=H, W=W)
    elif name == "frames":          # fix medium spatial, grow T (must be such that T%patch_t==0; patch_t=1)
        for T in [1, 5, 9, 13]:
            yield dict(**REAL, T=T, H=68, W=48)
    elif name == "depth":           # fix medium input, grow #blocks (keep 1:2 double:single)
        for nd, ns in [(2, 4), (5, 10), (10, 20), (20, 40)]:
            yield dict(hidden=3072, heads=24, nd=nd, ns=ns, T=5, H=68, W=48)
    elif name == "width":           # fix medium input+depth, grow width
        for hidden, heads in [(480, 5), (960, 5), (1536, 12), (3072, 24)]:
            yield dict(hidden=hidden, heads=heads, nd=10, ns=20, T=5, H=68, W=48)
    elif name == "real":            # the single real training point
        yield dict(**REAL, T=13, H=136, W=96)
    else:
        raise ValueError(name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", required=True, choices=["seqlen", "frames", "depth", "width", "real"])
    ap.add_argument("--ckpt", default="both", choices=["off", "on", "both"],
                    help="gradient-checkpoint setting(s) to measure")
    ap.add_argument("--no-primal", action="store_true", help="skip the plain model.forward baseline")
    ap.add_argument("--adapter", default="full", choices=["full", "lora"],
                    help="trainable scope. 'full' = all params (finetune upper bound); "
                         "'lora' = LoRA on backbone + base frozen, matching meanflow_train.sh "
                         "(only LoRA params get grads; activations still retained along LoRA paths).")
    ap.add_argument("--lora-rank", type=int, default=640, help="LoRA rank for --adapter lora")
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="freeze ALL params (requires_grad=False): forward-only timing + peak, "
                         "no gradient buffers, no retained graph. The lower-bound sanity check; "
                         "applied on top of --adapter. Forces gradient-checkpoint off (no backward).")
    ap.add_argument("--dtype", default="bf16", choices=list(_DTYPES))
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--txt-len", type=int, default=256)
    ap.add_argument("--in-ch", type=int, default=50,
                    help="latent_concat input channels; >=50 so model.forward's debug "
                         "channel slices (x[:, 49:]) are non-empty (timing-insensitive)")
    ap.add_argument("--out-ch", type=int, default=16)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "needs a CUDA device"
    device, dtype = "cuda", _DTYPES[args.dtype]
    patch = (1, 2, 2)
    ckpt_settings = {"off": [False], "on": [True], "both": [False, True]}[args.ckpt]
    if args.freeze_backbone:
        ckpt_settings = [False]   # no backward -> checkpointing is a no-op (and would warn per block)

    gpu = torch.cuda.get_device_properties(0)
    print(f"# GPU: {gpu.name}  {gpu.total_memory/1e9:.1f} GB | dtype={args.dtype} | sweep={args.sweep}")
    print(f"# patch={patch} batch={args.batch} txt_len={args.txt_len} | adapter={args.adapter}"
          + (f" rank={args.lora_rank}" if args.adapter == "lora" else "")
          + (" | FREEZE-BACKBONE (forward-only lower bound)" if args.freeze_backbone else ""))
    hdr = ["config", "ckpt", "tokens", "blocks", "primal_fwd_ms", "jvp_fwd_ms",
           "ratio", "jvp_bwd_ms", "peak_GB"]
    print("  ".join(f"{h:>13}" for h in hdr))

    model_cache = {}
    for cfg in sweep_configs(args.sweep):
        key = (cfg["hidden"], cfg["heads"], cfg["nd"], cfg["ns"])
        if key not in model_cache:
            model_cache.clear(); gc.collect(); torch.cuda.empty_cache()
            m = build_model(
                cfg["hidden"], cfg["heads"], cfg["nd"], cfg["ns"],
                args.in_ch, args.out_ch, patch, device, dtype,
            )
            if args.adapter == "lora":
                # mirror meanflow_train.sh: LoRA on backbone, base frozen
                apply_lora_to_hunyuan_video(
                    m, r=args.lora_rank, lora_alpha=args.lora_alpha,
                    lora_dropout=0.0, freeze_base=True,
                )
                m.to(device=torch.device(device), dtype=dtype)  # cast freshly-added LoRA layers
            if args.freeze_backbone:
                m.requires_grad_(False)
            n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in m.parameters())
            print(f"# built {key}: trainable {n_train/1e6:.1f}M / {n_total/1e6:.1f}M params "
                  f"(adapter={args.adapter}, freeze_backbone={args.freeze_backbone})")
            model_cache[key] = m
        model = model_cache[key]
        blocks = cfg["nd"] + cfg["ns"]
        label = f"h{cfg['hidden']}_d{blocks}_{cfg['T']}x{cfg['H']}x{cfg['W']}"

        inp = make_inputs(model, args.batch, cfg["T"], cfg["H"], cfg["W"],
                          args.txt_len, device, dtype)

        for ckpt in ckpt_settings:
            primal_ms = float("nan")
            try:
                if not args.no_primal:
                    primal_ms, _ = time_primal(model, inp, ckpt, dtype)
                jvp_fwd, jvp_bwd, peak = time_jvp(model, inp, ckpt, dtype)
                ratio = jvp_fwd / primal_ms if primal_ms == primal_ms and primal_ms > 0 else float("nan")
                row = [label, "ckpt" if ckpt else "no", inp["tokens"], blocks,
                       f"{primal_ms:.1f}", f"{jvp_fwd:.1f}", f"{ratio:.2f}",
                       f"{jvp_bwd:.1f}", f"{peak:.2f}"]
            except torch.cuda.OutOfMemoryError:
                _cleanup(model)
                row = [label, "ckpt" if ckpt else "no", inp["tokens"], blocks,
                       "OOM", "OOM", "-", "OOM", "OOM"]
            print("  ".join(f"{str(c):>13}" for c in row), flush=True)

    print("\n# Read: ratio = jvp_fwd/primal_fwd (functorch eager overhead). The hang is")
    print("#   in the FORWARD, so jvp_fwd_ms + ratio are the primary signal. Fit")
    print("#   time ~ a*N*S + b*N*S^2 across the seqlen (S) and depth (N) sweeps, then")
    print("#   extrapolate to the real point (blocks=60, tokens=42432).")
    print("# Memory regimes (peak_GB): --adapter full = finetune UPPER bound (full grad")
    print("#   buffers); --adapter lora = matches meanflow_train.sh (grads only for LoRA, but")
    print("#   activations still retained along LoRA paths); --freeze-backbone = forward-only")
    print("#   LOWER bound (no graph, no grad buffers). The real run sits at --adapter lora.")


if __name__ == "__main__":
    main()
