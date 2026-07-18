"""Offline calibration of TeaCache polynomial coefficients for Voyager.

Runs a *no-cache* (full-quality) forward over one or more input cases with the
transformer in calibration mode, collecting per-step pairs of
    x = relative-L1 of the timestep-modulated input
    y = relative-L1 of the transformer body residual (the cached quantity)
then fits y = f(x) with a degree-4 polynomial. The resulting coefficients can be
pasted into HYVideoDiffusionTransformer.teacache_coefficients (or loaded via a
future flag).

Usage (single case):
    python calibrate_teacache.py --model HYVideo-T/2 \
        --input-path examples/case1 --prompt "..." \
        --i2v-stability --flow-reverse --flow-shift 7.0 \
        --infer-steps 50 --embedded-cfg-scale 6.0 --seed 0 --use-cpu-offload \
        --with-teacache

Multiple cases: point --input-path at a parent directory that contains several
case sub-folders (each with rgb/ depth/ mask/), or pass --calib-glob.
More cases + more steps => a more robust fit. 3-5 clips is usually enough.
"""
import os
import sys
import glob
import json
import argparse
from pathlib import Path

import torch
import numpy as np
from loguru import logger

from voyager.config import parse_args
from voyager.inference import HunyuanVideoSampler


def list_cases(args):
    """Return a list of case directories to calibrate over."""
    calib_glob = getattr(args, "calib_glob", None)
    if calib_glob:
        cases = sorted(glob.glob(calib_glob))
    else:
        p = Path(args.input_path)
        # If input-path itself is a case (has rgb/), use it directly.
        if (p / "rgb").is_dir():
            cases = [str(p)]
        else:
            cases = sorted(
                str(c) for c in p.iterdir()
                if c.is_dir() and (c / "rgb").is_dir()
            )
    if not cases:
        raise ValueError(
            f"No calibration cases found under {args.input_path!r}. "
            "Each case needs rgb/ depth/ mask/ sub-folders."
        )
    return cases


def run_case(sampler, args, input_path):
    """Run one full no-skip forward pass over a case (records into cal_data)."""
    vl = args.video_length
    sampler.predict(
        prompt=args.prompt,
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=vl,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        with_teacache=True,          # gate must be active for the hook to run
        teacache_thresh=args.teacache_thresh,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=1,
        flow_shift=args.flow_shift,
        batch_size=1,
        embedded_guidance_scale=args.embedded_cfg_scale,
        i2v_mode=args.i2v_mode,
        i2v_resolution=args.i2v_resolution,
        i2v_image_path=args.i2v_image_path,
        i2v_condition_type=args.i2v_condition_type,
        i2v_stability=args.i2v_stability,
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        ref_images=[(
            os.path.join(input_path, "rgb", "000.png"),
            os.path.join(input_path, "depth", "000.exr"),
        )],
        partial_cond=[(
            os.path.join(input_path, "rgb", f"{j:03d}.png"),
            os.path.join(input_path, "depth", f"{j:03d}.exr"),
        ) for j in range(vl)],
        partial_mask=[(
            os.path.join(input_path, "mask", f"{j:03d}.png"),
            os.path.join(input_path, "mask", f"{j:03d}.png"),
        ) for j in range(vl)],
        use_kernel_indices=args.use_kernel_indices if args.use_kernel_indices is not None else None,
        step_sample=args.step_sample,
        attn_map=args.attn_map,
        dmd2_steps=getattr(args, "dmd2_steps", 0),
    )


def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    args = parse_args()

    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    args = sampler.args

    transformer = sampler.pipeline.transformer
    if getattr(transformer, "patch_sizes", None) is not None or \
            getattr(transformer, "use_second_branch", False):
        raise SystemExit(
            "Calibration is only meaningful for the standard single-branch "
            "path (the original Voyager checkpoint)."
        )

    # Enable calibration recording.
    transformer.enable_teacache = True
    transformer.teacache_calibrate = True
    transformer.teacache_cal_data = []

    cases = list_cases(args)
    logger.info(f"Calibrating over {len(cases)} case(s): {cases}")

    with torch.no_grad():
        for i, case in enumerate(cases):
            logger.info(f"[{i + 1}/{len(cases)}] {case}")
            run_case(sampler, args, case)

    data = transformer.teacache_cal_data
    if len(data) < 6:
        raise SystemExit(
            f"Only {len(data)} (x, y) samples collected; need more steps/cases "
            "for a stable degree-4 fit."
        )

    xs = np.array([d[0] for d in data], dtype=np.float64)
    ys = np.array([d[1] for d in data], dtype=np.float64)

    deg = getattr(args, "calib_degree", 4)
    coeffs = np.polyfit(xs, ys, deg)
    resid = ys - np.poly1d(coeffs)(xs)
    rmse = float(np.sqrt(np.mean(resid ** 2)))

    logger.info(f"Collected {len(data)} (x, y) pairs across {len(cases)} case(s).")
    logger.info(f"Fit degree-{deg} polynomial, RMSE={rmse:.6f}")
    coeff_list = [float(c) for c in coeffs]
    print("\nteacache_coefficients = [")
    print("    " + ", ".join(f"{c:.8e}" for c in coeff_list))
    print("]\n")

    out_path = os.path.join(
        args.save_path if args.save_path else ".", "teacache_coefficients.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {"coefficients": coeff_list, "degree": deg, "rmse": rmse,
             "num_samples": len(data), "cases": cases},
            f, indent=2)
    logger.info(f"Saved coefficients to {out_path}")


if __name__ == "__main__":
    main()
