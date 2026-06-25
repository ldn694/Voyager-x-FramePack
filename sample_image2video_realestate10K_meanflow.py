"""
Few-step MeanFlow evaluation over the RealEstate10K test set (dataset variant of
sample_image2video_meanflow.py, mirroring sample_image2video_realestate10K_dmd2.py).

Loads the base Hunyuan-Voyager DiT, attaches the trained MeanFlow ``r_in`` (and,
if given, the LoRA it was trained with), then iterates the dataset folders running
``HunyuanVideoSampler.predict_meanflow`` (``z_r = z_t - (t-r)*u``) per case and
writes per-case videos + metrics JSON.

See sample_realestate10K_meanflow_replay.sh for an invocation. Key flags:
  --meanflow-steps N   number of sampling intervals (1 = single NFE)
  --meanflow-path P    trained meanflow_last.pt (the r_in weights)
  --use-lora --lora-path P   the LoRA the student was trained with (if any)
  --dataset-path / --output-path   eval dataset root and results destination
"""

import os
import cv2
import numpy as np
import torch
import json
import time
from voyager.utils.file_utils import save_videos_grid
from voyager.config import parse_args
from voyager.inference import HunyuanVideoSampler
from voyager.diffusion.meanflow import apply_meanflow_to_hunyuan_video, load_meanflow_state_dict
from pathlib import Path
from utils.metrics import MergedMetric
from loguru import logger
import sys

def read_ground_truth_rgb(folder, num_frames):
    print(f"Reading ground truth from {folder} for {num_frames} frames")
    # In the folder, there would be file as id:04d
    rgb_images = os.listdir(os.path.join(folder, 'rgb'))
    rgb_images = sorted(rgb_images)
    rgbs = []
    # Read the first num_frames image, return a [num_frames, C, H, W] tensor in [0, 1]
    for i in range(num_frames):
        img_path = os.path.join(folder, 'rgb', rgb_images[i])
        img = cv2.imread(img_path)  # H, W, C in BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32') / 255.0
        rgbs.append(img)
    rgbs = np.stack(rgbs, axis=0)  # (T, H, W, C)
    rgbs = torch.from_numpy(rgbs).permute(0, 3, 1, 2)  # (T, C, H, W)
    return rgbs

if __name__ == "__main__":
    args = parse_args(mode="eval_realestate10K")
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    models_root_path = Path(args.model_base)

    meanflow_steps = getattr(args, "meanflow_steps", 0)
    if not meanflow_steps or meanflow_steps <= 0:
        raise ValueError("--meanflow-steps must be > 0 for MeanFlow inference.")
    logger.info(f"Running MeanFlow sampling with {meanflow_steps} interval(s).")

    # Load base model (+ LoRA via --use-lora/--lora-path inside from_pretrained).
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(
        models_root_path, args=args)
    args = hunyuan_video_sampler.args
    model = hunyuan_video_sampler.model

    # Attach + load the trained MeanFlow second-time (r) embedder.
    apply_meanflow_to_hunyuan_video(model)
    if args.meanflow_path is None or not os.path.isfile(args.meanflow_path):
        raise ValueError(f"--meanflow-path must point to a trained meanflow_last.pt, got {args.meanflow_path}")
    logger.info(f"Loading MeanFlow r_in from {args.meanflow_path}")
    ckpt = torch.load(args.meanflow_path, map_location="cpu", weights_only=False)
    load_meanflow_state_dict(model, ckpt["meanflow"], strict=False)
    model.eval()

    dataset_path = args.dataset_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    args_path = os.path.join(output_path, 'args.json')
    args_json = vars(args)
    args_json['dataset_path'] = dataset_path
    args_json['output_path'] = output_path
    with open(args_path, 'w') as f:
        json.dump(args_json, f, indent=4)

    merged_metric = MergedMetric(device='cuda')

    test_paths = sorted(os.listdir(dataset_path))
    start_time = time.time()
    total_tests = len(test_paths)
    for idx, test_path in enumerate(test_paths, 1):
        single_start_time = time.time()
        elapsed = time.time() - start_time
        avg_time = elapsed / idx
        eta = avg_time * (total_tests - idx)
        print(f"[{idx}/{total_tests}] ETA: {eta/60:.2f} min")
        full_input_path = os.path.join(dataset_path, test_path, 'input')
        save_path = os.path.join(output_path, test_path)
        os.makedirs(save_path, exist_ok=True)
        print(f'Processing {full_input_path}, saving to {save_path}')

        if os.path.exists(os.path.join(save_path, f'{test_path}.json')):
            print(f"Metrics already exist for {test_path}, skipping...")
            continue
        outputs = hunyuan_video_sampler.predict_meanflow(
            prompt=args.prompt,
            height=args.video_size[0],
            width=args.video_size[1],
            video_length=args.video_length,
            seed=args.seed,
            negative_prompt=args.neg_prompt,
            flow_shift=args.flow_shift,
            embedded_guidance_scale=args.embedded_cfg_scale,
            num_steps=meanflow_steps,
            i2v_condition_type=args.i2v_condition_type,
            ref_images=[(
                os.path.join(full_input_path, "rgb", "000.png"),
                os.path.join(full_input_path, "depth", "000.exr")
            )],
            partial_cond=[(
                os.path.join(full_input_path, "rgb", f"{j:03d}.png"),
                os.path.join(full_input_path, "depth", f"{j:03d}.exr")
            ) for j in range(args.video_length)],
            partial_mask=[(
                os.path.join(full_input_path, "mask", f"{j:03d}.png"),
                os.path.join(full_input_path, "mask", f"{j:03d}.png")
            ) for j in range(args.video_length)],
        )
        samples = outputs['samples'] # (B, C, T, H * 2, W)
        save_videos_grid(samples, os.path.join(save_path, f'{test_path}.mp4'), fps=10)
        samples = samples[0].permute(1, 0, 2, 3)  # (T, C, H * 2, W)
        T, C, H, W = samples.shape
        samples = samples[:, :, :H // 2, :]
        full_ground_truth_path = os.path.join(dataset_path, test_path, 'ground_truth')
        ground_truth = read_ground_truth_rgb(full_ground_truth_path, samples.shape[0])  # (T, C, H, W)
        assert samples.shape == ground_truth.shape, f"Shape mismatch: samples {samples.shape}, ground_truth {ground_truth.shape}"
        if args.first_clean_frame: # Whether to use the first clean frame for evaluation
            samples[0] = ground_truth[0]
        single_test_time = time.time() - single_start_time
        metrics = merged_metric.compute(samples, ground_truth)  # (T,)
        avg_metric = {}
        for metric in metrics:
            avg_metric[metric] = sum(metrics[metric]) / len(metrics[metric])
            print(metric, avg_metric[metric])
        final_metric = {
            'metrics': metrics,
            'avg_metric': avg_metric,
            'time': single_test_time
        }
        # Save metrics to json
        with open(os.path.join(save_path, f'{test_path}.json'), 'w') as f:
            json.dump(final_metric, f, indent=4)
