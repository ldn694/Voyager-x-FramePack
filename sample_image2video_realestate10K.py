import os
import cv2
import numpy as np
import torch
import json
import time
from voyager.utils.file_utils import save_videos_grid
from voyager.config import parse_args
from voyager.inference import HunyuanVideoSampler
from pathlib import Path
from utils.metrics import MergedMetric

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
    args = parse_args()
    models_root_path = Path(args.model_base)
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(
        models_root_path, args=args)
    
    dataset_path = '/raid/hvtham/ldnhuan/HunyuanWorld-Voyager/dataset/RealEstate10K/refined_test_150_768x512'
    output_path = 'evaluation/RealEstate10K/refined_test_150_768x512'
    os.makedirs(output_path, exist_ok=True)

    merged_metric = MergedMetric(device='cuda')

    test_paths = sorted(os.listdir(dataset_path))
    start_time = time.time()
    total_tests = len(test_paths)
    for idx, test_path in enumerate(test_paths, 1):
        elapsed = time.time() - start_time
        avg_time = elapsed / idx
        eta = avg_time * (total_tests - idx)
        print(f"[{idx}/{total_tests}] ETA: {eta/60:.2f} min")
        full_input_path = os.path.join(dataset_path, test_path, 'input')
        save_path = os.path.join(output_path, test_path)
        os.makedirs(save_path, exist_ok=True)
        print(f'Processing {full_input_path}, saving to {save_path}')

        outputs = hunyuan_video_sampler.predict(
            prompt=args.prompt,
            height=args.video_size[0],
            width=args.video_size[1],
            video_length=args.video_length,
            seed=args.seed,
            negative_prompt=args.neg_prompt,
            infer_steps=args.infer_steps,
            guidance_scale=args.cfg_scale,
            num_videos_per_prompt=args.num_videos,
            flow_shift=args.flow_shift,
            batch_size=args.batch_size,
            embedded_guidance_scale=args.embedded_cfg_scale,
            i2v_mode=args.i2v_mode,
            i2v_resolution=args.i2v_resolution,
            i2v_image_path=args.i2v_image_path,
            i2v_condition_type=args.i2v_condition_type,
            i2v_stability=args.i2v_stability,
            ulysses_degree=args.ulysses_degree,
            ring_degree=args.ring_degree,
            ref_images=[(
                os.path.join(full_input_path, "rgb", "000.png"),
                os.path.join(full_input_path, "depth", "000.exr")
            )],
            partial_cond=[(
                os.path.join(full_input_path, "rgb", f"{j:03d}.png"),
                os.path.join(full_input_path, "depth", f"{j:03d}.exr")
            ) for j in range(49)],
            partial_mask=[(
                os.path.join(full_input_path, "rgb", f"{j:03d}.png"),
                os.path.join(full_input_path, "mask", f"{j:03d}.png")
            ) for j in range(49)],
        )
        samples = outputs['samples'] # (B, C, T, H * 2, W)
        save_videos_grid(samples, os.path.join(save_path, f'{test_path}.mp4'), fps=10)
        samples = samples[0].permute(1, 0, 2, 3)  # (T, C, H * 2, W)
        T, C, H, W = samples.shape
        samples = samples[:, :, :H // 2, :]
        full_ground_truth_path = os.path.join(dataset_path, test_path, 'ground_truth')
        ground_truth = read_ground_truth_rgb(full_ground_truth_path, samples.shape[0])  # (T, C, H, W)
        assert samples.shape == ground_truth.shape, f"Shape mismatch: samples {samples.shape}, ground_truth {ground_truth.shape}"
        metrics = merged_metric.compute(samples, ground_truth)  # (T,)
        avg_metric = {}
        for metric in metrics:
            avg_metric[metric] = sum(metrics[metric]) / len(metrics[metric])
            print(metric, avg_metric[metric])
        final_metric = {
            'metrics': metrics,
            'avg_metric': avg_metric,
        }
        # Save metrics to json
        with open(os.path.join(save_path, f'{test_path}.json'), 'w') as f:
            json.dump(final_metric, f, indent=4)