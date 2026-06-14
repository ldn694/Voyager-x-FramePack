"""
Few-step MeanFlow inference entry (standalone of the teacher/DMD2 pipeline loop).

Loads the base Hunyuan-Voyager DiT, attaches the trained MeanFlow ``r_in`` (and,
if given, the LoRA it was trained with), then runs ``HunyuanVideoSampler``'s
``predict_meanflow`` — which reuses the normal conditioning prep but replaces the
scheduler denoise loop with ``meanflow_sample`` (``z_r = z_t - (t-r)*u``).

See ``meanflow_sample.sh`` for an invocation. Key flags:
  --meanflow-steps N   number of sampling intervals (1 = single NFE)
  --meanflow-path P    trained meanflow_last.pt (the r_in weights)
  --use-lora --lora-path P   the LoRA the student was trained with (if any)
"""

import os
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from loguru import logger

from voyager.config import parse_args
from voyager.inference import HunyuanVideoSampler
from voyager.diffusion.meanflow import apply_meanflow_to_hunyuan_video, load_meanflow_state_dict
from voyager.utils.file_utils import save_videos_grid


def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    args = parse_args()
    print(args)

    if getattr(args, "meanflow_steps", 0) <= 0:
        raise ValueError("--meanflow-steps must be > 0 for MeanFlow inference.")

    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    save_path = args.save_path if args.save_path_suffix == "" else f"{args.save_path}_{args.save_path_suffix}"
    os.makedirs(save_path, exist_ok=True)

    # Load base model (+ LoRA via --use-lora/--lora-path inside from_pretrained).
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
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

    outputs = hunyuan_video_sampler.predict_meanflow(
        prompt=args.prompt,
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        flow_shift=args.flow_shift,
        embedded_guidance_scale=args.embedded_cfg_scale,
        num_steps=args.meanflow_steps,
        i2v_condition_type=args.i2v_condition_type,
        ref_images=[(
            os.path.join(args.input_path, "rgb", "000.png"),
            os.path.join(args.input_path, "depth", "000.exr"),
        )],
        partial_cond=[(
            os.path.join(args.input_path, "rgb", f"{j:03d}.png"),
            os.path.join(args.input_path, "depth", f"{j:03d}.exr"),
        ) for j in range(args.video_length)],
        partial_mask=[(
            os.path.join(args.input_path, "mask", f"{j:03d}.png"),
            os.path.join(args.input_path, "mask", f"{j:03d}.png"),
        ) for j in range(args.video_length)],
    )

    samples = outputs["samples"]
    if "LOCAL_RANK" not in os.environ or int(os.environ["LOCAL_RANK"]) == 0:
        time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H-%M-%S")
        for i, sample_tensor in enumerate(samples):
            curr_seed = outputs["seeds"][i] if isinstance(outputs["seeds"], list) else outputs["seeds"]
            clean_prompt = str(outputs.get("prompts", "prompt"))[:100].replace("/", "").replace(" ", "_")
            cur_save_folder = f"{save_path}/{time_flag}_mf{args.meanflow_steps}_s{curr_seed}_{clean_prompt}"
            os.makedirs(cur_save_folder, exist_ok=True)
            cur_save_path = f"{cur_save_folder}/sample.mp4"
            save_videos_grid(sample_tensor.unsqueeze(0), cur_save_path, fps=24)
            with open(f"{cur_save_folder}/args.json", "w") as f:
                json.dump(vars(args), f, indent=4)
            logger.info(f"Sample saved to: {cur_save_path}")
        logger.info(f"Finished saving {len(samples)} samples to {save_path}")


if __name__ == "__main__":
    main()
