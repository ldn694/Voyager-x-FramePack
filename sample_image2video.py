import os
import time
from pathlib import Path
from loguru import logger
from datetime import datetime
import torch
import json
import sys
from tqdm import tqdm
from voyager.utils.file_utils import save_videos_grid
from voyager.config import parse_args
from voyager.inference import HunyuanVideoSampler
from spectralAnalyser import spatialFreq


def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    args = parse_args()
    if args.use_multiple_kernels and (args.multiple_kernels_path is not None):
        multiple_kernels_ckpt = torch.load(args.multiple_kernels_path, map_location="cpu", weights_only=False)
        multiple_kernels_args = multiple_kernels_ckpt["args"]
        args.use_kernel_sizes = multiple_kernels_args["kernel_sizes"]
    print(args)
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix == "" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(
        models_root_path, args=args)

    # Get the updated args
    args = hunyuan_video_sampler.args
    # Start sampling
    # TODO: batch inference check
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
        # ref_images=[(
        #     os.path.join(args.input_path, "ref_image.png"),
        #     os.path.join(args.input_path, "ref_depth.exr")
        # )],
        ref_images=[(
            os.path.join(args.input_path, "rgb", "000.png"),
            os.path.join(args.input_path, "depth", "000.exr")
        )],
        # ref_images=[(
        #     os.path.join(args.input_path, "final", "rendered_views", "render_0000.png"),
        #     os.path.join(args.input_path, "final", "output_inverse_depth_dir", "gt_0000_inverse.exr")
        # )],
        # partial_cond=[(
        #     os.path.join(args.input_path, "video_input", f"render_{j:04d}.png"),
        #     os.path.join(args.input_path, "video_input", f"depth_{j:04d}.exr")
        # ) for j in range(49)],
        partial_cond=[(
            os.path.join(args.input_path, "rgb", f"{j:03d}.png"),
            os.path.join(args.input_path, "depth", f"{j:03d}.exr")
        ) for j in range(args.video_length)],
        # partial_cond=[(
        #     os.path.join(args.input_path, "final", "rendered_views", f"render_{j:04d}.png"),
        #     os.path.join(args.input_path, "final", "output_inverse_depth_dir", f"gt_{j:04d}_inverse.exr")
        # ) for j in range(49)],
        # partial_mask=[(
        #     os.path.join(args.input_path, "video_input", f"mask_{j:04d}.png"),
        #     os.path.join(args.input_path, "video_input", f"mask_{j:04d}.png")
        # ) for j in range(49)],
        partial_mask=[(
            os.path.join(args.input_path, "mask", f"{j:03d}.png"),
            os.path.join(args.input_path, "mask", f"{j:03d}.png")
        ) for j in range(args.video_length)],
        # partial_mask=[(
        #     os.path.join(args.input_path, "final", "rendered_views", f"mask_{j:04d}.png"),
        #     os.path.join(args.input_path, "final", "rendered_views", f"mask_{j:04d}.png")
        # ) for j in range(49)],
        use_kernel_indices = args.use_kernel_indices if args.use_kernel_indices is not None else None,
        step_sample = args.step_sample,
        attn_map = args.attn_map,
        dmd2_steps = getattr(args, "dmd2_steps", 0),
    )
    samples = outputs['samples']

    if args.step_sample > 0:
        sample_list = outputs.get('sample_list', [])

    # Save generated videos to disk
    # Only save on the main process in distributed settings
    # if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
    #     for i, sample in enumerate(samples):
    #         sample = samples[i].unsqueeze(0)
    #         time_flag = datetime.fromtimestamp(
    #             time.time()).strftime("%Y-%m-%d-%H:%M:%S")
    #         # cur_save_path = \
    #         #     f"{save_path}/{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/', '')}.mp4"
    #         # save_videos_grid(sample, cur_save_path, fps=24)
    #         # logger.info(f'Sample save to: {cur_save_path}')
    #         cur_save_folder = \
    #             f"{save_path}/{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/', '')}"
    #         os.makedirs(cur_save_folder, exist_ok=True)
            
    #         cur_save_path = f"{cur_save_folder}/sample.mp4"
    #         save_videos_grid(sample, cur_save_path, fps=24)
    #         logger.info(f'Sample save to: {cur_save_path}')

    #         with open(f"{cur_save_folder}/args.json", "w") as f:
    #             vars_args = vars(args)
    #             json.dump(vars_args, f, indent=4)
    #         logger.info(f'Args save to: {cur_save_folder}/args.json')


    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
    # 1. Generate timestamp ONCE for the whole batch
    # Changed ":" to "-" for better filename compatibility
        time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H-%M-%S")
        
        # 2. Add a progress bar for the batch of videos
        # Initialize the new spatialFreq analyzer
        spectral_analyzer = spatialFreq(figsize=(10, 6))
                
        for i, sample_tensor in enumerate(tqdm(samples, desc="Saving final videos")):
            
            # Pull metadata safely
            curr_seed = outputs['seeds'][i] if isinstance(outputs['seeds'], list) else outputs['seeds']
            curr_prompt = outputs.get('prompts', ["prompt"])[i] if isinstance(outputs.get('prompts'), list) else outputs.get('prompts', "prompt")
            clean_prompt = str(curr_prompt)[:100].replace('/', '').replace(' ', '_')

            # Create folder hierarchy
            cur_save_folder = f"{save_path}/{time_flag}_s{curr_seed}_{clean_prompt}"
            os.makedirs(cur_save_folder, exist_ok=True)
            
            # Save final video (adding unsqueeze to make it a 4D/5D tensor for the grid saver)
            video_to_save = sample_tensor.unsqueeze(0) 
            cur_save_path = f"{cur_save_folder}/sample.mp4"
            save_videos_grid(video_to_save, cur_save_path, fps=24)
            
            # Save intermediate steps if requested
            if args.step_sample > 0 and 'sample_list' in outputs:
                steps_folder = f"{cur_save_folder}/steps"
                os.makedirs(steps_folder, exist_ok=True)
                
                # Nested progress bar for steps if you want to see progress within the sample
                for step_num, step_batch_tensor in tqdm(outputs['sample_list'], desc=f"Saving steps for sample {i}", leave=False):
                    # Create a subfolder for this specific step
                    step_subfolder = f"{steps_folder}/step_{step_num:03d}"
                    os.makedirs(step_subfolder, exist_ok=True)
                    
                    # Paths for step video and its TWO spectral analysis outputs
                    step_video_path = f"{step_subfolder}/video.mp4"
                    step_spectral_rgb_path = f"{step_subfolder}/spectral_rgb.mp4"
                    step_spectral_depth_path = f"{step_subfolder}/spectral_depth.mp4"
                    
                    # Save the step video
                    step_video = step_batch_tensor[i].unsqueeze(0)
                    save_videos_grid(step_video, step_video_path, fps=24)
                    
                    # 5. RUN SPECTRAL ANALYSIS ON STEP VIDEO
                    # Using the updated class with dual-output paths
                    spectral_analyzer.process_and_save(
                        step_video_path, 
                        step_spectral_rgb_path,
                        step_spectral_depth_path,
                        fps=10
                    )

            # Save args for reproducibility
            with open(f"{cur_save_folder}/args.json", "w") as f:
                json.dump(vars(args), f, indent=4)
                
        logger.info(f"Finished saving {len(samples)} samples to {save_path}")


if __name__ == "__main__":
    main()
