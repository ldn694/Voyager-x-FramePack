from dataset.RealEstate10K import RealEstate10K
from utils.render import Camera, Frame, tensor_to_mp4

import os
import numpy as np
import json
import cv2
import pyexr
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.transforms import ToPILImage 
import trimesh
import torch

def parse_arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True, help='Path to RealEstate10K dataset root')
    parser.add_argument('--output_root', type=str, required=True, help='Path to output directory')
    parser.add_argument('--split', type=str, default='refined_test_150', help='Dataset split to use')
    parser.add_argument('--width', type=int, default=360, help='Width of images to load')
    parser.add_argument('--height', type=int, default=384, help='Height of images to load')
    args = parser.parse_args()
    return args


# for i, (render, mask, depth) in enumerate(zip(render_list, mask_list, depth_list)):

#         # Sky part is the region where depth_max is, also included in mask
#         mask = mask > 0
#         # depth_max = np.max(depth)
#         # non_sky_mask = (depth != depth_max)
#         # mask = mask & non_sky_mask
#         depth[mask] = 1 / (depth[mask] + 1e-6)
#         depth_values = depth[mask]
        
#         min_percentile = np.percentile(depth_values, 2)
#         max_percentile = np.percentile(depth_values, 98)
#         value_list.append((min_percentile, max_percentile))

#         depth[mask] = (depth[mask] - min_percentile) / (max_percentile - min_percentile)
#         depth[~mask] = depth[mask].min()
        

#         # resize to 512x512
#         render = cv2.resize(render, (Width, Height), interpolation=cv2.INTER_LINEAR)
#         mask = cv2.resize((mask.astype(np.float32) * 255).astype(np.uint8), \
#             (Width, Height), interpolation=cv2.INTER_NEAREST)
#         depth = cv2.resize(depth, (Width, Height), interpolation=cv2.INTER_LINEAR)

#         # Save mask as png
#         mask_path = os.path.join(video_input_dir, f"mask_{i:04d}.png")
#         imageio.imwrite(mask_path, mask)

def norm_partial_render_output(render, mask, depth):
    mask_indices = mask > 0
    depth[mask_indices] = 1 / (depth[mask_indices] + 1e-6)
    depth_values = depth[mask_indices]
    min_percentile = np.percentile(depth_values, 2)
    max_percentile = np.percentile(depth_values, 98)
    print(f"Depth percentiles: min {min_percentile}, max {max_percentile}")
    depth[mask_indices] = (depth[mask_indices] - min_percentile) / (max_percentile - min_percentile)
    depth[~mask_indices] = depth[mask_indices].min()
    return render, mask, depth



if __name__ == "__main__":
    args = parse_arg()
    dataset_root = args.dataset_root
    output_root = args.output_root
    os.makedirs(output_root, exist_ok=True)
    dataset = RealEstate10K(dataset_root, set_name=args.split, width=args.width, height=args.height, return_inverse_depth=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16)

    for batch_idx, data in tqdm(enumerate(dataloader), desc="Iterating over dataset"):
        rgbs = data['rgb'][0]  # (3, T, H, W)
        inverse_depths = data['depth'][0]  # (1, T, H, W)
        intrinsics = data['intrinsic'][0]  # (T, 3, 3)
        w2cs = data['w2c'][0]  # (T, 4, 4)
        sample_id = data['sample_id'][0] # (str)

        test_folder = os.path.join(output_root, sample_id)
        os.makedirs(test_folder, exist_ok=True)

        _, T, H, W = rgbs.shape

        # Create ground truth
        ground_truth_folder = os.path.join(test_folder, 'ground_truth')
        os.makedirs(ground_truth_folder, exist_ok=True)
        ground_truth_rgb_folder = os.path.join(ground_truth_folder, 'rgb')
        os.makedirs(ground_truth_rgb_folder, exist_ok=True)
        for i in range(T):
            rgb = ToPILImage()(rgbs[:, i])
            rgb_path = os.path.join(ground_truth_rgb_folder, f'{i:03d}.png')
            if not os.path.exists(rgb_path):
                rgb.save(rgb_path)
        gt_video_path = os.path.join(ground_truth_folder, 'video.mp4')
        print(f'Saving ground truth video to {gt_video_path}')
        tensor_to_mp4(rgbs, gt_video_path, fps=10)

        # Create partial RGB and depth as input
        input_folder = os.path.join(test_folder, 'input')
        os.makedirs(input_folder, exist_ok=True)
        input_rgb_folder = os.path.join(input_folder, 'rgb')
        os.makedirs(input_rgb_folder, exist_ok=True)
        input_depth_folder = os.path.join(input_folder, 'depth')
        os.makedirs(input_depth_folder, exist_ok=True)
        input_mask_folder = os.path.join(input_folder, 'mask')
        os.makedirs(input_mask_folder, exist_ok=True)
        
        first_frame = Frame(
            rgb=ToPILImage()(rgbs[:, 0]),
            depth=inverse_depths[0, 0].numpy(),
            camera=Camera(intrinsics[0].numpy(), w2cs[0].numpy()),
            is_reverse_depth=True,
        )

        total_rendered_image_tensor = []

        for i in range(T):
            rgb = ToPILImage()(rgbs[:, i])
            depth = inverse_depths[0, i].numpy()
            frame = Frame(
                rgb=rgb,
                depth=depth,
                camera=Camera(intrinsics[i].numpy(), w2cs[i].numpy()),
                is_reverse_depth=True,
            )
            rendered_image, mask, depth_buffer = first_frame.render(frame.camera)
            rendered_image, mask, depth_buffer = norm_partial_render_output(
                rendered_image, mask, depth_buffer
            )
            print(f"Min depth buffer: {depth_buffer.min()}, Max depth buffer: {depth_buffer.max()}")
            # Save RGB
            rgb_path = os.path.join(input_rgb_folder, f'{i:03d}.png')
            if not os.path.exists(rgb_path):
                cv2.imwrite(rgb_path, cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR))
            # Save depth
            depth_path = os.path.join(input_depth_folder, f'{i:03d}.exr')
            if not os.path.exists(depth_path):
                pyexr.write(depth_path, depth_buffer.astype(np.float32))
            # Save mask
            mask_path = os.path.join(input_mask_folder, f'{i:03d}.png')
            if not os.path.exists(mask_path):
                cv2.imwrite(mask_path, mask)
            print(f"Min rgb: {rendered_image.min()}, Max rgb: {rendered_image.max()}")
            total_rendered_image_tensor.append((torch.from_numpy(rendered_image).float().permute(2,0,1) / 255.0).contiguous())  # (3, H, W)
        total_rendered_image_tensor = torch.stack(total_rendered_image_tensor, dim=1)  # (3, T, H, W)
        input_video_path = os.path.join(input_folder, 'video.mp4')
        print(f'Saving input video to {input_video_path}')
        tensor_to_mp4(total_rendered_image_tensor, input_video_path, fps=10)
