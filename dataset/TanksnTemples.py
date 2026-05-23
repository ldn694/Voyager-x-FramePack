import os
import numpy as np
import json
import cv2
import pyexr
import av
import random
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import ToPILImage 
from loguru import logger
import time
class TanksAndTemplesDataset(Dataset):
    def __init__(self, dataset_root, split='test_50', width=768, height=512, transform=None):
        self.dataset_root = dataset_root
        self.width = width
        self.height = height
        
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()
        
        # Paths based on your depth_calc_tanks.py & collect_rgb.py outputs
        self.test_dir = os.path.join(dataset_root, split)
        self.image_dir = os.path.join(dataset_root, 'transcode', 'images')
        self.depth_dir = os.path.join(dataset_root, 'depth_videos')
        self.range_dir = os.path.join(dataset_root, 'depth_ranges')
        self.camera_dir = os.path.join(dataset_root, 'cameras')
        
        self.samples = [f for f in sorted(os.listdir(self.test_dir)) if f.endswith('.json')]
        
    def __len__(self):
        return len(self.samples)

    def _compute_crop_and_scale(self, H0, W0):
        """
        Given original size H0, W0 and target self.height, self.width,
        compute center crop (x0, y0, Wc, Hc) and scaling (sx, sy).
        """
        if self.width is None or self.height is None:
            return 0, 0, W0, H0, 1.0, 1.0

        W_t, H_t = self.width, self.height
        target_ratio = W_t / H_t
        orig_ratio = W0 / H0

        assert target_ratio <= orig_ratio + 1e-6, \
            f"Target ratio {target_ratio} > original ratio {orig_ratio}. This would require padding."

        # Full height crop, horizontal center crop
        Hc = H0
        Wc = int(round(target_ratio * Hc))

        x0 = (W0 - Wc) // 2
        y0 = 0

        sx = W_t / Wc
        sy = H_t / Hc

        return x0, y0, Wc, Hc, sx, sy

    def __getitem__(self, idx):
        sample_file = self.samples[idx]
        sample_id = sample_file.replace('.json', '')
        
        with open(os.path.join(self.test_dir, sample_file), 'r') as f:
            test_info = json.load(f)
            
        video_name = test_info['video_name']
        frame_ids = test_info['frame_ids']
        T = len(frame_ids)
        
        # Determine crop & scale based on the first image
        first_img_path = os.path.join(self.image_dir, video_name, f"{frame_ids[0]}.png")
        with Image.open(first_img_path) as img0:
            img0 = img0.convert("RGB")
            W0, H0 = img0.size
        
        x0, y0, Wc, Hc, sx, sy = self._compute_crop_and_scale(H0, W0)
        
        # 1. Load RGBs
        images = []
        for fid in frame_ids:
            img_path = os.path.join(self.image_dir, video_name, f"{fid}.png")
            img = Image.open(img_path).convert("RGB")
            
            img = img.crop((x0, y0, x0 + Wc, y0 + Hc))
            if self.width is not None and self.height is not None:
                img = img.resize((self.width, self.height), resample=Image.BILINEAR)
                
            img = self.transform(img)
            images.append(img)
        images = torch.stack(images, dim=0) # (T, 3, H, W)
        
        # 2. Load Cameras
        cam_path = os.path.join(self.camera_dir, f"{sample_id}.json")
        with open(cam_path, 'r') as f:
            cams = json.load(f)
            
        intrinsics = []
        w2cs = []
        
        for i in range(T):
            cam = cams[i]
            # Standard metric3d/moge output uses 'intrinsic' and 'extrinsic' 
            # Fallbacks provided just in case
            K_np = np.array(cam.get('intrinsic', cam.get('K', np.eye(3))), dtype=np.float32)
            w2c_np = np.array(cam.get('extrinsic', cam.get('w2c', np.eye(4))), dtype=np.float32)
            
            K = torch.tensor(K_np, dtype=torch.float32)
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]

            # Adjust intrinsics for crop
            cx1 = cx - x0
            cy1 = cy - y0

            # Adjust intrinsics for resize
            fx2 = fx * sx
            fy2 = fy * sy
            cx2 = cx1 * sx
            cy2 = cy1 * sy

            K[0, 0] = fx2
            K[1, 1] = fy2
            K[0, 2] = cx2
            K[1, 2] = cy2
            
            intrinsics.append(K)
            w2cs.append(torch.tensor(w2c_np, dtype=torch.float32))
            
        intrinsics = torch.stack(intrinsics, dim=0)
        w2cs = torch.stack(w2cs, dim=0)

        # 3. Load Depth Ranges
        range_path = os.path.join(self.range_dir, f"{sample_id}.json")
        with open(range_path, 'r') as f:
            depth_ranges = json.load(f)
            
        # Format might be list of lists [min, max] or list of dicts {'min_inverse_depth': ..., 'max_inverse_depth': ...}
        # Handle both formats based on typical pipeline outputs
        lo_list = []
        hi_list = []
        for r in depth_ranges:
            if isinstance(r, dict):
                lo_list.append(r.get('min_inverse_depth', r.get('min', 0.0)))
                hi_list.append(r.get('max_inverse_depth', r.get('max', 1.0)))
            else:
                lo_list.append(r[0])
                hi_list.append(r[1])
                
        # 4. Load 16-bit Depth Video via PyAV
        depth_video_path = os.path.join(self.depth_dir, f"{sample_id}.avi")
        raw_depths = []
        
        with av.open(depth_video_path) as container:
            video_stream = container.streams.video[0]
            for frame_id, frame in enumerate(container.decode(video_stream)):
                if frame_id >= T:
                    break
                depth_u16 = frame.to_ndarray(format="gray16le")  # uint16
                raw_depths.append(depth_u16)
                
        # Batch linear transform: inverse_depth = raw/65535 * (hi - lo) + lo
        depth_np = np.stack(raw_depths, axis=0).astype(np.float32)  # (T, H, W)
        lo_arr = np.array(lo_list, dtype=np.float32).reshape(-1, 1, 1)
        hi_arr = np.array(hi_list, dtype=np.float32).reshape(-1, 1, 1)
        
        depth_np = depth_np / 65535.0 * (hi_arr - lo_arr) + lo_arr  # (T, H, W)
        depth_tensor = torch.from_numpy(depth_np)  # (T, H0, W0)
        
        # Crop Depth
        depth_tensor = depth_tensor[..., y0:y0+Hc, x0:x0+Wc]  # (T, Hc, Wc)

        # Resize Depth
        if self.width is not None and self.height is not None:
            depth_tensor = depth_tensor.unsqueeze(1)  # (T, 1, Hc, Wc)
            depth_tensor = F.interpolate(
                depth_tensor,
                size=(self.height, self.width),
                mode='bilinear',
                align_corners=False
            )
            depth_tensor = depth_tensor.squeeze(1)  # (T, H_t, W_t)
            
        depth_tensor = depth_tensor.unsqueeze(1)  # (T, 1, H_t, W_t)
        
        # Format identical to RealEstate10K layout 
        return {
            'rgb': images.permute(1, 0, 2, 3),        # (3, T, H, W)
            'depth': depth_tensor.permute(1, 0, 2, 3),# (1, T, H, W)
            'intrinsic': intrinsics,                  # (T, 3, 3)
            'w2c': w2cs,                              # (T, 4, 4)
            'sample_id': sample_id
        }