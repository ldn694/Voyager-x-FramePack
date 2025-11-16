import torch
from torch.utils.data import Dataset
import os
import glob
import json
from PIL import Image
import random
import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import av
import numpy as np
import torch.nn.functional as F

class RealEstate10K(Dataset):
    def __init__(self, 
                 root_path,
                 set_name, 
                 frame_per_sample=49, 
                 frames_intervals=[1],
                 transform=None,
                 width=None,
                 height=None,
                 return_inverse_depth=False
                 ):
        '''
        Args:
            root_path (str): Root path of the RealEstate10K dataset.
            set_name (str): Name of the dataset split, e.g., 'train', 'test'.
            frame_per_sample (int): Number of frames per sample.
            frames_intervals (list): List of frame intervals to sample. E.g. 1 is every frame, 4 is every 4th frame.
            transform (callable, optional): Optional transform to be applied on a sample.
        '''
        self.root_path = root_path
        self.set_name = set_name
        self.frame_per_sample = frame_per_sample
        self.frames_intervals = frames_intervals
        self.transform = transform
        self.return_inverse_depth = return_inverse_depth

        self.frame_path = os.path.join(self.root_path, 'transcode')
        self.depth_path = os.path.join(self.root_path, 'depth_videos')
        self.depth_ranges_path = os.path.join(self.root_path, 'depth_ranges')
        self.cameras_path = os.path.join(self.root_path, 'cameras')
        self.folder_path = os.path.join(self.root_path, self.set_name)
        self.width = width
        self.height = height

        self.media_files = glob.glob(os.path.join(self.folder_path, '*.txt'))
        self.data = []

        for media_file in tqdm.tqdm(self.media_files, desc="Loading media files"):
            # Read media file to get video path and timestamps
            timestamps = []
            with open(media_file) as f:
                videoPathURL = f.readline().rstrip()
                video_id = videoPathURL.split('=')[-1]
                for l in f.readlines():
                    line = l.split(' ')
                    timestamp = int(line[0])
                    timestamps.append(timestamp)
            
            if len(timestamps) < self.frame_per_sample:
                continue
            
            # Load camera data
            camera_file = os.path.join(self.cameras_path, os.path.basename(media_file).replace('.txt', '.json'))
            with open(camera_file) as f:
                camera_data = json.load(f)
            assert len(timestamps) == len(camera_data)
            for i in range(len(timestamps)):
                assert timestamps[i] == int(camera_data[i]['file_name']), f"Timestamps do not match, {timestamps[i]} vs {camera_data[i]['file_name']}"

            # Load depth video file
            depth_video_file = os.path.join(self.depth_path, os.path.basename(media_file).replace('.txt', '.avi'))
            assert os.path.exists(depth_video_file), f"Depth video file not found: {depth_video_file}"
            container = av.open(depth_video_file)
            video_stream = container.streams.video[0]
            num_frames = video_stream.frames
            assert num_frames == len(timestamps), f"Number of frames in depth video does not match timestamps for {media_file}"
            container.close()

            # Load depth ranges file
            depth_range_file = os.path.join(self.depth_ranges_path, os.path.basename(media_file).replace('.txt', '.json'))
            with open(depth_range_file) as f:
                depth_ranges = json.load(f)
            assert len(depth_ranges) == len(timestamps), f"Number of depth ranges does not match timestamps for {media_file}"
            for i in range(len(timestamps)):
                assert timestamps[i] == int(depth_ranges[i]['file_name']), f"Timestamps do not match in depth ranges, {timestamps[i]} vs {depth_ranges[i]['file_name']}"

            # Load RGB
            rgb_folder = os.path.join(self.frame_path, video_id)
            if not os.path.exists(rgb_folder):
                continue
            for i in range(len(timestamps)):
                rgb_image_path = os.path.join(rgb_folder, f"{timestamps[i]}.jpg")
                assert os.path.exists(rgb_image_path), f"RGB image not found: {rgb_image_path}"

            frame_infos = [{
                'timestamp': timestamps[i],
                'intrinsic': camera_data[i]['intrinsic'],
                'w2c': camera_data[i]['w2c'],
                'depth_range': (depth_ranges[i]['min_inverse_depth'], depth_ranges[i]['max_inverse_depth']),
                'frame_index': i,
            } for i in range(len(timestamps))]

            self.data.append({
                'sample_id': os.path.basename(media_file).split('.')[0],
                'video_id': video_id,
                'frame_infos': frame_infos,
                'rgb_folder': rgb_folder,
                'depth_video_file': depth_video_file,
            })
        print(f"Loaded {len(self.data)} samples from {self.set_name} set.")
    
    def __len__(self):
        return len(self.data)

    def _compute_crop_and_scale(self, H0, W0):
        """
        Given original size H0, W0 and target self.height, self.width,
        compute center crop (x0, y0, Wc, Hc) and scaling (sx, sy).
        """
        if self.width is None or self.height is None:
            # No geometry change
            return 0, 0, W0, H0, 1.0, 1.0

        W_t, H_t = self.width, self.height
        target_ratio = W_t / H_t
        orig_ratio = W0 / H0

        # Given your guarantee, this assert should always hold
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

    def __getitem__(self, index):
        frame_infos = self.data[index]['frame_infos']
        rgb_folder = self.data[index]['rgb_folder']
        depth_video_file = self.data[index]['depth_video_file']

        viable = sorted([iv for iv in self.frames_intervals
                        if len(frame_infos) >= 1 + (self.frame_per_sample - 1) * iv])
        if not viable:
            raise ValueError("Not enough frames for any requested interval.")
        frame_interval = random.choice(viable)

        max_start = len(frame_infos) - 1 - (self.frame_per_sample - 1) * frame_interval
        # start_index = random.randint(0, max_start)
        start_index = 0

        # indices actually used
        idxs = [start_index + i * frame_interval for i in range(self.frame_per_sample)]
        selected_infos = [frame_infos[i] for i in idxs]
        idxs = set(idxs)

        # read first image for width and height
        first_image_path = os.path.join(rgb_folder, f"{selected_infos[0]['timestamp']}.jpg")
        with Image.open(first_image_path) as img0:
            img0 = img0.convert("RGB")
            W0, H0 = img0.size
        x0, y0, Wc, Hc, sx, sy = self._compute_crop_and_scale(H0, W0)

        images = []
        intrinsics = []
        w2c = []
        for info in selected_infos:
            image_path = os.path.join(rgb_folder, f"{info['timestamp']}.jpg")
            if not os.path.exists(image_path):
                # policy: bail out (or you could resample or pad)
                raise FileNotFoundError(f"Missing frame: {image_path}")
            img = Image.open(image_path).convert("RGB")

            # 1) crop in original resolution
            # PIL crop box: (left, upper, right, lower)
            img = img.crop((x0, y0, x0 + Wc, y0 + Hc))
            # 2) resize to target size
            if self.width is not None and self.height is not None:
                img = img.resize((self.width, self.height), resample=Image.BILINEAR)
            # 3) apply non-geometric transforms (e.g. ToTensor, Normalize)
            if self.transform:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)
            images.append(img)

            # --- Adjust intrinsics for this frame ---
            K = torch.tensor(info['intrinsic'], dtype=torch.float32)  # (3, 3)
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]

            # after crop
            cx1 = cx - x0
            cy1 = cy - y0

            # after resize
            fx2 = fx * sx
            fy2 = fy * sy
            cx2 = cx1 * sx
            cy2 = cy1 * sy

            K[0, 0] = fx2
            K[1, 1] = fy2
            K[0, 2] = cx2
            K[1, 2] = cy2

            intrinsics.append(K)
            w2c.append(torch.tensor(info['w2c'], dtype=torch.float32))
        
        depths = []

        # Collect raw depth frames first (uint16), then process in batch
        raw_depths = []     # list of np.ndarray (H, W), dtype=uint16
        lo_list = []        # per-frame lo
        hi_list = []        # per-frame hi

        max_idx = max(idxs)
        with av.open(depth_video_file) as container:
            video_stream = container.streams.video[0]
            for frame_id, frame in enumerate(container.decode(video_stream)):
                if frame_id > max_idx:
                    break
                if frame_id not in idxs:
                    continue

                # Just decode to gray16le, no float/resize here
                depth_u16 = frame.to_ndarray(format="gray16le")  # uint16
                raw_depths.append(depth_u16)

                lo, hi = frame_infos[frame_id]['depth_range']
                lo_list.append(lo)
                hi_list.append(hi)

        # Now batch-process everything outside the decoding loop
        # raw_depths: list of (H, W), dtype uint16
        depth_np = np.stack(raw_depths, axis=0).astype(np.float32)  # (T, H, W)

        lo_arr = np.array(lo_list, dtype=np.float32).reshape(-1, 1, 1)  # (T, 1, 1)
        hi_arr = np.array(hi_list, dtype=np.float32).reshape(-1, 1, 1)  # (T, 1, 1)

        # Batch linear transform: inverse_depth = raw/65535 * (hi - lo) + lo
        depth_np = depth_np / 65535.0 * (hi_arr - lo_arr) + lo_arr      # (T, H, W)
        if not self.return_inverse_depth:
            depth_np = 1.0 / (depth_np + 1e-6)                              # (T, H, W)

        depth_tensor = torch.from_numpy(depth_np)  # (T, H0d, W0d)
        T_frames, H0d, W0d = depth_tensor.shape

        assert H0d == H0 and W0d == W0, \
            f"Depth size ({H0d}, {W0d}) != RGB size ({H0}, {W0})."
        
         # 1) crop with the same (x0, y0, Wc, Hc)
        depth_tensor = depth_tensor[..., y0:y0+Hc, x0:x0+Wc]  # (T, Hc, Wc)

        # 2) resize to target size
        if self.width is not None and self.height is not None:
            depth_tensor = depth_tensor.unsqueeze(1)  # (T, 1, Hc, Wc)
            depth_tensor = F.interpolate(
                depth_tensor,
                size=(self.height, self.width),
                mode='bilinear',
                align_corners=False
            )
            depth_tensor = depth_tensor.squeeze(1)  # (T, H_t, W_t)

        # images: list length T, each (3, H, W)
        images = torch.stack(images, dim=0)      # (T, 3, H, W)
        images = images.permute(1, 0, 2, 3)      # (3, T, H, W)

        # depth_tensor: (T, H, W)
        depth_tensor = depth_tensor.unsqueeze(0)   # (1, T, H, W)

        intrinsics = torch.stack(intrinsics, dim=0)  # (T, 3, 3)
        w2c = torch.stack(w2c, dim=0)                # (T, 4, 4)
        
        return {
            'sample_id': self.data[index]['sample_id'],
            'rgb': images,          # (3, T, H, W)
            'depth': depth_tensor,  # (1, T, H, W)
            'intrinsic': intrinsics,  # (T, 3, 3)
            'w2c': w2c,               # (T, 4, 4)
        }
    def __repr__(self):
        return f"RealEstate10K(dataset='{self.set_name}', num_samples={len(self)})"

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # dataset = RealEstate10K(root_path='.', set_name='refined_train_10pct', transform=transform)
    dataset = RealEstate10K(root_path='.', set_name='refined_test_150', transform=transform, width=1280, height=720)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=16, pin_memory=True)

    for batch in tqdm.tqdm(data_loader, desc="Iterating over data loader"):
        pass
    #    print(batch['sample_id'], len(batch['frames']), batch['intrinsics'][0], batch['pose'][0])
