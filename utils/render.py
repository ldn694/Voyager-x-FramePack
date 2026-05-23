import numpy as np
from PIL import Image
import cv2
import trimesh
import torch
import imageio.v2 as imageio  # pip install imageio[ffmpeg]
import numba

@numba.njit(parallel=False)
def numba_render_kernel(rows, cols, depths, rgbs, width, height):
    # Initialize buffers
    rendered_image = np.zeros((height, width, 3), dtype=np.uint8)
    depth_buffer = np.full((height, width), 1e8, dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.uint8)

    for i in range(len(depths)):
        r, c = rows[i], cols[i]
        d = depths[i]
        
        # Check bounds (safety)
        if r < 0 or r >= height or c < 0 or c >= width:
            continue
            
        # Z-buffer check: only draw if point is closer than what's already there
        if 0 < d < depth_buffer[r, c]:
            depth_buffer[r, c] = d
            rendered_image[r, c, 0] = rgbs[i, 0]
            rendered_image[r, c, 1] = rgbs[i, 1]
            rendered_image[r, c, 2] = rgbs[i, 2]
            mask[r, c] = 255
            
    return rendered_image, mask, depth_buffer

class Camera:
    @staticmethod
    def _se3_inverse(w2c: np.ndarray) -> np.ndarray:
        # w2c: (4,4)
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        R_T = R.T
        t_inv = -R_T @ t
        c2w = np.eye(4, dtype=w2c.dtype)
        c2w[:3, :3] = R_T
        c2w[:3, 3] = t_inv
        return c2w
    def __init__(self, intrinsic_matrix, w2c_matrix):
        self.intrinsic_matrix = intrinsic_matrix
        self.w2c_matrix = w2c_matrix
        # self.c2w_matrix = Camera._se3_inverse(w2c_matrix)
    def project_2d_to_3d(self, points_2d, depth):
        """
        Projects 2D image points to 3D world coordinates using the camera intrinsic and extrinsic parameters.
        
        Args:
            points_2d (np.ndarray): Nx2 array of 2D image points.
            depth (np.ndarray): N array of depth values corresponding to each 2D point.
        """
        K_tensor = self.intrinsic_matrix
        w2c = self.w2c_matrix
        # c2w = self.c2w_matrix
        fx, fy = K_tensor[0, 0], K_tensor[1, 1]
        cx, cy = K_tensor[0, 2], K_tensor[1, 2]
        x = (points_2d[:, 0] - cx) * depth / fx
        y = (points_2d[:, 1] - cy) * depth / fy
        z = depth
        points_3d_camera = np.vstack((x, y, z, np.ones_like(z))).T
        # points_3d_world = (c2w @ points_3d_camera.T).T[:, :3]
        points_3d_world = (np.linalg.inv(w2c) @ points_3d_camera.T).T[:, :3]
        return points_3d_world
    def project_3d_to_2d(self, points_3d):
        """
        Projects 3D world coordinates to 2D image points using the camera intrinsic and extrinsic parameters.

        Args:
            points_3d (np.ndarray): Nx3 array of 3D world points.
        """
        K_tensor = self.intrinsic_matrix
        w2c = self.w2c_matrix
        num_points = points_3d.shape[0]
        points_3d_homogeneous = np.hstack((points_3d, np.ones((num_points, 1))))
        points_3d_camera = (w2c @ points_3d_homogeneous.T).T
        points_2d_homogeneous = (K_tensor @ points_3d_camera[:, :3].T).T
        # To avoid division by zero, we can add a small epsilon to the depth values or filter out points with very small depth before division.
        valid_depth_mask = np.abs(points_2d_homogeneous[:, 2:3]) > 1e-6
        z_safe = np.where(valid_depth_mask, points_2d_homogeneous[:, 2:3], 1e-6) 
        points_2d = points_2d_homogeneous[:, :2] / z_safe
        # points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:3]
        z = points_3d_camera[:, 2] # not actual depth, but depth in camera coordinates
        # Create a mask of valid rows (where both x and y are finite: not NaN, not Inf)
        valid_mask = np.all(np.isfinite(points_2d), axis=1)
        
        # Filter both the 2D points and the Z values using the mask
        points_2d_clean = points_2d[valid_mask]
        z_clean = z[valid_mask]
        
        return points_2d_clean, z_clean
        # if np.any(np.isnan(points_2d)) and np.any(np.isinf(points_2d)):
        #     raise ValueError("NaN and Infs in projection!")
        # elif np.any(np.isnan(points_2d)):
        #     raise ValueError("NaN in projection!")
        # elif np.any(np.isinf(points_2d)):
        #     raise ValueError("Infs in projection!")
        
        # return points_2d, z

class Frame:
    def __init__(self, rgb: Image.Image, 
                 depth: np.ndarray, 
                 camera: Camera, 
                 is_reverse_depth: bool = True, 
                 depth_calibration=None):
        assert rgb.size[::-1] == depth.shape, "RGB and depth dimensions do not match."
        assert isinstance(is_reverse_depth, bool), "is_reverse_depth should be a boolean."
        assert depth_calibration is None or (isinstance(depth_calibration, tuple) and len(depth_calibration) == 2), \
            "depth_calibration should be None or a tuple of (scale, shift)."
        self.rgb = rgb
        if is_reverse_depth:
            self.depth = 1.0 / (depth + 1e-6)
        else:
            self.depth = depth
        if depth_calibration is not None:
            scale, shift = depth_calibration
            self.depth = self.depth * scale + shift
        # print(f"Min depth: {self.depth.min()}, Max depth: {self.depth.max()}")
        self.camera = camera
    def get_point_cloud(self):
        width, height = self.rgb.size
        u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))
        points_2d = np.vstack((u_coords.flatten(), v_coords.flatten())).T
        depth_values = self.depth.flatten()
        points_3d = self.camera.project_2d_to_3d(points_2d, depth_values)
        return points_3d
    def get_rgb_point_cloud(self, color=None):
        points_3d = self.get_point_cloud()
        rgb_values = np.array(self.rgb).reshape(-1, 3)
        if color is not None:
            # blend with provided color
            blend_ratio = 0.5
            rgb_values = (rgb_values * (1 - blend_ratio) + np.array(color) * blend_ratio).astype(np.uint8)
        point_cloud = trimesh.PointCloud(points_3d, colors=rgb_values)
        return point_cloud
    def render(self, camera):
        points_3d = self.get_point_cloud()
        rgb_values = np.array(self.rgb).reshape(-1, 3)
        projected_2d, depth = camera.project_3d_to_2d(points_3d)
        width, height = self.rgb.size

        # Get valid pixel coordinates
        pixel_coords = np.round(projected_2d)
        pixel_coords = np.clip(pixel_coords, np.iinfo(np.int32).min, np.iinfo(np.int32).max).astype(np.int32)
        cols = pixel_coords[:, 0]
        rows = pixel_coords[:, 1]
        valid_pixels = (
            (cols >= 0) & (cols < width) &
            (rows >= 0) & (rows < height)
        )
        if not np.any(valid_pixels):
            rendered_image = np.zeros((height, width, 3), dtype=np.uint8)
            rendered_depth = np.full((height, width), np.inf, dtype=np.float32)
            return rendered_image, np.zeros((height, width), dtype=np.uint8), rendered_depth
        rows = rows[valid_pixels]
        cols = cols[valid_pixels]
        rgb_values = rgb_values[valid_pixels]
        depth = depth[valid_pixels]

        # Get inbound pixels
        valid_mask = (depth > 0) & (depth < 60000)
        if not np.any(valid_mask):
            rendered_image = np.zeros((height, width, 3), dtype=np.uint8)
            rendered_depth = np.full((height, width), np.inf, dtype=np.float32)
            return rendered_image, np.zeros((height, width), dtype=np.uint8), rendered_depth
        rgb_values = rgb_values[valid_mask]
        depth = depth[valid_mask]
        rows = rows[valid_mask]
        cols = cols[valid_mask]

        # Sort near to far
        sorted_indices = np.argsort(depth)
        rows = rows[sorted_indices]
        cols = cols[sorted_indices]
        depth = depth[sorted_indices]
        rgb_values = rgb_values[sorted_indices]

        # Depth buffer
        depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
        rendered_image = np.zeros((height, width, 3), dtype=np.uint8)

        flat = rows * width + cols
        uniq, first = np.unique(flat, return_index=True)
        final_rows = uniq // width
        final_cols = uniq % width

        depth_buffer[final_rows, final_cols] = depth[first]
        rendered_image[final_rows, final_cols] = rgb_values[first]

        mask = np.zeros_like(depth_buffer, dtype=np.uint8)
        mask[depth_buffer != np.inf] = 255
        return rendered_image, mask, depth_buffer
    
    def render_numba(self, camera):
        points_3d = self.get_point_cloud()
        rgb_values = np.array(self.rgb).reshape(-1, 3)
        projected_2d, depth = camera.project_3d_to_2d(points_3d)
        width, height = self.rgb.size

        # 1. Basic filtering (keep points in front of camera)
        pixel_coords = np.round(projected_2d).astype(np.int32)
        cols = pixel_coords[:, 0]
        rows = pixel_coords[:, 1]
        
        # 2. Filter valid indices to reduce work for the kernel
        valid = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height) & (depth > 0)
        
        # 3. Call the fast Numba kernel
        # No more np.argsort or np.unique!
        img, mask, d_buf = numba_render_kernel(
            rows[valid], 
            cols[valid], 
            depth[valid].astype(np.float32), 
            rgb_values[valid], 
            width, 
            height
        )
        
        # Replace 1e8 with inf for consistency with your original code
        d_buf[d_buf == 1e8] = np.inf
        
        return img, mask, d_buf

def tensor_to_mp4(tensor: torch.Tensor, out_path: str, fps: int = 25) -> None:
    """
    tensor: (3, T, H, W) from torchvision.transforms.ToTensor (values in [0,1])
    out_path: e.g. "video.mp4"
    """
    # (3, T, H, W) -> (T, H, W, 3)
    vid = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 3, 0)
    # to uint8
    vid = (vid * 255).to(torch.uint8).numpy()

    # write mp4 (readable in VS Code / any standard player)
    imageio.mimsave(out_path, vid, fps=fps, format="FFMPEG", codec="libx264")