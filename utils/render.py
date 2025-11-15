import numpy as np
from PIL import Image
import cv2
import trimesh

class Camera:
    def __init__(self, intrinsic_matrix, w2c_matrix):
        self.intrinsic_matrix = intrinsic_matrix
        self.w2c_matrix = w2c_matrix
    def project_2d_to_3d(self, points_2d, depth):
        """
        Projects 2D image points to 3D world coordinates using the camera intrinsic and extrinsic parameters.
        
        Args:
            points_2d (np.ndarray): Nx2 array of 2D image points.
            depth (np.ndarray): N array of depth values corresponding to each 2D point.
        """
        K_tensor = self.intrinsic_matrix
        w2c = self.w2c_matrix
        fx, fy = K_tensor[0, 0], K_tensor[1, 1]
        cx, cy = K_tensor[0, 2], K_tensor[1, 2]
        x = (points_2d[:, 0] - cx) * depth / fx
        y = (points_2d[:, 1] - cy) * depth / fy
        z = depth
        points_3d_camera = np.vstack((x, y, z, np.ones_like(z))).T
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
        points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:3]
        z = points_3d_camera[:, 2] # not actual depth, but depth in camera coordinates
        return points_2d, z

class Frame:
    def __init__(self, rgb_path, depth_path=None, depth_range=None, is_reverse_depth=True, camera=None, depth_calibration=None):
        self.rgb = Image.open(rgb_path).convert("RGB")
        if is_reverse_depth:
            inverse_depth_normalized = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            lo, hi = depth_range
            inverse_depth = inverse_depth_normalized.astype(np.float32) / 65535.0 * (hi - lo) + lo
            self.depth = 1.0 / (inverse_depth + 1e-6)
        else:
            self.depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            lo, hi = depth_range
            self.depth = self.depth.astype(np.float32) / 65535.0 * (hi - lo) + lo
        if depth_calibration is not None:
            scale, shift = depth_calibration
            self.depth = self.depth * scale + shift
        print(f"Min depth: {self.depth.min()}, Max depth: {self.depth.max()}")
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
        pixel_coords = np.round(projected_2d).astype(np.int32)
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