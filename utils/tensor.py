import torch
import torch.distributed as dist
from loguru import logger

def pad_for_3d_conv(x, kernel_size):
    b, c, t, h, w = x.shape
    pt, ph, pw = kernel_size
    pad_t = (pt - (t % pt)) % pt
    pad_h = (ph - (h % ph)) % ph
    pad_w = (pw - (w % pw)) % pw
    return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode='replicate')

def center_down_sample_3d(x, kernel_size):
    return torch.nn.functional.avg_pool3d(x, kernel_size, stride=kernel_size)

def check_nan_inf(tensors_to_check):
    for name, tensor in tensors_to_check.items():
        assert torch.isfinite(tensor).all(), f"Found NaN or Inf in {name}!"

def is_tensor_valid(tensors_to_check) -> bool:
    """Returns True if all tensors are finite, False otherwise."""
    for name, tensor in tensors_to_check.items():
        if not torch.isfinite(tensor).all():
            logger.warning(f"Found NaN/Inf in {name}!")
            return False
    return True