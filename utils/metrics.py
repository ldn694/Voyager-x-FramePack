import lpips
import torch
from skimage.metrics import structural_similarity as ssim


class LPIPSMetric:
    def __init__(self, device='cuda'):
        self.loss_fn = lpips.LPIPS(net='alex').to(device).eval()
        self.device = device

    def compute(self, img1, img2):
        """
        img1, img2: (B, C, H, W), either in [-1, 1] or [0, 1]
        returns: (B,)
        """
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)

        # If inputs are [0,1], rescale to [-1,1]
        if img1.min() >= 0 and img1.max() <= 1:
            img1 = img1 * 2 - 1
        if img2.min() >= 0 and img2.max() <= 1:
            img2 = img2 * 2 - 1

        with torch.no_grad():
            distances = self.loss_fn(img1, img2)  # (B, 1, 1, 1)
        return distances.view(distances.size(0))  # (B,)


class PSNRMetric:
    @staticmethod
    def compute(img1, img2):
        """
        img1, img2: (B, C, H, W) in [0, 1]
        returns: (B,)
        """
        mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-10))
        return psnr


def compute_ssim_batch(img1, img2):
    """
    img1, img2: (B, C, H, W) in [0,1]
    returns: torch.Tensor (B,)
    """
    B = img1.size(0)
    scores = []

    for i in range(B):
        x = img1[i].permute(1, 2, 0).detach().cpu().numpy()
        y = img2[i].permute(1, 2, 0).detach().cpu().numpy()

        score = ssim(x, y, channel_axis=2, data_range=1.0)
        scores.append(score)

    return torch.tensor(scores, dtype=img1.dtype, device=img1.device)


class MergedMetric:
    def __init__(self, device='cuda'):
        self.device = device
        self.lpips_metric = LPIPSMetric(device)

    def compute(self, img1, img2):
        # img1, img2: (B, C, H, W) in [0, 1]
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        print(f'Computing metrics on images of shape {img1.shape}')

        with torch.no_grad():
            lpips_values = self.lpips_metric.compute(img1, img2)      # (B,)
            psnr_values = PSNRMetric.compute(img1, img2)              # (B,)
            ssim_values = compute_ssim_batch(img1, img2)              # (B,)

        return {
            'LPIPS': lpips_values.cpu().tolist(),
            'PSNR': psnr_values.cpu().tolist(),
            'SSIM': ssim_values.cpu().tolist(),
        }
