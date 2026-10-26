"""Metrics used by the final DMD-Net evaluation scripts."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> float:
    mse = F.mse_loss(x, y, reduction="none").flatten(1).mean(1)
    mse = mse.clamp(min=1e-12)
    return (10.0 * torch.log10(data_range ** 2 / mse)).mean().item()


def _gaussian_window(size: int, sigma: float, device) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.outer(g)


def ssim(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0,
         window_size: int = 11) -> float:
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    window = _gaussian_window(window_size, 1.5, x.device)
    window = window.view(1, 1, window_size, window_size).expand(
        x.size(1), 1, -1, -1
    )

    def mean_filter(tensor):
        return F.conv2d(
            tensor, window, padding=window_size // 2, groups=tensor.size(1)
        )

    mu_x, mu_y = mean_filter(x), mean_filter(y)
    sigma_x = mean_filter(x * x) - mu_x ** 2
    sigma_y = mean_filter(y * y) - mu_y ** 2
    sigma_xy = mean_filter(x * y) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    )
    return score.mean().item()


class LPIPSMetric:
    def __init__(self, device):
        import lpips
        self.model = lpips.LPIPS(net="alex").to(device).eval()

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> float:
        return self.model(x * 2.0 - 1.0, y * 2.0 - 1.0).mean().item()
