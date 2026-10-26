"""AWGN channel and power normalization with per-sample SNR (dB).

Provides the common channel interface used by the Stage-1 latent DJSCC encoder,
receiver reconstruction, and later restoration stages. It is self-contained so
training and evaluation share the same normalization and AWGN model.
"""

import math

import torch
import torch.nn as nn


def snr_db_to_linear(snr_db: torch.Tensor) -> torch.Tensor:
    """Convert dB SNR to linear scale: snr_lin = 10^(snr_db/10)."""
    return torch.pow(10.0, snr_db / 10.0)


def snr_db_to_noise_std(snr_db: torch.Tensor, signal_power: float = 1.0) -> torch.Tensor:
    """Noise standard deviation for AWGN given per-symbol power P and SNR (dB).

    SNR = P / sigma^2  =>  sigma = sqrt(P / SNR_lin).
    """
    return torch.sqrt(signal_power / snr_db_to_linear(snr_db))


def power_normalize(x: torch.Tensor, power: float = 1.0, eps: float = 1e-9) -> torch.Tensor:
    """Normalize the transmitted signal so the average per-symbol power equals `power`.

    x: (B, C, H, W) -> normalized tensor of the same shape.
    """
    mean_power = x.pow(2).mean(dim=(1, 2, 3), keepdim=True)
    return x * math.sqrt(power) / torch.sqrt(mean_power + eps)


class AWGNChannel(nn.Module):
    """AWGN channel with per-sample SNR (dB) conditioning.

    Forward:
        y = x + n,  n ~ N(0, sigma^2),  sigma^2 = P / 10^(snr_db/10)
    """

    def __init__(self, signal_power: float = 1.0):
        super().__init__()
        self.signal_power = signal_power

    def forward(self, x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        if not isinstance(snr_db, torch.Tensor):
            snr_db = torch.full((b,), float(snr_db), device=x.device, dtype=x.dtype)
        else:
            snr_db = snr_db.to(device=x.device, dtype=x.dtype).reshape(-1)
            if snr_db.numel() == 1 and b > 1:
                snr_db = snr_db.expand(b)
        sigma = snr_db_to_noise_std(snr_db, self.signal_power).view(b, 1, 1, 1)
        noise = torch.randn_like(x) * sigma
        return x + noise


def sample_snr(batch_size: int, snr_min: float, snr_max: float,
               device=None, dtype=None) -> torch.Tensor:
    """Sample a batch of SNRs uniformly in [snr_min, snr_max] (dB)."""
    snr = torch.rand(batch_size, device=device, dtype=dtype) * (snr_max - snr_min) + snr_min
    return snr
