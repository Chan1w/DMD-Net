"""Diffusion latent conversion used by the Demand x0 gate."""

from __future__ import annotations

import torch


def x0_from_eps(z_t: torch.Tensor, eps: torch.Tensor, t: torch.Tensor,
                alphas_cumprod: torch.Tensor) -> torch.Tensor:
    alpha_bar = alphas_cumprod.to(device=z_t.device, dtype=torch.float32)[t]
    alpha_bar = alpha_bar.reshape(-1, 1, 1, 1).to(z_t.dtype)
    return (
        z_t - (1.0 - alpha_bar).sqrt() * eps
    ) / alpha_bar.sqrt().clamp_min(1e-8)
