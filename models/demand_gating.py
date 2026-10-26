"""Explicit demand gating of the per-step clean-latent (x0) correction."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .loss import x0_from_eps


@dataclass
class GatedX0Output:
    eps: torch.Tensor
    x0: torch.Tensor
    x0_reference: torch.Tensor
    x0_full: torch.Tensor
    delta_x0: torch.Tensor
    gate: torch.Tensor


def eps_from_x0(z_t, x0, timestep, alphas_cumprod):
    if not torch.is_tensor(timestep):
        timestep = torch.tensor([timestep], device=z_t.device, dtype=torch.long)
    timestep = timestep.to(device=z_t.device, dtype=torch.long)
    if timestep.ndim == 0:
        timestep = timestep[None]
    timestep = timestep.expand(z_t.shape[0])
    alpha = alphas_cumprod.to(z_t.device, torch.float32)[timestep]
    alpha = alpha.reshape(-1, 1, 1, 1).to(z_t.dtype)
    return (z_t - alpha.sqrt() * x0) / (1.0 - alpha).sqrt().clamp_min(1e-8)


def gate_x0_correction(z_t, eps_full, z_r, demand, timestep,
                       alphas_cumprod, curriculum_mix: float = 1.0):
    """Blend the full Control proposal toward received latent ``z_r``.

    The inference rule is ``x0 = z_r + D * (x0_full - z_r)``.  Thus ``D=0``
    preserves the received latent and ``D=1`` accepts the full Control result.

    ``curriculum_mix=0`` uses the full correction everywhere during Control
    warm-up; ``curriculum_mix=1`` gives the desired inference rule exactly.
    """
    if demand is None:
        raise ValueError("x0 correction gating requires a demand map")
    t = timestep
    if not torch.is_tensor(t):
        t = torch.tensor([t], device=z_t.device, dtype=torch.long)
    if t.ndim == 0:
        t = t[None]
    t = t.to(z_t.device).expand(z_t.shape[0])
    x0_full = x0_from_eps(z_t, eps_full, t, alphas_cumprod)
    x0_reference = z_r.to(device=z_t.device, dtype=z_t.dtype)
    if x0_reference.shape[-2:] != z_t.shape[-2:]:
        x0_reference = F.interpolate(
            x0_reference, size=z_t.shape[-2:], mode="bilinear",
            align_corners=False,
        )
    if x0_reference.shape != z_t.shape:
        raise ValueError(
            f"z_r shape {tuple(x0_reference.shape)} must match "
            f"z_t shape {tuple(z_t.shape)}"
        )
    gate = F.interpolate(demand, size=z_t.shape[-2:], mode="bilinear",
                         align_corners=False).clamp(0.0, 1.0)
    mix = float(max(0.0, min(1.0, curriculum_mix)))
    gate = (1.0 - mix) + mix * gate
    delta = x0_full - x0_reference
    x0 = x0_reference + gate * delta
    eps = eps_from_x0(z_t, x0, t, alphas_cumprod)
    return GatedX0Output(eps=eps, x0=x0, x0_reference=x0_reference,
                         x0_full=x0_full, delta_x0=delta, gate=gate)
