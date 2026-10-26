"""Frozen Stable Diffusion VAE adapter.

Stable Diffusion VAEs map between pixel space (``B x 3 x H x W`` in ``[-1, 1]``) and a
4-channel latent at ``H/8 x W/8``.  The DMD-Net pipeline uses the following latent convention:

* encoding uses the **fixed posterior mean** (``latent_dist.mode()``) so that no
  extra sampling noise is introduced;
* the latent is scaled by the checkpoint's ``scaling_factor`` when handed to the
  UNet, and unscaled when handed to the VAE decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers import AutoencoderKL

SD21_MODEL_ID = "LanguageMachines/stable-diffusion-2-1-base"
DEFAULT_SCALING_FACTOR = 0.18215


def load_vae(device, dtype=torch.float32, model_id: str = SD21_MODEL_ID) -> AutoencoderKL:
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", local_files_only=True, torch_dtype=dtype
    )
    vae.requires_grad_(False)
    return vae.eval().to(device)


class VAEAdapter(nn.Module):
    """Frozen VAE wrapper with fixed-mean encoding and scaling.

    ``scaling_factor`` is read from the loaded VAE config when available and
    falls back to the SD 2.1-compatible default otherwise.
    """

    def __init__(self, vae: AutoencoderKL, scaling_factor: float | None = None):
        super().__init__()
        self.vae = vae
        if scaling_factor is None:
            scaling_factor = float(getattr(vae.config, "scaling_factor", DEFAULT_SCALING_FACTOR))
        self.scaling_factor = scaling_factor

    @torch.no_grad()
    def encode_mean(self, image: torch.Tensor) -> torch.Tensor:
        """Pixel image (B,3,H,W in [-1,1]) -> raw VAE latent mode (B,4,H/8,W/8)."""
        return self.vae.encode(image).latent_dist.mode()

    @torch.no_grad()
    def encode_mean_scaled(self, image: torch.Tensor) -> torch.Tensor:
        """Pixel image -> *scaled* latent (the space the UNet expects)."""
        return self.encode_mean(image) * self.scaling_factor

    def scale_raw(self, latent_raw: torch.Tensor) -> torch.Tensor:
        """Raw VAE/DJSCC latent -> scaled latent expected by the diffusion UNet."""
        return latent_raw * self.scaling_factor

    def unscale(self, latent_scaled: torch.Tensor) -> torch.Tensor:
        """Diffusion-scaled latent -> raw VAE latent."""
        return latent_scaled / self.scaling_factor

    @torch.no_grad()
    def decode_scaled(self, latent_scaled: torch.Tensor) -> torch.Tensor:
        """Scaled latent (B,4,h,w) -> pixel image (B,3,H,W in [-1,1])."""
        return self.vae.decode(self.unscale(latent_scaled)).sample

    def decode_scaled_with_grad(self, latent_scaled: torch.Tensor) -> torch.Tensor:
        """Decode while preserving gradients to the latent input.

        VAE parameters remain frozen through ``requires_grad_(False)``. This
        variant lets an image-space loss optimize an upstream demand map.
        """
        return self.vae.decode(self.unscale(latent_scaled)).sample

    @torch.no_grad()
    def decode_raw(self, latent_raw: torch.Tensor) -> torch.Tensor:
        """Raw (unscaled) latent -> pixel image (B,3,H,W in [-1,1])."""
        return self.vae.decode(latent_raw).sample

    def forward(self):
        raise NotImplementedError("VAEAdapter is used via its encode/decode helpers.")

