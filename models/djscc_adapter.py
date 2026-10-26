"""Frozen SD2.1 latent DJSCC receiver adapter."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .djscc import JSCCModel
from .vae_adapter import VAEAdapter


@dataclass
class DJSCCObservation:
    """Quantities available after one DJSCC transmission."""

    x_base: torch.Tensor
    z_hat_raw: torch.Tensor
    csi: torch.Tensor
    s: torch.Tensor | None


class DJSCCAdapter(nn.Module):
    def __init__(self, jscc: JSCCModel, vae_adapter: VAEAdapter):
        super().__init__()
        self.jscc = jscc
        self.vae = vae_adapter
        self.jscc.requires_grad_(False)

    @torch.no_grad()
    def transmit(self, x_gt: torch.Tensor,
                 snr_db: torch.Tensor) -> DJSCCObservation:
        z_raw = self.vae.encode_mean(x_gt)
        z_hat_raw, symbols = self.jscc(z_raw, snr_db)
        x_base = self.vae.decode_raw(z_hat_raw)
        return DJSCCObservation(
            x_base=x_base,
            z_hat_raw=z_hat_raw,
            csi=snr_db.reshape(-1, 1).float(),
            s=symbols,
        )


def load_jscc(checkpoint_path: str, device, **kwargs) -> JSCCModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    bandwidth = kwargs.pop("bandwidth", None)
    if bandwidth is None:
        bandwidth = state["decoder.stem.weight"].shape[1]
    model_config = (
        checkpoint.get("model_config", {})
        if isinstance(checkpoint, dict) else {}
    )
    for name in ("latent_ch", "mid_ch", "emb_dim", "n_blocks", "signal_power"):
        if name not in kwargs and name in model_config:
            kwargs[name] = model_config[name]
    latent_ch = kwargs.pop("latent_ch", 4)
    model = JSCCModel(latent_ch=latent_ch, bandwidth=bandwidth, **kwargs)
    model.load_state_dict(state)
    model.checkpoint_metadata = {
        key: checkpoint[key]
        for key in ("model_family", "model_config", "vae_model_id")
        if isinstance(checkpoint, dict) and key in checkpoint
    }
    return model.to(device).eval()
