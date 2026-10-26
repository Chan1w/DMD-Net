"""SNR-conditioned JSCC encoder / decoder (latent domain).

This module defines the Stage-1 latent DJSCC codec whose weights are reused by
the later restoration stages.  The encoder compresses a
(latent) feature map into C channel symbols and normalizes transmit power; the
decoder reconstructs the feature map from the noisy received symbols.  SNR (dB)
is injected into both networks via FiLM so a single set of weights generalizes
across a continuous range of channel SNRs.

NOTE: the module/attribute names here MUST stay identical to the checkpoint keys
(``encoder.*`` / ``decoder.*`` / ``channel.*``) so ``state_dict`` loads directly.
"""

from dataclasses import asdict, dataclass
from typing import Sequence

import torch
import torch.nn as nn

from .channel import AWGNChannel, power_normalize


@dataclass(frozen=True)
class DJSCCRateInfo:
    """Rate information for the real-valued AWGN implementation.

    ``image_cbr`` is the channel bandwidth ratio reported by the training and
    inference scripts:

        CBR = number of transmitted real channel symbols
              / number of source-image scalar samples
            = (C_s * H_s * W_s) / (C_x * H_x * W_x).

    The channel in :mod:`models.channel` is real-valued, so one tensor element
    is one real channel use.  If a future implementation packs I/Q pairs into
    complex symbols, the numerator must be divided by two and the convention
    must be stated explicitly.

    ``latent_cbr`` is also exposed for debugging.  It measures compression
    relative to the VAE latent rather than to RGB pixels and is therefore not
    the primary paper CBR.
    """

    source_shape: tuple[int, int, int]
    latent_shape: tuple[int, int, int]
    symbol_shape: tuple[int, int, int]
    source_values: int
    latent_values: int
    real_channel_uses: int
    image_cbr: float
    latent_cbr: float

    def to_dict(self) -> dict:
        return asdict(self)


def calculate_cbr(source_shape: Sequence[int], latent_shape: Sequence[int],
                  symbol_shape: Sequence[int]) -> DJSCCRateInfo:
    """Calculate image-domain and latent-domain CBR from ``(C,H,W)`` shapes.

    This shape-based function is intentionally independent of a forward pass,
    making the CBR definition easy to test and include in checkpoint metadata.
    Batch dimensions must be removed before calling it.
    """

    source_shape = tuple(int(v) for v in source_shape)
    latent_shape = tuple(int(v) for v in latent_shape)
    symbol_shape = tuple(int(v) for v in symbol_shape)
    if len(source_shape) != 3 or len(latent_shape) != 3 or len(symbol_shape) != 3:
        raise ValueError("source_shape, latent_shape and symbol_shape must be (C,H,W)")
    if any(v <= 0 for v in source_shape + latent_shape + symbol_shape):
        raise ValueError("all CBR shape dimensions must be positive")

    source_values = source_shape[0] * source_shape[1] * source_shape[2]
    latent_values = latent_shape[0] * latent_shape[1] * latent_shape[2]
    channel_uses = symbol_shape[0] * symbol_shape[1] * symbol_shape[2]
    return DJSCCRateInfo(
        source_shape=source_shape,
        latent_shape=latent_shape,
        symbol_shape=symbol_shape,
        source_values=source_values,
        latent_values=latent_values,
        real_channel_uses=channel_uses,
        image_cbr=channel_uses / source_values,
        latent_cbr=channel_uses / latent_values,
    )


class SNREmbedding(nn.Module):
    """Maps a scalar SNR (dB) to a conditioning embedding."""

    def __init__(self, emb_dim: int = 128, snr_scale: float = 20.0):
        super().__init__()
        self.snr_scale = snr_scale
        self.mlp = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, snr_db: torch.Tensor) -> torch.Tensor:
        snr_db = snr_db.to(dtype=torch.float32).reshape(-1) / self.snr_scale
        return self.mlp(snr_db.unsqueeze(-1))


class FiLM(nn.Module):
    """Feature-wise affine conditioning from an SNR embedding."""

    def __init__(self, emb_dim: int, feat_dim: int):
        super().__init__()
        self.gamma = nn.Linear(emb_dim, feat_dim)
        self.beta = nn.Linear(emb_dim, feat_dim)
        # initialize so FiLM is near identity at start of training
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        g = self.gamma(emb).unsqueeze(-1).unsqueeze(-1)
        b = self.beta(emb).unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + g) + b


class ResBlockFiLM(nn.Module):
    """Residual conv block with SNR-conditioned FiLM modulation."""

    def __init__(self, ch: int, emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, ch), ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.norm2 = nn.GroupNorm(min(32, ch), ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.film = FiLM(emb_dim, ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        h = self.film(h, emb)
        return x + self.act(h)


class JSCCEncoder(nn.Module):
    def __init__(self, in_ch: int = 4, out_ch: int = 8, mid_ch: int = 128,
                 emb_dim: int = 128, n_blocks: int = 6):
        super().__init__()
        self.snr_emb = SNREmbedding(emb_dim)
        self.stem = nn.Conv2d(in_ch, mid_ch, 3, 1, 1)
        self.blocks = nn.ModuleList([ResBlockFiLM(mid_ch, emb_dim) for _ in range(n_blocks)])
        self.head = nn.Conv2d(mid_ch, out_ch, 3, 1, 1)

    def forward(self, z: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        emb = self.snr_emb(snr_db)
        h = self.stem(z)
        for blk in self.blocks:
            h = blk(h, emb)
        return self.head(h)


class JSCCDecoder(nn.Module):
    def __init__(self, in_ch: int = 8, out_ch: int = 4, mid_ch: int = 128,
                 emb_dim: int = 128, n_blocks: int = 6):
        super().__init__()
        self.snr_emb = SNREmbedding(emb_dim)
        self.stem = nn.Conv2d(in_ch, mid_ch, 3, 1, 1)
        self.blocks = nn.ModuleList([ResBlockFiLM(mid_ch, emb_dim) for _ in range(n_blocks)])
        self.head = nn.Conv2d(mid_ch, out_ch, 3, 1, 1)

    def forward(self, y: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        emb = self.snr_emb(snr_db)
        h = self.stem(y)
        for blk in self.blocks:
            h = blk(h, emb)
        return self.head(h)


class JSCCModel(nn.Module):
    """Full JSCC codec: latent -> symbols -> AWGN -> reconstructed latent.

    Only this module's parameters are trained in Stage 1. During Stages 2 and 3,
    they stay frozen and act as the fixed transmitter and base receiver.
    """

    def __init__(self, latent_ch: int = 4, bandwidth: int = 8, mid_ch: int = 128,
                 emb_dim: int = 128, n_blocks: int = 6, signal_power: float = 1.0):
        super().__init__()
        self.bandwidth = bandwidth
        self.encoder = JSCCEncoder(in_ch=latent_ch, out_ch=bandwidth,
                                   mid_ch=mid_ch, emb_dim=emb_dim, n_blocks=n_blocks)
        self.decoder = JSCCDecoder(in_ch=bandwidth, out_ch=latent_ch,
                                   mid_ch=mid_ch, emb_dim=emb_dim, n_blocks=n_blocks)
        self.channel = AWGNChannel(signal_power=signal_power)

    def forward(self, z: torch.Tensor, snr_db: torch.Tensor):
        """Returns (reconstructed_latent, transmitted_symbols)."""
        s = self.encoder(z, snr_db)
        s = power_normalize(s)
        y = self.channel(s, snr_db)
        z_hat = self.decoder(y, snr_db)
        return z_hat, s

    def cbr_info(self, image_shape: Sequence[int], latent_shape: Sequence[int]) -> DJSCCRateInfo:
        """Return CBR metadata for this model without running the channel.

        Args:
            image_shape: source RGB shape ``(C,H,W)``.
            latent_shape: VAE latent shape ``(C_z,H_z,W_z)``.
        """

        latent_shape = tuple(int(v) for v in latent_shape)
        symbol_shape = (self.bandwidth, latent_shape[-2], latent_shape[-1])
        return calculate_cbr(image_shape, latent_shape, symbol_shape)
