"""Predict a dense diffusion-demand map from DJSCC receiver features and SNR.

The estimator deliberately uses only information available at the receiver:
``z_r``, the scaled 4-channel DJSCC reconstruction in SD latent space, and
``csi``, the channel SNR in dB. Multi-kernel residual blocks preserve local
corruption patterns, while low-resolution self-attention models global region
relationships. U-Net skips restore that context at the input resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DemandOutput:
    logits: torch.Tensor
    demand: torch.Tensor
    diagnostics: dict = field(default_factory=dict)


def _group_norm(channels: int) -> nn.GroupNorm:
    """Choose the largest conventional group count that divides channels."""
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    raise ValueError(f"invalid channel count: {channels}")


class SNREmbedding(nn.Module):
    """Embed scalar SNR after mapping its usual dB range to a stable scale."""

    def __init__(self, embed_channels: int, snr_scale: float = 20.0):
        super().__init__()
        self.snr_scale = float(snr_scale)
        self.net = nn.Sequential(
            nn.Linear(1, embed_channels), nn.SiLU(),
            nn.Linear(embed_channels, embed_channels), nn.SiLU(),
        )

    def forward(self, snr_db: torch.Tensor) -> torch.Tensor:
        snr = snr_db.float().reshape(-1, 1) / self.snr_scale
        return self.net(snr)


class SNRFiLM(nn.Module):
    """Apply channel-wise SNR conditioning without destroying spatial detail."""

    def __init__(self, embed_channels: int, feature_channels: int):
        super().__init__()
        self.affine = nn.Linear(embed_channels, feature_channels * 2)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, x: torch.Tensor, snr_embedding: torch.Tensor) -> torch.Tensor:
        scale, bias = self.affine(snr_embedding).chunk(2, dim=1)
        return x * (1.0 + scale[:, :, None, None]) + bias[:, :, None, None]


class LocalImageBlock(nn.Module):
    """Residual local/dilated filtering with channel selection and refinement."""

    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        hidden = channels * expansion
        self.norm1 = _group_norm(channels)
        self.local = nn.Conv2d(channels, channels, 3, padding=1,
                               groups=channels)
        self.dilated = nn.Conv2d(channels, channels, 3, padding=2,
                                 dilation=2, groups=channels)
        self.mix = nn.Conv2d(channels * 2, channels, 1)
        gate_hidden = max(channels // 4, 16)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, gate_hidden, 1), nn.SiLU(),
            nn.Conv2d(gate_hidden, channels, 1), nn.Sigmoid(),
        )
        self.norm2 = _group_norm(channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.SiLU(),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.mix(torch.cat((self.local(h), self.dilated(h)), dim=1))
        x = x + h * self.channel_gate(h)
        return x + self.ffn(self.norm2(x))


class GlobalUnderstandingBlock(nn.Module):
    """Transformer block for long-range region and semantic relationships."""

    def __init__(self, channels: int, num_heads: int = 8,
                 mlp_ratio: int = 2, dropout: float = 0.0):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("context channels must be divisible by attention heads")
        self.position = nn.Conv2d(channels, channels, 3, padding=1,
                                  groups=channels)
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(channels)
        hidden = channels * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, channels), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.position(x)
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        normalized = self.norm1(tokens)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        tokens = tokens + attended
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class EncoderStage(nn.Module):
    def __init__(self, channels: int, snr_channels: int, num_blocks: int):
        super().__init__()
        self.film = SNRFiLM(snr_channels, channels)
        self.blocks = nn.Sequential(
            *[LocalImageBlock(channels) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor, snr_embedding: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.film(x, snr_embedding))


class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 snr_channels: int, num_blocks: int):
        super().__init__()
        self.project = nn.Conv2d(in_channels, out_channels, 1)
        self.fuse = nn.Conv2d(out_channels + skip_channels, out_channels, 3,
                              padding=1)
        self.film = SNRFiLM(snr_channels, out_channels)
        self.blocks = nn.Sequential(
            *[LocalImageBlock(out_channels) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor,
                snr_embedding: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                          align_corners=False)
        x = self.project(x)
        x = self.fuse(torch.cat((x, skip), dim=1))
        return self.blocks(self.film(x, snr_embedding))


class DemandEstimator(nn.Module):
    """SNR-conditioned convolution/attention U-Net for demand-map prediction."""

    architecture_version = "demand_estimator_zr_snr_v3"

    def __init__(self, latent_channels: int = 4,
                 base_channels: int = 64,
                 mid_channels: int = 96,
                 context_channels: int = 128,
                 snr_embed_channels: int = 64,
                 local_blocks: int = 2,
                 context_blocks: int = 2,
                 attention_heads: int = 8,
                 mlp_ratio: int = 2,
                 dropout: float = 0.0,
                 out_channels: int = 1):
        super().__init__()
        self.snr_embedding = SNREmbedding(snr_embed_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(latent_channels, base_channels, 3, padding=1),
            _group_norm(base_channels), nn.SiLU(),
        )
        self.encoder_high = EncoderStage(
            base_channels, snr_embed_channels, local_blocks
        )
        self.down_high = nn.Conv2d(base_channels, mid_channels, 3,
                                   stride=2, padding=1)
        self.encoder_mid = EncoderStage(
            mid_channels, snr_embed_channels, local_blocks
        )
        self.down_mid = nn.Conv2d(mid_channels, context_channels, 3,
                                  stride=2, padding=1)

        self.context_film = SNRFiLM(snr_embed_channels, context_channels)
        context_layers = [LocalImageBlock(context_channels)]
        context_layers.extend(
            GlobalUnderstandingBlock(
                context_channels, num_heads=attention_heads,
                mlp_ratio=mlp_ratio, dropout=dropout,
            )
            for _ in range(context_blocks)
        )
        context_layers.append(LocalImageBlock(context_channels))
        self.context = nn.Sequential(*context_layers)

        self.decoder_mid = DecoderStage(
            context_channels, mid_channels, mid_channels,
            snr_embed_channels, local_blocks,
        )
        self.decoder_high = DecoderStage(
            mid_channels, base_channels, base_channels,
            snr_embed_channels, local_blocks,
        )
        self.refine = LocalImageBlock(base_channels)
        self.head = nn.Sequential(
            _group_norm(base_channels), nn.SiLU(),
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.SiLU(), nn.Conv2d(base_channels // 2, out_channels, 1),
        )

    def forward(self, z_r: torch.Tensor, csi: torch.Tensor) -> DemandOutput:
        if z_r.ndim != 4:
            raise ValueError(f"z_r must be [B,C,H,W], got {tuple(z_r.shape)}")
        if csi is None:
            raise ValueError("DemandEstimator requires receiver SNR (csi)")
        if csi.numel() != z_r.shape[0]:
            raise ValueError("csi must contain one scalar SNR per input sample")

        snr_embedding = self.snr_embedding(csi)
        high = self.encoder_high(self.stem(z_r), snr_embedding)
        mid = self.encoder_mid(self.down_high(high), snr_embedding)
        context = self.down_mid(mid)
        context = self.context(self.context_film(context, snr_embedding))
        decoded = self.decoder_mid(context, mid, snr_embedding)
        decoded = self.decoder_high(decoded, high, snr_embedding)
        logits = self.head(self.refine(decoded))
        demand = torch.sigmoid(logits)
        diagnostics = {
            "demand_mean": demand.detach().mean(),
            "demand_std": demand.detach().std(),
            "logit_abs_mean": logits.detach().abs().mean(),
        }
        return DemandOutput(logits=logits, demand=demand,
                            diagnostics=diagnostics)
