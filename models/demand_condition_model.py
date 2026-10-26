"""DemandConditionModel: demand-gated ControlNet for the frozen diffusion U-Net.

The Control encoder receives ``z_r`` once through ``cond_embedding``.  A
``DemandPathAdapter`` itself receives only the Control feature ``H_l`` and the
resized demand map ``D_l``; it never reads or reprojects ``z_r``.  At every
skip scale, Stage 2 sends the raw Control feature through only the original
zero-conv. Stage 3 adds a parallel D-dependent correction branch whose final
convolution is also zero-initialized:

    R_l_stage2          = ZeroConv_stage2(H_l)
    U_l                 = Fuse(P_feature(H_l), P_demand(D_l))
    gate_l              = sigmoid(P_gate(U_l))
    scale_l, shift_l    = P_mod(U_l)
    V_l                 = gate_l * ((1 + tanh(scale_l)) * H_l + shift_l)
    delta_R_l_D         = ZeroConv_D(V_l)
    R_l_stage3          = R_l_stage2 + delta_R_l_D

Key invariants:

* D does not enter the Control encoder and only modulates outgoing skip
  residuals;
* Stage 2 has no learned skip adapter beyond its original zero-conv;
* each Stage-3 skip owns a separate D projection, gate and zero-conv branch;
* every D-branch zero-conv starts at exactly zero, so adding Stage 3 initially
  leaves the complete Stage-2 network output unchanged;
* the separate final x0 gate guarantees ``D == 0`` returns exactly ``z_r``;
* every output residual still passes a zero-initialized 1x1 conv, so at
  initialization the frozen U-Net output is unchanged (verified by
  ``tests/test_demand_condition.py``).

The condition image is exactly ``z_r`` (4 channels).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.controlnets.controlnet import ControlNetModel
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

from .cond_encoder import CSIEmbedding, ConditionEncoder


def zero_conv(channels: int) -> nn.Conv2d:
    """1x1 convolution with zero-initialized weight and bias."""
    conv = nn.Conv2d(channels, channels, kernel_size=1)
    nn.init.zeros_(conv.weight)
    nn.init.zeros_(conv.bias)
    return conv


@dataclass
class DemandControlOutput:
    down_residuals: tuple            # 12 tensors, aligned with UNet down skips
    mid_residual: torch.Tensor       # 1 tensor, aligned with UNet mid residual
    diagnostics: dict = field(default_factory=dict)


def _group_count(channels: int) -> int:
    """Largest practical GroupNorm group count that divides ``channels``."""
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class DemandGate(nn.Module):
    """Learned per-scale fusion of a Control feature and a demand map.

    The gate is conditioned jointly on feature content and projected D.  Its
    output has one value per batch, channel and spatial position rather than a
    single learned scalar. The complete branch ends in a zero-initialized 1x1
    convolution. Therefore its initial correction is exactly zero for every D,
    preserving the frozen Stage-2 output bit-for-bit.
    """

    def __init__(self, channels: int, mode: str = "learned",
                 gate_init: float = 0.9):
        super().__init__()
        if mode != "learned":
            raise ValueError(
                "DemandGate supports only the learned projection gate; "
                f"got mode={mode!r}"
            )
        if not 0.0 < gate_init < 1.0:
            raise ValueError("gate_init must be in (0, 1)")
        self.mode = "learned"
        hidden = min(max(channels // 8, 32), 128)
        groups = _group_count(hidden)
        self.demand_proj = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
        )
        self.feature_proj = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * hidden, hidden, 1),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.Conv2d(hidden, hidden, 1),
            nn.SiLU(),
        )
        self.gate_head = nn.Conv2d(hidden, channels, 1)
        self.modulation_head = nn.Conv2d(hidden, 2 * channels, 1)
        self.demand_zero_conv = zero_conv(channels)

        # A small nonzero gate projection lets the new zero-conv learn from a
        # D-dependent input immediately, while the zero-conv itself guarantees
        # zero perturbation of Stage 2.
        nn.init.normal_(self.gate_head.weight, mean=0.0, std=2e-2)
        nn.init.constant_(
            self.gate_head.bias,
            torch.logit(torch.tensor(float(gate_init))).item(),
        )
        nn.init.zeros_(self.modulation_head.weight)
        nn.init.zeros_(self.modulation_head.bias)
        self.register_buffer(
            "initial_gate", torch.tensor(float(gate_init)), persistent=True
        )
        self.last_gate_mean: torch.Tensor | None = None
        self.last_correction_norm: torch.Tensor | None = None

    def forward(self, features: torch.Tensor,
                demand: torch.Tensor) -> torch.Tensor:
        demand_features = self.demand_proj(demand.clamp(0.0, 1.0))
        feature_context = self.feature_proj(features)
        fused = self.fuse(torch.cat((feature_context, demand_features), dim=1))
        gate = torch.sigmoid(self.gate_head(fused))
        scale, shift = self.modulation_head(fused).chunk(2, dim=1)
        modulated = (1.0 + torch.tanh(scale)) * features + shift
        self.last_gate_mean = gate.detach().mean()
        correction = self.demand_zero_conv(gate * modulated)
        self.last_correction_norm = correction.detach().norm()
        return correction


class DemandPathAdapter(nn.Module):
    """Stage-2 zero-conv plus an optional Stage-3 D correction branch.

    ``z_r`` is deliberately absent from this module.  It has already entered
    the Control encoder through ``cond_embedding`` and is not injected again at
    any skip.
    """

    def __init__(self, channels: int, gate_mode: str = "learned",
                 gate_init: float = 0.9):
        super().__init__()
        self.gate = None if gate_mode == "none" else DemandGate(
            channels, mode=gate_mode, gate_init=gate_init,
        )
        self.zero_conv = zero_conv(channels)

    def forward(self, h_l: torch.Tensor,
                D_l: torch.Tensor | None = None):
        # This is the complete Stage-2 skip path. There are no other per-skip
        # trainable transformations in Stage 2.
        residual = self.zero_conv(h_l)
        # Stage 3 adds a parallel zero-initialized D correction. Its output is
        # exactly zero when first attached to a trained Stage-2 checkpoint.
        if D_l is not None and self.gate is not None:
            residual = residual + self.gate(h_l, D_l)
        return residual


class DemandConditionModel(nn.Module):
    """Demand-gated ControlNet built from a frozen SD U-Net encoder."""

    STAGE2_ARCHITECTURE = "control_condition_zr_zero_conv_v3"
    STAGE3_ARCHITECTURE = "demand_condition_zr_zero_init_skip_delta_v5"

    def __init__(self, unet: UNet2DConditionModel, cond_channels: int = 4,
                 csi_dim: int = 1, zr_channels: int = 4,
                 gate_mode: str = "learned", gate_init: float = 0.9):
        super().__init__()
        if cond_channels != zr_channels:
            raise ValueError(
                "DemandConditionModel Control input must contain z_r only; "
                f"expected cond_channels={zr_channels}, got {cond_channels}"
            )
        self.architecture_version = (
            self.STAGE2_ARCHITECTURE
            if gate_mode == "none"
            else self.STAGE3_ARCHITECTURE
        )

        # --- shared encoder: copy UNet encoder weights (trainable, ControlNet-style)
        base = ControlNetModel.from_unet(
            unet, conditioning_channels=cond_channels, load_weights_from_unet=True
        )
        self.conv_in = base.conv_in
        self.time_proj = base.time_proj
        self.time_embedding = base.time_embedding
        self.down_blocks = base.down_blocks
        self.mid_block = base.mid_block

        self.block_out_channels = tuple(unet.config.block_out_channels)
        self.layers_per_block = int(unet.config.layers_per_block)
        self.cross_attention_dim = int(unet.config.cross_attention_dim)

        time_embed_dim = self.block_out_channels[0] * 4
        self.cond_embedding = ConditionEncoder(cond_channels, self.block_out_channels[0])
        self.csi_embedding = CSIEmbedding(csi_dim, time_embed_dim)

        channels = self._residual_channels()
        self.adapters = nn.ModuleList([
            DemandPathAdapter(c, gate_mode=gate_mode,
                              gate_init=gate_init)
            for c in channels
        ])
        self.n_injections = len(channels)  # 13

    def freeze_encoder(self):
        """Freeze the UNet-copied encoder, leaving condition encoders and the
        original Stage-2 zero-convs trainable (low-VRAM variant)."""
        for m in (self.conv_in, self.time_proj, self.time_embedding,
                  self.down_blocks, self.mid_block):
            m.requires_grad_(False)

    def forward_control(self, noisy_latents: torch.Tensor, timestep,
                        encoder_hidden_states: torch.Tensor | None = None,
                        rx_latents: torch.Tensor | None = None,
                        csi: torch.Tensor | None = None):
        """Run ordinary Control training/inference without any demand map.

        No D tensor is created and every multi-scale residual passes at full
        strength.  Parameters learned here are directly compatible with the
        later demand-aware model because its complete Stage-2 path is shared.
        """
        return self.forward(
            noisy_latents, timestep,
            encoder_hidden_states=encoder_hidden_states,
            rx_latents=rx_latents, demand_map=None, csi=csi,
        )

    def _residual_channels(self) -> list:
        chans = [self.block_out_channels[0]]  # conv_in output
        for i, ch in enumerate(self.block_out_channels):
            is_final = i == len(self.block_out_channels) - 1
            for _ in range(self.layers_per_block):
                chans.append(ch)
            if not is_final:
                chans.append(ch)  # downsample output
        chans.append(self.block_out_channels[-1])  # mid block output
        return chans

    def forward(self, noisy_latents: torch.Tensor, timestep,
                encoder_hidden_states: torch.Tensor | None = None,
                rx_latents: torch.Tensor | None = None,
                demand_map: torch.Tensor | None = None,
                csi: torch.Tensor | None = None,
                return_dict: bool = True):
        """Run the Stage-2/3 Condition model with optional Demand gating."""
        b = noisy_latents.shape[0]
        device = noisy_latents.device
        dtype = noisy_latents.dtype

        if rx_latents is None:
            raise ValueError("DemandConditionModel needs rx_latents")

        # ---- timestep -> time embedding -------------------------------------
        if not torch.is_tensor(timestep):
            timesteps = torch.tensor([timestep], device=device, dtype=torch.long)
        else:
            timesteps = timestep.to(device=device, dtype=torch.long)
        if timesteps.ndim == 0:
            timesteps = timesteps[None]
        timesteps = timesteps.expand(b)

        t_emb = self.time_proj(timesteps).to(dtype=dtype)
        emb = self.time_embedding(t_emb)
        if csi is not None:
            emb = emb + self.csi_embedding(csi).to(emb.dtype)

        # ---- identical teacher/student Control input: z_r only --------------
        if encoder_hidden_states is None:
            encoder_hidden_states = torch.zeros(
                b, 1, self.cross_attention_dim, device=device, dtype=dtype
            )

        sample = self.conv_in(noisy_latents)
        sample = sample + self.cond_embedding(rx_latents)

        # ---- down blocks -----------------------------------------------------
        down_res = (sample,)
        for block in self.down_blocks:
            if getattr(block, "has_cross_attention", False):
                sample, res = block(
                    hidden_states=sample, temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample, res = block(hidden_states=sample, temb=emb)
            down_res = down_res + tuple(res)

        # ---- mid block -------------------------------------------------------
        if getattr(self.mid_block, "has_cross_attention", False):
            mid = self.mid_block(sample, emb,
                                 encoder_hidden_states=encoder_hidden_states)
        else:
            mid = self.mid_block(sample, emb)

        # ---- per-scale demand gating ----------------------------------------
        all_res = tuple(down_res) + (mid,)          # 13 tensors
        residuals = []
        gate_means = []
        res_norms = []

        for i, h_l in enumerate(all_res):
            D_l = None
            if demand_map is not None:
                D_l = F.interpolate(
                    demand_map, size=h_l.shape[-2:], mode="bilinear",
                    align_corners=False,
                )  # 双线性插值缩放到和 `h_l` 一样的空间分辨率（H,W）
            delta = self.adapters[i](h_l, D_l)
            residuals.append(delta)
            if D_l is not None:
                gate_means.append(
                    self.adapters[i].gate.last_gate_mean
                )
            if i < self.n_injections - 1:
                res_norms.append(delta.detach().norm())

        down_residuals = tuple(residuals[:-1])      # 12
        mid_residual = residuals[-1]                # 1

        diagnostics = {
            "demand_gate_mean": torch.stack(gate_means).mean() if gate_means else torch.tensor(0.0, device=device),
            "res_norm_mean": torch.stack(res_norms).mean() if res_norms else torch.tensor(0.0, device=device),
        }
        if gate_means:
            diagnostics["learned_gate_mean"] = torch.stack(gate_means).mean()

        return DemandControlOutput(
            down_residuals=down_residuals, mid_residual=mid_residual,
            diagnostics=diagnostics,
        )


