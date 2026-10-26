"""Stage-2 and Stage-3 SD2.1 diffusion restorers."""

from __future__ import annotations

import torch
import torch.nn as nn

from .builder import apply_classifier_free_guidance
from .demand_gating import gate_x0_correction


class DemandRestorer(nn.Module):
    """img2img restorer driven by a Demand-gated ControlNet.

    ``D`` is the Diffusion Demand Map (high = intervene more).  The restorer is
    ``DemandConditionModel`` with explicit keyword arguments (``rx_latents`` /
    ``demand_map``), which is the diffusers-style interface of the new model.
    """

    def __init__(self, unet, scheduler, vae_adapter, condition=None,
                 guidance_scale: float = 1.0, guidance_rescale: float = 0.0):
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler
        self.vae = vae_adapter
        self.condition = condition  # DemandConditionModel or None
        self.guidance_scale = guidance_scale
        self.guidance_rescale = guidance_rescale

    def make_timesteps(self, start_rel: float, steps: int):
        self.scheduler.set_timesteps(steps, device=self.unet.device)
        timesteps = self.scheduler.timesteps
        init_timestep = min(int(steps * start_rel), steps)
        t_start = max(steps - init_timestep, 0)
        return timesteps[t_start:]

    @torch.no_grad()
    def restore(self, z_r: torch.Tensor, D: torch.Tensor | None,
                csi: torch.Tensor | None, prompt_embeds: torch.Tensor,
                start_rel: float, steps: int,
                return_latent: bool = False,
                uncond_embeds: torch.Tensor | None = None) -> torch.Tensor:
        timesteps = self.make_timesteps(start_rel, steps)
        b = z_r.shape[0]
        noise = torch.randn_like(z_r)
        z_t = self.scheduler.add_noise(z_r, noise, timesteps[0])

        for t in timesteps:
            model_input = self.scheduler.scale_model_input(z_t, t)
            if self.condition is not None:
                if D is None:
                    raise ValueError("DemandRestorer requires D when condition is enabled")
                # The Control encoder input is z_r alone. D is projected and
                # fused only at the outgoing multi-scale skip residuals; the
                # explicit x0 gate remains the final preservation rule.
                ctrl = self.condition(
                    model_input, t,
                    encoder_hidden_states=prompt_embeds,
                    rx_latents=z_r,
                    demand_map=D,
                    csi=csi,
                )
                eps_full = apply_classifier_free_guidance(
                    self.unet, model_input, t, prompt_embeds, uncond_embeds,
                    self.guidance_scale, self.guidance_rescale,
                    down_block_additional_residuals=ctrl.down_residuals,
                    mid_block_additional_residual=ctrl.mid_residual,
                )
                eps = gate_x0_correction(
                    z_t, eps_full, z_r, D, t,
                    self.scheduler.alphas_cumprod,
                ).eps
            else:
                eps = apply_classifier_free_guidance(
                    self.unet, model_input, t, prompt_embeds, uncond_embeds,
                    self.guidance_scale, self.guidance_rescale,
                )
            z_t = self.scheduler.step(eps, t, z_t).prev_sample

        if return_latent:
            return z_t
        return self.vae.decode_scaled(z_t)


class ControlRestorer(nn.Module):
    """SD img2img restorer with optional full-strength Stage-2 Condition.

    Unlike :class:`DemandRestorer`, this path has no demand-map input, no
    residual gate, and no per-step x0 gate.  It is the inference counterpart of
    ``scripts/stage2_train_control_sd21.py``.
    """

    def __init__(self, unet, scheduler, vae_adapter, condition=None,
                 guidance_scale: float = 1.0, guidance_rescale: float = 0.0):
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler
        self.vae = vae_adapter
        self.condition = condition
        self.guidance_scale = guidance_scale
        self.guidance_rescale = guidance_rescale

    def make_timesteps(self, start_rel: float, steps: int):
        self.scheduler.set_timesteps(steps, device=self.unet.device)
        timesteps = self.scheduler.timesteps
        init_timestep = min(int(steps * start_rel), steps)
        t_start = max(steps - init_timestep, 0)
        return timesteps[t_start:]

    @torch.no_grad()
    def restore(self, z_r: torch.Tensor, csi: torch.Tensor | None,
                prompt_embeds: torch.Tensor, start_rel: float, steps: int,
                return_latent: bool = False,
                uncond_embeds: torch.Tensor | None = None) -> torch.Tensor:
        timesteps = self.make_timesteps(start_rel, steps)
        if len(timesteps) == 0:
            return z_r if return_latent else self.vae.decode_scaled(z_r)

        noise = torch.randn_like(z_r)
        z_t = self.scheduler.add_noise(z_r, noise, timesteps[0])
        for timestep in timesteps:
            model_input = self.scheduler.scale_model_input(z_t, timestep)
            kwargs = {}
            if self.condition is not None:
                control = self.condition.forward_control(
                    model_input,
                    timestep,
                    encoder_hidden_states=prompt_embeds,
                    rx_latents=z_r,
                    csi=csi,
                )
                kwargs = {
                    "down_block_additional_residuals": control.down_residuals,
                    "mid_block_additional_residual": control.mid_residual,
                }
            prediction = apply_classifier_free_guidance(
                self.unet,
                model_input,
                timestep,
                prompt_embeds,
                uncond_embeds,
                self.guidance_scale,
                self.guidance_rescale,
                **kwargs,
            )
            z_t = self.scheduler.step(prediction, timestep, z_t).prev_sample

        if return_latent:
            return z_t
        return self.vae.decode_scaled(z_t)


