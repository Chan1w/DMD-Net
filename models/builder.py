"""Builders for the final SD2.1 DMD-Net pipeline."""

from __future__ import annotations

import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from config import BLIP_MODEL_ID, SD21_MODEL_ID
from .djscc_adapter import DJSCCAdapter, load_jscc
from .demand_condition_model import DemandConditionModel
from .demand_estimator import DemandEstimator
from .vae_adapter import VAEAdapter


def patch_diffusers_attention_signature():
    """Cache ``inspect.signature`` in ``Attention.forward``.

    diffusers 0.40.0 calls ``inspect.signature(self.processor.__call__)`` on
    *every* attention forward.  Under torch 2.7.1 + Python 3.10 this
    intermittently segfaults (a CPython ``inspect.signature`` bug).  The
    processor's parameter set is invariant for the lifetime of a run, so caching
    it per processor class is safe and removes the crash entirely.
    """
    import inspect as _inspect

    import diffusers.models.attention_processor as _ap

    if getattr(_ap.Attention, "_aris_sig_patched", False):
        return

    _sig_cache: dict = {}

    def _cached_attn_parameters(processor):
        cls = type(processor)
        if cls not in _sig_cache:
            _sig_cache[cls] = set(_inspect.signature(processor.__call__).parameters.keys())
        return _sig_cache[cls]

    _orig_forward = _ap.Attention.forward

    def _patched_forward(self, hidden_states, encoder_hidden_states=None,
                         attention_mask=None, **cross_attention_kwargs):
        attn_parameters = _cached_attn_parameters(self.processor)
        quiet_attn_parameters = {"ip_adapter_masks", "ip_hidden_states"}
        unused_kwargs = [
            k for k, _ in cross_attention_kwargs.items()
            if k not in attn_parameters and k not in quiet_attn_parameters
        ]
        if len(unused_kwargs) > 0:
            _ap.logger.warning(
                f"cross_attention_kwargs {unused_kwargs} are not expected by "
                f"{self.processor.__class__.__name__} and will be ignored."
            )
        cross_attention_kwargs = {
            k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters
        }
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

    _ap.Attention.forward = _patched_forward
    _ap.Attention._aris_sig_patched = True


def disable_flash_attention():
    """Disable unstable GPU kernels on Blackwell (RTX 5090).

    torch 2.7.1 + cu128 on sm_120 intermittently crashes (segfault / illegal
    instruction) inside the flash SDPA kernels *and* inside cuDNN convolutions.
    We fall back to the most conservative, stable paths:

    * SDPA -> math backend (explicit matmul attention);
    * conv  -> PyTorch native CUDA kernels (cuDNN disabled).
    """
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False


def load_vae(device, dtype=torch.float32,
             model_id: str = SD21_MODEL_ID) -> AutoencoderKL:
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", local_files_only=True, torch_dtype=dtype
    )
    vae.requires_grad_(False)
    return vae.eval().to(device)


def load_unet(device, dtype=torch.float32, model_id: str = SD21_MODEL_ID) -> UNet2DConditionModel:
    patch_diffusers_attention_signature()
    disable_flash_attention()
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", local_files_only=True, torch_dtype=dtype
    )
    unet.requires_grad_(False)
    return unet.eval().to(device)


def load_text_encoder(device, dtype=torch.float32, model_id: str = SD21_MODEL_ID):
    # Stable Diffusion conditions its U-Net with per-token hidden states.  The
    # SD-2.1 Diffusers checkpoint declares ``CLIPTextModel`` in
    # text_encoder/config.json and intentionally contains no text-projection
    # weight.  Loading it as CLIPTextModelWithProjection would therefore create
    # an unused random ``text_projection`` and emit a misleading warning.
    te = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", local_files_only=True, torch_dtype=dtype
    )
    # SD-2.1 uses the penultimate token state followed by CLIP's final LayerNorm.
    # The runtime-only marker avoids coupling encode_prompt to a model class.
    te._prompt_hidden_state_index = -2 if model_id == SD21_MODEL_ID else None
    te.requires_grad_(False)
    return te.eval().to(device)


def load_tokenizer(model_id: str = SD21_MODEL_ID) -> CLIPTokenizer:
    return CLIPTokenizer.from_pretrained(
        model_id, subfolder="tokenizer", local_files_only=True
    )


def load_scheduler(model_id: str = SD21_MODEL_ID) -> DDIMScheduler:
    try:
        return DDIMScheduler.from_pretrained(
            model_id, subfolder="scheduler", local_files_only=True
        )
    except OSError:
        if model_id != SD21_MODEL_ID:
            raise
        # Some local mirrors contain the SD-2.1 weights but omit the tiny
        # scheduler directory.  Construct the published SD-2.1-base noise
        # construct the scheduler explicitly when no cached scheduler config is available.
        return DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
            prediction_type="epsilon",
        )


def build_vae_adapter(device, dtype=torch.float32,
                      model_id: str = SD21_MODEL_ID) -> VAEAdapter:
    return VAEAdapter(load_vae(device, dtype, model_id=model_id))


def build_djscc_adapter(checkpoint_path: str, device,
                        vae_dtype=torch.float32,
                        vae_model_id: str | None = None, **kwargs) -> DJSCCAdapter:
    jscc = load_jscc(checkpoint_path, device, **kwargs)
    checkpoint_vae = getattr(jscc, "checkpoint_metadata", {}).get("vae_model_id")
    if vae_model_id is None:
        vae_model_id = checkpoint_vae or SD21_MODEL_ID
    elif checkpoint_vae is not None and vae_model_id != checkpoint_vae:
        raise ValueError(
            f"DJSCC checkpoint was trained with VAE {checkpoint_vae!r}, "
            f"but {vae_model_id!r} was requested"
        )
    vae_adapter = build_vae_adapter(device, vae_dtype, model_id=vae_model_id)
    return DJSCCAdapter(jscc, vae_adapter)


def build_demand_estimator(device, latent_channels: int = 4,
                           **kwargs) -> DemandEstimator:
    """Build the learned Diffusion Demand Map estimator."""
    return DemandEstimator(latent_channels=latent_channels, **kwargs).to(device)


def build_demand_condition_model(unet: UNet2DConditionModel, device,
                                 **kwargs) -> DemandConditionModel:
    """Build the Stage-3 demand-gated Condition model."""
    return DemandConditionModel(unet, **kwargs).to(device)


def build_control_condition_model(unet: UNet2DConditionModel, device,
                                  **kwargs) -> DemandConditionModel:
    """Build the Stage-2 z_r-conditioned model without Demand gates."""
    if "gate_mode" in kwargs and kwargs["gate_mode"] != "none":
        raise ValueError("plain Control model requires gate_mode='none'")
    kwargs["gate_mode"] = "none"
    return DemandConditionModel(unet, **kwargs).to(device)


def encode_prompt(text_encoder, tokenizer, prompts, device, dtype):
    """Encode a list of prompt strings into text embeddings.

    SD 1.x returns CLIP ViT-L/14 embeddings ``[B,77,768]`` (last hidden state);
    SD 2.x returns OpenCLIP ViT-H/14 embeddings ``[B,77,1024]``.  Its
    penultimate hidden state must pass through CLIP's final LayerNorm before it
    is used for U-Net cross-attention.
    """
    tokens = tokenizer(
        list(prompts), padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    with torch.no_grad():
        hidden_state_index = getattr(
            text_encoder, "_prompt_hidden_state_index", None
        )
        if hidden_state_index is None:
            embeds = text_encoder(input_ids)[0]
        else:
            output = text_encoder(input_ids, output_hidden_states=True)
            embeds = output.hidden_states[hidden_state_index]
            text_model = getattr(text_encoder, "text_model", text_encoder)
            embeds = text_model.final_layer_norm(embeds)
        embeds = embeds.to(dtype)
    return embeds


def apply_classifier_free_guidance(unet, sample, timestep, prompt_embeds,
                                   uncond_embeds, guidance_scale,
                                   guidance_rescale=0.0, **unet_kwargs):
    """Classifier-free guidance (CFG) UNet forward.

    Combines conditional and unconditional epsilon predictions:

        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

    * ``guidance_scale == 1.0`` or ``uncond_embeds is None`` -> plain
      conditional forward (backward-compatible, no extra UNet call).
    * otherwise the batch is doubled (uncond || cond) so the UNet runs once.

    Extra kwargs (e.g. ``down_block_additional_residuals`` /
    ``mid_block_additional_residual`` from a ControlNet) are applied to BOTH
    branches identically, matching the standard diffusers ControlNet CFG scheme.
    """
    if uncond_embeds is None or guidance_scale == 1.0:
        return unet(sample, timestep, encoder_hidden_states=prompt_embeds,
                    **unet_kwargs).sample

    b = sample.shape[0]
    sample_cat = torch.cat([sample, sample], dim=0)

    if torch.is_tensor(timestep):
        ts = timestep.unsqueeze(0) if timestep.dim() == 0 else timestep
        ts_cat = torch.cat([ts, ts], dim=0)
    else:
        ts_cat = torch.tensor([timestep] * (2 * b), device=sample.device)

    emb_cat = torch.cat([uncond_embeds, prompt_embeds], dim=0)

    kw = {}
    for k, v in unet_kwargs.items():
        if isinstance(v, tuple):
            kw[k] = tuple(torch.cat([x, x], dim=0) if torch.is_tensor(x) else x
                          for x in v)
        elif torch.is_tensor(v):
            kw[k] = torch.cat([v, v], dim=0)
        else:
            kw[k] = v

    eps_cat = unet(sample_cat, ts_cat, encoder_hidden_states=emb_cat, **kw).sample
    eps_uncond, eps_cond = eps_cat.chunk(2, dim=0)
    eps_cfg = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

    # Very large CFG can inflate the prediction variance and produce saturated
    # or chaotic images.  Guidance rescale (Common Diffusion Noise Schedules
    # are Flawed, section 3.4) restores the conditional prediction's standard
    # deviation while retaining a configurable share of ordinary CFG.
    rescale = float(max(0.0, min(1.0, guidance_rescale)))
    if rescale > 0.0:
        dimensions = tuple(range(1, eps_cfg.ndim))
        std_cond = eps_cond.std(dim=dimensions, keepdim=True)
        std_cfg = eps_cfg.std(dim=dimensions, keepdim=True).clamp_min(1e-6)
        eps_rescaled = eps_cfg * (std_cond / std_cfg)
        eps_cfg = rescale * eps_rescaled + (1.0 - rescale) * eps_cfg
    return eps_cfg


def build_blip_captioner(device, model_id: str = BLIP_MODEL_ID):
    """Load BLIP image captioner (processor + model) offline from the HF cache."""
    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = BlipProcessor.from_pretrained(model_id, local_files_only=True)
    blip = BlipForConditionalGeneration.from_pretrained(
        model_id, local_files_only=True
    )
    blip.requires_grad_(False)
    return processor, blip.eval().to(device)


def caption_image(processor, blip, image, device, max_length: int = 50):
    """Turn a [0,1] RGB image tensor [3,H,W] into a text caption with BLIP."""
    from PIL import Image

    if image.dim() == 4:
        image = image[0]
    pil = Image.fromarray(
        (image.clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
    )
    inputs = processor(images=pil, return_tensors="pt").to(device)
    with torch.no_grad():
        out = blip.generate(**inputs, max_length=max_length, num_beams=5)
    return processor.decode(out[0], skip_special_tokens=True).strip()


