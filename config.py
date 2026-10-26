"""Final DMD-Net configuration for the SD2.1 Stage-1/2/3 pipeline."""

SD21_MODEL_ID = "LanguageMachines/stable-diffusion-2-1-base"
BLIP_MODEL_ID = "Salesforce/blip-image-captioning-base"

DJSCC_BANDWIDTH = 8
DJSCC_MID_CH = 128
DJSCC_N_BLOCKS = 6
DJSCC_EMB_DIM = 128
VAL_SNRS = [0.0, 5.0, 10.0, 15.0, 20.0]

DDM_DEFAULTS = {
    "base_channels": 64,
    "mid_channels": 96,
    "context_channels": 128,
    "snr_embed_channels": 64,
    "local_blocks": 2,
    "context_blocks": 2,
    "attention_heads": 8,
    "mlp_ratio": 2,
    "dropout": 0.0,
    "out_channels": 1,
}
