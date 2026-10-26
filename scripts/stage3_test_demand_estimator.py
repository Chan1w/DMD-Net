"""Evaluate Stage-3 DemandEstimator and learned D-gated Control skips.

The four paths share the same channel observation and diffusion seed:

* DJSCC direct reconstruction;
* frozen SD2.1 img2img without Control;
* full pretrained Control with both D gates bypassed;
* Stage-3 predicted D in residual gates and per-step x0 guidance.

Reports contain PSNR, SSIM, LPIPS, demand statistics, per-SNR aggregates, and
visual comparison panels.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from config import DDM_DEFAULTS, SD21_MODEL_ID
from data.paired_channel_dataset import from_vae_output, to_vae_input
from metrics import LPIPSMetric
from models.builder import (
    build_blip_captioner, build_demand_condition_model,
    build_demand_estimator, build_djscc_adapter, caption_image, encode_prompt,
    load_scheduler, load_text_encoder, load_tokenizer, load_unet,
)
from models.restorer import ControlRestorer, DemandRestorer
from evaluation_utils import (
    METHODS, aggregate, atomic_json, build_dataset, image_metrics,
    parse_snr_values, run_restorer, save_csv, seed_diffusion,
)
from utils import set_seed


DEFAULT_STAGE3 = (
    "scripts/checkpoints/"
    "stage3_demand_control_SD21_xbase/best.pth"
)
DEFAULT_DJSCC = (
    "scripts/checkpoints/DJSCC_SD21/best.pth"
)
# DEFAULT_DATA = "/home/wangch/projects/Dataset/Kodak24"
DEFAULT_DATA = "/home/wangch/projects/Dataset/picture"
DEFAULT_OUTPUT = (
    "outputs/test_stage3_demand_control_SD21_xbase"
)
EXPECTED_STAGE = "stage3_demand_skip_finetuning"
PROMPT_EMBEDDING_VERSION = "sd21_penultimate_final_layer_norm_v1"
GATE_RULE = (
    "residual: frozen stage2_zero_conv(H) + zero-initialized "
    "D_skip_correction(H,D); x0: z_r + D*(x0_conditioned-z_r)"
)


def save_visualization(
    image_id,
    snr_value,
    target,
    outputs,
    demand,
    metrics,
    original_prompt,
    receiver_prompt,
    prompt_source,
    path,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = {
        "djscc": "DJSCC",
        "sd21_base": "SD2.1 base",
        "full_control": "Full Control",
        "joint_gated": "Joint gated",
    }
    figure, axes = plt.subplots(1, 6, figsize=(22, 5.2))
    target_image = target[0].detach().float().cpu().permute(1, 2, 0).numpy()
    axes[0].imshow(target_image)
    axes[0].set_title("Ground truth")
    for index, method in enumerate(METHODS, start=1):
        shown = outputs[method][0].detach().float().cpu()
        axes[index].imshow(shown.permute(1, 2, 0).numpy())
        item = metrics[method]
        lpips_text = (
            "" if item["lpips"] is None else f"\nLPIPS {item['lpips']:.3f}"
        )
        axes[index].set_title(
            f"{names[method]}\nPSNR {item['psnr']:.2f}{lpips_text}"
        )
    demand_image = demand[0, 0].detach().float().cpu().numpy()
    demand_axis = axes[5].imshow(
        demand_image, cmap="inferno", vmin=0.0, vmax=1.0
    )
    axes[5].set_title(f"Predicted D\nmean {demand.mean().item():.3f}")
    figure.colorbar(demand_axis, ax=axes[5], fraction=0.046, pad=0.04)
    for axis in axes:
        axis.axis("off")

    selected_original = " [USED]" if prompt_source == "original" else ""
    selected_receiver = " [USED]" if prompt_source == "receiver" else ""
    original_text = textwrap.fill(
        f"Original prompt{selected_original}: {original_prompt or '<empty>'}",
        width=150,
    )
    receiver_text = textwrap.fill(
        f"x_base prompt{selected_receiver}: {receiver_prompt or '<empty>'}",
        width=150,
    )
    figure.suptitle(f"{image_id} | SNR={snr_value:g} dB", y=0.99)
    figure.text(0.5, 0.925, original_text, ha="center", va="top", fontsize=9)
    figure.text(0.5, 0.855, receiver_text, ha="center", va="top", fontsize=9)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.78))
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test Stage-3 demand and learned skip-gate fine-tuning"
    )
    parser.add_argument("--stage3-checkpoint", default=DEFAULT_STAGE3)
    parser.add_argument("--djscc-checkpoint", default=DEFAULT_DJSCC)
    parser.add_argument("--data-dir", default=DEFAULT_DATA)
    parser.add_argument(
        "--voc-root", default="",
        help="optional VOC root; takes precedence over --data-dir",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--out-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--snr-values", default="-10,-5,0,5,10")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--start-rel", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--guidance-rescale", type=float, default=0.0)
    parser.add_argument(
        "--prompt-mode", choices=("blip", "empty"), default="blip"
    )
    parser.add_argument(
        "--prompt-source",
        choices=("original", "receiver"),
        default="receiver",
        help=(
            "image used by BLIP to generate the prompt: original uses the "
            "ground-truth image; receiver uses the DJSCC x_base at each SNR"
        ),
    )
    parser.add_argument("--num-visualizations", type=int, default=120)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def validate_args(args):
    if args.steps < 1:
        raise SystemExit("--steps must be at least 1")
    if not 0.0 <= args.start_rel <= 1.0:
        raise SystemExit("--start-rel must be in [0, 1]")
    if args.guidance_scale < 1.0:
        raise SystemExit("--guidance-scale must be at least 1")
    if not 0.0 <= args.guidance_rescale <= 1.0:
        raise SystemExit("--guidance-rescale must be in [0, 1]")


def main():
    args = parse_args()
    validate_args(args)
    snr_values = parse_snr_values(args.snr_values)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    visualization_dir = os.path.join(args.out_dir, "visualizations")
    os.makedirs(visualization_dir, exist_ok=True)

    print(f"[init] loading Stage-3 checkpoint: {args.stage3_checkpoint}")
    checkpoint = torch.load(args.stage3_checkpoint, map_location="cpu")
    if checkpoint.get("stage") != EXPECTED_STAGE:
        raise RuntimeError(
            f"expected checkpoint stage {EXPECTED_STAGE!r}, got "
            f"{checkpoint.get('stage')!r}"
        )
    if checkpoint.get("gate_rule") != GATE_RULE:
        raise RuntimeError(
            f"checkpoint gate rule {checkpoint.get('gate_rule')!r} does not "
            f"match runtime {GATE_RULE!r}"
        )
    if checkpoint.get("estimator_inputs") != ["z_r", "snr"]:
        raise RuntimeError(
            f"unexpected estimator inputs: {checkpoint.get('estimator_inputs')!r}"
        )
    checkpoint_prompt_embedding = checkpoint.get("prompt_embedding")
    if checkpoint_prompt_embedding != PROMPT_EMBEDDING_VERSION:
        print("[warn] checkpoint prompt-embedding version differs from runtime")

    djscc = build_djscc_adapter(
        args.djscc_checkpoint, device, vae_model_id=SD21_MODEL_ID
    )
    vae = djscc.vae
    unet = load_unet(device, torch.float32, model_id=SD21_MODEL_ID)
    unet.requires_grad_(False).eval()
    try:
        scheduler = load_scheduler(model_id=SD21_MODEL_ID)
    except OSError:
        scheduler = load_scheduler()
    text_encoder = load_text_encoder(
        device, torch.float32, model_id=SD21_MODEL_ID
    )
    tokenizer = load_tokenizer(model_id=SD21_MODEL_ID)

    with torch.no_grad():
        probe = torch.zeros(1, 3, args.image_size, args.image_size, device=device)
        probe_observation = djscc.transmit(
            probe, torch.tensor([[10.0]], device=device)
        )
    estimator = build_demand_estimator(
        device,
        latent_channels=probe_observation.z_hat_raw.shape[1],
        **DDM_DEFAULTS,
    )
    estimator_architecture = checkpoint.get("architecture_version")
    if estimator_architecture != estimator.architecture_version:
        raise RuntimeError(
            f"estimator architecture {estimator_architecture!r} does not "
            f"match runtime {estimator.architecture_version!r}"
        )
    estimator.load_state_dict(checkpoint["demand_estimator"], strict=True)
    estimator.requires_grad_(False).eval()

    checkpoint_args = checkpoint.get("args", {})
    condition = build_demand_condition_model(
        unet,
        device,
        cond_channels=4,
        gate_mode="learned",
        gate_init=float(
            checkpoint_args.get("internal_gate_init", 0.9)
        ),
    )
    condition_architecture = checkpoint.get("condition_architecture_version")
    if condition_architecture != condition.architecture_version:
        raise RuntimeError(
            f"condition architecture {condition_architecture!r} does not "
            f"match runtime {condition.architecture_version!r}"
        )
    condition.load_state_dict(checkpoint["demand_condition"], strict=True)
    condition.requires_grad_(False).eval()

    base_restorer = DemandRestorer(
        unet, scheduler, vae, condition=None,
        guidance_scale=args.guidance_scale,
        guidance_rescale=args.guidance_rescale,
    )
    gated_restorer = DemandRestorer(
        unet, scheduler, vae, condition=condition,
        guidance_scale=args.guidance_scale,
        guidance_rescale=args.guidance_rescale,
    )
    full_control_restorer = ControlRestorer(
        unet, scheduler, vae, condition=condition,
        guidance_scale=args.guidance_scale,
        guidance_rescale=args.guidance_rescale,
    )

    blip_processor = blip = None
    if args.prompt_mode == "blip":
        blip_processor, blip = build_blip_captioner(device)
    lpips_metric = None if args.no_lpips else LPIPSMetric(device)
    dataset = build_dataset(args)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"[data] {len(dataset)} images x {len(snr_values)} SNR values")

    records = []
    visualized = 0
    gate_sum = [0.0] * len(condition.adapters)
    gate_observations = 0
    total = len(dataset) * len(snr_values)
    progress = tqdm(total=total, desc="Evaluating", dynamic_ncols=True)
    with torch.inference_mode():
        for image_index, batch in enumerate(loader):
            image = batch["image"].to(device, non_blocking=True)
            image_id = batch["id"][0]
            original_prompt = (
                caption_image(blip_processor, blip, image[0], device)
                if args.prompt_mode == "blip"
                else ""
            )
            original_prompt_embeds = encode_prompt(
                text_encoder,
                tokenizer,
                [original_prompt],
                device,
                torch.float32,
            )
            uncond_embeds = None
            if args.guidance_scale != 1.0:
                uncond_embeds = encode_prompt(
                    text_encoder, tokenizer, [""], device, torch.float32
                )
            x_gt = to_vae_input(image)
            target = from_vae_output(x_gt)

            for snr_index, snr_value in enumerate(snr_values):
                snr = torch.tensor([[snr_value]], device=device)
                observation = djscc.transmit(x_gt, snr)
                z_r = vae.scale_raw(observation.z_hat_raw)
                receiver_prompt = (
                    caption_image(
                        blip_processor,
                        blip,
                        from_vae_output(observation.x_base)[0],
                        device,
                    )
                    if args.prompt_mode == "blip"
                    else ""
                )
                if args.prompt_mode == "empty":
                    prompt = ""
                    prompt_embeds = original_prompt_embeds
                elif args.prompt_source == "receiver":
                    prompt = receiver_prompt
                    prompt_embeds = encode_prompt(
                        text_encoder,
                        tokenizer,
                        [prompt],
                        device,
                        torch.float32,
                    )
                else:
                    prompt = original_prompt
                    prompt_embeds = original_prompt_embeds
                demand = estimator(z_r, observation.csi).demand
                diffusion_seed = (
                    args.seed + image_index * len(snr_values) + snr_index
                )

                base_output = run_restorer(
                    base_restorer, z_r, None, observation.csi,
                    prompt_embeds, uncond_embeds, args, diffusion_seed,
                )
                seed_diffusion(diffusion_seed)
                full_output = full_control_restorer.restore(
                    z_r, observation.csi, prompt_embeds,
                    start_rel=args.start_rel, steps=args.steps,
                    uncond_embeds=uncond_embeds,
                )
                gated_output = run_restorer(
                    gated_restorer, z_r, demand, observation.csi,
                    prompt_embeds, uncond_embeds, args, diffusion_seed,
                )
                scale_gate_means = [
                    float(adapter.gate.last_gate_mean.item())
                    for adapter in condition.adapters
                ]
                for index, value in enumerate(scale_gate_means):
                    gate_sum[index] += value
                gate_observations += 1
                internal_gate_mean = sum(scale_gate_means) / len(
                    scale_gate_means
                )
                outputs = {
                    "djscc": from_vae_output(observation.x_base),
                    "sd21_base": from_vae_output(base_output),
                    "full_control": from_vae_output(full_output),
                    "joint_gated": from_vae_output(gated_output),
                }
                metric_values = {
                    method: image_metrics(output, target, lpips_metric)
                    for method, output in outputs.items()
                }
                for method in METHODS:
                    records.append(
                        {
                            "sample_id": image_id,
                            "snr_db": snr_value,
                            "method": method,
                            **metric_values[method],
                            "demand_mean": float(demand.mean().item()),
                            "demand_min": float(demand.min().item()),
                            "demand_max": float(demand.max().item()),
                            "internal_gate_mean": internal_gate_mean,
                            "prompt": prompt,
                            "prompt_source": args.prompt_source,
                            "original_prompt": original_prompt,
                            "receiver_prompt": receiver_prompt,
                        }
                    )

                gated_metric = metric_values["joint_gated"]
                lpips_text = (
                    "" if gated_metric["lpips"] is None
                    else f" LPIPS={gated_metric['lpips']:.4f}"
                )
                progress.set_postfix_str(
                    f"{image_id} snr={snr_value:g} "
                    f"gated_PSNR={gated_metric['psnr']:.2f}{lpips_text}"
                )
                if visualized < args.num_visualizations:
                    safe_snr = (
                        f"{snr_value:g}".replace("-", "m").replace(".", "p")
                    )
                    save_visualization(
                        image_id, snr_value, target, outputs, demand,
                        metric_values,
                        original_prompt, receiver_prompt, args.prompt_source,
                        os.path.join(
                            visualization_dir, f"{image_id}_snr{safe_snr}.png"
                        ),
                    )
                    visualized += 1
                progress.update(1)
    progress.close()

    overall, by_snr, improvements = aggregate(records)
    gate_means_by_scale = [
        value / max(gate_observations, 1) for value in gate_sum
    ]
    summary = {
        "stage3_checkpoint": args.stage3_checkpoint,
        "checkpoint_stage": checkpoint.get("stage"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_loss": checkpoint.get("best_loss"),
        "architecture_version": estimator.architecture_version,
        "condition_architecture_version": condition.architecture_version,
        "estimator_inputs": ["z_r", "snr"],
        "control_condition_inputs": ["z_r"],
        "gate_rule": GATE_RULE,
        "learned_gate_means_by_scale": gate_means_by_scale,
        "learned_gate_mean": (
            sum(gate_means_by_scale) / len(gate_means_by_scale)
        ),
        "loss_definition": checkpoint.get("loss_definition"),
        "loss_weights": checkpoint.get("loss_weights"),
        "runtime_prompt_embedding": PROMPT_EMBEDDING_VERSION,
        "checkpoint_prompt_embedding": checkpoint_prompt_embedding,
        "snr_values_db": snr_values,
        "overall": overall,
        "by_snr": by_snr,
        "improvements": improvements,
        "args": vars(args),
    }
    atomic_json(summary, os.path.join(args.out_dir, "summary.json"))
    atomic_json(records, os.path.join(args.out_dir, "per_sample_metrics.json"))
    save_csv(records, os.path.join(args.out_dir, "per_sample_metrics.csv"))

    print("\n[result] overall averages")
    for method in METHODS:
        values = overall[method]
        lpips_text = (
            "disabled" if values["lpips"] is None
            else f"{values['lpips']:.4f}"
        )
        print(
            f"  {method:12s} PSNR={values['psnr']:.3f} "
            f"SSIM={values['ssim']:.4f} LPIPS={lpips_text}"
        )
    for comparison, delta in improvements.items():
        print(
            f"[delta] {comparison}: PSNR={delta['psnr_delta_db']:+.3f} dB "
            f"SSIM={delta['ssim_delta']:+.4f} "
            f"LPIPS={delta['lpips_delta']}"
        )
    print(
        f"[gate] learned activation mean={summary['learned_gate_mean']:.4f} "
        f"min-scale={min(gate_means_by_scale):.4f} "
        f"max-scale={max(gate_means_by_scale):.4f}"
    )
    print(f"[done] reports and visualizations saved to {args.out_dir}")


if __name__ == "__main__":
    main()



