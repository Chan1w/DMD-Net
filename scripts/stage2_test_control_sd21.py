"""Evaluate the best Stage-2 SD2.1 Condition checkpoint without D gating.

For every image/SNR pair, the script compares:

* direct DJSCC reconstruction;
* frozen SD2.1 img2img without Control;
* the trained full-strength Control model.

The two diffusion paths use the same random seed and denoising configuration.
Outputs include PSNR/SSIM/LPIPS summaries, per-sample JSON/CSV records, and
visual comparison panels.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import textwrap
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SD21_MODEL_ID
from data.paired_channel_dataset import (
    ImageFolderDataset,
    VOCImageDataset,
    from_vae_output,
    to_vae_input,
)
from metrics import LPIPSMetric, psnr, ssim
from models.builder import (
    build_blip_captioner,
    build_control_condition_model,
    build_djscc_adapter,
    caption_image,
    encode_prompt,
    load_scheduler,
    load_text_encoder,
    load_tokenizer,
    load_unet,
)
from models.restorer import ControlRestorer
from utils import set_seed


DEFAULT_CONTROL = (
    "scripts/checkpoints/"
    "stage2_control_SD21_xbase/control_best.pth"
)
DEFAULT_DJSCC = (
    "scripts/checkpoints/DJSCC_SD21/best.pth"
)
DEFAULT_DATA = "/home/wangch/projects/Dataset/Kodak24"
DEFAULT_OUTPUT = (
    "outputs/test_stage2_control_SD21_xbase"
)
METHODS = ("djscc", "sd21_base", "trained_control")
PROMPT_EMBEDDING_VERSION = "sd21_penultimate_final_layer_norm_v1"
EXPECTED_STAGE = "stage2_condition_without_demand"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test the best SD2.1 Control checkpoint without D"
    )
    parser.add_argument("--control-checkpoint", default=DEFAULT_CONTROL)
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
    parser.add_argument("--snr-values", default="-5,0,5,10")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--start-rel", type=float, default=1)
    parser.add_argument("--guidance-scale", type=float, default=1)
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
    parser.add_argument("--num-visualizations", type=int, default=12)
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


def parse_snr_values(text):
    try:
        values = [float(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as error:
        raise SystemExit(f"invalid --snr-values: {text!r}") from error
    if not values:
        raise SystemExit("--snr-values must contain at least one number")
    return values


def build_dataset(args):
    if args.voc_root:
        dataset = VOCImageDataset(
            args.voc_root, split=args.split, image_size=args.image_size
        )
    else:
        dataset = ImageFolderDataset(args.data_dir, image_size=args.image_size)
    if args.max_samples > 0:
        dataset.ids = dataset.ids[:args.max_samples]
        if isinstance(dataset, ImageFolderDataset):
            dataset.files = dataset.files[:args.max_samples]
    if len(dataset) == 0:
        raise SystemExit("the selected test dataset contains no images")
    return dataset


def atomic_json(data, path):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def save_csv(records, path):
    temporary = path + ".tmp"
    fieldnames = list(records[0].keys()) if records else []
    with open(temporary, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, path)


def seed_diffusion(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_restorer(restorer, z_r, csi, prompt_embeds,
                 uncond_embeds, args, seed):
    seed_diffusion(seed)
    return restorer.restore(
        z_r,
        csi,
        prompt_embeds,
        start_rel=args.start_rel,
        steps=args.steps,
        uncond_embeds=uncond_embeds,
    )


def image_metrics(output, target, lpips_metric):
    return {
        "psnr": psnr(output, target),
        "ssim": ssim(output, target),
        "lpips": (
            lpips_metric(output, target) if lpips_metric is not None else None
        ),
    }


def aggregate(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["snr_db"], record["method"])].append(record)

    def summarize(items):
        result = {"num_samples": len(items)}
        for metric in ("psnr", "ssim", "lpips"):
            values = [item[metric] for item in items if item[metric] is not None]
            result[metric] = sum(values) / len(values) if values else None
        return result

    snr_values = sorted({record["snr_db"] for record in records})
    by_snr = {
        f"{snr:g}": {
            method: summarize(grouped[(snr, method)]) for method in METHODS
        }
        for snr in snr_values
    }
    overall = {
        method: summarize([r for r in records if r["method"] == method])
        for method in METHODS
    }
    control = overall["trained_control"]
    improvements = {}
    for reference in ("djscc", "sd21_base"):
        baseline = overall[reference]
        improvements[f"trained_control_vs_{reference}"] = {
            "psnr_delta_db": control["psnr"] - baseline["psnr"],
            "ssim_delta": control["ssim"] - baseline["ssim"],
            "lpips_delta": (
                control["lpips"] - baseline["lpips"]
                if control["lpips"] is not None else None
            ),
        }
    return overall, by_snr, improvements


def save_visualization(
    image_id,
    snr_value,
    target,
    outputs,
    metrics,
    original_prompt,
    receiver_prompt,
    prompt_source,
    path,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "djscc": "DJSCC",
        "sd21_base": "SD2.1 base",
        "trained_control": "Trained Control",
    }
    figure, axes = plt.subplots(1, 4, figsize=(16, 5.2))
    target_image = target[0].detach().float().cpu().permute(1, 2, 0).numpy()
    axes[0].imshow(target_image)
    axes[0].set_title("Ground truth")
    for axis_index, method in enumerate(METHODS, start=1):
        image = outputs[method][0].detach().float().cpu()
        axes[axis_index].imshow(image.permute(1, 2, 0).numpy())
        item = metrics[method]
        lpips_text = (
            "" if item["lpips"] is None else f"\nLPIPS {item['lpips']:.3f}"
        )
        axes[axis_index].set_title(
            f"{labels[method]}\nPSNR {item['psnr']:.2f}{lpips_text}"
        )
    for axis in axes:
        axis.axis("off")
    selected_original = " [USED]" if prompt_source == "original" else ""
    selected_receiver = " [USED]" if prompt_source == "receiver" else ""
    original_text = textwrap.fill(
        f"Original prompt{selected_original}: {original_prompt or '<empty>'}",
        width=120,
    )
    receiver_text = textwrap.fill(
        f"x_base prompt{selected_receiver}: {receiver_prompt or '<empty>'}",
        width=120,
    )
    figure.suptitle(f"{image_id} | SNR={snr_value:g} dB", y=0.99)
    figure.text(0.5, 0.925, original_text, ha="center", va="top", fontsize=9)
    figure.text(0.5, 0.855, receiver_text, ha="center", va="top", fontsize=9)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.78))
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    validate_args(args)
    snr_values = parse_snr_values(args.snr_values)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    visualization_dir = os.path.join(args.out_dir, "visualizations")
    os.makedirs(visualization_dir, exist_ok=True)

    print(f"[init] loading Control checkpoint: {args.control_checkpoint}")
    checkpoint = torch.load(args.control_checkpoint, map_location="cpu")
    if checkpoint.get("stage") != EXPECTED_STAGE:
        raise RuntimeError(
            f"expected stage {EXPECTED_STAGE!r}, got {checkpoint.get('stage')!r}"
        )
    if checkpoint.get("demand_enabled") is not False:
        raise RuntimeError("checkpoint is not a demand-free Control checkpoint")
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

    condition = build_control_condition_model(unet, device, cond_channels=4)
    checkpoint_architecture = checkpoint.get("condition_architecture_version")
    if checkpoint_architecture != condition.architecture_version:
        raise RuntimeError(
            f"checkpoint architecture {checkpoint_architecture!r} does not "
            f"match runtime {condition.architecture_version!r}"
        )
    condition.load_state_dict(checkpoint["control_model"], strict=True)
    condition.requires_grad_(False).eval()

    base_restorer = ControlRestorer(
        unet,
        scheduler,
        vae,
        condition=None,
        guidance_scale=args.guidance_scale,
        guidance_rescale=args.guidance_rescale,
    )
    control_restorer = ControlRestorer(
        unet,
        scheduler,
        vae,
        condition=condition,
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
                diffusion_seed = (
                    args.seed + image_index * len(snr_values) + snr_index
                )

                base_output = run_restorer(
                    base_restorer, z_r, observation.csi, prompt_embeds,
                    uncond_embeds, args, diffusion_seed,
                )
                control_output = run_restorer(
                    control_restorer, z_r, observation.csi, prompt_embeds,
                    uncond_embeds, args, diffusion_seed,
                )
                outputs = {
                    "djscc": from_vae_output(observation.x_base),
                    "sd21_base": from_vae_output(base_output),
                    "trained_control": from_vae_output(control_output),
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
                            "prompt": prompt,
                            "prompt_source": args.prompt_source,
                            "original_prompt": original_prompt,
                            "receiver_prompt": receiver_prompt,
                        }
                    )

                control_metric = metric_values["trained_control"]
                lpips_text = (
                    "" if control_metric["lpips"] is None
                    else f" LPIPS={control_metric['lpips']:.4f}"
                )
                progress.set_postfix_str(
                    f"{image_id} snr={snr_value:g} "
                    f"control_PSNR={control_metric['psnr']:.2f}{lpips_text}"
                )
                if visualized < args.num_visualizations:
                    safe_snr = (
                        f"{snr_value:g}".replace("-", "m").replace(".", "p")
                    )
                    save_visualization(
                        image_id,
                        snr_value,
                        target,
                        outputs,
                        metric_values,
                        original_prompt,
                        receiver_prompt,
                        args.prompt_source,
                        os.path.join(
                            visualization_dir, f"{image_id}_snr{safe_snr}.png"
                        ),
                    )
                    visualized += 1
                progress.update(1)
    progress.close()

    overall, by_snr, improvements = aggregate(records)
    summary = {
        "control_checkpoint": args.control_checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_loss": checkpoint.get("best_loss"),
        "condition_architecture_version": condition.architecture_version,
        "control_condition_inputs": checkpoint.get("control_condition_inputs"),
        "demand_enabled": False,
        "runtime_prompt_embedding": PROMPT_EMBEDDING_VERSION,
        "checkpoint_prompt_embedding": checkpoint_prompt_embedding,
        "loss_definition": checkpoint.get("loss_definition"),
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
            f"  {method:15s} PSNR={values['psnr']:.3f} "
            f"SSIM={values['ssim']:.4f} LPIPS={lpips_text}"
        )
    for comparison, delta in improvements.items():
        print(
            f"[delta] {comparison}: PSNR={delta['psnr_delta_db']:+.3f} dB "
            f"SSIM={delta['ssim_delta']:+.4f} "
            f"LPIPS={delta['lpips_delta']}"
        )
    print(f"[done] reports and visualizations saved to {args.out_dir}")


if __name__ == "__main__":
    main()

