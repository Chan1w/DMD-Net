"""Evaluation helpers shared by the final Stage-3 test script."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import torch

from data.paired_channel_dataset import ImageFolderDataset, VOCImageDataset
from metrics import psnr, ssim


METHODS = ("djscc", "sd21_base", "full_control", "joint_gated")


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


def run_restorer(restorer, z_r, demand, csi, prompt_embeds,
                 uncond_embeds, args, seed):
    seed_diffusion(seed)
    return restorer.restore(
        z_r, demand, csi, prompt_embeds,
        start_rel=args.start_rel,
        steps=args.steps,
        uncond_embeds=uncond_embeds,
    )


def image_metrics(output, target, lpips_metric):
    values = {
        "psnr": psnr(output, target),
        "ssim": ssim(output, target),
    }
    values["lpips"] = (
        lpips_metric(output, target) if lpips_metric is not None else None
    )
    return values


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

    by_snr = {
        f"{snr:g}": {
            method: summarize(grouped[(snr, method)]) for method in METHODS
        }
        for snr in sorted({record["snr_db"] for record in records})
    }
    overall = {
        method: summarize([r for r in records if r["method"] == method])
        for method in METHODS
    }
    improvements = {}
    for reference in ("djscc", "sd21_base", "full_control"):
        target = overall["joint_gated"]
        base = overall[reference]
        improvements[f"joint_gated_vs_{reference}"] = {
            "psnr_delta_db": target["psnr"] - base["psnr"],
            "ssim_delta": target["ssim"] - base["ssim"],
            "lpips_delta": (
                target["lpips"] - base["lpips"]
                if target["lpips"] is not None else None
            ),
        }
    return overall, by_snr, improvements
