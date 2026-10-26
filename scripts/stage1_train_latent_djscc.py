"""Stage 1: train the project's VAE-latent DJSCC codec with a selected SD VAE.

This is the training counterpart of ``models.djscc_adapter``.  The VAE is
frozen and deterministic (posterior mode); only the existing ``JSCCModel`` is
optimized.  Checkpoints record the VAE model id and full codec configuration so
the receiver can automatically load the matching VAE later.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (DJSCC_BANDWIDTH, DJSCC_EMB_DIM, DJSCC_MID_CH,
                    DJSCC_N_BLOCKS, SD21_MODEL_ID, VAL_SNRS)
from data.paired_channel_dataset import VOCImageDataset, to_vae_input
from models.djscc import JSCCModel
from models.vae_adapter import VAEAdapter, load_vae
from utils import AverageMeter, set_seed


class CachedLatentDataset(Dataset):
    def __init__(self, payload: dict):
        self.ids = list(payload["ids"])
        self.latents = payload["latents"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return {"id": self.ids[index], "latent": self.latents[index]}


def parse_args():
    p = argparse.ArgumentParser(description="Train SD-VAE latent DJSCC")
    p.add_argument("--vae-model-id", default=SD21_MODEL_ID)
    p.add_argument("--data-root", default="/home/wangch/projects/Dataset/VOC2012")
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--cache-dir", default="outputs/latent_cache_sd21")
    p.add_argument("--out-dir", default="scripts/checkpoints/DJSCC_SD21")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--cache-batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--snr-min", type=float, default=0.0)
    p.add_argument("--snr-max", type=float, default=20.0)
    p.add_argument("--val-snrs", default=",".join(str(v) for v in VAL_SNRS))
    p.add_argument("--bandwidth", type=int, default=DJSCC_BANDWIDTH)
    p.add_argument("--mid-ch", type=int, default=DJSCC_MID_CH)
    p.add_argument("--n-blocks", type=int, default=DJSCC_N_BLOCKS)
    p.add_argument("--emb-dim", type=int, default=DJSCC_EMB_DIM)
    p.add_argument("--signal-power", type=float, default=1.0)
    p.add_argument("--max-train-samples", type=int, default=-1)
    p.add_argument("--max-val-samples", type=int, default=-1)
    p.add_argument("--resume", default=None,
                   help="path to a checkpoint produced by this script")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def limited(dataset, count):
    return Subset(dataset, range(min(count, len(dataset)))) if count > 0 else dataset


def autocast_context(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def cache_path(args, split, limit):
    safe_model = args.vae_model_id.replace("/", "--")
    suffix = f"_n{limit}" if limit > 0 else ""
    return os.path.join(
        args.cache_dir, f"{safe_model}_{split}_{args.image_size}{suffix}.pt"
    )


@torch.no_grad()
def build_latent_cache(args, split, limit, vae, device):
    path = cache_path(args, split, limit)
    if os.path.isfile(path) and not args.rebuild_cache:
        payload = torch.load(path, map_location="cpu")
        if payload.get("vae_model_id") != args.vae_model_id:
            raise ValueError(f"latent cache VAE mismatch: {path}")
        print(f"[cache] loaded {len(payload['ids'])} {split} latents from {path}")
        return CachedLatentDataset(payload)

    dataset = limited(
        VOCImageDataset(args.data_root, split=split, image_size=args.image_size), limit
    )
    loader = DataLoader(
        dataset, batch_size=args.cache_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda"
    )
    ids, chunks = [], []
    for batch in tqdm(loader, desc=f"cache {split}", dynamic_ncols=True):
        image = to_vae_input(batch["image"].to(device, non_blocking=True))
        chunks.append(vae.encode_mean(image).float().cpu())
        ids.extend(batch["id"])
    if not chunks:
        raise RuntimeError(f"empty {split} split")
    payload = {
        "vae_model_id": args.vae_model_id,
        "image_size": args.image_size,
        "split": split,
        "ids": ids,
        "latents": torch.cat(chunks),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)
    print(f"[cache] saved {len(ids)} {split} latents to {path}")
    return CachedLatentDataset(payload)


def sample_snr(count, low, high, device, dtype):
    if low == high:
        return torch.full((count,), low, device=device, dtype=dtype)
    return torch.empty(count, device=device, dtype=dtype).uniform_(low, high)


@torch.no_grad()
def validate(model, loader, snrs, device, amp_enabled):
    model.eval()
    meters = {snr: AverageMeter() for snr in snrs}
    for batch in tqdm(loader, desc="validate", leave=False, dynamic_ncols=True):
        latent = batch["latent"].to(device, non_blocking=True)
        for snr_value, meter in meters.items():
            snr = torch.full(
                (latent.size(0),), snr_value, device=device, dtype=latent.dtype
            )
            with autocast_context(device, amp_enabled):
                prediction, _ = model(latent, snr)
                loss = F.mse_loss(prediction, latent)
            meter.update(loss.float().item(), latent.size(0))
    return {f"snr_{snr:g}": meter.avg for snr, meter in meters.items()}


def checkpoint_payload(model, optimizer, epoch, train_loss, val_report,
                       args, best_val):
    model_config = {
        "latent_ch": 4,
        "bandwidth": args.bandwidth,
        "mid_ch": args.mid_ch,
        "emb_dim": args.emb_dim,
        "n_blocks": args.n_blocks,
        "signal_power": args.signal_power,
    }
    return {
        "stage": "stage1_djscc",
        "stage_definition": "train DJSCC only; all other modules frozen",
        "model_family": "vae_latent_djscc",
        "vae_model_id": args.vae_model_id,
        "model_config": model_config,
        "image_size": args.image_size,
        "epoch": epoch,
        "loss": train_loss,
        "best_val_loss": best_val,
        "val_report": val_report,
        "snr_range": [args.snr_min, args.snr_max],
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "training_args": vars(args),
    }


def main():
    args = parse_args()
    if args.snr_min > args.snr_max:
        raise SystemExit("--snr-min must be <= --snr-max")
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[vae] loading {args.vae_model_id}")
    vae = VAEAdapter(load_vae(device, torch.float32, args.vae_model_id))
    train_dataset = build_latent_cache(
        args, args.train_split, args.max_train_samples, vae, device
    )
    val_dataset = build_latent_cache(
        args, args.val_split, args.max_val_samples, vae, device
    )
    del vae
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda"
    )
    model = JSCCModel(
        latent_ch=4, bandwidth=args.bandwidth, mid_ch=args.mid_ch,
        emb_dim=args.emb_dim, n_blocks=args.n_blocks,
        signal_power=args.signal_power
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    val_snrs = [float(value) for value in args.val_snrs.split(",") if value.strip()]
    if not val_snrs:
        raise SystemExit("--val-snrs must contain at least one value")

    start_epoch, best_val = 1, float("inf")
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu")
        if saved.get("vae_model_id") != args.vae_model_id:
            raise ValueError("resume checkpoint was trained with a different VAE")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = int(saved["epoch"]) + 1
        best_val = float(saved.get("best_val_loss", best_val))
        print(f"[resume] {args.resume}, next epoch {start_epoch}")

    history_path = os.path.join(args.out_dir, "history.jsonl")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        meter = AverageMeter()
        progress = tqdm(
            train_loader, desc=f"latent DJSCC {epoch}/{args.epochs}",
            dynamic_ncols=True
        )
        for batch in progress:
            latent = batch["latent"].to(device, non_blocking=True)
            snr = sample_snr(
                latent.size(0), args.snr_min, args.snr_max, device, latent.dtype
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp_enabled):
                prediction, _ = model(latent, snr)
                loss = F.mse_loss(prediction, latent)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            meter.update(loss.detach().float().item(), latent.size(0))
            progress.set_postfix(mse=f"{meter.avg:.6f}", snr=f"{snr.mean().item():.1f}")

        val_report = validate(model, val_loader, val_snrs, device, amp_enabled)
        mean_val = sum(val_report.values()) / len(val_report)
        improved = mean_val < best_val
        best_val = min(best_val, mean_val)
        payload = checkpoint_payload(
            model, optimizer, epoch, meter.avg, val_report, args, best_val
        )
        torch.save(payload, os.path.join(args.out_dir, "last.pth"))
        if improved:
            torch.save(payload, os.path.join(args.out_dir, "best.pth"))
        record = {
            "epoch": epoch, "train_mse": meter.avg,
            "mean_val_mse": mean_val, "val_report": val_report,
            "best_val_mse": best_val,
        }
        with open(history_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"[epoch {epoch}] train_MSE={meter.avg:.6f} "
            f"val_MSE={mean_val:.6f} best={best_val:.6f}"
        )
    print(f"[done] checkpoints written to {args.out_dir}")


if __name__ == "__main__":
    main()
