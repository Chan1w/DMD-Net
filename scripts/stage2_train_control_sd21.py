"""Stage 2: train the SD-2.1 Condition Model without DemandEstimator or D.

The Condition encoder receives the DJSCC recovered latent ``z_r`` and channel
state information. Each multi-scale feature is sent directly through the
original zero-conv and injected into the frozen SD-2.1 U-Net. There is no
per-skip ResBlock, z_r projection, D projection or gate. The only objective is
the standard diffusion target loss::

    L_control = MSE(model_prediction, diffusion_target)

For epsilon-prediction schedulers the target is the sampled noise.  For
v-prediction schedulers it is the corresponding velocity target.  Demand gates
are absent from every skip, so this checkpoint initializes Stage 3. DJSCC,
VAE, text encoder and U-Net are frozen; only the Condition Model is trained.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SD21_MODEL_ID
from data.paired_channel_dataset import (
    VOCImageDataset,
    from_vae_output,
    to_vae_input,
)
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
from utils import AverageMeter, set_seed
from utils_distributed import (
    cleanup_distributed, distributed_barrier, launch_training, parse_gpu_ids,
    reduce_meters, setup_distributed, unwrap_ddp, wrap_ddp,
)


PROMPT_EMBEDDING_VERSION = "sd21_penultimate_final_layer_norm_v1"
TRAINING_STAGE = "stage2_condition_without_demand"
LOSS_DEFINITION = "mse(model_prediction, diffusion_target)"
CURVE_UPDATE_INTERVAL = 200


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the SD2.1 Control model without DemandEstimator or D"
    )
    parser.add_argument(
        "--data-root", default="/home/wangch/projects/Dataset/VOC2012"
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--djscc-checkpoint",
        default=(
            "scripts/checkpoints/"
            "DJSCC_SD21/best.pth"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "scripts/checkpoints/"
            "stage2_control_SD21_xbase"
        ),
    )
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--snr-min", type=int, default=-5)
    parser.add_argument("--snr-max", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--prompt-mode", choices=("blip", "empty"), default="blip"
    )
    parser.add_argument(
        "--prompt-source",
        choices=("original", "receiver"),
        default="receiver",
        help=(
            "image used by BLIP to generate the prompt: original uses the "
            "ground-truth image; receiver uses the current DJSCC x_base"
        ),
    )
    parser.add_argument(
        "--freeze-control-encoder",
        action="store_true",
        help="train only condition encoders and original zero-convs",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--gpu-ids", default="0",
        help="physical GPU indices, e.g. 0 for one GPU or 0,1 for two GPUs",
    )
    parser.add_argument(
        "--master-port", type=int, default=29521,
        help="local NCCL rendezvous port used only for multi-GPU training",
    )
    return parser.parse_args()


def validate_args(args):
    if args.epochs < 1:
        raise SystemExit("--epochs must be positive")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.snr_min > args.snr_max:
        raise SystemExit("--snr-min must be no greater than --snr-max")
    if args.learning_rate <= 0:
        raise SystemExit("--learning-rate must be positive")
    if args.max_grad_norm <= 0:
        raise SystemExit("--max-grad-norm must be positive")
    try:
        parse_gpu_ids(args.gpu_ids)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not 1 <= args.master_port <= 65535:
        raise SystemExit("--master-port must be in [1, 65535]")


def diffusion_target(scheduler, z0, noise, timestep):
    prediction_type = getattr(scheduler.config, "prediction_type", "epsilon")
    if prediction_type == "epsilon":
        return noise
    if prediction_type == "v_prediction":
        return scheduler.get_velocity(z0, noise, timestep)
    if prediction_type == "sample":
        return z0
    raise ValueError(f"unsupported scheduler prediction_type: {prediction_type!r}")


def save_checkpoint(path, condition, optimizer, epoch, global_step,
                    best_loss, history, curve_history, args, prediction_type):
    temporary = path + ".tmp"
    torch.save(
        {
            "stage": TRAINING_STAGE,
            "epoch": epoch,
            "global_step": global_step,
            "best_loss": best_loss,
            "control_model": condition.state_dict(),
            "optimizer": optimizer.state_dict(),
            "condition_architecture_version": condition.architecture_version,
            "control_condition_inputs": ["z_r", "snr"],
            "demand_enabled": False,
            "stage_definition": (
                "train Condition Model only; each skip contains only its "
                "original zero-conv and has no D-related module"
            ),
            "skip_structure": "zero_conv_only",
            "residual_gate": "bypassed",
            "prediction_type": prediction_type,
            "prompt_embedding": PROMPT_EMBEDDING_VERSION,
            "loss_definition": LOSS_DEFINITION,
            "history": history,
            "curve_history": curve_history,
            "args": vars(args),
        },
        temporary,
    )
    os.replace(temporary, path)


def save_training_artifacts(out_dir, history, curve_history):
    history_path = os.path.join(out_dir, "training_history.json")
    history_temporary = history_path + ".tmp"
    with open(history_temporary, "w", encoding="utf-8") as file:
        json.dump(
            {
                "epoch_history": history,
                "curve_history": curve_history,
                "curve_update_interval_batches": CURVE_UPDATE_INTERVAL,
            },
            file,
            indent=2,
        )
    os.replace(history_temporary, history_path)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [item["global_step"] for item in curve_history]
    batch_losses = [item["diffusion_loss"] for item in curve_history]
    figure, axis = plt.subplots(figsize=(8, 5))
    if steps:
        axis.plot(
            steps, batch_losses, marker="o", markersize=3,
            label="batch-interval mean",
        )
    if history:
        epoch_steps = [item["global_step"] for item in history]
        epoch_losses = [item["diffusion_loss"] for item in history]
        axis.plot(
            epoch_steps, epoch_losses, marker="s", linewidth=2,
            label="epoch mean",
        )
    axis.set_xlabel("Global batch step")
    axis.set_ylabel("MSE loss")
    axis.set_title("SD2.1 Control Training Without Demand")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    curve_path = os.path.join(out_dir, "convergence_curve.png")
    curve_temporary = curve_path + ".tmp"
    figure.savefig(curve_temporary, format="png", dpi=160)
    plt.close(figure)
    os.replace(curve_temporary, curve_path)


def train_worker(local_rank, world_size, args):
    context = setup_distributed(local_rank, world_size, args)
    device = context.device
    set_seed(args.seed + context.rank)
    if context.is_main:
        os.makedirs(args.out_dir, exist_ok=True)
    distributed_barrier(context)

    if context.is_main:
        print(
            "[init] loading frozen DJSCC and SD2.1 components | "
            f"GPUs={args.gpu_ids} world_size={context.world_size} "
            f"batch_per_gpu={args.batch_size} "
            f"global_batch={args.batch_size * context.world_size}"
        )
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

    blip_processor = blip = None
    if args.prompt_mode == "blip":
        blip_processor, blip = build_blip_captioner(device)

    condition = build_control_condition_model(unet, device, cond_channels=4)
    if args.freeze_control_encoder:
        condition.freeze_encoder()

    trainable_parameters = [
        parameter for parameter in condition.parameters()
        if parameter.requires_grad
    ]
    condition = wrap_ddp(condition, context)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    dataset = VOCImageDataset(args.data_root, args.split, args.image_size)
    if args.max_samples > 0:
        dataset.ids = dataset.ids[:args.max_samples]
    sampler = (
        DistributedSampler(
            dataset, num_replicas=context.world_size, rank=context.rank,
            shuffle=True, seed=args.seed, drop_last=False,
        )
        if context.distributed else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    captions = {}

    def get_prompts(sample_ids, original_images, receiver_images):
        if args.prompt_mode == "empty":
            return [""] * len(sample_ids)
        if args.prompt_source == "receiver":
            return [
                caption_image(blip_processor, blip, image, device)
                for image in receiver_images
            ]
        prompts = []
        for sample_id, image in zip(sample_ids, original_images):
            if sample_id not in captions:
                captions[sample_id] = caption_image(
                    blip_processor, blip, image, device
                )
            prompts.append(captions[sample_id])
        return prompts

    prediction_type = getattr(scheduler.config, "prediction_type", "epsilon")
    num_timesteps = int(scheduler.config.num_train_timesteps)
    global_step = 0
    best_loss = float("inf")
    history = []
    curve_history = []
    curve_meter = AverageMeter()
    if context.is_main:
        print(
            f"[train] samples={len(dataset)} prediction_type={prediction_type} "
            f"trainable_parameters={sum(p.numel() for p in trainable_parameters):,}"
        )

    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        condition.train()
        loss_meter = AverageMeter()
        progress = tqdm(
            loader, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True,
            disable=not context.is_main,
        )

        batch_index = 0
        for batch_index, batch in enumerate(progress, start=1):
            images = batch["image"]
            x_gt = to_vae_input(images.to(device, non_blocking=True))
            batch_size = x_gt.shape[0]
            snr = torch.randint(
                args.snr_min,
                args.snr_max + 1,
                (batch_size, 1),
                device=device,
            ).float()

            with torch.no_grad():
                observation = djscc.transmit(x_gt, snr)
                z_gt = vae.encode_mean_scaled(x_gt)
                z_r = vae.scale_raw(observation.z_hat_raw)
                prompts = get_prompts(
                    batch["id"], images, from_vae_output(observation.x_base)
                )
                prompt_embeds = encode_prompt(
                    text_encoder,
                    tokenizer,
                    prompts,
                    device,
                    torch.float32,
                )

            noise = torch.randn_like(z_gt)
            timestep = torch.randint(
                0, num_timesteps, (batch_size,), device=device
            )
            z_t = scheduler.add_noise(z_gt, noise, timestep)
            target = diffusion_target(
                scheduler, z_gt, noise, timestep
            ).detach()

            # Use forward rather than forward_control so DDP owns the forward
            # call and synchronizes Condition gradients across workers.
            control = condition(
                z_t,
                timestep,
                encoder_hidden_states=prompt_embeds,
                rx_latents=z_r,
                demand_map=None,
                csi=observation.csi,
            )
            model_prediction = unet(
                z_t,
                timestep,
                encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals=control.down_residuals,
                mid_block_additional_residual=control.mid_residual,
            ).sample
            loss = F.mse_loss(model_prediction.float(), target.float())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters, args.max_grad_norm
            )
            optimizer.step()

            loss_value = float(loss.detach().item())
            loss_meter.update(loss_value, batch_size)
            curve_meter.update(loss_value, batch_size)
            global_step += 1
            progress.set_postfix(loss=f"{loss_meter.avg:.5f}")
            if batch_index % CURVE_UPDATE_INTERVAL == 0:
                curve_values = reduce_meters({"loss": curve_meter}, context)
                if context.is_main:
                    curve_history.append(
                        {
                            "global_step": global_step,
                            "epoch": epoch,
                            "batch_in_epoch": batch_index,
                            "diffusion_loss": curve_values["loss"],
                            "window_samples": (
                                curve_meter.count * context.world_size
                            ),
                        }
                    )
                curve_meter = AverageMeter()
                if context.is_main:
                    save_training_artifacts(
                        args.out_dir, history, curve_history
                    )
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        epoch_loss = reduce_meters({"loss": loss_meter}, context)["loss"]
        if context.is_main:
            history.append(
                {
                    "epoch": epoch,
                    "diffusion_loss": epoch_loss,
                    "global_step": global_step,
                }
            )
        # Preserve the final partial interval so the curve reaches the current
        # epoch even when its batch count is not divisible by 200.
        if curve_meter.count > 0:
            curve_values = reduce_meters({"loss": curve_meter}, context)
            if context.is_main:
                curve_history.append(
                    {
                        "global_step": global_step,
                        "epoch": epoch,
                        "batch_in_epoch": batch_index,
                        "diffusion_loss": curve_values["loss"],
                        "window_samples": (
                            curve_meter.count * context.world_size
                        ),
                    }
                )
            curve_meter = AverageMeter()
        is_best = epoch_loss < best_loss
        best_loss = min(best_loss, epoch_loss)

        if context.is_main:
            save_checkpoint(
                os.path.join(args.out_dir, "control_last.pth"),
                unwrap_ddp(condition),
                optimizer,
                epoch,
                global_step,
                best_loss,
                history,
                curve_history,
                args,
                prediction_type,
            )
            if is_best:
                save_checkpoint(
                    os.path.join(args.out_dir, "control_best.pth"),
                    unwrap_ddp(condition),
                    optimizer,
                    epoch,
                    global_step,
                    best_loss,
                    history,
                    curve_history,
                    args,
                    prediction_type,
                )
            save_training_artifacts(args.out_dir, history, curve_history)
            best_status = "updated" if is_best else "kept"
            print(
                f"[epoch {epoch}] diffusion_loss={epoch_loss:.6f} "
                f"best={best_loss:.6f} | last=updated best={best_status} "
                "curve=updated"
            )
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    if context.is_main:
        print(f"[done] Control checkpoints saved in {args.out_dir}")
    cleanup_distributed(context)


def main():
    args = parse_args()
    validate_args(args)
    launch_training(train_worker, args)


if __name__ == "__main__":
    main()

