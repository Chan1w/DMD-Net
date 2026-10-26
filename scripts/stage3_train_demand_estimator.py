"""Stage 3: train DemandEstimator and D-affected skip modules only.

There is one Control model and no teacher/student construction. The model is
initialized from ``stage2_train_control_sd21.py`` and all pretrained Control weights
remain frozen. Only DemandEstimator and the per-skip learned D projection,
modulation, gate and zero-initialized correction-conv branches are updated.

D is projected independently at every Control skip and fused with the local
Control feature to predict a spatial/channel gate and FiLM modulation. Their
result enters a new zero-initialized correction conv and is added in parallel
to the frozen Stage-2 residual. D also controls
``x0=z_r+D(x0_conditioned-z_r)`` for per-step guidance. The gated x0 estimate
is decoded by the frozen VAE. Maximizing PSNR is equivalent to minimizing
pixel MSE, so the differentiable objective is a weighted mean of pixel MSE and
LPIPS. A standard noise-prediction MSE is added to retain the
diffusion denoising objective while the Stage-2 Condition backbone stays
frozen::

    L_image = weighted_mean(pixel_mse, lpips)
    L_diff  = MSE(conditioned_epsilon, sampled_noise)
    L_total = L_image + lambda_diff * L_diff
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

from config import DDM_DEFAULTS, SD21_MODEL_ID
from data.paired_channel_dataset import (
    VOCImageDataset,
    from_vae_output,
    to_vae_input,
)
from models.builder import (
    build_blip_captioner, build_demand_condition_model,
    build_demand_estimator, build_djscc_adapter, caption_image, encode_prompt,
    load_scheduler, load_text_encoder, load_tokenizer, load_unet,
)
from models.demand_gating import gate_x0_correction
from utils import AverageMeter, set_seed
from utils_distributed import (
    all_workers_true, cleanup_distributed, distributed_barrier,
    launch_training, parse_gpu_ids, reduce_meters, setup_distributed,
    unwrap_ddp, wrap_ddp,
)


PROMPT_EMBEDDING_VERSION = "sd21_penultimate_final_layer_norm_v1"
TRAINING_STAGE = "stage3_demand_skip_finetuning"
GATE_RULE = (
    "residual: frozen stage2_zero_conv(H) + zero-initialized "
    "D_skip_correction(H,D); x0: z_r + D*(x0_conditioned-z_r)"
)
LOSS_DEFINITION = (
    "weighted_mean(pixel_mse, lpips) + lambda_diff * "
    "mse(conditioned_epsilon, sampled_noise)"
)
CURVE_UPDATE_INTERVAL = 200


class FrozenLPIPSLoss(torch.nn.Module):
    """Frozen LPIPS network that keeps gradients with respect to its input."""

    def __init__(self, device):
        super().__init__()
        try:
            import lpips
        except ImportError as error:
            raise RuntimeError(
                "LPIPS training requires the `lpips` package"
            ) from error
        self.metric = lpips.LPIPS(net="alex").to(device).eval()
        self.metric.requires_grad_(False)

    def forward(self, prediction, target):
        return self.metric(
            prediction.float().clamp(-1.0, 1.0),
            target.float().clamp(-1.0, 1.0),
        ).mean()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train D directly using pixel MSE and LPIPS"
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
        "--control-checkpoint",
        default=(
            "scripts/checkpoints/"
            "stage2_control_SD21_xbase/control_best.pth"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "scripts/checkpoints/"
            "stage3_demand_control_SD21_xbase"
        ),
    )
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--snr-min", type=int, default=-5)
    parser.add_argument("--snr-max", type=int, default=20)
    parser.add_argument("--lr-demand", type=float, default=1e-4)
    parser.add_argument("--lr-gate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--pixel-weight", type=float, default=0.5)
    parser.add_argument("--lpips-weight", type=float, default=0.5)
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
        "--diff-weight", type=float, default=1.0,
        help="lambda_diff for the standard epsilon-prediction MSE",
    )
    parser.add_argument(
        "--internal-gate-init", "--internal-gate-floor",
        dest="internal_gate_init", type=float, default=0.9,
        help=(
            "initial opening probability of each learned skip gate; the old "
            "--internal-gate-floor spelling is accepted as an alias"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--gpu-ids", default="0",
        help="physical GPU indices, e.g. 0 for one GPU or 0,1 for two GPUs",
    )
    parser.add_argument(
        "--master-port", type=int, default=29522,
        help="local NCCL rendezvous port used only for multi-GPU training",
    )
    return parser.parse_args()


def validate_args(args):
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("--epochs and --batch-size must be positive")
    if args.snr_min > args.snr_max:
        raise SystemExit("--snr-min must be no greater than --snr-max")
    if args.lr_demand <= 0 or args.lr_gate <= 0:
        raise SystemExit("learning rates must be positive")
    if args.max_grad_norm <= 0:
        raise SystemExit("--max-grad-norm must be positive")
    if args.pixel_weight < 0 or args.lpips_weight < 0:
        raise SystemExit("loss weights must be non-negative")
    if args.pixel_weight + args.lpips_weight <= 0:
        raise SystemExit("at least one image-loss weight must be positive")
    if args.diff_weight <= 0:
        raise SystemExit("--diff-weight must be positive")
    if not 0.0 < args.internal_gate_init < 1.0:
        raise SystemExit("--internal-gate-init must be in (0, 1)")
    try:
        parse_gpu_ids(args.gpu_ids)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not 1 <= args.master_port <= 65535:
        raise SystemExit("--master-port must be in [1, 65535]")


def load_control_into_gated_model(model, state_dict, architecture_version):
    if architecture_version != model.STAGE2_ARCHITECTURE:
        raise RuntimeError(
            "Stage-3 requires a Stage-2 checkpoint trained with only the "
            "original per-scale zero-convs. Expected architecture "
            f"{model.STAGE2_ARCHITECTURE!r}, got {architecture_version!r}. "
            "Retrain scripts/stage2_train_control_sd21.py first."
        )
    result = model.load_state_dict(state_dict, strict=False)
    expected_missing = {
        f"adapters.{index}.gate.{name}"
        for index, adapter in enumerate(model.adapters)
        for name in adapter.gate.state_dict().keys()
    }
    if set(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise RuntimeError(
            "demand-free Control checkpoint is incompatible with the gated "
            f"model; missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )
    for index, adapter in enumerate(model.adapters):
        weight = adapter.gate.demand_zero_conv.weight
        bias = adapter.gate.demand_zero_conv.bias
        if torch.count_nonzero(weight).item() or torch.count_nonzero(bias).item():
            raise RuntimeError(
                f"adapter {index} D correction is not zero-initialized"
            )


def freeze_control_except_d_gates(condition):
    """Freeze pretrained Control and return learned D-fusion parameters."""
    condition.requires_grad_(False)
    gate_parameters = []
    for index, adapter in enumerate(condition.adapters):
        if adapter.gate is None or adapter.gate.mode != "learned":
            raise RuntimeError(f"adapter {index} has no learned D gate")
        adapter.gate.requires_grad_(True)
        gate_parameters.extend(adapter.gate.parameters())
    return gate_parameters


def weighted_image_loss(pixel_loss, lpips_loss, pixel_weight, lpips_weight):
    weight_sum = pixel_weight + lpips_weight
    return (
        pixel_weight * pixel_loss + lpips_weight * lpips_loss
    ) / weight_sum


def save_checkpoint(path, estimator, condition, optimizer, epoch, global_step,
                    best_loss, history, curve_history, args):
    temporary = path + ".tmp"
    torch.save(
        {
            "stage": TRAINING_STAGE,
            "epoch": epoch,
            "global_step": global_step,
            "best_loss": best_loss,
            "demand_estimator": estimator.state_dict(),
            "demand_condition": condition.state_dict(),
            "optimizer": optimizer.state_dict(),
            "architecture_version": estimator.architecture_version,
            "condition_architecture_version": condition.architecture_version,
            "estimator_inputs": ["z_r", "snr"],
            "control_condition_inputs": ["z_r"],
            "frozen_control_checkpoint": args.control_checkpoint,
            "frozen_condition_backbone": True,
            "trainable_control_parameters": ["adapters.*.gate.*"],
            "internal_demand_rule": (
                "frozen Stage-2 zero-conv residual plus a parallel learned "
                "D correction ending in a zero-initialized 1x1 convolution"
            ),
            "stage3_initial_perturbation": "exactly_zero",
            "prompt_embedding": PROMPT_EMBEDDING_VERSION,
            "gate_rule": GATE_RULE,
            "loss_definition": LOSS_DEFINITION,
            "loss_weights": {
                "pixel_mse": args.pixel_weight,
                "lpips": args.lpips_weight,
                "diffusion_mse": args.diff_weight,
            },
            "history": history,
            "curve_history": curve_history,
            "args": vars(args),
        },
        temporary,
    )
    os.replace(temporary, path)


def save_training_artifacts(out_dir, history, curve_history):
    history_path = os.path.join(out_dir, "training_history.json")
    temporary = history_path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(
            {
                "epoch_history": history,
                "curve_history": curve_history,
                "curve_update_interval_batches": CURVE_UPDATE_INTERVAL,
            },
            file,
            indent=2,
        )
    os.replace(temporary, history_path)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [item["global_step"] for item in curve_history]
    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for key, label in (
        ("loss", "total"),
        ("image", "weighted image"),
        ("diff", "diffusion MSE"),
        ("pixel", "pixel MSE"),
        ("lpips", "LPIPS"),
    ):
        axes[0].plot(steps, [item[key] for item in curve_history], label=label)
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Stage 3 Demand/Skip Fine-tuning Convergence")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for key, label in (
        ("psnr_proxy", "PSNR proxy"),
        ("demand_mean", "mean D"),
        ("internal_gate_mean", "mean learned skip gate"),
    ):
        axes[1].plot(steps, [item[key] for item in curve_history], label=label)
    axes[1].set_xlabel("Global batch step")
    axes[1].set_ylabel("Diagnostic")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    figure.tight_layout()

    curve_path = os.path.join(out_dir, "convergence_curve.png")
    curve_temporary = curve_path + ".tmp"
    figure.savefig(curve_temporary, format="png", dpi=160)
    plt.close(figure)
    os.replace(curve_temporary, curve_path)


def meter_snapshot(meters, global_step, epoch, batch_index, context):
    values = reduce_meters(meters, context)
    return {
        "global_step": global_step,
        "epoch": epoch,
        "batch_in_epoch": batch_index,
        **values,
    }


def train_worker(local_rank, world_size, args):
    context = setup_distributed(local_rank, world_size, args)
    device = context.device
    set_seed(args.seed + context.rank)
    if context.is_main:
        os.makedirs(args.out_dir, exist_ok=True)
        print(
            f"[init] GPUs={args.gpu_ids} world_size={context.world_size} "
            f"batch_per_gpu={args.batch_size} "
            f"global_batch={args.batch_size * context.world_size}"
        )
    distributed_barrier(context)

    control_checkpoint = torch.load(args.control_checkpoint, map_location="cpu")
    if control_checkpoint.get("demand_enabled") is not False:
        raise RuntimeError(
            "--control-checkpoint must come from demand-free Stage 2 training"
        )
    control_state = control_checkpoint["control_model"]

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
    prediction_type = getattr(scheduler.config, "prediction_type", "epsilon")
    if prediction_type != "epsilon":
        raise RuntimeError(
            "x0 guidance requires epsilon prediction; scheduler reports "
            f"{prediction_type!r}"
        )
    text_encoder = load_text_encoder(
        device, torch.float32, model_id=SD21_MODEL_ID
    )
    tokenizer = load_tokenizer(model_id=SD21_MODEL_ID)
    blip_processor, blip = build_blip_captioner(device)
    lpips_loss = FrozenLPIPSLoss(device)

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
    condition = build_demand_condition_model(
        unet,
        device,
        cond_channels=4,
        gate_mode="learned",
        gate_init=args.internal_gate_init,
    )
    load_control_into_gated_model(
        condition,
        control_state,
        control_checkpoint.get("condition_architecture_version"),
    )
    del control_state, control_checkpoint
    gate_parameters = freeze_control_except_d_gates(condition)
    condition.eval()

    optimizer = torch.optim.AdamW(
        [
            {"params": estimator.parameters(), "lr": args.lr_demand},
            {"params": gate_parameters, "lr": args.lr_gate},
        ],
        weight_decay=args.weight_decay,
    )
    trainable_parameters = list(estimator.parameters()) + gate_parameters
    estimator = wrap_ddp(estimator, context)
    condition = wrap_ddp(condition, context)

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

    def get_caption(sample_id, original_image, receiver_image):
        if args.prompt_source == "receiver":
            return caption_image(blip_processor, blip, receiver_image, device)
        if sample_id not in captions:
            captions[sample_id] = caption_image(
                blip_processor, blip, original_image, device
            )
        return captions[sample_id]

    alphas = scheduler.alphas_cumprod.to(device)
    num_timesteps = int(scheduler.config.num_train_timesteps)
    global_step = 0
    best_loss = float("inf")
    history = []
    curve_history = []
    meter_names = (
        "loss", "image", "diff", "pixel", "lpips", "psnr_proxy",
        "demand_mean", "internal_gate_mean",
    )
    if context.is_main:
        print(
            "[trainable] DemandEstimator + "
            f"{sum(parameter.numel() for parameter in gate_parameters):,} "
            "learned per-skip D-correction parameters"
        )

    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        estimator.train()
        condition.eval()
        meters = {name: AverageMeter() for name in meter_names}
        curve_meters = {name: AverageMeter() for name in meter_names}
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
                receiver_images = from_vae_output(observation.x_base)
                prompts = [
                    get_caption(sample_id, original_image, receiver_image)
                    for sample_id, original_image, receiver_image in zip(
                        batch["id"], images, receiver_images
                    )
                ]
                prompt_embeds = encode_prompt(
                    text_encoder, tokenizer, prompts, device, torch.float32
                )

            noise = torch.randn_like(z_gt)
            timestep = torch.randint(
                0, num_timesteps, (batch_size,), device=device
            )
            z_t = scheduler.add_noise(z_gt, noise, timestep)

            demand = estimator(z_r, observation.csi).demand
            control = condition(
                z_t,
                timestep,
                encoder_hidden_states=prompt_embeds,
                rx_latents=z_r,
                demand_map=demand,
                csi=observation.csi,
            )
            conditioned_eps = unet(
                z_t,
                timestep,
                encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals=control.down_residuals,
                mid_block_additional_residual=control.mid_residual,
            ).sample
            gated = gate_x0_correction(
                z_t, conditioned_eps, z_r, demand, timestep, alphas
            )
            prediction_image = vae.decode_scaled_with_grad(gated.x0)
            pixel_loss = F.mse_loss(prediction_image.float(), x_gt.float())
            perceptual_loss = lpips_loss(prediction_image, x_gt)
            image_loss = weighted_image_loss(
                pixel_loss, perceptual_loss,
                args.pixel_weight, args.lpips_weight,
            )
            # Standard SD2.1 epsilon objective. U-Net and the pretrained
            # Condition backbone are frozen, so this term can update only the
            # DemandEstimator and newly introduced D skip modules.
            diffusion_loss = F.mse_loss(
                conditioned_eps.float(), noise.float()
            )
            loss = image_loss + args.diff_weight * diffusion_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if global_step == 0:
                estimator_has_gradient = any(
                    parameter.grad is not None
                    and torch.isfinite(parameter.grad).all()
                    and parameter.grad.abs().sum() > 0
                    for parameter in estimator.parameters()
                )
                gates_have_gradients = all(
                    parameter.grad is not None
                    and torch.isfinite(parameter.grad).all()
                    for parameter in gate_parameters
                )
                gradients_are_valid = all_workers_true(
                    estimator_has_gradient and gates_have_gradients, context
                )
                if not gradients_are_valid:
                    raise RuntimeError(
                        "Stage-3 loss did not reach DemandEstimator and all "
                        "D skip modules"
                    )
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters, args.max_grad_norm
            )
            optimizer.step()

            # [-1,1] MSE / 4 equals the corresponding [0,1] MSE for PSNR.
            psnr_proxy = -10.0 * torch.log10(
                (pixel_loss.detach() / 4.0).clamp_min(1e-12)
            )
            internal_gate_mean = control.diagnostics[
                "learned_gate_mean"
            ].detach()
            values = {
                "loss": loss,
                "image": image_loss,
                "diff": diffusion_loss,
                "pixel": pixel_loss,
                "lpips": perceptual_loss,
                "psnr_proxy": psnr_proxy,
                "demand_mean": demand.detach().mean(),
                "internal_gate_mean": internal_gate_mean,
            }
            for name, value in values.items():
                scalar = float(value.detach().item())
                meters[name].update(scalar, batch_size)
                curve_meters[name].update(scalar, batch_size)

            global_step += 1
            progress.set_postfix(
                loss=f"{meters['loss'].avg:.4f}",
                diff=f"{meters['diff'].avg:.4f}",
                psnr=f"{meters['psnr_proxy'].avg:.2f}",
                lpips=f"{meters['lpips'].avg:.3f}",
                D=f"{meters['demand_mean'].avg:.3f}",
            )
            if batch_index % CURVE_UPDATE_INTERVAL == 0:
                snapshot = meter_snapshot(
                    curve_meters, global_step, epoch, batch_index, context
                )
                if context.is_main:
                    curve_history.append(snapshot)
                curve_meters = {
                    name: AverageMeter() for name in meter_names
                }
                if context.is_main:
                    save_training_artifacts(
                        args.out_dir, history, curve_history
                    )
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        if curve_meters["loss"].count > 0:
            snapshot = meter_snapshot(
                curve_meters, global_step, epoch, batch_index, context
            )
            if context.is_main:
                curve_history.append(snapshot)
        epoch_values = reduce_meters(meters, context)
        epoch_loss = epoch_values["loss"]
        if context.is_main:
            history.append(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    **epoch_values,
                }
            )
        is_best = epoch_loss < best_loss
        best_loss = min(best_loss, epoch_loss)
        if context.is_main:
            save_checkpoint(
                os.path.join(args.out_dir, "last.pth"),
                unwrap_ddp(estimator), unwrap_ddp(condition), optimizer,
                epoch, global_step,
                best_loss, history, curve_history, args,
            )
            if is_best:
                save_checkpoint(
                    os.path.join(args.out_dir, "best.pth"),
                    unwrap_ddp(estimator), unwrap_ddp(condition), optimizer,
                    epoch, global_step,
                    best_loss, history, curve_history, args,
                )
            save_training_artifacts(args.out_dir, history, curve_history)
            print(
                f"[epoch {epoch}] loss={epoch_loss:.6f} "
                f"image={epoch_values['image']:.6f} "
                f"diff={epoch_values['diff']:.6f} "
                f"pixel={epoch_values['pixel']:.6f} "
                f"LPIPS={epoch_values['lpips']:.6f} "
                f"best={best_loss:.6f} last=updated "
                f"best={'updated' if is_best else 'kept'} curve=updated"
            )
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    if context.is_main:
        print(
            f"[done] Stage-3 demand/skip checkpoints saved in {args.out_dir}"
        )
    cleanup_distributed(context)


def main():
    args = parse_args()
    validate_args(args)
    launch_training(train_worker, args)


if __name__ == "__main__":
    main()

