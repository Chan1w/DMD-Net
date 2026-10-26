"""Small DDP launcher and metric helpers for the Stage-2/Stage-3 trainers."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def parse_gpu_ids(text: str) -> list[int]:
    try:
        ids = [int(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError(f"invalid --gpu-ids value: {text!r}") from error
    if not ids:
        raise ValueError("--gpu-ids must contain at least one GPU index")
    if any(index < 0 for index in ids) or len(set(ids)) != len(ids):
        raise ValueError("--gpu-ids must contain unique non-negative indices")
    return ids


def _spawn_entry(local_rank, worker, args, world_size):
    worker(local_rank, world_size, args)


def launch_training(worker, args):
    """Run one worker directly or auto-spawn one NCCL worker per selected GPU."""
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    if args.device == "cpu":
        if len(gpu_ids) != 1:
            raise RuntimeError("multi-GPU training requires CUDA")
        worker(0, 1, args)
        return

    # This must be set before any CUDA API call.  In particular, do not call
    # torch.cuda.is_available()/device_count() in this parent process: on
    # servers using exclusive-process compute mode that creates a CUDA context
    # which can prevent the spawned DDP workers from claiming their GPUs.
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )
    if len(gpu_ids) == 1:
        # No process group is needed for the one-GPU path.
        args.single_gpu_id = gpu_ids[0]
        worker(0, 1, args)
        return

    # Restrict child visibility so local ranks 0..N-1 map exactly to the
    # requested physical devices and torchrun is not required.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(args.master_port)
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    mp.spawn(
        _spawn_entry,
        nprocs=len(gpu_ids),
        args=(worker, args, len(gpu_ids)),
        join=True,
    )


def setup_distributed(local_rank: int, world_size: int,
                      args) -> DistributedContext:
    if args.device == "cpu":
        return DistributedContext(0, 0, 1, torch.device("cpu"))
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable in this training worker. Check --gpu-ids "
            "and whether another process owns the requested GPU."
        )
    if world_size > 1:
        available = torch.cuda.device_count()
        if local_rank >= available:
            raise RuntimeError(
                f"DDP local rank {local_rank} cannot be mapped because only "
                f"{available} selected CUDA devices are visible"
            )
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            rank=local_rank,
            world_size=world_size,
        )
        return DistributedContext(
            local_rank, local_rank, world_size,
            torch.device("cuda", local_rank),
        )
    gpu_id = int(getattr(args, "single_gpu_id", 0))
    torch.cuda.set_device(gpu_id)
    return DistributedContext(0, 0, 1, torch.device("cuda", gpu_id))


def cleanup_distributed(context: DistributedContext):
    if context.distributed and dist.is_initialized():
        dist.destroy_process_group()


def distributed_barrier(context: DistributedContext):
    if context.distributed:
        dist.barrier()


def wrap_ddp(module, context: DistributedContext):
    if not context.distributed:
        return module
    return DistributedDataParallel(
        module,
        device_ids=[context.local_rank],
        output_device=context.local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )


def unwrap_ddp(module):
    return module.module if isinstance(module, DistributedDataParallel) else module


def reduce_meters(meters: dict, context: DistributedContext) -> dict[str, float]:
    """Return sample-weighted global averages for a dict of AverageMeters."""
    names = list(meters)
    values = []
    for name in names:
        values.extend((float(meters[name].sum), float(meters[name].count)))
    tensor = torch.tensor(values, dtype=torch.float64, device=context.device)
    if context.distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    result = {}
    for index, name in enumerate(names):
        total = tensor[2 * index].item()
        count = tensor[2 * index + 1].item()
        result[name] = total / max(count, 1.0)
    return result


def all_workers_true(value: bool, context: DistributedContext) -> bool:
    tensor = torch.tensor(
        1 if value else 0, dtype=torch.int32, device=context.device
    )
    if context.distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return bool(tensor.item())
