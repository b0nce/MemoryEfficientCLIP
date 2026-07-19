"""Timing, peak memory, and torch.profiler traces for the embedding loss.

Single GPU:  python bench_embedding_loss.py [--local-batch N] [--dim D] [--k K]
Multi GPU:   torchrun --nproc-per-node=4 bench_embedding_loss.py

Runs the embedding-loss variants (plain q->d, +hard negatives, +qq/dd) plus the CLIP
loss as a baseline, prints ms/iter (forward+backward) and peak memory, and exports one
chrome trace per config to ./traces/ (openable in chrome://tracing or Perfetto).
"""
import argparse
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.profiler import profile, schedule, ProfilerActivity

from clip_embedding_loss import (
    MemoryEfficientEmbeddingLoss, DistributedMemoryEfficientEmbeddingLoss)
from lit_embedding_loss import (
    MemoryEfficientLiTEmbeddingLoss, DistributedMemoryEfficientLiTEmbeddingLoss)
from clip_loss import MemoryEfficientCLIPLoss
from distributed_clip_loss import DistributedMemoryEfficientCLIPLoss

WARMUP, ITERS = 5, 10                       # timing loop (warmup also compiles Triton)
PROF_STEPS = 6                              # profiler: wait=1, warmup=2, active=3


def make_config_list(distributed):
    emb = DistributedMemoryEfficientEmbeddingLoss if distributed else MemoryEfficientEmbeddingLoss
    lit = (DistributedMemoryEfficientLiTEmbeddingLoss if distributed
           else MemoryEfficientLiTEmbeddingLoss)
    clip = DistributedMemoryEfficientCLIPLoss if distributed else MemoryEfficientCLIPLoss
    return [
        ("clip_baseline", clip(temperature=0.05, normalized_inputs=True), False),
        ("emb_base", emb(temperature=0.05, margin=0.1, normalized_inputs=True), False),
        ("emb_hardneg", emb(temperature=0.05, margin=0.1, normalized_inputs=True), True),
        ("emb_full_qq_dd", emb(temperature=0.05, margin=0.1, use_qq_negatives=True,
                               use_dd_negatives=True, normalized_inputs=True), True),
        ("lit_hardneg", lit(temperature=0.05, margin=0.1, normalized_inputs=True), True),
        ("lit_qq", lit(temperature=0.05, margin=0.1, use_qq_negatives=True,
                       normalized_inputs=True), True),
    ]


def make_inputs(local_batch, k, dim, device, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    q = F.normalize(torch.randn(local_batch, dim, device=device, generator=gen), dim=-1)
    d = F.normalize(torch.randn(local_batch, dim, device=device, generator=gen), dim=-1)
    h = F.normalize(torch.randn(local_batch, k, dim, device=device, generator=gen), dim=-1)
    return (q.requires_grad_(True), d.requires_grad_(True), h.requires_grad_(True))


def one_step(loss_fn, q, d, h, use_h):
    q.grad = d.grad = h.grad = None
    if isinstance(loss_fn, (MemoryEfficientCLIPLoss, DistributedMemoryEfficientCLIPLoss)):
        loss = loss_fn(q, d)
    else:
        loss = loss_fn(q, d, h if use_h else None)
    loss.backward()
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-batch", type=int, default=16384)
    parser.add_argument("--dim", type=int, default=1152)
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
    else:
        rank = 0
    device = torch.device("cuda", rank)
    tag = f"dist{dist.get_world_size()}" if distributed else "single"
    os.makedirs("traces", exist_ok=True)

    if rank == 0:
        print(f"{tag}: local_batch={args.local_batch} dim={args.dim} k={args.k} "
              f"gpu={torch.cuda.get_device_name(device)}")

    for name, loss_fn, use_h in make_config_list(distributed):
        q, d, h = make_inputs(args.local_batch, args.k, args.dim, device, seed=17 + rank)

        for _ in range(WARMUP):
            one_step(loss_fn, q, d, h, use_h)
        torch.cuda.synchronize()
        if distributed:
            dist.barrier()

        torch.cuda.reset_peak_memory_stats(device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITERS):
            one_step(loss_fn, q, d, h, use_h)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / ITERS
        peak_gb = torch.cuda.max_memory_allocated(device) / 2**30
        if distributed:
            dist.barrier()
        if rank == 0:
            print(f"  {name:16s} {ms:8.2f} ms/iter (fwd+bwd)   peak mem {peak_gb:6.2f} GiB")

        trace_path = f"traces/{tag}_{name}_rank{rank}.json.gz"
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=2, active=3),
            on_trace_ready=lambda p, path=trace_path: p.export_chrome_trace(path),
        ) as prof:
            for _ in range(PROF_STEPS):
                one_step(loss_fn, q, d, h, use_h)
                torch.cuda.synchronize()
                prof.step()
        if distributed:
            dist.barrier()
        if rank == 0:
            print(f"  {'':16s} trace -> {trace_path}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
