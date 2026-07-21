"""Naive dense MRL (materialized B x B similarity per dim, torch autograd) vs
the fused kernels vs the eager wrapper: fwd+bwd time and peak memory across
batch sizes, same loss semantics (Qwen3 row softmax + false-negative mask,
in-batch negatives only, biased, no label smoothing).

    python bench_mrl_naive.py [d_model]
"""
import sys
import time
import torch
import torch.nn.functional as F

from memeff import (MatryoshkaLoss, MemoryEfficientMatryoshkaQwen3Loss,
                    MemoryEfficientQwen3Loss)

D = int(sys.argv[1]) if len(sys.argv) > 1 else 384
DIMS = tuple(m for m in (64, 128, 256, 384, 512, 768, 1024) if m < D) + (D,)
TAU, MARGIN = 0.05, 0.1


def naive_mrl(q, d):
    total = 0.0
    for m in DIMS:
        qm = F.normalize(q[:, :m], dim=-1)
        dm = F.normalize(d[:, :m], dim=-1)
        S = (qm @ dm.T).float()
        pos = S.diagonal()
        keep = (S <= (pos + MARGIN)[:, None]).detach()
        Z = (keep * torch.exp((S - 1.0) / TAU)).sum(1)
        total = total + (torch.log(Z) - (pos - 1.0) / TAU).mean()
    return total


def bench(B, name, loss_fn, iters=5):
    q0 = torch.randn(B, D, device="cuda", dtype=torch.bfloat16)
    d0 = torch.randn(B, D, device="cuda", dtype=torch.bfloat16)
    try:
        for _ in range(2):
            q, dd = q0.clone().requires_grad_(True), d0.clone().requires_grad_(True)
            loss_fn(q, dd).backward()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(iters):
            q, dd = q0.clone().requires_grad_(True), d0.clone().requires_grad_(True)
            loss_fn(q, dd).backward()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters * 1e3
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"  {name:>8}: {dt:9.1f} ms/iter   peak {peak:7.2f} GiB")
        return dt, peak
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"  {name:>8}:       OOM (> 40 GiB)")
        return None, None


if __name__ == "__main__":
    kw = dict(temperature=TAU, margin=MARGIN, normalized_inputs=False)
    print(f"D={D} dims={DIMS} bf16, fwd+bwd, Qwen3 semantics (biased, masked)")
    for B in (4096, 8192, 16384, 32768, 65536, 131072):
        print(f"B={B}")
        bench(B, "naive", naive_mrl)
        bench(B, "fused", MemoryEfficientMatryoshkaQwen3Loss(DIMS, **kw))
        bench(B, "wrapper", MatryoshkaLoss(MemoryEfficientQwen3Loss(**kw), DIMS))
