"""Fused MRL loss vs eager wrapper vs plain loss: fwd+bwd time and peak memory.

    python bench_mrl_loss.py [batch] [d_model]

Defaults: batch 32768, d_model 384, dims (64, 128, 256, 384), bf16 features,
Qwen3 pair mode (no hard negatives), stable=True, tau_plus=1e-4. The point to
verify: fused ~ plain in time and memory while the wrapper pays ~sum(dims)/D.
"""
import sys
import time
import torch

from memeff import (MatryoshkaLoss, MemoryEfficientMatryoshkaQwen3Loss,
                    MemoryEfficientQwen3Loss)

B = int(sys.argv[1]) if len(sys.argv) > 1 else 32768
D = int(sys.argv[2]) if len(sys.argv) > 2 else 384
DIMS = tuple(m for m in (64, 128, 256, 384, 512, 768, 1024) if m < D) + (D,)
KW = dict(temperature=0.05, stable=True, tau_plus=1e-4, normalized_inputs=False)


def bench(name, loss_fn, iters=10):
    q0 = torch.randn(B, D, device="cuda", dtype=torch.bfloat16)
    d0 = torch.randn(B, D, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):   # warmup + autotune/compile
        q, d = q0.clone().requires_grad_(True), d0.clone().requires_grad_(True)
        loss_fn(q, d).backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        q, d = q0.clone().requires_grad_(True), d0.clone().requires_grad_(True)
        loss_fn(q, d).backward()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"{name:>10}: {dt * 1e3:8.1f} ms/iter   peak {peak:6.2f} GiB")
    return dt


if __name__ == "__main__":
    print(f"B={B} D={D} dims={DIMS} bf16, fwd+bwd")
    t_plain = bench("plain", MemoryEfficientQwen3Loss(**KW))
    t_fused = bench("fused", MemoryEfficientMatryoshkaQwen3Loss(DIMS, **KW))
    t_wrap = bench("wrapper", MatryoshkaLoss(MemoryEfficientQwen3Loss(**KW), DIMS))
    print(f"fused/plain {t_fused / t_plain:.2f}x   wrapper/fused "
          f"{t_wrap / t_fused:.2f}x   (sum(dims)/D = {sum(DIMS) / D:.2f})")
