"""Correctness tests for the embedding losses (needs a CUDA GPU).

Single GPU:  python test_embedding_loss.py
Multi GPU:   torchrun --nproc-per-node=2 test_embedding_loss.py

Compares losses and gradients against a dense autograd reference. Kernels run with
ieee fp32 matmuls here (and torch tf32 stays off) so the false-negative mask makes
the same boundary decisions in the kernel and the reference: under tf32 a flipped
near-threshold negative carries up to exp(margin/temp) times the positive's softmax
weight, which would drown the comparison in mask-flip noise.
"""
import math
import os

os.environ.setdefault("MEMEFF_INPUT_PRECISION", "ieee")   # must precede the imports

import torch
import torch.distributed as dist
import torch.nn.functional as F

from clip_embedding_loss import (
    MemoryEfficientEmbeddingLoss, DistributedMemoryEfficientEmbeddingLoss)
from lit_embedding_loss import (
    MemoryEfficientLiTEmbeddingLoss, DistributedMemoryEfficientLiTEmbeddingLoss)

TAU, MARGIN, DIM = 0.05, 0.1, 256


def reference_loss(q, d, h, tau=TAU, margin=MARGIN, use_qq=False, use_dd=False):
    B = q.shape[0]
    pos = (q * d).sum(-1)
    thr = (pos + margin).detach()   # the mask is a stop-gradient

    def masked_exp(S, excl_diag):
        m = (S.detach() <= thr[:, None]).float()
        if excl_diag:
            m = m * (1.0 - torch.eye(B, device=S.device))
        return m * torch.exp((S - 1.0) / tau)

    Z = masked_exp(q @ d.T, False).sum(1)
    if h is not None:
        S_h = torch.einsum('bkd,bd->bk', h, q)
        Z = Z + ((S_h.detach() <= thr[:, None]).float() * torch.exp((S_h - 1.0) / tau)).sum(1)
    if use_qq:
        Z = Z + masked_exp(q @ q.T, True).sum(1)
    if use_dd:
        Z = Z + masked_exp(d @ d.T, True).sum(1)
    return -((pos - 1.0) / tau - torch.log(Z)).mean()


def make_inputs(B, K, dtype, seed=0):
    gen = torch.Generator().manual_seed(seed)
    q = F.normalize(torch.randn(B, DIM, generator=gen), dim=-1)
    d = F.normalize(torch.randn(B, DIM, generator=gen), dim=-1)
    # near-duplicates so the false-negative mask fires on every negative group
    d[1] = F.normalize(q[0] + 0.01 * torch.randn(DIM, generator=gen), dim=-1)
    q[3] = F.normalize(q[2] + 0.01 * torch.randn(DIM, generator=gen), dim=-1)
    h = None
    if K:
        h = F.normalize(torch.randn(B, K, DIM, generator=gen), dim=-1)
        h[4, 0] = F.normalize(d[4] + 0.005 * torch.randn(DIM, generator=gen), dim=-1)
    to = lambda t: t.cuda().to(dtype).contiguous()
    return to(q), to(d), (to(h) if h is not None else None)


def rel_err(a, b):
    a = a.float()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()


def check(name, a, b, tol):
    err = rel_err(a, b)
    print(f"  {'OK ' if err < tol else 'FAIL'} {name}: rel err {err:.2e}")
    assert err < tol, f"{name}: {err:.2e} >= {tol:.0e}"


def leafs(q, d, h, dtype=None):
    cast = (lambda t: t.to(dtype)) if dtype is not None else (lambda t: t)
    ql = cast(q).detach().requires_grad_(True)
    dl = cast(d).detach().requires_grad_(True)
    hl = cast(h).detach().requires_grad_(True) if h is not None else None
    return ql, dl, hl


def run_single_clip():
    configs = [  # (B, K, use_qq, use_dd, stable)
        (1024, 0, False, False, False),
        (1024, 4, False, False, False),
        (1024, 4, True, False, False),
        (1024, 4, False, True, False),
        (1024, 4, True, True, False),
        (1000, 3, True, True, False),   # batch not a multiple of the block sizes
        (1024, 4, True, True, True),    # stable gradient rescaling
    ]
    for dtype, tol in ((torch.float32, 1e-3), (torch.bfloat16, 5e-2)):
        for B, K, use_qq, use_dd, stable in configs:
            q, d, h = make_inputs(B, K, dtype)
            qr, dr, hr = leafs(q, d, h, torch.float32)
            ref = reference_loss(qr, dr, hr, use_qq=use_qq, use_dd=use_dd)
            ref.backward()

            loss_fn = MemoryEfficientEmbeddingLoss(
                temperature=TAU, margin=MARGIN, use_qq_negatives=use_qq,
                use_dd_negatives=use_dd, normalized_inputs=True, stable=stable)
            qk, dk, hk = leafs(q, d, h)
            loss = loss_fn(qk, dk, hk)
            loss.backward()

            sc = math.sqrt(B * TAU) if stable else 1.0   # stable rescales grads only
            print(f"clip {str(dtype).split('.')[-1]} B={B} K={K} "
                  f"qq={use_qq} dd={use_dd} stable={stable}")
            check("loss", loss.detach(), ref.detach(), tol)
            check("dQ", qk.grad, qr.grad * sc, tol)
            check("dD", dk.grad, dr.grad * sc, tol)
            if hk is not None:
                check("dH", hk.grad, hr.grad * sc, tol)


def run_single_lit():
    configs = [  # (B, K, use_qq)
        (1024, 0, False),
        (1024, 4, False),
        (1024, 4, True),
        (1000, 3, True),
    ]
    for dtype, tol in ((torch.float32, 1e-3), (torch.bfloat16, 5e-2)):
        for B, K, use_qq in configs:
            q, d, h = make_inputs(B, K, dtype)
            qr = q.float().detach().requires_grad_(True)
            ref = reference_loss(qr, d.float(), h.float() if h is not None else None,
                                 use_qq=use_qq)
            ref.backward()

            loss_fn = MemoryEfficientLiTEmbeddingLoss(
                temperature=TAU, margin=MARGIN, use_qq_negatives=use_qq,
                normalized_inputs=True)
            qk = q.detach().requires_grad_(True)
            loss = loss_fn(qk, d, h)
            loss.backward()

            print(f"lit {str(dtype).split('.')[-1]} B={B} K={K} qq={use_qq}")
            check("loss", loss.detach(), ref.detach(), tol)
            check("dQ", qk.grad, qr.grad, tol)


def run_distributed():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    local_b = 512
    B = world * local_b
    rows = slice(rank * local_b, (rank + 1) * local_b)
    tol = 1e-3

    for K, use_qq, use_dd in [(0, False, False), (4, False, False),
                              (4, True, False), (4, False, True), (4, True, True)]:
        # identical global batch on every rank, each takes its shard
        q, d, h = make_inputs(B, K, torch.float32, seed=7)
        qr, dr, hr = leafs(q, d, h)
        ref = reference_loss(qr, dr, hr, use_qq=use_qq, use_dd=use_dd)
        ref.backward()

        loss_fn = DistributedMemoryEfficientEmbeddingLoss(
            temperature=TAU, margin=MARGIN, use_qq_negatives=use_qq,
            use_dd_negatives=use_dd, normalized_inputs=True)
        qs, ds, hs = leafs(q[rows], d[rows], h[rows] if h is not None else None)
        partial = loss_fn(qs, ds, hs)
        partial.backward()
        total = partial.detach().clone()
        dist.all_reduce(total)

        if rank == 0:
            print(f"clip dist world={world} K={K} qq={use_qq} dd={use_dd}")
        check(f"loss(rank{rank})", total, ref.detach(), tol)
        check(f"dQ(rank{rank})", qs.grad, qr.grad[rows], tol)
        check(f"dD(rank{rank})", ds.grad, dr.grad[rows], tol)
        if hs is not None:
            check(f"dH(rank{rank})", hs.grad, hr.grad[rows], tol)
        dist.barrier()

    for K, use_qq in [(0, False), (4, False), (4, True)]:
        q, d, h = make_inputs(B, K, torch.float32, seed=11)
        qr = q.detach().requires_grad_(True)
        ref = reference_loss(qr, d, h, use_qq=use_qq)
        ref.backward()

        loss_fn = DistributedMemoryEfficientLiTEmbeddingLoss(
            temperature=TAU, margin=MARGIN, use_qq_negatives=use_qq,
            normalized_inputs=True)
        qs = q[rows].detach().requires_grad_(True)
        partial = loss_fn(qs, d[rows], h[rows] if h is not None else None)
        partial.backward()
        total = partial.detach().clone()
        dist.all_reduce(total)

        if rank == 0:
            print(f"lit dist world={world} K={K} qq={use_qq}")
        check(f"loss(rank{rank})", total, ref.detach(), tol)
        check(f"dQ(rank{rank})", qs.grad, qr.grad[rows], tol)
        dist.barrier()
    dist.destroy_process_group()


def run_world1_fallbacks():
    q, d, h = make_inputs(1024, 4, torch.float32)
    qr, dr, hr = leafs(q, d, h)
    ref = reference_loss(qr, dr, hr, use_qq=True, use_dd=True)
    ref.backward()
    loss_fn = DistributedMemoryEfficientEmbeddingLoss(
        temperature=TAU, margin=MARGIN, use_qq_negatives=True,
        use_dd_negatives=True, normalized_inputs=True)
    qk, dk, hk = leafs(q, d, h)
    loss = loss_fn(qk, dk, hk)
    loss.backward()
    print("clip dist world=1 fallback")
    check("loss", loss.detach(), ref.detach(), 1e-3)
    check("dQ", qk.grad, qr.grad, 1e-3)
    check("dD", dk.grad, dr.grad, 1e-3)
    check("dH", hk.grad, hr.grad, 1e-3)

    qr = q.detach().requires_grad_(True)
    ref = reference_loss(qr, d, h, use_qq=True)
    ref.backward()
    loss_fn = DistributedMemoryEfficientLiTEmbeddingLoss(
        temperature=TAU, margin=MARGIN, use_qq_negatives=True, normalized_inputs=True)
    qk = q.detach().requires_grad_(True)
    loss = loss_fn(qk, d, h)
    loss.backward()
    print("lit dist world=1 fallback")
    check("loss", loss.detach(), ref.detach(), 1e-3)
    check("dQ", qk.grad, qr.grad, 1e-3)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        run_distributed()
    else:
        run_single_clip()
        run_single_lit()
        run_world1_fallbacks()
        print("ALL OK")
