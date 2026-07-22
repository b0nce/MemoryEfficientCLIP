"""Correctness tests for the matryoshka CLIP/LiT losses (needs a CUDA GPU).

    python test_mrl_clip_loss.py

Reference: the per-dim weighted sum of test_clip_lit_loss.reference_clip /
reference_lit on re-normalized prefixes. Runs the fused modules across
fp32 + bf16, custom weights, stable, debiasing (float / per-row / clamp),
label smoothing, unaligned batch sizes, the degenerate dims=(D,) case which
must reproduce the plain loss, and both forced backward kernels; also checks
the eager MatryoshkaLoss wrapper around the plain CLIP loss against the same
reference.
"""
import math
import os

os.environ.setdefault("MEMEFF_INPUT_PRECISION", "ieee")   # must precede the imports

import torch
import torch.distributed as dist
import torch.nn.functional as F

from memeff import (MatryoshkaLoss, MemoryEfficientCLIPLoss,
                    MemoryEfficientMatryoshkaCLIPLoss,
                    MemoryEfficientMatryoshkaLiTLoss,
                    DistributedMemoryEfficientMatryoshkaCLIPLoss,
                    DistributedMemoryEfficientMatryoshkaLiTLoss)
import memeff.mrl_qwen3_loss as _mrl
from test_clip_lit_loss import (TAU, DIM, check, debias_on, make_inputs,
                                make_tau_row, reference_clip, reference_lit)

DIMS2 = (64, 256)
DIMS3 = (64, 128, 256)


def reference_mrl(x, y, dims, weights, ref, **kw):
    total = 0.0
    for m, w in zip(dims, weights):
        total = total + w * ref(F.normalize(x[:, :m], dim=-1),
                                F.normalize(y[:, :m], dim=-1), **kw)
    return total


def leafs(x, y, cast=None):
    c = (lambda t: t) if cast is None else (lambda t: t.to(cast))
    return (c(x).detach().requires_grad_(True), c(y).detach().requires_grad_(True))


def run_fused():
    configs = [  # (B, dims, weights, stable, tau_plus, ls)
        (1024, DIMS3, None, False, 0.0, 0.0),
        (1000, DIMS2, None, False, 0.0, 0.0),    # unaligned batch
        (1024, DIMS3, (0.5, 0.3, 0.2), False, 0.0, 0.0),
        (1024, DIMS3, None, True, 0.0, 0.0),     # stable rescaling
        (1024, DIMS3, None, False, 0.3, 0.0),    # debiased
        (1024, DIMS2, None, False, 0.5, 0.0),    # estimator clamp (dupes)
        (1024, DIMS3, None, False, "row", 0.0),  # per-row priors
        (1024, DIMS3, None, False, 0.0, 0.1),    # label smoothing
        (1024, DIMS3, None, True, 0.3, 0.1),     # everything at once
        (1024, (DIM,), None, False, 0.3, 0.0),   # degenerate = plain
    ]
    for dtype, tol in ((torch.float32, 1e-3), (torch.bfloat16, 5e-2)):
        for B, dims, weights, stable, tau_label, ls in configs:
            w = weights or (1.0,) * len(dims)
            per_row = tau_label == "row"
            tau_plus = make_tau_row(B) if per_row else tau_label
            x, y = make_inputs(B, dtype, pos_dupes=debias_on(tau_plus))
            sc = math.sqrt(B * TAU) if stable else 1.0

            xr, yr = leafs(x, y, torch.float32)
            ref = reference_mrl(xr, yr, dims, w, reference_clip,
                                tau_plus=tau_plus, label_smoothing=ls)
            ref.backward()
            loss_fn = MemoryEfficientMatryoshkaCLIPLoss(
                dims, weights=weights, temperature=TAU, normalized_inputs=True,
                stable=stable, tau_plus=0.0 if per_row else tau_plus,
                label_smoothing=ls)
            xk, yk = leafs(x, y)
            loss = loss_fn(xk, yk, tau_plus=tau_plus if per_row else None)
            loss.backward()
            print(f"mrl-clip {str(dtype).split('.')[-1]} B={B} dims={dims} "
                  f"w={'custom' if weights else 'unit'} stable={stable} "
                  f"tau+={tau_label} ls={ls}")
            check("loss", loss.detach(), ref.detach(), tol)
            check("dX", xk.grad, xr.grad * sc, tol)
            check("dY", yk.grad, yr.grad * sc, tol)

            xr, _ = leafs(x, y, torch.float32)
            ref = reference_mrl(xr, y.float(), dims, w, reference_lit,
                                tau_plus=tau_plus, label_smoothing=ls)
            ref.backward()
            lit_fn = MemoryEfficientMatryoshkaLiTLoss(
                dims, weights=weights, temperature=TAU, normalized_inputs=True,
                stable=stable, tau_plus=0.0 if per_row else tau_plus,
                label_smoothing=ls)
            xk = x.detach().requires_grad_(True)
            loss = lit_fn(xk, y, tau_plus=tau_plus if per_row else None)
            loss.backward()
            print(f"mrl-lit  {str(dtype).split('.')[-1]} B={B} dims={dims} "
                  f"w={'custom' if weights else 'unit'} stable={stable} "
                  f"tau+={tau_label} ls={ls}")
            check("loss", loss.detach(), ref.detach(), tol)
            check("dX", xk.grad, xr.grad * sc, tol)


def run_wrapper():
    for B, dims, tau_plus, ls in [(1024, DIMS3, 0.0, 0.0),
                                  (1024, DIMS2, 0.3, 0.1),
                                  (1000, (32, 96, 256), 0.0, 0.0)]:
        w = (1.0,) * len(dims)
        x, y = make_inputs(B, torch.float32, pos_dupes=debias_on(tau_plus))
        xr, yr = leafs(x, y)
        ref = reference_mrl(xr, yr, dims, w, reference_clip,
                            tau_plus=tau_plus, label_smoothing=ls)
        ref.backward()
        loss_fn = MatryoshkaLoss(MemoryEfficientCLIPLoss(
            temperature=TAU, normalized_inputs=False, stable=False,
            tau_plus=tau_plus, label_smoothing=ls), dims)
        xk, yk = leafs(x, y)
        loss = loss_fn(xk, yk)
        loss.backward()
        print(f"wrapper B={B} dims={dims} tau+={tau_plus} ls={ls}")
        check("loss", loss.detach(), ref.detach(), 1e-3)
        check("dX", xk.grad, xr.grad, 1e-3)
        check("dY", yk.grad, yr.grad, 1e-3)


def run_dist_world1():
    """The fused distributed modules at world=1 (no process group): every
    kernel launch and eager term minus the collectives."""
    for tau_plus, ls in [(0.0, 0.0), (0.3, 0.1), ("row", 0.0)]:
        per_row = tau_plus == "row"
        tp = make_tau_row(1024) if per_row else tau_plus
        w = (1.0,) * len(DIMS3)
        x, y = make_inputs(1024, torch.float32, pos_dupes=debias_on(tp))

        xr, yr = leafs(x, y, torch.float32)
        ref = reference_mrl(xr, yr, DIMS3, w, reference_clip,
                            tau_plus=tp, label_smoothing=ls)
        ref.backward()
        loss_fn = DistributedMemoryEfficientMatryoshkaCLIPLoss(
            DIMS3, temperature=TAU, normalized_inputs=True, stable=False,
            tau_plus=0.0 if per_row else tp, label_smoothing=ls)
        xk, yk = leafs(x, y)
        loss = loss_fn(xk, yk, tau_plus=tp if per_row else None)
        loss.backward()
        print(f"mrl-clip dist world=1 tau+={tau_plus} ls={ls}")
        check("loss", loss.detach(), ref.detach(), 1e-3)
        check("dX", xk.grad, xr.grad, 1e-3)
        check("dY", yk.grad, yr.grad, 1e-3)

        xr, _ = leafs(x, y, torch.float32)
        ref = reference_mrl(xr, y.float(), DIMS3, w, reference_lit,
                            tau_plus=tp, label_smoothing=ls)
        ref.backward()
        lit_fn = DistributedMemoryEfficientMatryoshkaLiTLoss(
            DIMS3, temperature=TAU, normalized_inputs=True, stable=False,
            tau_plus=0.0 if per_row else tp, label_smoothing=ls)
        xk = x.detach().requires_grad_(True)
        loss = lit_fn(xk, y, tau_plus=tp if per_row else None)
        loss.backward()
        print(f"mrl-lit dist world=1 tau+={tau_plus} ls={ls}")
        check("loss", loss.detach(), ref.detach(), 1e-3)
        check("dX", xk.grad, xr.grad, 1e-3)


def run_distributed():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    local_b = 512
    B = world * local_b
    rows = slice(rank * local_b, (rank + 1) * local_b)
    tol = 1e-3

    for force in ("prefix", "telescope"):   # both tile backends over the ring
        _mrl._FORCE_BACKWARD = force
        if rank == 0:
            print(f"--- mrl distributed suites, {force} backward ---")
        for dims, weights, stable, tau_label, ls in [
                (DIMS3, None, False, 0.0, 0.0),
                (DIMS3, (0.5, 0.3, 0.2), False, 0.0, 0.0),
                (DIMS2, None, True, 0.0, 0.0),
                (DIMS3, None, False, 0.3, 0.0),
                (DIMS3, None, False, "row", 0.0),
                (DIMS3, None, False, 0.0, 0.1),
                (DIMS3, None, True, 0.3, 0.1)]:
            # identical global batch on every rank, each takes its shard
            w = weights or (1.0,) * len(dims)
            per_row = tau_label == "row"
            tau_plus = make_tau_row(B) if per_row else tau_label   # same seed everywhere
            x, y = make_inputs(B, torch.float32, seed=7, pos_dupes=debias_on(tau_plus))
            sc = math.sqrt(B * TAU) if stable else 1.0

            xr, yr = leafs(x, y, torch.float32)
            ref = reference_mrl(xr, yr, dims, w, reference_clip,
                                tau_plus=tau_plus, label_smoothing=ls)
            ref.backward()
            loss_fn = DistributedMemoryEfficientMatryoshkaCLIPLoss(
                dims, weights=weights, temperature=TAU, normalized_inputs=True,
                stable=stable, tau_plus=0.0 if per_row else tau_plus,
                label_smoothing=ls)
            xs = x[rows].detach().requires_grad_(True)
            ys = y[rows].detach().requires_grad_(True)
            partial = loss_fn(xs, ys, tau_plus=tau_plus[rows] if per_row else None)
            partial.backward()
            total = partial.detach().clone()
            dist.all_reduce(total)

            if rank == 0:
                print(f"mrl-clip dist world={world} dims={dims} "
                      f"w={'custom' if weights else 'unit'} stable={stable} "
                      f"tau+={tau_label} ls={ls}")
            check(f"loss(rank{rank})", total, ref.detach(), tol)
            check(f"dX(rank{rank})", xs.grad, xr.grad[rows] * sc, tol)
            check(f"dY(rank{rank})", ys.grad, yr.grad[rows] * sc, tol)
            dist.barrier()

        for dims, tau_label, ls in [(DIMS3, 0.0, 0.0), (DIMS3, 0.3, 0.0),
                                    (DIMS3, "row", 0.0), (DIMS2, 0.0, 0.1)]:
            w = (1.0,) * len(dims)
            per_row = tau_label == "row"
            tau_plus = make_tau_row(B) if per_row else tau_label
            x, y = make_inputs(B, torch.float32, seed=11, pos_dupes=debias_on(tau_plus))

            xr, _ = leafs(x, y, torch.float32)
            ref = reference_mrl(xr, y.float(), dims, w, reference_lit,
                                tau_plus=tau_plus, label_smoothing=ls)
            ref.backward()
            lit_fn = DistributedMemoryEfficientMatryoshkaLiTLoss(
                dims, temperature=TAU, normalized_inputs=True, stable=False,
                tau_plus=0.0 if per_row else tau_plus, label_smoothing=ls)
            xs = x[rows].detach().requires_grad_(True)
            ys = y[rows].detach().requires_grad_(True)
            partial = lit_fn(xs, ys, tau_plus=tau_plus[rows] if per_row else None)
            partial.backward()
            total = partial.detach().clone()
            dist.all_reduce(total)

            if rank == 0:
                print(f"mrl-lit dist world={world} dims={dims} "
                      f"tau+={tau_label} ls={ls}")
            check(f"loss(rank{rank})", total, ref.detach(), tol)
            check(f"dX(rank{rank})", xs.grad, xr.grad[rows], tol)
            assert ys.grad is None, "locked image tower must receive no gradient"
            dist.barrier()
    _mrl._FORCE_BACKWARD = None
    dist.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        run_distributed()
    else:
        run_wrapper()
        run_dist_world1()
        for force in ("prefix", "telescope", "fa"):   # every backward kernel
            _mrl._FORCE_BACKWARD = force
            print(f"--- fused suites, {force} backward ---")
            run_fused()
        _mrl._FORCE_BACKWARD = None
        print("ALL OK")
