"""Correctness tests for the matryoshka Qwen3 losses (needs a CUDA GPU).

    python test_mrl_qwen3_loss.py

Reference: the per-dim sum of test_qwen3_loss.reference_loss on re-normalized
prefixes. Checks both the eager MatryoshkaLoss wrapper (which is itself the
oracle for the fused kernels' design) and the fused
MemoryEfficientMatryoshka(LiT)Qwen3Loss modules across the full feature matrix:
hard negatives, q-q/d-d negatives, stable, debiasing (float / per-row / clamp),
label smoothing, custom weights, unaligned batch sizes, fp32 + bf16, and the
degenerate dims=(D,) case which must reproduce the plain loss. Mask-tie
ambiguity handling follows test_qwen3_loss (unioned across dims: a tie at any
prefix taints the row).
"""
import math
import os

os.environ.setdefault("MEMEFF_INPUT_PRECISION", "ieee")   # must precede the imports

import torch
import torch.distributed as dist
import torch.nn.functional as F

from memeff import (MatryoshkaLoss, MemoryEfficientMatryoshkaQwen3Loss,
                    MemoryEfficientMatryoshkaLiTQwen3Loss,
                    MemoryEfficientQwen3Loss,
                    DistributedMemoryEfficientMatryoshkaQwen3Loss,
                    DistributedMemoryEfficientMatryoshkaLiTQwen3Loss)
from test_qwen3_loss import (TAU, MARGIN, DIM, ambiguous_rows, check, debias_on,
                             leafs, make_inputs, make_tau_row, reference_loss)

DIMS2 = (64, 256)
DIMS3 = (64, 128, 256)


def prefixes(q, d, h, m):
    qm = F.normalize(q[:, :m], dim=-1)
    dm = F.normalize(d[:, :m], dim=-1)
    hm = F.normalize(h[..., :m], dim=-1) if h is not None else None
    return qm, dm, hm


def reference_mrl(q, d, h, dims, weights, **kw):
    total = 0.0
    for m, w in zip(dims, weights):
        qm, dm, hm = prefixes(q, d, h, m)
        total = total + w * reference_loss(qm, dm, hm, **kw)
    return total


def mrl_ambiguous(q, d, h, dims, use_qq=False, use_dd=False):
    bq = bd = None
    for m in dims:
        qm, dm, hm = prefixes(q.float(), d.float(), h.float() if h is not None else None, m)
        a, b = ambiguous_rows(qm, dm, hm, use_qq, use_dd)
        bq = a if bq is None else bq | a
        bd = b if bd is None else bd | b
    return bq, bd


def run_fused_clip():
    configs = [  # (B, dims, weights, K, use_qq, use_dd, stable, tau_plus, ls)
        (1024, DIMS3, None, 0, False, False, False, 0.0, 0.0),
        (1024, DIMS3, None, 4, False, False, False, 0.0, 0.0),
        (1024, DIMS3, None, 4, True, True, False, 0.0, 0.0),
        (1000, DIMS2, None, 3, True, True, False, 0.0, 0.0),   # unaligned batch
        (1024, DIMS3, (0.5, 0.3, 0.2), 4, True, True, False, 0.0, 0.0),
        (1024, DIMS3, None, 4, True, True, True, 0.0, 0.0),    # stable rescaling
        (1024, DIMS3, None, 4, True, True, False, 0.3, 0.0),   # debiased
        (1024, DIMS2, None, 0, False, False, False, 0.5, 0.0), # estimator clamp
        (1024, DIMS3, None, 4, True, True, False, "row", 0.0), # per-row priors
        (1024, DIMS3, None, 4, True, True, False, 0.0, 0.1),   # label smoothing
        (1024, DIMS3, None, 4, True, True, True, 0.3, 0.1),    # everything at once
        (1024, (DIM,), None, 4, True, True, False, 0.3, 0.0),  # degenerate = plain
    ]
    for dtype, tol in ((torch.float32, 1e-3), (torch.bfloat16, 5e-2)):
        for B, dims, weights, K, use_qq, use_dd, stable, tau_label, ls in configs:
            w = weights or (1.0,) * len(dims)
            per_row = tau_label == "row"
            tau_plus = make_tau_row(B) if per_row else tau_label
            q, d, h = make_inputs(B, K, dtype, pos_dupes=debias_on(tau_plus))
            bad_q, bad_d = mrl_ambiguous(q, d, h, dims, use_qq, use_dd)
            kq, kd = ~bad_q, ~bad_d
            qr, dr, hr = leafs(q, d, h, torch.float32)
            ref = reference_mrl(qr, dr, hr, dims, w, use_qq=use_qq, use_dd=use_dd,
                                tau_plus=tau_plus, label_smoothing=ls)
            ref.backward()

            loss_fn = MemoryEfficientMatryoshkaQwen3Loss(
                dims, weights=weights, temperature=TAU, margin=MARGIN,
                use_qq_negatives=use_qq, use_dd_negatives=use_dd,
                normalized_inputs=True, stable=stable,
                tau_plus=0.0 if per_row else tau_plus, label_smoothing=ls)
            qk, dk, hk = leafs(q, d, h)
            loss = loss_fn(qk, dk, hk, tau_plus=tau_plus if per_row else None)
            loss.backward()

            sc = math.sqrt(B * TAU) if stable else 1.0
            print(f"mrl-clip {str(dtype).split('.')[-1]} B={B} dims={dims} "
                  f"w={'custom' if weights else 'unit'} K={K} qq={use_qq} "
                  f"dd={use_dd} stable={stable} tau+={tau_label} ls={ls} "
                  f"amb={int(bad_q.sum())}q/{int(bad_d.sum())}d")
            check("loss", loss.detach(), ref.detach(), tol)
            check("dQ", qk.grad[kq], qr.grad[kq] * sc, tol)
            check("dD", dk.grad[kd], dr.grad[kd] * sc, tol)
            if hk is not None:
                check("dH", hk.grad[kq], hr.grad[kq] * sc, tol)


def run_fused_lit():
    configs = [  # (B, dims, K, use_qq, tau_plus, ls)
        (1024, DIMS3, 0, False, 0.0, 0.0),
        (1024, DIMS3, 4, True, 0.0, 0.0),
        (1000, DIMS2, 3, True, 0.0, 0.0),
        (1024, DIMS3, 4, True, 0.3, 0.0),
        (1024, DIMS3, 4, True, "row", 0.0),
        (1024, DIMS3, 4, True, 0.0, 0.1),
    ]
    for dtype, tol in ((torch.float32, 1e-3), (torch.bfloat16, 5e-2)):
        for B, dims, K, use_qq, tau_label, ls in configs:
            w = (1.0,) * len(dims)
            per_row = tau_label == "row"
            tau_plus = make_tau_row(B) if per_row else tau_label
            q, d, h = make_inputs(B, K, dtype, pos_dupes=debias_on(tau_plus))
            bad_q, _ = mrl_ambiguous(q, d, h, dims, use_qq)
            kq = ~bad_q
            qr = q.float().detach().requires_grad_(True)
            ref = reference_mrl(qr, d.float(),
                                h.float() if h is not None else None,
                                dims, w, use_qq=use_qq, tau_plus=tau_plus,
                                label_smoothing=ls)
            ref.backward()

            loss_fn = MemoryEfficientMatryoshkaLiTQwen3Loss(
                dims, temperature=TAU, margin=MARGIN, use_qq_negatives=use_qq,
                normalized_inputs=True, stable=False,
                tau_plus=0.0 if per_row else tau_plus, label_smoothing=ls)
            qk = q.detach().requires_grad_(True)
            loss = loss_fn(qk, d, h, tau_plus=tau_plus if per_row else None)
            loss.backward()

            print(f"mrl-lit {str(dtype).split('.')[-1]} B={B} dims={dims} K={K} "
                  f"qq={use_qq} tau+={tau_label} ls={ls} amb={int(bad_q.sum())}q")
            check("loss", loss.detach(), ref.detach(), tol)
            check("dQ", qk.grad[kq], qr.grad[kq], tol)


def run_wrapper():
    """The eager wrapper against the same reference (it is the design oracle
    for the fused kernels, so it must be independently correct)."""
    for B, dims, K, use_qq, use_dd, tau_label, ls in [
            (1024, DIMS3, 4, True, True, 0.0, 0.0),
            (1024, DIMS2, 4, True, True, 0.3, 0.1),
            (1000, (32, 96, 256), 3, False, False, 0.0, 0.0)]:  # unaligned dims OK here
        w = (1.0,) * len(dims)
        tau_plus = tau_label
        q, d, h = make_inputs(B, K, torch.float32, pos_dupes=debias_on(tau_plus))
        bad_q, bad_d = mrl_ambiguous(q, d, h, dims, use_qq, use_dd)
        kq, kd = ~bad_q, ~bad_d
        qr, dr, hr = leafs(q, d, h, torch.float32)
        ref = reference_mrl(qr, dr, hr, dims, w, use_qq=use_qq, use_dd=use_dd,
                            tau_plus=tau_plus, label_smoothing=ls)
        ref.backward()

        loss_fn = MatryoshkaLoss(MemoryEfficientQwen3Loss(
            temperature=TAU, margin=MARGIN, use_qq_negatives=use_qq,
            use_dd_negatives=use_dd, normalized_inputs=False, stable=False,
            tau_plus=tau_plus, label_smoothing=ls), dims)
        qk, dk, hk = leafs(q, d, h)
        loss = loss_fn(qk, dk, hk)
        loss.backward()

        print(f"wrapper B={B} dims={dims} K={K} qq={use_qq} dd={use_dd} "
              f"tau+={tau_label} ls={ls} amb={int(bad_q.sum())}q/{int(bad_d.sum())}d")
        check("loss", loss.detach(), ref.detach(), 1e-3)
        check("dQ", qk.grad[kq], qr.grad[kq], 1e-3)
        check("dD", dk.grad[kd], dr.grad[kd], 1e-3)
        if hk is not None:
            check("dH", hk.grad[kq], hr.grad[kq], 1e-3)


def run_dist_world1():
    """The fused distributed modules at world=1 (no process group): every
    kernel launch and eager term minus the collectives."""
    for K, use_qq, use_dd, tau_plus, ls in [(4, True, True, 0.0, 0.0),
                                            (4, True, True, 0.3, 0.1),
                                            (0, False, False, 0.0, 0.0)]:
        w = (1.0,) * len(DIMS3)
        q, d, h = make_inputs(1024, K, torch.float32, pos_dupes=debias_on(tau_plus))
        bad_q, bad_d = mrl_ambiguous(q, d, h, DIMS3, use_qq, use_dd)
        kq, kd = ~bad_q, ~bad_d
        qr, dr, hr = leafs(q, d, h)
        ref = reference_mrl(qr, dr, hr, DIMS3, w, use_qq=use_qq, use_dd=use_dd,
                            tau_plus=tau_plus, label_smoothing=ls)
        ref.backward()
        loss_fn = DistributedMemoryEfficientMatryoshkaQwen3Loss(
            DIMS3, temperature=TAU, margin=MARGIN, use_qq_negatives=use_qq,
            use_dd_negatives=use_dd, normalized_inputs=True, stable=False,
            tau_plus=tau_plus, label_smoothing=ls)
        qk, dk, hk = leafs(q, d, h)
        loss = loss_fn(qk, dk, hk)
        loss.backward()
        print(f"mrl-clip dist world=1 K={K} qq={use_qq} dd={use_dd} "
              f"tau+={tau_plus} ls={ls} amb={int(bad_q.sum())}q/{int(bad_d.sum())}d")
        check("loss", loss.detach(), ref.detach(), 1e-3)
        check("dQ", qk.grad[kq], qr.grad[kq], 1e-3)
        check("dD", dk.grad[kd], dr.grad[kd], 1e-3)
        if hk is not None:
            check("dH", hk.grad[kq], hr.grad[kq], 1e-3)

    q, d, h = make_inputs(1024, 4, torch.float32)
    bad_q, _ = mrl_ambiguous(q, d, h, DIMS3, use_qq=True)
    kq = ~bad_q
    w = (1.0,) * len(DIMS3)
    qr = q.detach().requires_grad_(True)
    ref = reference_mrl(qr, d, h, DIMS3, w, use_qq=True)
    ref.backward()
    loss_fn = DistributedMemoryEfficientMatryoshkaLiTQwen3Loss(
        DIMS3, temperature=TAU, margin=MARGIN, use_qq_negatives=True,
        normalized_inputs=True, stable=False)
    qk = q.detach().requires_grad_(True)
    loss = loss_fn(qk, d, h)
    loss.backward()
    print(f"mrl-lit dist world=1 qq=True amb={int(bad_q.sum())}q")
    check("loss", loss.detach(), ref.detach(), 1e-3)
    check("dQ", qk.grad[kq], qr.grad[kq], 1e-3)


def run_distributed():
    import memeff.mrl_qwen3_loss as _mrl

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
        for dims, K, use_qq, use_dd, stable, tau_label, ls in [
                (DIMS3, 0, False, False, False, 0.0, 0.0),
                (DIMS3, 4, True, True, False, 0.0, 0.0),
                (DIMS2, 4, True, True, True, 0.0, 0.0),
                (DIMS3, 4, True, True, False, 0.3, 0.0),
                (DIMS3, 4, True, True, False, "row", 0.0),
                (DIMS3, 4, True, True, False, 0.0, 0.1),
                (DIMS3, 4, True, True, True, 0.3, 0.1)]:
            # identical global batch on every rank, each takes its shard
            w = (1.0,) * len(dims)
            per_row = tau_label == "row"
            tau_plus = make_tau_row(B) if per_row else tau_label   # same seed everywhere
            q, d, h = make_inputs(B, K, torch.float32, seed=7,
                                  pos_dupes=debias_on(tau_plus))
            bad_q, bad_d = mrl_ambiguous(q, d, h, dims, use_qq, use_dd)
            kq, kd = ~bad_q[rows], ~bad_d[rows]
            qr, dr, hr = leafs(q, d, h)
            ref = reference_mrl(qr, dr, hr, dims, w, use_qq=use_qq, use_dd=use_dd,
                                tau_plus=tau_plus, label_smoothing=ls)
            ref.backward()

            loss_fn = DistributedMemoryEfficientMatryoshkaQwen3Loss(
                dims, temperature=TAU, margin=MARGIN, use_qq_negatives=use_qq,
                use_dd_negatives=use_dd, normalized_inputs=True, stable=stable,
                tau_plus=0.0 if per_row else tau_plus, label_smoothing=ls)
            qs, ds, hs = leafs(q[rows], d[rows], h[rows] if h is not None else None)
            partial = loss_fn(qs, ds, hs,
                              tau_plus=tau_plus[rows] if per_row else None)
            partial.backward()
            total = partial.detach().clone()
            dist.all_reduce(total)

            sc = math.sqrt(B * TAU) if stable else 1.0
            if rank == 0:
                print(f"mrl-clip dist world={world} dims={dims} K={K} qq={use_qq} "
                      f"dd={use_dd} stable={stable} tau+={tau_label} ls={ls} "
                      f"amb={int(bad_q.sum())}q/{int(bad_d.sum())}d")
            check(f"loss(rank{rank})", total, ref.detach(), tol)
            check(f"dQ(rank{rank})", qs.grad[kq], qr.grad[rows][kq] * sc, tol)
            check(f"dD(rank{rank})", ds.grad[kd], dr.grad[rows][kd] * sc, tol)
            if hs is not None:
                check(f"dH(rank{rank})", hs.grad[kq], hr.grad[rows][kq] * sc, tol)
            dist.barrier()

        for dims, K, use_qq, tau_label, ls in [
                (DIMS3, 0, False, 0.0, 0.0), (DIMS3, 4, True, 0.0, 0.0),
                (DIMS3, 4, True, 0.3, 0.0), (DIMS3, 4, True, "row", 0.0),
                (DIMS2, 4, True, 0.0, 0.1)]:
            w = (1.0,) * len(dims)
            per_row = tau_label == "row"
            tau_plus = make_tau_row(B) if per_row else tau_label
            q, d, h = make_inputs(B, K, torch.float32, seed=11,
                                  pos_dupes=debias_on(tau_plus))
            bad_q, _ = mrl_ambiguous(q, d, h, dims, use_qq)
            kq = ~bad_q[rows]
            qr = q.detach().requires_grad_(True)
            ref = reference_mrl(qr, d, h, dims, w, use_qq=use_qq,
                                tau_plus=tau_plus, label_smoothing=ls)
            ref.backward()

            loss_fn = DistributedMemoryEfficientMatryoshkaLiTQwen3Loss(
                dims, temperature=TAU, margin=MARGIN, use_qq_negatives=use_qq,
                normalized_inputs=True, stable=False,
                tau_plus=0.0 if per_row else tau_plus, label_smoothing=ls)
            qs = q[rows].detach().requires_grad_(True)
            partial = loss_fn(qs, d[rows], h[rows] if h is not None else None,
                              tau_plus=tau_plus[rows] if per_row else None)
            partial.backward()
            total = partial.detach().clone()
            dist.all_reduce(total)

            if rank == 0:
                print(f"mrl-lit dist world={world} dims={dims} K={K} qq={use_qq} "
                      f"tau+={tau_label} ls={ls} amb={int(bad_q.sum())}q")
            check(f"loss(rank{rank})", total, ref.detach(), tol)
            check(f"dQ(rank{rank})", qs.grad[kq], qr.grad[rows][kq], tol)
            dist.barrier()
    _mrl._FORCE_BACKWARD = None
    dist.destroy_process_group()


def run_asserts():
    """Misaligned / unsorted / non-terminal dims must fail loudly."""
    q, d, _ = make_inputs(256, 0, torch.float32)
    ctor_bad = [(128, 64), (64, 64, 256), ()]
    fwd_bad = [(60, 256), (64, 128), (16, 256), (64, 192, 256, 320)]
    for dims in ctor_bad:
        try:
            MemoryEfficientMatryoshkaQwen3Loss(dims)
            raise AssertionError(f"ctor accepted dims={dims}")
        except ValueError:
            pass
    for dims in fwd_bad:
        loss_fn = MemoryEfficientMatryoshkaQwen3Loss(dims, normalized_inputs=True)
        try:
            loss_fn(q, d)
            raise AssertionError(f"forward accepted dims={dims}")
        except ValueError:
            pass
    try:
        MatryoshkaLoss(MemoryEfficientQwen3Loss(normalized_inputs=True), (64, 256))
        raise AssertionError("wrapper accepted normalized_inputs=True")
    except ValueError:
        pass
    print("asserts OK (misaligned/unsorted/non-terminal dims and "
          "normalized_inputs wrapper all raise)")


if __name__ == "__main__":
    import memeff.mrl_qwen3_loss as _mrl

    torch.backends.cuda.matmul.allow_tf32 = False
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        run_distributed()
    else:
        run_asserts()
        run_wrapper()
        run_dist_world1()
        for force in ("prefix", "telescope", "fa"):   # every backward kernel
            _mrl._FORCE_BACKWARD = force
            print(f"--- fused suites, {force} backward ---")
            run_fused_clip()
            run_fused_lit()
        _mrl._FORCE_BACKWARD = None
        print("ALL OK")
