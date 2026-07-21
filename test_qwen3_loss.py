"""Correctness tests for the Qwen3 losses (needs a CUDA GPU).

Single GPU:  python test_qwen3_loss.py
Multi GPU:   torchrun --nproc-per-node=2 test_qwen3_loss.py

Compares losses and gradients against a dense autograd reference. Kernels run with
ieee fp32 matmuls here (and torch tf32 stays off) so the false-negative mask makes
the same boundary decisions in the kernel and the reference: under tf32 a flipped
near-threshold negative carries up to exp(margin/temp) times the positive's softmax
weight, which would drown the comparison in mask-flip noise.

Even under ieee the kernel's block-chunked matmul and torch's cublas can round a
similarity to opposite sides of the mask threshold when it lands within a few ulps
of it, and at B ~ 1000 the ~10^6 pair similarities make such ties likely for any
seed. Rows touched by a near-tie may therefore legitimately disagree with the
reference, and are excluded from the gradient comparisons (see ambiguous_rows).
"""
import math
import os

os.environ.setdefault("MEMEFF_INPUT_PRECISION", "ieee")   # must precede the imports

import torch
import torch.distributed as dist
import torch.nn.functional as F

from memeff.clip_qwen3_loss import (
    MemoryEfficientQwen3Loss, DistributedMemoryEfficientQwen3Loss)
from memeff.lit_qwen3_loss import (
    MemoryEfficientLiTQwen3Loss, DistributedMemoryEfficientLiTQwen3Loss)

TAU, MARGIN, DIM = 0.05, 0.1, 256


def debias_on(tau_plus):
    return isinstance(tau_plus, torch.Tensor) or bool(tau_plus)


def make_tau_row(B, seed=3):
    """Per-row priors mixing zero and nonzero rows; the pos-dupe row (5) gets a
    prior high enough to fire the estimator clamp."""
    gen = torch.Generator().manual_seed(seed)
    t = torch.rand(B, generator=gen) * 0.5
    t[::7] = 0.0
    t[5] = 0.45
    return t.cuda()


def reference_loss(q, d, h, tau=TAU, margin=MARGIN, use_qq=False, use_dd=False,
                   tau_plus=0.0, label_smoothing=0.0):
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
    N = (B - 1) * (1 + use_qq + use_dd) + (h.shape[1] if h is not None else 0)
    if debias_on(tau_plus):   # debiased denominator (2007.00224), nominal count
        pos_e = torch.exp((pos - 1.0) / tau)
        g = ((Z - pos_e) / N - tau_plus * pos_e) / (1.0 - tau_plus)
        Z = pos_e + N * g.clamp_min(math.exp(-2.0 / tau))
    lnE_pos = (pos - 1.0) / tau
    if not label_smoothing:
        return (torch.log(Z) - lnE_pos).mean()
    # smoothed target: (1-eps) on the positive, eps/C uniform over the C = N + 1
    # nominal candidates (the false-negative mask does not reshape the target).
    cand = ((q @ d.T - 1.0) / tau).sum(1)
    if use_qq:
        cand = cand + ((q @ q.T - 1.0) / tau).sum(1) - ((q * q).sum(1) - 1.0) / tau
    if use_dd:
        cand = cand + ((d @ d.T - 1.0) / tau).sum(1) - ((d * d).sum(1) - 1.0) / tau
    if h is not None:
        cand = cand + ((torch.einsum('bkd,bd->bk', h, q) - 1.0) / tau).sum(1)
    blend = (1.0 - label_smoothing) * lnE_pos + (label_smoothing / (N + 1)) * cand
    return (torch.log(Z) - blend).mean()


def make_inputs(B, K, dtype, seed=0, pos_dupes=False):
    gen = torch.Generator().manual_seed(seed)
    q = F.normalize(torch.randn(B, DIM, generator=gen), dim=-1)
    d = F.normalize(torch.randn(B, DIM, generator=gen), dim=-1)
    # near-duplicates so the false-negative mask fires on every negative group
    d[1] = F.normalize(q[0] + 0.01 * torch.randn(DIM, generator=gen), dim=-1)
    q[3] = F.normalize(q[2] + 0.01 * torch.randn(DIM, generator=gen), dim=-1)
    if pos_dupes:   # pos ~ 1 rows make the debiased clamp fire
        d[5] = q[5]
    h = None
    if K:
        h = F.normalize(torch.randn(B, K, DIM, generator=gen), dim=-1)
        h[4, 0] = F.normalize(d[4] + 0.005 * torch.randn(DIM, generator=gen), dim=-1)
    to = lambda t: t.cuda().to(dtype).contiguous()
    return to(q), to(d), (to(h) if h is not None else None)


def ambiguous_rows(q, d, h, use_qq=False, use_dd=False, eps=1e-6):
    """Rows whose gradients may legitimately differ from the reference because a
    similarity in them sits within eps of the mask threshold (kernel and reference
    matmuls can round such a tie to opposite sides). Returns boolean masks (bad_q,
    bad_d) over the rows of dQ/dH and dD; the loss itself moves by at most ~1/B of
    one softmax weight per tie, far under the tolerances, so it is always checked."""
    q, d = q.float(), d.float()
    B = q.shape[0]
    thr = (q * d).sum(-1) + MARGIN
    eye = torch.eye(B, dtype=torch.bool, device=q.device)

    def ties(S, excl_diag):
        t = (S - thr[:, None]).abs() < eps
        return t & ~eye if excl_diag else t

    bad_q = torch.zeros(B, dtype=torch.bool, device=q.device)
    bad_d = torch.zeros(B, dtype=torch.bool, device=q.device)
    t = ties(q @ d.T, False)
    bad_q |= t.any(1)                     # row r's denominator moved
    bad_d |= t.any(1) | t.any(0)          # ... and column c received the flipped weight
    if use_qq:
        t = ties(q @ q.T, True)
        bad_q |= t.any(1) | t.any(0)
        bad_d |= t.any(1)
    if use_dd:
        t = ties(d @ d.T, True)
        bad_d |= t.any(1) | t.any(0)
        bad_q |= t.any(1)
    if h is not None:
        t = (torch.einsum('bkd,bd->bk', h.float(), q) - thr[:, None]).abs() < eps
        bad_q |= t.any(1)
        bad_d |= t.any(1)
    return bad_q, bad_d


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
    configs = [  # (B, K, use_qq, use_dd, stable, tau_plus, label_smoothing)
        (1024, 0, False, False, False, 0.0, 0.0),
        (1024, 4, False, False, False, 0.0, 0.0),
        (1024, 4, True, False, False, 0.0, 0.0),
        (1024, 4, False, True, False, 0.0, 0.0),
        (1024, 4, True, True, False, 0.0, 0.0),
        (1000, 3, True, True, False, 0.0, 0.0),   # batch not a multiple of block sizes
        (1024, 4, True, True, True, 0.0, 0.0),    # stable gradient rescaling
        (1024, 4, True, True, False, 0.3, 0.0),   # debiased contrastive loss
        (1024, 0, False, False, False, 0.5, 0.0), # ... with the estimator clamp firing
        (1024, 4, True, True, False, "row", 0.0), # per-row priors passed per call
        (1024, 0, False, False, False, 0.0, 0.1), # label smoothing, plain
        (1024, 4, True, True, False, 0.0, 0.1),   # ... with every negative group
        (1024, 4, True, True, True, 0.3, 0.1),    # ... + stable + debiasing
    ]
    for dtype, tol in ((torch.float32, 1e-3), (torch.bfloat16, 5e-2)):
        for B, K, use_qq, use_dd, stable, tau_label, ls in configs:
            per_row = tau_label == "row"
            tau_plus = make_tau_row(B) if per_row else tau_label
            q, d, h = make_inputs(B, K, dtype, pos_dupes=debias_on(tau_plus))
            bad_q, bad_d = ambiguous_rows(q, d, h, use_qq, use_dd)
            kq, kd = ~bad_q, ~bad_d
            qr, dr, hr = leafs(q, d, h, torch.float32)
            ref = reference_loss(qr, dr, hr, use_qq=use_qq, use_dd=use_dd,
                                 tau_plus=tau_plus, label_smoothing=ls)
            ref.backward()

            loss_fn = MemoryEfficientQwen3Loss(
                temperature=TAU, margin=MARGIN, use_qq_negatives=use_qq,
                use_dd_negatives=use_dd, normalized_inputs=True, stable=stable,
                tau_plus=0.0 if per_row else tau_plus, label_smoothing=ls)
            qk, dk, hk = leafs(q, d, h)
            loss = loss_fn(qk, dk, hk, tau_plus=tau_plus if per_row else None)
            loss.backward()

            sc = math.sqrt(B * TAU) if stable else 1.0   # stable rescales grads only
            print(f"clip {str(dtype).split('.')[-1]} B={B} K={K} "
                  f"qq={use_qq} dd={use_dd} stable={stable} tau+={tau_label} ls={ls} "
                  f"amb={int(bad_q.sum())}q/{int(bad_d.sum())}d")
            check("loss", loss.detach(), ref.detach(), tol)
            check("dQ", qk.grad[kq], qr.grad[kq] * sc, tol)
            check("dD", dk.grad[kd], dr.grad[kd] * sc, tol)
            if hk is not None:
                check("dH", hk.grad[kq], hr.grad[kq] * sc, tol)


def run_single_lit():
    configs = [  # (B, K, use_qq, tau_plus, label_smoothing)
        (1024, 0, False, 0.0, 0.0),
        (1024, 4, False, 0.0, 0.0),
        (1024, 4, True, 0.0, 0.0),
        (1000, 3, True, 0.0, 0.0),
        (1024, 4, True, 0.3, 0.0),   # debiased contrastive loss
        (1024, 4, True, "row", 0.0), # per-row priors passed per call
        (1024, 4, True, 0.0, 0.1),   # label smoothing
        (1024, 4, True, 0.3, 0.1),   # ... composed with debiasing
    ]
    for dtype, tol in ((torch.float32, 1e-3), (torch.bfloat16, 5e-2)):
        for B, K, use_qq, tau_label, ls in configs:
            per_row = tau_label == "row"
            tau_plus = make_tau_row(B) if per_row else tau_label
            q, d, h = make_inputs(B, K, dtype, pos_dupes=debias_on(tau_plus))
            bad_q, _ = ambiguous_rows(q, d, h, use_qq)
            kq = ~bad_q
            qr = q.float().detach().requires_grad_(True)
            ref = reference_loss(qr, d.float(), h.float() if h is not None else None,
                                 use_qq=use_qq, tau_plus=tau_plus, label_smoothing=ls)
            ref.backward()

            loss_fn = MemoryEfficientLiTQwen3Loss(
                temperature=TAU, margin=MARGIN, use_qq_negatives=use_qq,
                normalized_inputs=True, stable=False,
                tau_plus=0.0 if per_row else tau_plus, label_smoothing=ls)
            qk = q.detach().requires_grad_(True)
            loss = loss_fn(qk, d, h, tau_plus=tau_plus if per_row else None)
            loss.backward()

            print(f"lit {str(dtype).split('.')[-1]} B={B} K={K} qq={use_qq} "
                  f"tau+={tau_label} ls={ls} amb={int(bad_q.sum())}q")
            check("loss", loss.detach(), ref.detach(), tol)
            check("dQ", qk.grad[kq], qr.grad[kq], tol)


def run_distributed():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    local_b = 512
    B = world * local_b
    rows = slice(rank * local_b, (rank + 1) * local_b)
    tol = 1e-3

    for K, use_qq, use_dd, tau_label, ls in [
            (0, False, False, 0.0, 0.0), (4, False, False, 0.0, 0.0),
            (4, True, False, 0.0, 0.0), (4, False, True, 0.0, 0.0),
            (4, True, True, 0.0, 0.0), (4, True, True, 0.3, 0.0),
            (4, True, True, "row", 0.0), (4, True, True, 0.0, 0.1),
            (4, True, True, "row", 0.1)]:
        # identical global batch on every rank, each takes its shard
        per_row = tau_label == "row"
        tau_plus = make_tau_row(B) if per_row else tau_label   # same seed on every rank
        q, d, h = make_inputs(B, K, torch.float32, seed=7, pos_dupes=debias_on(tau_plus))
        bad_q, bad_d = ambiguous_rows(q, d, h, use_qq, use_dd)
        kq, kd = ~bad_q[rows], ~bad_d[rows]
        qr, dr, hr = leafs(q, d, h)
        ref = reference_loss(qr, dr, hr, use_qq=use_qq, use_dd=use_dd,
                             tau_plus=tau_plus, label_smoothing=ls)
        ref.backward()

        loss_fn = DistributedMemoryEfficientQwen3Loss(
            temperature=TAU, margin=MARGIN, use_qq_negatives=use_qq,
            use_dd_negatives=use_dd, normalized_inputs=True, stable=False,
            tau_plus=0.0 if per_row else tau_plus, label_smoothing=ls)
        qs, ds, hs = leafs(q[rows], d[rows], h[rows] if h is not None else None)
        partial = loss_fn(qs, ds, hs, tau_plus=tau_plus[rows] if per_row else None)
        partial.backward()
        total = partial.detach().clone()
        dist.all_reduce(total)

        if rank == 0:
            print(f"clip dist world={world} K={K} qq={use_qq} dd={use_dd} "
                  f"tau+={tau_label} ls={ls} amb={int(bad_q.sum())}q/{int(bad_d.sum())}d")
        check(f"loss(rank{rank})", total, ref.detach(), tol)
        check(f"dQ(rank{rank})", qs.grad[kq], qr.grad[rows][kq], tol)
        check(f"dD(rank{rank})", ds.grad[kd], dr.grad[rows][kd], tol)
        if hs is not None:
            check(f"dH(rank{rank})", hs.grad[kq], hr.grad[rows][kq], tol)
        dist.barrier()

    for K, use_qq, tau_label, ls in [(0, False, 0.0, 0.0), (4, False, 0.0, 0.0),
                                     (4, True, 0.0, 0.0), (4, True, 0.3, 0.0),
                                     (4, True, "row", 0.0), (4, True, 0.0, 0.1)]:
        per_row = tau_label == "row"
        tau_plus = make_tau_row(B) if per_row else tau_label   # same seed on every rank
        q, d, h = make_inputs(B, K, torch.float32, seed=11, pos_dupes=debias_on(tau_plus))
        bad_q, _ = ambiguous_rows(q, d, h, use_qq)
        kq = ~bad_q[rows]
        qr = q.detach().requires_grad_(True)
        ref = reference_loss(qr, d, h, use_qq=use_qq, tau_plus=tau_plus,
                             label_smoothing=ls)
        ref.backward()

        loss_fn = DistributedMemoryEfficientLiTQwen3Loss(
            temperature=TAU, margin=MARGIN, use_qq_negatives=use_qq,
            normalized_inputs=True, stable=False,
            tau_plus=0.0 if per_row else tau_plus, label_smoothing=ls)
        qs = q[rows].detach().requires_grad_(True)
        partial = loss_fn(qs, d[rows], h[rows] if h is not None else None,
                          tau_plus=tau_plus[rows] if per_row else None)
        partial.backward()
        total = partial.detach().clone()
        dist.all_reduce(total)

        if rank == 0:
            print(f"lit dist world={world} K={K} qq={use_qq} tau+={tau_label} ls={ls} "
                  f"amb={int(bad_q.sum())}q")
        check(f"loss(rank{rank})", total, ref.detach(), tol)
        check(f"dQ(rank{rank})", qs.grad[kq], qr.grad[rows][kq], tol)
        dist.barrier()
    dist.destroy_process_group()


def run_world1_fallbacks():
    q, d, h = make_inputs(1024, 4, torch.float32)
    bad_q, bad_d = ambiguous_rows(q, d, h, use_qq=True, use_dd=True)
    kq, kd = ~bad_q, ~bad_d
    qr, dr, hr = leafs(q, d, h)
    ref = reference_loss(qr, dr, hr, use_qq=True, use_dd=True)
    ref.backward()
    loss_fn = DistributedMemoryEfficientQwen3Loss(
        temperature=TAU, margin=MARGIN, use_qq_negatives=True,
        use_dd_negatives=True, normalized_inputs=True, stable=False)
    qk, dk, hk = leafs(q, d, h)
    loss = loss_fn(qk, dk, hk)
    loss.backward()
    print("clip dist world=1 fallback")
    check("loss", loss.detach(), ref.detach(), 1e-3)
    check("dQ", qk.grad[kq], qr.grad[kq], 1e-3)
    check("dD", dk.grad[kd], dr.grad[kd], 1e-3)
    check("dH", hk.grad[kq], hr.grad[kq], 1e-3)

    bad_q, _ = ambiguous_rows(q, d, h, use_qq=True)
    kq = ~bad_q
    qr = q.detach().requires_grad_(True)
    ref = reference_loss(qr, d, h, use_qq=True)
    ref.backward()
    loss_fn = DistributedMemoryEfficientLiTQwen3Loss(
        temperature=TAU, margin=MARGIN, use_qq_negatives=True,
        normalized_inputs=True, stable=False)
    qk = q.detach().requires_grad_(True)
    loss = loss_fn(qk, d, h)
    loss.backward()
    print("lit dist world=1 fallback")
    check("loss", loss.detach(), ref.detach(), 1e-3)
    check("dQ", qk.grad[kq], qr.grad[kq], 1e-3)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        run_distributed()
    else:
        import memeff.clip_qwen3_loss as _cq
        import memeff.lit_qwen3_loss as _lq
        for force in (False, True):
            _cq._FORCE_FA = _lq._FORCE_FA = force
            print(f"=== backward: {'FA' if force else 'atomic tile-grid'} ===")
            run_single_clip()
            run_single_lit()
        _cq._FORCE_FA = _lq._FORCE_FA = None
        run_world1_fallbacks()
        print("ALL OK")
