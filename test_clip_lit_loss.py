"""Correctness tests for the CLIP and LiT losses (needs a CUDA GPU).

Single GPU:  python test_clip_lit_loss.py
Multi GPU:   torchrun --nproc-per-node=2 test_clip_lit_loss.py

Compares losses and gradients against a dense autograd reference. Kernels run with
ieee fp32 matmuls here (and torch tf32 stays off) so the comparison is not drowned
in tensor-core rounding noise.
"""
import math
import os

os.environ.setdefault("MEMEFF_INPUT_PRECISION", "ieee")   # must precede the imports

import torch
import torch.distributed as dist
import torch.nn.functional as F

from memeff.clip_loss import MemoryEfficientCLIPLoss, StableMemoryEfficientCLIPLoss
from memeff.lit_loss import MemoryEfficientLiTLoss, StableMemoryEfficientLiTLoss
from memeff.distributed_clip_loss import DistributedMemoryEfficientCLIPLoss
from memeff.distributed_lit_loss import DistributedMemoryEfficientLiTLoss

TAU, DIM = 0.07, 256


def debiased_denominator(sums, pos, num_negatives, tau=TAU, tau_plus=0.0):
    """Debiased softmax denominator (arXiv 2007.00224) in the shifted exp domain;
    the clamp's zero gradient is exactly the paper's 'g held at the floor'.
    tau_plus broadcasts, so a (B,) tensor gives per-row priors."""
    g = ((sums - pos) / num_negatives - tau_plus * pos) / (1.0 - tau_plus)
    return pos + num_negatives * g.clamp_min(math.exp(-2.0 / tau))


def debias_on(tau_plus):
    return isinstance(tau_plus, torch.Tensor) or bool(tau_plus)


def make_tau_row(B, seed=3):
    """Per-row priors mixing zero and nonzero rows; the dupe rows (0, 1) get a
    prior high enough to fire the estimator clamp."""
    gen = torch.Generator().manual_seed(seed)
    t = torch.rand(B, generator=gen) * 0.5
    t[::7] = 0.0
    t[:2] = 0.45
    return t.cuda()


def smoothed_targets(B, label_smoothing, device):
    """The F.cross_entropy label-smoothing target matrix: (1-eps) I + eps/B."""
    return (torch.full((B, B), label_smoothing / B, device=device)
            + (1.0 - label_smoothing) * torch.eye(B, device=device))


def reference_clip(x, y, tau=TAU, tau_plus=0.0, label_smoothing=0.0):
    B = x.shape[0]
    if not debias_on(tau_plus):
        logits = x @ y.T / tau
        target = torch.arange(B, device=x.device)
        return 0.5 * (F.cross_entropy(logits, target, label_smoothing=label_smoothing)
                      + F.cross_entropy(logits.T, target, label_smoothing=label_smoothing))
    lnE = (x @ y.T - 1.0) / tau
    E = torch.exp(lnE)
    pos = E.diagonal()
    Dr = debiased_denominator(E.sum(1), pos, B - 1, tau, tau_plus)
    Dc = debiased_denominator(E.sum(0), pos, B - 1, tau, tau_plus)
    t = smoothed_targets(B, label_smoothing, x.device)
    return 0.5 * ((torch.log(Dr) - (t * lnE).sum(1))
                  + (torch.log(Dc) - (t * lnE).sum(0))).mean()


def reference_lit(x, y, tau=TAU, tau_plus=0.0, label_smoothing=0.0):
    B = x.shape[0]
    if not debias_on(tau_plus):
        logits = x @ y.T / tau
        target = torch.arange(B, device=x.device)
        return F.cross_entropy(logits, target, label_smoothing=label_smoothing)
    lnE = (x @ y.T - 1.0) / tau
    E = torch.exp(lnE)
    pos = E.diagonal()
    D = debiased_denominator(E.sum(1), pos, B - 1, tau, tau_plus)
    t = smoothed_targets(B, label_smoothing, x.device)
    return (torch.log(D) - (t * lnE).sum(1)).mean()


def make_inputs(B, dtype, seed=0, pos_dupes=False):
    gen = torch.Generator().manual_seed(seed)
    x = F.normalize(torch.randn(B, DIM, generator=gen), dim=-1)
    y = F.normalize(torch.randn(B, DIM, generator=gen), dim=-1)
    if pos_dupes:   # pos ~ 1 rows make the debiased clamp fire
        y[0] = x[0]
        y[1] = F.normalize(x[1] + 0.01 * torch.randn(DIM, generator=gen), dim=-1)
    to = lambda t: t.cuda().to(dtype).contiguous()
    return to(x), to(y)


def rel_err(a, b):
    a = a.float()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()


def check(name, a, b, tol):
    err = rel_err(a, b)
    print(f"  {'OK ' if err < tol else 'FAIL'} {name}: rel err {err:.2e}")
    assert err < tol, f"{name}: {err:.2e} >= {tol:.0e}"


def run_single():
    configs = [  # (B, stable, tau_plus, label_smoothing)
        (1024, False, 0.0, 0.0),
        (1000, False, 0.0, 0.0),   # batch not a multiple of the block sizes
        (1024, True, 0.0, 0.0),    # stable gradient rescaling
        (1024, False, 0.1, 0.0),   # debiased contrastive loss
        (1024, False, 0.5, 0.0),   # ... with the estimator clamp firing on dupe rows
        (1024, True, 0.3, 0.0),    # debiased + stable
        (1024, False, "row", 0.0), # per-row priors passed per call
        (1024, False, 0.0, 0.1),   # label smoothing
        (1024, True, 0.0, 0.1),    # ... with the stable gradient rescale
        (1024, False, 0.3, 0.1),   # ... composed with debiasing
        (1024, False, "row", 0.1), # ... and with per-row priors
    ]
    for dtype, tol in ((torch.float32, 1e-3), (torch.bfloat16, 5e-2)):
        for B, stable, tau_label, ls in configs:
            per_row = tau_label == "row"
            tau_plus = make_tau_row(B) if per_row else tau_label
            x, y = make_inputs(B, dtype, pos_dupes=debias_on(tau_plus))
            sc = math.sqrt(B * TAU) if stable else 1.0   # stable rescales grads only

            xr = x.float().detach().requires_grad_(True)
            yr = y.float().detach().requires_grad_(True)
            ref = reference_clip(xr, yr, tau_plus=tau_plus, label_smoothing=ls)
            ref.backward()

            loss_fn = MemoryEfficientCLIPLoss(
                temperature=TAU, normalized_inputs=True, stable=stable,
                tau_plus=0.0 if per_row else tau_plus, label_smoothing=ls)
            xk = x.detach().requires_grad_(True)
            yk = y.detach().requires_grad_(True)
            loss = loss_fn(xk, yk, tau_plus=tau_plus if per_row else None)
            loss.backward()

            print(f"clip {str(dtype).split('.')[-1]} B={B} stable={stable} "
                  f"tau+={tau_label} ls={ls}")
            check("loss", loss.detach(), ref.detach(), tol)
            check("dX", xk.grad, xr.grad * sc, tol)
            check("dY", yk.grad, yr.grad * sc, tol)

            xr = x.float().detach().requires_grad_(True)
            ref = reference_lit(xr, y.float(), tau_plus=tau_plus, label_smoothing=ls)
            ref.backward()

            lit_fn = MemoryEfficientLiTLoss(
                temperature=TAU, normalized_inputs=True, stable=stable,
                tau_plus=0.0 if per_row else tau_plus, label_smoothing=ls)
            xk = x.detach().requires_grad_(True)
            yk = y.detach().requires_grad_(True)
            loss = lit_fn(xk, yk, tau_plus=tau_plus if per_row else None)
            loss.backward()

            print(f"lit  {str(dtype).split('.')[-1]} B={B} stable={stable} "
                  f"tau+={tau_label} ls={ls}")
            check("loss", loss.detach(), ref.detach(), tol)
            check("dX", xk.grad, xr.grad * sc, tol)
            assert yk.grad is None, "locked image tower must receive no gradient"


def run_aliases():
    """The deprecated Stable* classes must behave exactly as stable=True."""
    B = 1024
    x, y = make_inputs(B, torch.float32)
    for alias_cls, flag_cls, unidirectional in (
            (StableMemoryEfficientCLIPLoss, MemoryEfficientCLIPLoss, False),
            (StableMemoryEfficientLiTLoss, MemoryEfficientLiTLoss, True)):
        xa = x.detach().requires_grad_(True)
        ya = y.detach().requires_grad_(True)
        loss_a = alias_cls(temperature=TAU, normalized_inputs=True)(xa, ya)
        loss_a.backward()

        xf = x.detach().requires_grad_(True)
        yf = y.detach().requires_grad_(True)
        loss_f = flag_cls(temperature=TAU, normalized_inputs=True, stable=True)(xf, yf)
        loss_f.backward()

        print(f"alias {alias_cls.__name__}")
        check("loss", loss_a.detach(), loss_f.detach(), 1e-6)
        check("dX", xa.grad, xf.grad, 1e-6)
        if not unidirectional:
            check("dY", ya.grad, yf.grad, 1e-6)


def run_world1_fallbacks():
    B, tol = 1024, 1e-3
    for tau_label, ls in ((0.0, 0.0), (0.3, 0.0), ("row", 0.0), (0.0, 0.1), (0.3, 0.1)):
        per_row = tau_label == "row"
        tau_plus = make_tau_row(B) if per_row else tau_label
        x, y = make_inputs(B, torch.float32, pos_dupes=debias_on(tau_plus))

        xr = x.detach().requires_grad_(True)
        yr = y.detach().requires_grad_(True)
        ref = reference_clip(xr, yr, tau_plus=tau_plus, label_smoothing=ls)
        ref.backward()
        loss_fn = DistributedMemoryEfficientCLIPLoss(
            temperature=TAU, normalized_inputs=True,
            tau_plus=0.0 if per_row else tau_plus, label_smoothing=ls)
        xk = x.detach().requires_grad_(True)
        yk = y.detach().requires_grad_(True)
        loss = loss_fn(xk, yk, tau_plus=tau_plus if per_row else None)
        loss.backward()
        print(f"clip dist world=1 fallback tau+={tau_label} ls={ls}")
        check("loss", loss.detach(), ref.detach(), tol)
        check("dX", xk.grad, xr.grad, tol)
        check("dY", yk.grad, yr.grad, tol)

        xr = x.detach().requires_grad_(True)
        ref = reference_lit(xr, y, tau_plus=tau_plus, label_smoothing=ls)
        ref.backward()
        loss_fn = DistributedMemoryEfficientLiTLoss(
            temperature=TAU, normalized_inputs=True,
            tau_plus=0.0 if per_row else tau_plus, label_smoothing=ls)
        xk = x.detach().requires_grad_(True)
        loss = loss_fn(xk, y, tau_plus=tau_plus if per_row else None)
        loss.backward()
        print(f"lit dist world=1 fallback tau+={tau_label} ls={ls}")
        check("loss", loss.detach(), ref.detach(), tol)
        check("dX", xk.grad, xr.grad, tol)


def run_distributed():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    local_b = 512
    B = world * local_b
    rows = slice(rank * local_b, (rank + 1) * local_b)
    tol = 1e-3

    for stable, tau_label, ls in ((False, 0.0, 0.0), (True, 0.0, 0.0), (False, 0.3, 0.0),
                                  (False, "row", 0.0), (False, 0.0, 0.1),
                                  (True, "row", 0.1)):
        # identical global batch on every rank, each takes its shard
        per_row = tau_label == "row"
        tau_plus = make_tau_row(B) if per_row else tau_label   # same seed on every rank
        x, y = make_inputs(B, torch.float32, seed=7, pos_dupes=debias_on(tau_plus))
        sc = math.sqrt(B * TAU) if stable else 1.0
        xr = x.detach().requires_grad_(True)
        yr = y.detach().requires_grad_(True)
        ref = reference_clip(xr, yr, tau_plus=tau_plus, label_smoothing=ls)
        ref.backward()

        loss_fn = DistributedMemoryEfficientCLIPLoss(
            temperature=TAU, normalized_inputs=True, stable=stable,
            tau_plus=0.0 if per_row else tau_plus, label_smoothing=ls)
        xs = x[rows].detach().requires_grad_(True)
        ys = y[rows].detach().requires_grad_(True)
        partial = loss_fn(xs, ys, tau_plus=tau_plus[rows] if per_row else None)
        partial.backward()
        total = partial.detach().clone()
        dist.all_reduce(total)

        if rank == 0:
            print(f"clip dist world={world} stable={stable} tau+={tau_label} ls={ls}")
        check(f"loss(rank{rank})", total, ref.detach(), tol)
        check(f"dX(rank{rank})", xs.grad, xr.grad[rows] * sc, tol)
        check(f"dY(rank{rank})", ys.grad, yr.grad[rows] * sc, tol)
        dist.barrier()

    for tau_label, ls in ((0.0, 0.0), (0.3, 0.0), ("row", 0.0), (0.0, 0.1), ("row", 0.1)):
        per_row = tau_label == "row"
        tau_plus = make_tau_row(B) if per_row else tau_label   # same seed on every rank
        x, y = make_inputs(B, torch.float32, seed=11, pos_dupes=debias_on(tau_plus))
        xr = x.detach().requires_grad_(True)
        ref = reference_lit(xr, y, tau_plus=tau_plus, label_smoothing=ls)
        ref.backward()

        loss_fn = DistributedMemoryEfficientLiTLoss(
            temperature=TAU, normalized_inputs=True,
            tau_plus=0.0 if per_row else tau_plus, label_smoothing=ls)
        xs = x[rows].detach().requires_grad_(True)
        ys = y[rows].detach().requires_grad_(True)
        partial = loss_fn(xs, ys, tau_plus=tau_plus[rows] if per_row else None)
        partial.backward()
        total = partial.detach().clone()
        dist.all_reduce(total)

        if rank == 0:
            print(f"lit dist world={world} tau+={tau_label} ls={ls}")
        check(f"loss(rank{rank})", total, ref.detach(), tol)
        check(f"dX(rank{rank})", xs.grad, xr.grad[rows], tol)
        assert ys.grad is None, "locked image tower must receive no gradient"
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        run_distributed()
    else:
        run_single()
        run_aliases()
        run_world1_fallbacks()
        print("ALL OK")
