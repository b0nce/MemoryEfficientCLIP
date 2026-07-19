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

from clip_loss import MemoryEfficientCLIPLoss, StableMemoryEfficientCLIPLoss
from lit_loss import MemoryEfficientLiTLoss, StableMemoryEfficientLiTLoss
from distributed_clip_loss import DistributedMemoryEfficientCLIPLoss
from distributed_lit_loss import DistributedMemoryEfficientLiTLoss

TAU, DIM = 0.07, 256


def debiased_denominator(sums, pos, num_negatives, tau=TAU, tau_plus=0.0):
    """Debiased softmax denominator (arXiv 2007.00224) in the shifted exp domain;
    the clamp's zero gradient is exactly the paper's 'g held at the floor'."""
    g = ((sums - pos) / num_negatives - tau_plus * pos) / (1.0 - tau_plus)
    return pos + num_negatives * g.clamp_min(math.exp(-2.0 / tau))


def reference_clip(x, y, tau=TAU, tau_plus=0.0):
    if not tau_plus:
        logits = x @ y.T / tau
        target = torch.arange(x.shape[0], device=x.device)
        return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
    E = torch.exp((x @ y.T - 1.0) / tau)
    pos = E.diagonal()
    Dr = debiased_denominator(E.sum(1), pos, x.shape[0] - 1, tau, tau_plus)
    Dc = debiased_denominator(E.sum(0), pos, x.shape[0] - 1, tau, tau_plus)
    return -0.5 * ((torch.log(pos) - torch.log(Dr))
                   + (torch.log(pos) - torch.log(Dc))).mean()


def reference_lit(x, y, tau=TAU, tau_plus=0.0):
    if not tau_plus:
        logits = x @ y.T / tau
        target = torch.arange(x.shape[0], device=x.device)
        return F.cross_entropy(logits, target)
    E = torch.exp((x @ y.T - 1.0) / tau)
    pos = E.diagonal()
    D = debiased_denominator(E.sum(1), pos, x.shape[0] - 1, tau, tau_plus)
    return -(torch.log(pos) - torch.log(D)).mean()


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
    configs = [  # (B, stable, tau_plus)
        (1024, False, 0.0),
        (1000, False, 0.0),   # batch not a multiple of the block sizes
        (1024, True, 0.0),    # stable gradient rescaling
        (1024, False, 0.1),   # debiased contrastive loss
        (1024, False, 0.5),   # ... with the estimator clamp firing on dupe rows
        (1024, True, 0.3),    # debiased + stable
    ]
    for dtype, tol in ((torch.float32, 1e-3), (torch.bfloat16, 5e-2)):
        for B, stable, tau_plus in configs:
            x, y = make_inputs(B, dtype, pos_dupes=tau_plus > 0)
            sc = math.sqrt(B * TAU) if stable else 1.0   # stable rescales grads only

            xr = x.float().detach().requires_grad_(True)
            yr = y.float().detach().requires_grad_(True)
            ref = reference_clip(xr, yr, tau_plus=tau_plus)
            ref.backward()

            loss_fn = MemoryEfficientCLIPLoss(
                temperature=TAU, normalized_inputs=True, stable=stable, tau_plus=tau_plus)
            xk = x.detach().requires_grad_(True)
            yk = y.detach().requires_grad_(True)
            loss = loss_fn(xk, yk)
            loss.backward()

            print(f"clip {str(dtype).split('.')[-1]} B={B} stable={stable} tau+={tau_plus}")
            check("loss", loss.detach(), ref.detach(), tol)
            check("dX", xk.grad, xr.grad * sc, tol)
            check("dY", yk.grad, yr.grad * sc, tol)

            xr = x.float().detach().requires_grad_(True)
            ref = reference_lit(xr, y.float(), tau_plus=tau_plus)
            ref.backward()

            lit_fn = MemoryEfficientLiTLoss(
                temperature=TAU, normalized_inputs=True, stable=stable, tau_plus=tau_plus)
            xk = x.detach().requires_grad_(True)
            yk = y.detach().requires_grad_(True)
            loss = lit_fn(xk, yk)
            loss.backward()

            print(f"lit  {str(dtype).split('.')[-1]} B={B} stable={stable} tau+={tau_plus}")
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
    for tau_plus in (0.0, 0.3):
        x, y = make_inputs(B, torch.float32, pos_dupes=tau_plus > 0)

        xr = x.detach().requires_grad_(True)
        yr = y.detach().requires_grad_(True)
        ref = reference_clip(xr, yr, tau_plus=tau_plus)
        ref.backward()
        loss_fn = DistributedMemoryEfficientCLIPLoss(
            temperature=TAU, normalized_inputs=True, tau_plus=tau_plus)
        xk = x.detach().requires_grad_(True)
        yk = y.detach().requires_grad_(True)
        loss = loss_fn(xk, yk)
        loss.backward()
        print(f"clip dist world=1 fallback tau+={tau_plus}")
        check("loss", loss.detach(), ref.detach(), tol)
        check("dX", xk.grad, xr.grad, tol)
        check("dY", yk.grad, yr.grad, tol)

        xr = x.detach().requires_grad_(True)
        ref = reference_lit(xr, y, tau_plus=tau_plus)
        ref.backward()
        loss_fn = DistributedMemoryEfficientLiTLoss(
            temperature=TAU, normalized_inputs=True, tau_plus=tau_plus)
        xk = x.detach().requires_grad_(True)
        loss = loss_fn(xk, y)
        loss.backward()
        print(f"lit dist world=1 fallback tau+={tau_plus}")
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

    for stable, tau_plus in ((False, 0.0), (True, 0.0), (False, 0.3)):
        # identical global batch on every rank, each takes its shard
        x, y = make_inputs(B, torch.float32, seed=7, pos_dupes=tau_plus > 0)
        sc = math.sqrt(B * TAU) if stable else 1.0
        xr = x.detach().requires_grad_(True)
        yr = y.detach().requires_grad_(True)
        ref = reference_clip(xr, yr, tau_plus=tau_plus)
        ref.backward()

        loss_fn = DistributedMemoryEfficientCLIPLoss(
            temperature=TAU, normalized_inputs=True, stable=stable, tau_plus=tau_plus)
        xs = x[rows].detach().requires_grad_(True)
        ys = y[rows].detach().requires_grad_(True)
        partial = loss_fn(xs, ys)
        partial.backward()
        total = partial.detach().clone()
        dist.all_reduce(total)

        if rank == 0:
            print(f"clip dist world={world} stable={stable} tau+={tau_plus}")
        check(f"loss(rank{rank})", total, ref.detach(), tol)
        check(f"dX(rank{rank})", xs.grad, xr.grad[rows] * sc, tol)
        check(f"dY(rank{rank})", ys.grad, yr.grad[rows] * sc, tol)
        dist.barrier()

    for tau_plus in (0.0, 0.3):
        x, y = make_inputs(B, torch.float32, seed=11, pos_dupes=tau_plus > 0)
        xr = x.detach().requires_grad_(True)
        ref = reference_lit(xr, y, tau_plus=tau_plus)
        ref.backward()

        loss_fn = DistributedMemoryEfficientLiTLoss(
            temperature=TAU, normalized_inputs=True, tau_plus=tau_plus)
        xs = x[rows].detach().requires_grad_(True)
        ys = y[rows].detach().requires_grad_(True)
        partial = loss_fn(xs, ys)
        partial.backward()
        total = partial.detach().clone()
        dist.all_reduce(total)

        if rank == 0:
            print(f"lit dist world={world} tau+={tau_plus}")
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
