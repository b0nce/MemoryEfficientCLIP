"""Multi-GPU (DDP) fused matryoshka CLIP and LiT losses: every prefix dim in
ONE ring pass.

Same one-ring-pass design as distributed_mrl_qwen3_loss with the plain
CLIP/LiT semantics. CLIP: the ring rotates only the Y tower, assembling it
together with its (K, batch) prefix inverse-norm table (each arriving block's
table is recomputed locally -- no extra communication) while the fused MRL
denom kernel accumulates all K row and column denominators per hop; the
column tables (an O(K x batch) matrix, versus O(batch * dim) for gathering
the tower again) are completed by one all-reduce. The backward is a single
rectangular launch of the single-GPU MRL tile kernels over local rows x all
columns, emitting dX (final) and dY_partial (reduce-scattered to its owning
shard) plus both rho tables (the column one all-reduced). LiT keeps the plain
distributed LiT contract -- the image tower is locked and NOTHING is ever
assembled: two ring laps stream the image blocks past the home texts (each
hop one fused MRL launch with that block's recomputed norm table), peak
memory stays O(local_batch * dim) and no gradient communication happens at
all.

The FA-shaped backward stays single-GPU only (see mrl_qwen3_loss.py). dims
follow the fused single-GPU rules; for anything else wrap the plain
distributed losses in memeff.MatryoshkaLoss.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from ._common import (
    LN2 as _LN2,
    LOG2E as _LOG2E,
    clip_smoothing_term as _clip_smoothing_term,
    debias_denominators as _debias_denominators,
    resolve_tau_plus as _resolve_tau_plus,
    ring_post as _ring_post,
    validate_features as _validate_features,
    validate_label_smoothing as _validate_label_smoothing,
    validate_tau_plus as _validate_tau_plus,
    world_and_rank as _world_and_rank,
)
from .mrl_clip_loss import (
    launch_mrl_clip_denom as _launch_mrl_clip_denom,
    launch_mrl_clip_grad as _launch_mrl_clip_grad,
)
from .mrl_qwen3_loss import (
    check_mrl_dims,
    _dims_tensor,
    _prefix_inv_norms,
    _prefix_pos,
)


def mrl_clip_assemble_ring(x_local, y_local, dims, dims_t, a_inv, b_inv,
                           inv_temperature, group):
    """Assembles the full Y tower and its (K, batch) prefix inverse-norm table
    while accumulating all K denominators: the row tables are complete locally
    (home rows see every column), the column tables hold this rank's
    contributions and are all-reduced at the end. X never leaves its rank.
    Returns (y_full, b_inv_full, row, col)."""
    world, rank = _world_and_rank(group)
    K = len(dims)
    local_batch, d_model = x_local.shape
    device = x_local.device
    row = torch.zeros(K, local_batch, device=device, dtype=torch.float32)

    if world == 1:
        col = torch.zeros(K, local_batch, device=device, dtype=torch.float32)
        _launch_mrl_clip_denom(x_local, y_local, dims_t, a_inv, b_inv, row, col,
                               inv_temperature, two_sided=True)
        return y_local.contiguous(), b_inv, row, col

    batch_size = world * local_batch
    y_full = torch.empty(batch_size, d_model, device=device, dtype=y_local.dtype)
    b_inv_full = torch.empty(K, batch_size, device=device, dtype=torch.float32)
    col = torch.zeros(K, batch_size, device=device, dtype=torch.float32)
    send_to, recv_from = (rank + 1) % world, (rank - 1) % world

    # The next block's transfer is posted before this hop's denom launch so it
    # overlaps the matmul; the kernel writes each hop's column sums straight
    # into that block's slice of the global tables (the slices share the full
    # tables' row stride, which is all the kernels' pointer math needs).
    cur_y = y_local.contiguous().clone()
    for hop in range(world):
        src = (rank - hop) % world
        off = src * local_batch
        reqs = None
        if hop + 1 < world:
            recv_y, reqs = _ring_post(cur_y, send_to, recv_from, group)
        y_full.narrow(0, off, local_batch).copy_(cur_y)
        b_inv_full[:, off:off + local_batch] = (
            b_inv if hop == 0 else _prefix_inv_norms(cur_y, dims))
        _launch_mrl_clip_denom(x_local, cur_y, dims_t, a_inv,
                               b_inv_full[:, off:off + local_batch], row,
                               col[:, off:off + local_batch],
                               inv_temperature, two_sided=True)
        if reqs is not None:
            for req in reqs:
                req.wait()
            cur_y = recv_y
    dist.all_reduce(col, group=group)   # sum column contributions across ranks
    return y_full, b_inv_full, row, col


class DistributedMemoryEfficientMRLCLIPLossNormed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, y_full, b_inv_full, a_inv, b_inv, pos, row, col,
                weights, dims, offset, n, inv_temperature, inv_temperature_orig,
                tau_plus, floor, batch_size, group):
        device = x.device
        K = len(dims)
        w_vec = torch.tensor(weights, device=device, dtype=torch.float32)
        dims_t = _dims_tensor(dims, device)
        world, _ = _world_and_rank(group)

        sv = pos * inv_temperature - inv_temperature
        den_row, div_row, seed = row, row, None
        den_col, div_col = col, col
        if tau_plus is not None:
            pos_exp = torch.exp2(sv)
            tau_col = tau_plus
            if world > 1:   # the column transform needs every column's positive
                gathered = torch.empty(world * K, n, device=device,
                                       dtype=torch.float32)
                dist.all_gather_into_tensor(gathered, pos_exp.contiguous(),
                                            group=group)
                pos_full = (gathered.view(world, K, n).permute(1, 0, 2)
                            .reshape(K, batch_size))
                if isinstance(tau_plus, torch.Tensor):   # ... and its prior
                    tau_col = torch.empty(batch_size, device=device,
                                          dtype=torch.float32)
                    dist.all_gather_into_tensor(tau_col, tau_plus, group=group)
            else:
                pos_full = pos_exp
            den_row, div_row, seed_row = _debias_denominators(
                pos_exp, row, batch_size - 1, tau_plus, floor)
            den_col, div_col, seed_col = _debias_denominators(
                pos_full, col, batch_size - 1, tau_col, floor)
            seed = seed_row + seed_col[:, offset:offset + n]

        saved = ((x, y, y_full, b_inv_full, a_inv, b_inv, div_row, div_col)
                 + (() if seed is None else (seed,)))
        ctx.save_for_backward(*saved)
        ctx.w_vec, ctx.dims_t = w_vec, dims_t
        ctx.debiased = seed is not None
        ctx.dims, ctx.weights = dims, weights
        ctx.offset, ctx.n, ctx.group = offset, n, group
        ctx.inv_temperature = inv_temperature
        ctx.inv_temperature_orig = inv_temperature_orig
        ctx.batch_size = batch_size
        ctx.in_dtype = x.dtype
        per_dim = (2.0 * sv * _LN2 - torch.log(den_row)
                   - torch.log(den_col[:, offset:offset + n])).sum(dim=1)
        return -(per_dim * w_vec).sum() / (2.0 * batch_size)

    @staticmethod
    def backward(ctx, grad_output):
        tensors = list(ctx.saved_tensors)
        x, y, y_full, b_inv_full, a_inv, b_inv, div_row, div_col = tensors[:8]
        seed = tensors[8] if ctx.debiased else None
        dims, weights = ctx.dims, ctx.weights
        w_vec, dims_t = ctx.w_vec, ctx.dims_t
        K = len(dims)
        offset, n, batch_size = ctx.offset, ctx.n, ctx.batch_size
        inv_temperature = ctx.inv_temperature
        grad_scale = ctx.inv_temperature_orig / (2.0 * batch_size)
        device, d_model = x.device, x.shape[1]
        world, _ = _world_and_rank(ctx.group)

        dX = torch.zeros(n, d_model, device=device, dtype=torch.float32)
        dY_partial = torch.zeros(batch_size, d_model, device=device,
                                 dtype=torch.float32)
        rho_x = torch.zeros(K, n, device=device, dtype=torch.float32)
        rho_y_partial = torch.zeros(K, batch_size, device=device,
                                    dtype=torch.float32)
        _launch_mrl_clip_grad(x, y_full, dims, dims_t, a_inv, b_inv_full,
                              div_row, div_col, w_vec, dX, dY_partial,
                              rho_x, rho_y_partial, inv_temperature, grad_scale,
                              two_sided=True)
        if world > 1:
            dY = torch.empty(n, d_model, device=device, dtype=torch.float32)
            dist.reduce_scatter_tensor(dY, dY_partial, group=ctx.group)
            dist.all_reduce(rho_y_partial, group=ctx.group)
        else:
            dY = dY_partial
        rho_y = rho_y_partial[:, offset:offset + n]

        # Eager positive-pair seeds and rank-one re-normalization corrections
        # exactly as in the single-GPU fused backward: every positive pair is
        # a home pair, so nothing here communicates.
        xf, yf = x.float(), y.float()
        if seed is None:
            cvec = (w_vec[:, None] * (-2.0 * grad_scale)).expand(K, n)
        else:
            cvec = seed * (w_vec[:, None] * grad_scale)
        for k, m in enumerate(dims):
            a = a_inv[k][:, None]
            b = b_inv[k][:, None]
            px = xf[:, :m] * a
            py = yf[:, :m] * b
            vx = py * cvec[k][:, None]
            dX[:, :m] += a * (vx - (vx * px).sum(-1, keepdim=True) * px
                              - rho_x[k][:, None] * px)
            vy = px * cvec[k][:, None]
            dY[:, :m] += b * (vy - (vy * py).sum(-1, keepdim=True) * py
                              - rho_y[k][:, None] * py)

        dX = (dX * grad_output).to(ctx.in_dtype)
        dY = (dY * grad_output).to(ctx.in_dtype)
        return (dX, dY) + (None,) * 17


class DistributedMemoryEfficientMRLLiTLossNormed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_text, y_img, a_inv, b_inv, pos, weights, dims,
                inv_temperature, inv_temperature_orig, tau_plus, floor,
                batch_size, group):
        device = x_text.device
        n = x_text.shape[0]
        w_vec = torch.tensor(weights, device=device, dtype=torch.float32)
        dims_t = _dims_tensor(dims, device)
        world, rank = _world_and_rank(group)
        send_to, recv_from = (rank + 1) % world, (rank - 1) % world

        # Lap 1: stream image blocks past the home texts, each hop feeding the
        # fused MRL denom kernel with that block's recomputed norm table ->
        # all K row denominators complete locally after `world` hops.
        row = torch.zeros(len(dims), n, device=device, dtype=torch.float32)
        cur_y = y_img.contiguous().clone()
        for hop in range(world):
            reqs = None
            if hop + 1 < world:
                recv_y, reqs = _ring_post(cur_y, send_to, recv_from, group)
            b_blk = b_inv if hop == 0 else _prefix_inv_norms(cur_y, dims)
            _launch_mrl_clip_denom(x_text, cur_y, dims_t, a_inv, b_blk, row,
                                   None, inv_temperature, two_sided=False)
            if reqs is not None:
                for req in reqs:
                    req.wait()
                cur_y = recv_y

        sv = pos * inv_temperature - inv_temperature
        denom, div, seed = row, row, None
        if tau_plus is not None:
            denom, div, seed = _debias_denominators(
                torch.exp2(sv), row, batch_size - 1, tau_plus, floor)

        saved = (x_text, y_img, a_inv, b_inv, div) + (() if seed is None else (seed,))
        ctx.save_for_backward(*saved)
        ctx.w_vec, ctx.dims_t = w_vec, dims_t
        ctx.debiased = seed is not None
        ctx.dims, ctx.weights = dims, weights
        ctx.group = group
        ctx.inv_temperature = inv_temperature
        ctx.inv_temperature_orig = inv_temperature_orig
        ctx.batch_size = batch_size
        ctx.in_dtype = x_text.dtype
        return -((sv * _LN2 - torch.log(denom)).sum(dim=1) * w_vec).sum() / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        tensors = list(ctx.saved_tensors)
        x_text, y_img, a_inv, b_inv, div = tensors[:5]
        seed = tensors[5] if ctx.debiased else None
        dims, weights = ctx.dims, ctx.weights
        w_vec, dims_t = ctx.w_vec, ctx.dims_t
        K, batch_size = len(dims), ctx.batch_size
        inv_temperature = ctx.inv_temperature
        grad_scale = ctx.inv_temperature_orig / batch_size
        device, (n, d_model) = x_text.device, x_text.shape
        world, rank = _world_and_rank(ctx.group)
        send_to, recv_from = (rank + 1) % world, (rank - 1) % world

        # Lap 2: with the complete divisors, stream the image blocks past the
        # home texts again -> dX and the rho rows complete locally, no
        # gradient communication (the image tower is locked).
        dX = torch.zeros(n, d_model, device=device, dtype=torch.float32)
        rho_x = torch.zeros(K, n, device=device, dtype=torch.float32)
        cur_y = y_img.contiguous().clone()
        for hop in range(world):
            reqs = None
            if hop + 1 < world:
                recv_y, reqs = _ring_post(cur_y, send_to, recv_from, ctx.group)
            b_blk = b_inv if hop == 0 else _prefix_inv_norms(cur_y, dims)
            _launch_mrl_clip_grad(x_text, cur_y, dims, dims_t, a_inv, b_blk,
                                  div, None, w_vec, dX, None, rho_x, None,
                                  inv_temperature, grad_scale, two_sided=False)
            if reqs is not None:
                for req in reqs:
                    req.wait()
                cur_y = recv_y

        # Eager seeds and corrections against the HOME images (the positives).
        xf, yf = x_text.float(), y_img.float()
        if seed is None:
            cvec = (w_vec[:, None] * (-grad_scale)).expand(K, n)
        else:
            cvec = seed * (w_vec[:, None] * grad_scale)
        for k, m in enumerate(dims):
            a = a_inv[k][:, None]
            b = b_inv[k][:, None]
            px = xf[:, :m] * a
            py = yf[:, :m] * b
            vx = py * cvec[k][:, None]
            dX[:, :m] += a * (vx - (vx * px).sum(-1, keepdim=True) * px
                              - rho_x[k][:, None] * px)

        dX = (dX * grad_output).to(ctx.in_dtype)
        return (dX,) + (None,) * 12


class DistributedMemoryEfficientMatryoshkaCLIPLoss(nn.Module):
    """DDP counterpart of MemoryEfficientMatryoshkaCLIPLoss: the bidirectional
    softmax loss at every nested prefix dim, both towers trained, one ring
    pass for all dims. Feed each rank its own shard; forward returns this
    rank's partial loss (all-reduce SUM for the global value), backward fills
    the shard's gradients.

    dims/weights follow the fused single-GPU rules: strictly increasing
    multiples of the kernels' feature chunk ending exactly at d_model.
    stable / tau_plus (float, per-row tensor for this rank's shard, or
    per-call override) / label_smoothing compose per dim exactly as in the
    plain distributed CLIP loss; per-dim column debiasing adds one O(K x
    batch) all-gather of the positive exponentials.
    """
    def __init__(self, dims, weights=None, temperature=0.07,
                 normalized_inputs=False, stable=True, tau_plus=0.0,
                 label_smoothing=0.0, group=None):
        super().__init__()
        _validate_tau_plus(tau_plus)
        _validate_label_smoothing(label_smoothing)
        self.dims = tuple(int(m) for m in dims)
        if not self.dims or list(self.dims) != sorted(set(self.dims)):
            raise ValueError(f"dims must be strictly increasing, got {dims}")
        if weights is None:
            weights = (1.0,) * len(self.dims)
        self.weights = tuple(float(w) for w in weights)
        if len(self.weights) != len(self.dims):
            raise ValueError(f"got {len(self.weights)} weights for "
                             f"{len(self.dims)} dims")
        self.temperature = temperature
        self.normalized_inputs = normalized_inputs
        self.stable = stable
        self.tau_plus = tau_plus
        self.label_smoothing = label_smoothing
        self.group = group

    def forward(self, image_features, text_features, tau_plus=None):
        x = image_features if self.normalized_inputs else F.normalize(image_features, dim=1)
        y = text_features if self.normalized_inputs else F.normalize(text_features, dim=1)
        x, y = x.contiguous(), y.contiguous()
        _validate_features(x, y)

        world, rank = _world_and_rank(self.group)
        local_batch, d_model = x.shape
        batch_size = world * local_batch
        dims = check_mrl_dims(self.dims, d_model)
        tau_plus = _resolve_tau_plus(self.tau_plus, tau_plus, local_batch, x.device)
        inv_temperature = _LOG2E / self.temperature
        inv_temperature_orig = (math.sqrt(batch_size / self.temperature)
                                if self.stable else 1.0 / self.temperature)

        a_inv = _prefix_inv_norms(x, dims)
        b_inv = _prefix_inv_norms(y, dims)
        pos = _prefix_pos(x, y, a_inv, b_inv, dims)
        dims_t = _dims_tensor(dims, x.device)

        y_full, b_inv_full, row, col = mrl_clip_assemble_ring(
            x.detach(), y.detach(), dims, dims_t, a_inv, b_inv,
            inv_temperature, self.group)
        offset = rank * local_batch
        loss = DistributedMemoryEfficientMRLCLIPLossNormed.apply(
            x, y, y_full, b_inv_full, a_inv, b_inv, pos, row, col,
            self.weights, dims, offset, local_batch, inv_temperature,
            inv_temperature_orig, tau_plus, math.exp(-2.0 / self.temperature),
            batch_size, self.group)
        if self.label_smoothing:
            grad_factor = self.temperature * inv_temperature_orig if self.stable else 1.0
            for m, w in zip(dims, self.weights):
                loss = loss + w * _clip_smoothing_term(
                    F.normalize(x[:, :m], dim=-1), F.normalize(y[:, :m], dim=-1),
                    self.label_smoothing, batch_size, self.temperature,
                    grad_factor, self.group, world)
        return loss


class DistributedMemoryEfficientMatryoshkaLiTLoss(nn.Module):
    """DDP counterpart of MemoryEfficientMatryoshkaLiTLoss: the row-softmax
    loss at every nested prefix dim, only the text tower trained, one ring
    pass (well, two laps of it -- denominators forward, gradient backward)
    for all dims. The image tower is locked, never assembled, and receives no
    gradient: peak memory stays O(local_batch * dim) and there is no gradient
    communication. forward(text_features, image_features) as in the plain
    distributed LiT loss; same dims/weights semantics as the CLIP class.
    """
    def __init__(self, dims, weights=None, temperature=0.07,
                 normalized_inputs=False, stable=True, tau_plus=0.0,
                 label_smoothing=0.0, group=None):
        super().__init__()
        _validate_tau_plus(tau_plus)
        _validate_label_smoothing(label_smoothing)
        self.dims = tuple(int(m) for m in dims)
        if not self.dims or list(self.dims) != sorted(set(self.dims)):
            raise ValueError(f"dims must be strictly increasing, got {dims}")
        if weights is None:
            weights = (1.0,) * len(self.dims)
        self.weights = tuple(float(w) for w in weights)
        if len(self.weights) != len(self.dims):
            raise ValueError(f"got {len(self.weights)} weights for "
                             f"{len(self.dims)} dims")
        self.temperature = temperature
        self.normalized_inputs = normalized_inputs
        self.stable = stable
        self.tau_plus = tau_plus
        self.label_smoothing = label_smoothing
        self.group = group

    def forward(self, text_features, image_features, tau_plus=None):
        x = text_features if self.normalized_inputs else F.normalize(text_features, dim=1)
        y = image_features if self.normalized_inputs else F.normalize(image_features, dim=1)
        x, y = x.contiguous(), y.detach().contiguous()
        _validate_features(x, y)

        world, rank = _world_and_rank(self.group)
        local_batch, d_model = x.shape
        batch_size = world * local_batch
        dims = check_mrl_dims(self.dims, d_model)
        tau_plus = _resolve_tau_plus(self.tau_plus, tau_plus, local_batch, x.device)
        inv_temperature = _LOG2E / self.temperature
        inv_temperature_orig = (math.sqrt(batch_size / self.temperature)
                                if self.stable else 1.0 / self.temperature)

        a_inv = _prefix_inv_norms(x, dims)
        b_inv = _prefix_inv_norms(y, dims)
        pos = _prefix_pos(x, y, a_inv, b_inv, dims)

        loss = DistributedMemoryEfficientMRLLiTLossNormed.apply(
            x, y, a_inv, b_inv, pos, self.weights, dims, inv_temperature,
            inv_temperature_orig, tau_plus, math.exp(-2.0 / self.temperature),
            batch_size, self.group)
        if self.label_smoothing:
            grad_factor = self.temperature * inv_temperature_orig if self.stable else 1.0
            for m, w in zip(dims, self.weights):
                loss = loss + w * _clip_smoothing_term(
                    F.normalize(x[:, :m], dim=-1), F.normalize(y[:, :m], dim=-1),
                    self.label_smoothing, batch_size, self.temperature,
                    grad_factor, self.group, world)
        return loss
