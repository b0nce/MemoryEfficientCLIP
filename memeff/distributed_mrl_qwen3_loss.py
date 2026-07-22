"""Multi-GPU (DDP) fused matryoshka Qwen3 losses: every prefix dim in ONE
ring pass.

The eager MatryoshkaLoss wrapper around the distributed Qwen3 losses runs K
separate ring passes -- K x the feature communication and K independent
similarity walks. Here the ring travels once: each hop's block feeds the
fused MRL denom kernel (cumulative prefix dots, all K denominators per hop),
and the traveling towers' (K, batch) prefix inverse-norm tables are assembled
alongside the features themselves -- each arriving block's table is
recomputed locally (O(local_batch * sum(dims)) eager math), so no extra
feature-sized communication. The backward is one rectangular launch of the
single-GPU MRL tile kernels per similarity block, local rows x global
columns, exactly like the plain distributed Qwen3 backward: the column-side
gradient is partial per rank and goes home by reduce-scatter; the new O(K x
batch) rho tables (the re-normalization correction row sums) go home by
all-reduce. All eager per-dim math (positive-pair seeds through the prefix
Jacobian, hard negatives, debiasing) involves only home rows and stays local.

The FA-shaped backward stays single-GPU only: the tile kernels here see the
same rectangular shapes as the plain distributed backward, and the FA design
needs the whole opposite tower streamed inside one launch (see
mrl_qwen3_loss.py). dims follow the fused single-GPU rules (strictly
increasing chunk-aligned prefixes ending at d_model); for anything else wrap
the plain distributed losses in memeff.MatryoshkaLoss.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from ._common import (
    LN2 as _LN2,
    LOG2E as _LOG2E,
    debias_denominators as _debias_denominators,
    qwen3_num_negatives as _num_negatives,
    qwen3_smoothing_term as _qwen3_smoothing_term,
    resolve_tau_plus as _resolve_tau_plus,
    validate_features as _validate_features,
    validate_hard_negatives as _validate_hard_negatives,
    validate_label_smoothing as _validate_label_smoothing,
    validate_tau_plus as _validate_tau_plus,
    world_and_rank as _world_and_rank,
)
from .mrl_qwen3_loss import (
    check_mrl_dims,
    launch_mrl_denom as _launch_mrl_denom,
    launch_mrl_grad as _launch_mrl_grad,
    _dims_tensor,
    _mrl_hard_negative_exp,
    _prefix_inv_norms,
    _prefix_pos,
)


def mrl_assemble_ring(q_local, d_local, dims, dims_t, a_inv, b_inv, pos,
                      inv_temperature, margin, use_qq, use_dd, group):
    """Assembles the document tower (and query tower iff use_qq) together with
    their (K, batch) prefix inverse-norm tables while accumulating all K row
    denominators for the home rows. Only features travel: each block's norm
    table is recomputed on arrival. Returns
    (d_full, b_inv_full, q_full, a_inv_full, sum_exp)."""
    world, rank = _world_and_rank(group)
    local_batch, d_model = q_local.shape
    device = q_local.device
    row_offset = rank * local_batch
    sum_exp = torch.zeros(len(dims), local_batch, device=device, dtype=torch.float32)

    if world == 1:
        _launch_mrl_denom(q_local, d_local, dims_t, a_inv, b_inv, pos, sum_exp,
                          inv_temperature, margin)
        if use_qq:
            _launch_mrl_denom(q_local, q_local, dims_t, a_inv, a_inv, pos, sum_exp,
                              inv_temperature, margin, exclude_diag=True)
        if use_dd:
            _launch_mrl_denom(d_local, d_local, dims_t, b_inv, b_inv, pos, sum_exp,
                              inv_temperature, margin, exclude_diag=True)
        return (d_local.contiguous(), b_inv,
                q_local.contiguous() if use_qq else None,
                a_inv if use_qq else None, sum_exp)

    batch_size = world * local_batch
    d_full = torch.empty(batch_size, d_model, device=device, dtype=d_local.dtype)
    b_inv_full = torch.empty(len(dims), batch_size, device=device, dtype=torch.float32)
    q_full = a_inv_full = None
    if use_qq:
        q_full = torch.empty(batch_size, d_model, device=device, dtype=q_local.dtype)
        a_inv_full = torch.empty(len(dims), batch_size, device=device,
                                 dtype=torch.float32)
    send_to, recv_from = (rank + 1) % world, (rank - 1) % world

    # Travelling blocks are prefetched one hop ahead so the transfer overlaps
    # this hop's denom launches (as in _common.qwen3_assemble_ring). The norm
    # tables are written into their slot before the launches so the kernels'
    # col_offset-indexed loads see them.
    cur_d = d_local.contiguous().clone()
    cur_q = q_local.contiguous().clone() if use_qq else None
    for hop in range(world):
        src = (rank - hop) % world
        col_offset = src * local_batch
        reqs = None
        if hop + 1 < world:
            recv_d = torch.empty_like(cur_d)
            ops = [dist.P2POp(dist.isend, cur_d, send_to, group=group),
                   dist.P2POp(dist.irecv, recv_d, recv_from, group=group)]
            if use_qq:
                recv_q = torch.empty_like(cur_q)
                ops += [dist.P2POp(dist.isend, cur_q, send_to, group=group),
                        dist.P2POp(dist.irecv, recv_q, recv_from, group=group)]
            reqs = dist.batch_isend_irecv(ops)
        d_full.narrow(0, col_offset, local_batch).copy_(cur_d)
        b_inv_full[:, col_offset:col_offset + local_batch] = (
            b_inv if hop == 0 else _prefix_inv_norms(cur_d, dims))
        _launch_mrl_denom(q_local, cur_d, dims_t, a_inv, b_inv_full, pos, sum_exp,
                          inv_temperature, margin, col_offset=col_offset)
        if use_dd:
            _launch_mrl_denom(d_local, cur_d, dims_t, b_inv, b_inv_full, pos, sum_exp,
                              inv_temperature, margin, exclude_diag=True,
                              row_offset=row_offset, col_offset=col_offset)
        if use_qq:
            q_full.narrow(0, col_offset, local_batch).copy_(cur_q)
            a_inv_full[:, col_offset:col_offset + local_batch] = (
                a_inv if hop == 0 else _prefix_inv_norms(cur_q, dims))
            _launch_mrl_denom(q_local, cur_q, dims_t, a_inv, a_inv_full, pos, sum_exp,
                              inv_temperature, margin, exclude_diag=True,
                              row_offset=row_offset, col_offset=col_offset)
        if reqs is not None:
            for req in reqs:
                req.wait()
            cur_d = recv_d
            if use_qq:
                cur_q = recv_q
    return d_full, b_inv_full, q_full, a_inv_full, sum_exp


class DistributedMemoryEfficientMRLQwen3LossNormed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, d, h, d_full, b_inv_full, q_full, a_inv_full,
                a_inv, b_inv, h_inv, pos, sum_exp, weights, dims, offset, n,
                inv_temperature, inv_temperature_orig, margin, use_qq, use_dd,
                tau_plus, floor, batch_size, lit, group):
        device = q.device
        w_vec = torch.tensor(weights, device=device, dtype=torch.float32)
        dims_t = _dims_tensor(dims, device)
        if h is not None:
            sum_exp += _mrl_hard_negative_exp(
                q, h, a_inv, h_inv, pos, inv_temperature, margin, dims).sum(-1)

        # the row denominators are complete on their home rank, so the
        # debiased transform needs no communication (as in the plain
        # distributed Qwen3 loss), just per-dim rows.
        sv = pos * inv_temperature - inv_temperature
        denom, div, seed = sum_exp, sum_exp, None
        if tau_plus is not None:
            denom, div, seed = _debias_denominators(
                torch.exp2(sv), sum_exp,
                _num_negatives(batch_size, h, use_qq, use_dd), tau_plus, floor)

        saved = ((q, d, d_full, b_inv_full, a_inv, b_inv, pos, div)
                 + (() if seed is None else (seed,))
                 + ((q_full, a_inv_full) if use_qq else ())
                 + (() if h is None else (h, h_inv)))
        ctx.save_for_backward(*saved)
        ctx.w_vec, ctx.dims_t = w_vec, dims_t
        ctx.debiased = seed is not None
        ctx.has_h = h is not None
        ctx.dims, ctx.weights = dims, weights
        ctx.offset, ctx.n, ctx.group = offset, n, group
        ctx.inv_temperature = inv_temperature
        ctx.inv_temperature_orig = inv_temperature_orig
        ctx.margin = margin
        ctx.use_qq, ctx.use_dd, ctx.lit = use_qq, use_dd, lit
        ctx.batch_size = batch_size
        ctx.in_dtype = q.dtype
        return -((sv * _LN2 - torch.log(denom)).sum(dim=1) * w_vec).sum() / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        tensors = list(ctx.saved_tensors)
        q, d, d_full, b_inv_full, a_inv, b_inv, pos, div = tensors[:8]
        idx = 8
        seed = None
        if ctx.debiased:
            seed, idx = tensors[idx], idx + 1
        q_full = a_inv_full = None
        if ctx.use_qq:
            q_full, a_inv_full, idx = tensors[idx], tensors[idx + 1], idx + 2
        h = h_inv = None
        if ctx.has_h:
            h, h_inv = tensors[idx], tensors[idx + 1]
        dims, weights = ctx.dims, ctx.weights
        w_vec, dims_t = ctx.w_vec, ctx.dims_t
        K, lit = len(dims), ctx.lit
        offset, n, batch_size = ctx.offset, ctx.n, ctx.batch_size
        inv_temperature, margin = ctx.inv_temperature, ctx.margin
        grad_scale = ctx.inv_temperature_orig / batch_size
        device, d_model = q.device, q.shape[1]
        world, _ = _world_and_rank(ctx.group)

        # dQ / rho_q cover the home rows and are complete after the launches
        # below; the *_partial buffers hold this rank's contribution to every
        # global column (features by reduce-scatter, the O(K x batch) rho
        # tables by all-reduce). The d-d row direction writes into this rank's
        # own slice of dD_partial so it rides the same reduce-scatter; its rho
        # rows need a separate (K, n) buffer because the kernel indexes every
        # row table with one shared stride.
        dQ = torch.zeros(n, d_model, device=device, dtype=torch.float32)
        rho_q = torch.zeros(K, n, device=device, dtype=torch.float32)
        dD_partial = rho_d_partial = None
        if not lit:
            dD_partial = torch.zeros(batch_size, d_model, device=device,
                                     dtype=torch.float32)
            rho_d_partial = torch.zeros(K, batch_size, device=device,
                                        dtype=torch.float32)
        _launch_mrl_grad(q, d_full, dims, dims_t, a_inv, b_inv_full, pos, div,
                         w_vec, dQ, dQ if lit else dD_partial,
                         rho_q, rho_q if lit else rho_d_partial,
                         inv_temperature, margin, grad_scale, two_sided=not lit)
        rho_d_row = None
        if ctx.use_dd:
            rho_d_row = torch.zeros(K, n, device=device, dtype=torch.float32)
            _launch_mrl_grad(d, d_full, dims, dims_t, b_inv, b_inv_full, pos, div,
                             w_vec, dD_partial.narrow(0, offset, n), dD_partial,
                             rho_d_row, rho_d_partial,
                             inv_temperature, margin, grad_scale,
                             exclude_diag=True, row_offset=offset, two_sided=True)
        dQ_partial = rho_q_partial = None
        if ctx.use_qq:
            dQ_partial = torch.zeros(batch_size, d_model, device=device,
                                     dtype=torch.float32)
            rho_q_partial = torch.zeros(K, batch_size, device=device,
                                        dtype=torch.float32)
            _launch_mrl_grad(q, q_full, dims, dims_t, a_inv, a_inv_full, pos, div,
                             w_vec, dQ, dQ_partial, rho_q, rho_q_partial,
                             inv_temperature, margin, grad_scale,
                             exclude_diag=True, row_offset=offset, two_sided=True)

        dD = None
        if world > 1:
            if not lit:
                dD = torch.empty(n, d_model, device=device, dtype=torch.float32)
                dist.reduce_scatter_tensor(dD, dD_partial, group=ctx.group)
                dist.all_reduce(rho_d_partial, group=ctx.group)
            if ctx.use_qq:
                dQ_cols = torch.empty(n, d_model, device=device, dtype=torch.float32)
                dist.reduce_scatter_tensor(dQ_cols, dQ_partial, group=ctx.group)
                dQ += dQ_cols
                dist.all_reduce(rho_q_partial, group=ctx.group)
        else:
            if not lit:
                dD = dD_partial
            if ctx.use_qq:
                dQ += dQ_partial
        if ctx.use_qq:
            rho_q = rho_q + rho_q_partial[:, offset:offset + n]
        rho_d = None
        if not lit:
            rho_d = rho_d_partial[:, offset:offset + n]
            if ctx.use_dd:
                rho_d = rho_d + rho_d_row

        # Eager per-dim terms exactly as in the single-GPU fused backward --
        # positive pairs, hard negatives and the rank-one re-normalization
        # corrections all live on the home rows.
        qf, df = q.float(), d.float()
        hf = h.float() if h is not None else None
        if seed is None:
            cvec = (w_vec[:, None] * (-grad_scale)).expand(K, n)
        else:
            cvec = seed * (w_vec[:, None] * grad_scale)
        e_h = (_mrl_hard_negative_exp(q, h, a_inv, h_inv, pos, inv_temperature,
                                      margin, dims)
               if h is not None else None)
        dH = None
        if h is not None and not lit:
            dH = torch.zeros(n, h.shape[1], d_model, device=device,
                             dtype=torch.float32)
        for k, (m, w) in enumerate(zip(dims, weights)):
            a = a_inv[k][:, None]
            b = b_inv[k][:, None]
            p = qf[:, :m] * a
            pd = df[:, :m] * b
            vq = pd * cvec[k][:, None]
            if h is not None:
                p_h = e_h[k] / div[k][:, None]
                ph = hf[:, :, :m] * h_inv[k][:, :, None]
                vq = vq + (w * grad_scale) * torch.einsum('bk,bkd->bd', p_h, ph)
                if dH is not None:
                    u = (w * grad_scale) * p_h[:, :, None] * p[:, None, :]
                    dH[:, :, :m] += h_inv[k][:, :, None] * (
                        u - (u * ph).sum(-1, keepdim=True) * ph)
            dQ[:, :m] += a * (vq - (vq * p).sum(-1, keepdim=True) * p
                              - rho_q[k][:, None] * p)
            if not lit:
                vd = p * cvec[k][:, None]
                dD[:, :m] += b * (vd - (vd * pd).sum(-1, keepdim=True) * pd
                                  - rho_d[k][:, None] * pd)

        dQ = (dQ * grad_output).to(ctx.in_dtype)
        dD = None if lit else (dD * grad_output).to(ctx.in_dtype)
        dH = None if dH is None else (dH * grad_output).to(h.dtype)
        return (dQ, dD, dH) + (None,) * 23


class DistributedMemoryEfficientMatryoshkaQwen3Loss(nn.Module):
    """DDP counterpart of MemoryEfficientMatryoshkaQwen3Loss: every prefix dim
    trained in one ring pass. Feed each rank its own shard of the batch; hard
    negatives stay on their query's rank. forward returns this rank's partial
    loss (all-reduce SUM for the global value), backward fills the shard's
    gradients.

    dims/weights follow the fused single-GPU rules: strictly increasing
    multiples of the kernels' feature chunk ending exactly at d_model. All of
    margin / stable / tau_plus (float, per-row tensor for this rank's shard,
    or per-call override) / label_smoothing compose per dim exactly as in the
    plain distributed loss; the row denominators live on their home rank, so
    debiasing adds no communication. Compared with wrapping the plain
    distributed loss in memeff.MatryoshkaLoss this sends the towers around
    the ring once instead of K times and walks each similarity block once.
    """
    _lit = False

    def __init__(self, dims, weights=None, temperature=0.07, margin=0.1,
                 use_qq_negatives=False, use_dd_negatives=False,
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
        self.margin = margin
        self.use_qq_negatives = use_qq_negatives
        self.use_dd_negatives = use_dd_negatives
        self.normalized_inputs = normalized_inputs
        self.stable = stable
        self.tau_plus = tau_plus
        self.label_smoothing = label_smoothing
        self.group = group

    def forward(self, query_features, doc_features, hard_negative_features=None,
                tau_plus=None):
        q, d, h = query_features, doc_features, hard_negative_features
        if not self.normalized_inputs:
            q = F.normalize(q, dim=-1)
            d = F.normalize(d, dim=-1)
            h = F.normalize(h, dim=-1) if h is not None else None
        if self._lit:
            d = d.detach()
            h = h.detach() if h is not None else None
        q, d = q.contiguous(), d.contiguous()
        h = h.contiguous() if h is not None else None

        _validate_features(q, d)
        if h is not None:
            _validate_hard_negatives(q, h)

        world, rank = _world_and_rank(self.group)
        local_batch, d_model = q.shape
        batch_size = world * local_batch
        dims = check_mrl_dims(self.dims, d_model)
        tau_plus = _resolve_tau_plus(self.tau_plus, tau_plus, local_batch, q.device)
        inv_temperature = _LOG2E / self.temperature
        inv_temperature_orig = (math.sqrt(batch_size / self.temperature)
                                if self.stable else 1.0 / self.temperature)

        a_inv = _prefix_inv_norms(q, dims)
        b_inv = _prefix_inv_norms(d, dims)
        h_inv = _prefix_inv_norms(h, dims) if h is not None else None
        pos = _prefix_pos(q, d, a_inv, b_inv, dims)
        dims_t = _dims_tensor(dims, q.device)

        use_dd = (not self._lit) and self.use_dd_negatives
        d_full, b_inv_full, q_full, a_inv_full, sum_exp = mrl_assemble_ring(
            q.detach(), d.detach(), dims, dims_t, a_inv, b_inv, pos,
            inv_temperature, self.margin, self.use_qq_negatives, use_dd,
            self.group)
        offset = rank * local_batch
        loss = DistributedMemoryEfficientMRLQwen3LossNormed.apply(
            q, d, h, d_full, b_inv_full, q_full, a_inv_full, a_inv, b_inv,
            h_inv, pos, sum_exp, self.weights, dims, offset, local_batch,
            inv_temperature, inv_temperature_orig, self.margin,
            self.use_qq_negatives, use_dd, tau_plus,
            math.exp(-2.0 / self.temperature), batch_size, self._lit,
            self.group)
        if self.label_smoothing:
            grad_factor = self.temperature * inv_temperature_orig if self.stable else 1.0
            n_neg = _num_negatives(batch_size, h, self.use_qq_negatives, use_dd)
            for m, w in zip(dims, self.weights):
                loss = loss + w * _qwen3_smoothing_term(
                    F.normalize(q[:, :m], dim=-1),
                    F.normalize(d[:, :m], dim=-1),
                    (F.normalize(h[:, :, :m], dim=-1) if h is not None else None),
                    self.label_smoothing, n_neg, batch_size, self.temperature,
                    grad_factor, self.use_qq_negatives, use_dd,
                    self.group, world)
        return loss


class DistributedMemoryEfficientMatryoshkaLiTQwen3Loss(
        DistributedMemoryEfficientMatryoshkaQwen3Loss):
    """DDP counterpart of MemoryEfficientMatryoshkaLiTQwen3Loss: documents and
    hard negatives are locked at every prefix dim, only the query shard gets a
    gradient -- so the only gradient communication is the q-q reduce-scatter
    when q-q negatives are enabled. No d-d negatives (locked documents repel
    nothing). Same dims/weights/feature semantics as the CLIP-style class.
    """
    _lit = True

    def __init__(self, dims, weights=None, temperature=0.07, margin=0.1,
                 use_qq_negatives=False, normalized_inputs=False, stable=True,
                 tau_plus=0.0, label_smoothing=0.0, group=None):
        super().__init__(dims, weights=weights, temperature=temperature,
                         margin=margin, use_qq_negatives=use_qq_negatives,
                         use_dd_negatives=False,
                         normalized_inputs=normalized_inputs, stable=stable,
                         tau_plus=tau_plus, label_smoothing=label_smoothing,
                         group=group)
