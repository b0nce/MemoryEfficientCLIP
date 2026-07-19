"""Qwen3-style embedding loss, LiT-style: the document tower is locked.

Same query->document InfoNCE with false-negative masking, hard negatives and
optional q-q negatives as clip_embedding_loss.py, but documents and hard negatives
receive no gradient. There is no d-d option: with locked documents its repulsion
gradient has nowhere to land, it only inflates the denominator. The backward emits
dQ only, so the DDP version needs no gradient communication unless q-q is enabled.
Includes single-GPU and DDP modules.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

try:
    from ._common import (
        LN2 as _LN2,
        LOG2E as _LOG2E,
        hard_negative_exp as _hard_negative_exp,
        launch_emb_denom as _launch_denom,
        launch_emb_grad as _launch_grad,
        validate_features as _validate_features,
        validate_hard_negatives as _validate_hard_negatives,
        world_and_rank as _world_and_rank,
    )
except ImportError:   # running as a flat module from inside the repo
    from _common import (
        LN2 as _LN2,
        LOG2E as _LOG2E,
        hard_negative_exp as _hard_negative_exp,
        launch_emb_denom as _launch_denom,
        launch_emb_grad as _launch_grad,
        validate_features as _validate_features,
        validate_hard_negatives as _validate_hard_negatives,
        world_and_rank as _world_and_rank,
    )


class MemoryEfficientLiTEmbeddingLossNormed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, d, h, inv_temperature, inv_temperature_orig, margin, use_qq):
        batch_size, d_model = q.shape
        pos = (q.float() * d.float()).sum(dim=1)

        sum_exp_row = torch.zeros(batch_size, device=q.device, dtype=torch.float32)
        _launch_denom(q, d, pos, sum_exp_row, inv_temperature, margin)
        if use_qq:
            _launch_denom(q, q, pos, sum_exp_row, inv_temperature, margin, exclude_diag=True)
        if h is not None:
            sum_exp_row += _hard_negative_exp(q, h, pos, inv_temperature, margin).sum(dim=1)

        saved = (q, d, pos, sum_exp_row) + ((h,) if h is not None else ())
        ctx.save_for_backward(*saved)
        ctx.has_h = h is not None
        ctx.inv_temperature = inv_temperature
        ctx.inv_temperature_orig = inv_temperature_orig
        ctx.margin = margin
        ctx.use_qq = use_qq
        ctx.batch_size = batch_size
        ctx.in_dtype = q.dtype

        sv = pos * inv_temperature - inv_temperature
        return -(sv * _LN2 - torch.log(sum_exp_row)).mean()

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.has_h:
            q, d, pos, sum_exp_row, h = ctx.saved_tensors
        else:
            (q, d, pos, sum_exp_row), h = ctx.saved_tensors, None
        inv_temperature, margin = ctx.inv_temperature, ctx.margin
        grad_scale = ctx.inv_temperature_orig / ctx.batch_size

        dQ = d.float() * (-grad_scale)
        _launch_grad(q, d, pos, sum_exp_row, dQ, dQ, inv_temperature, margin, grad_scale,
                     two_sided=False)
        if ctx.use_qq:
            _launch_grad(q, q, pos, sum_exp_row, dQ, dQ, inv_temperature, margin, grad_scale,
                         exclude_diag=True, two_sided=True)
        if h is not None:
            p_h = _hard_negative_exp(q, h, pos, inv_temperature, margin) / sum_exp_row[:, None]
            dQ += grad_scale * torch.einsum('bk,bkd->bd', p_h, h.float())

        dQ = dQ * grad_output
        return dQ.to(ctx.in_dtype), None, None, None, None, None, None


class MemoryEfficientLiTEmbeddingLoss(nn.Module):
    """forward(query_features, doc_features, hard_negative_features=None).

    Documents and hard negatives are treated as locked and receive no gradient.
    Hard negatives have shape (batch, K, dim). margin >= 2 disables the mask.
    stable=True rescales the gradient by sqrt(batch / temperature), see
    clip_embedding_loss.py.
    """
    def __init__(self, temperature=0.07, margin=0.1, use_qq_negatives=False,
                 normalized_inputs=False, stable=False):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.use_qq_negatives = use_qq_negatives
        self.normalized_inputs = normalized_inputs
        self.stable = stable

    def forward(self, query_features, doc_features, hard_negative_features=None):
        q, d, h = query_features, doc_features, hard_negative_features
        if not self.normalized_inputs:
            q = F.normalize(q, dim=-1)
            d = F.normalize(d, dim=-1)
            h = F.normalize(h, dim=-1) if h is not None else None
        q, d = q.contiguous(), d.contiguous()
        h = h.contiguous() if h is not None else None

        _validate_features(q, d)
        if h is not None:
            _validate_hard_negatives(q, h)

        batch_size = q.shape[0]
        inv_temperature = _LOG2E / self.temperature
        inv_temperature_orig = (math.sqrt(batch_size / self.temperature)
                                if self.stable else 1.0 / self.temperature)
        return MemoryEfficientLiTEmbeddingLossNormed.apply(
            q, d, h, inv_temperature, inv_temperature_orig, self.margin,
            self.use_qq_negatives)


def _assemble_ring(q_local, d_local, pos, inv_temperature, margin, use_qq, group):
    """Assembles the document tower (and query tower iff use_qq) while accumulating
    the home rows' denominators."""
    world, rank = _world_and_rank(group)
    local_batch, d_model = q_local.shape
    device = q_local.device
    row_offset = rank * local_batch
    sum_exp_row = torch.zeros(local_batch, device=device, dtype=torch.float32)

    if world == 1:
        _launch_denom(q_local, d_local, pos, sum_exp_row, inv_temperature, margin)
        if use_qq:
            _launch_denom(q_local, q_local, pos, sum_exp_row, inv_temperature, margin,
                          exclude_diag=True)
        return d_local.contiguous(), (q_local.contiguous() if use_qq else None), sum_exp_row

    batch_size = world * local_batch
    d_full = torch.empty(batch_size, d_model, device=device, dtype=d_local.dtype)
    q_full = (torch.empty(batch_size, d_model, device=device, dtype=q_local.dtype)
              if use_qq else None)
    send_to, recv_from = (rank + 1) % world, (rank - 1) % world

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
        _launch_denom(q_local, cur_d, pos, sum_exp_row, inv_temperature, margin)
        if use_qq:
            q_full.narrow(0, col_offset, local_batch).copy_(cur_q)
            _launch_denom(q_local, cur_q, pos, sum_exp_row, inv_temperature, margin,
                          exclude_diag=True, row_offset=row_offset, col_offset=col_offset)
        if reqs is not None:
            for req in reqs:
                req.wait()
            cur_d = recv_d
            if use_qq:
                cur_q = recv_q
    return d_full, q_full, sum_exp_row


class DistributedMemoryEfficientLiTEmbeddingLossNormed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_local, d_local, h_local, d_full, q_full, pos, sum_exp_row,
                offset, n, inv_temperature, inv_temperature_orig, margin,
                use_qq, batch_size, group):
        if h_local is not None:
            sum_exp_row += _hard_negative_exp(
                q_local, h_local, pos, inv_temperature, margin).sum(dim=1)

        saved = ((q_local, d_local, d_full, pos, sum_exp_row)
                 + ((q_full,) if use_qq else ()) + ((h_local,) if h_local is not None else ()))
        ctx.save_for_backward(*saved)
        ctx.has_h = h_local is not None
        ctx.offset, ctx.n, ctx.group = offset, n, group
        ctx.inv_temperature, ctx.inv_temperature_orig = inv_temperature, inv_temperature_orig
        ctx.margin, ctx.use_qq = margin, use_qq
        ctx.batch_size, ctx.in_dtype = batch_size, q_local.dtype

        sv = pos * inv_temperature - inv_temperature
        return -(sv * _LN2 - torch.log(sum_exp_row)).sum() / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        tensors = list(ctx.saved_tensors)
        q_local, d_local, d_full, pos, sum_exp_row = tensors[:5]
        q_full = tensors[5] if ctx.use_qq else None
        h_local = tensors[-1] if ctx.has_h else None
        offset, n, batch_size = ctx.offset, ctx.n, ctx.batch_size
        inv_temperature, margin = ctx.inv_temperature, ctx.margin
        grad_scale = ctx.inv_temperature_orig / batch_size
        device, d_model = q_local.device, q_local.shape[1]
        world, _ = _world_and_rank(ctx.group)

        dQ = d_local.float() * (-grad_scale)
        _launch_grad(q_local, d_full, pos, sum_exp_row, dQ, dQ,
                     inv_temperature, margin, grad_scale, two_sided=False)
        if ctx.use_qq:
            dQ_partial = torch.zeros(batch_size, d_model, device=device, dtype=torch.float32)
            _launch_grad(q_local, q_full, pos, sum_exp_row, dQ, dQ_partial,
                         inv_temperature, margin, grad_scale,
                         exclude_diag=True, row_offset=offset, two_sided=True)
            if world > 1:
                dQ_cols = torch.empty(n, d_model, device=device, dtype=torch.float32)
                dist.reduce_scatter_tensor(dQ_cols, dQ_partial, group=ctx.group)
                dQ += dQ_cols
            else:
                dQ += dQ_partial

        if h_local is not None:
            p_h = (_hard_negative_exp(q_local, h_local, pos, inv_temperature, margin)
                   / sum_exp_row[:, None])
            dQ += grad_scale * torch.einsum('bk,bkd->bd', p_h, h_local.float())

        dQ = dQ * grad_output
        return (dQ.to(ctx.in_dtype), None, None,
                None, None, None, None, None, None, None, None, None, None, None, None)


class DistributedMemoryEfficientLiTEmbeddingLoss(nn.Module):
    """DDP counterpart, feed each rank its shard of the global batch. Only the query
    shard gets a gradient, so no gradient communication happens unless q-q negatives
    are enabled (one reduce-scatter then). forward returns this rank's partial loss
    (all-reduce SUM for the global value).
    """
    def __init__(self, temperature=0.07, margin=0.1, use_qq_negatives=False,
                 normalized_inputs=False, stable=False, group=None):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.use_qq_negatives = use_qq_negatives
        self.normalized_inputs = normalized_inputs
        self.stable = stable
        self.group = group

    def forward(self, query_features, doc_features, hard_negative_features=None):
        q, d, h = query_features, doc_features, hard_negative_features
        if not self.normalized_inputs:
            q = F.normalize(q, dim=-1)
            d = F.normalize(d, dim=-1)
            h = F.normalize(h, dim=-1) if h is not None else None
        q, d = q.contiguous(), d.contiguous()
        h = h.contiguous() if h is not None else None

        _validate_features(q, d)
        if h is not None:
            _validate_hard_negatives(q, h)

        world, rank = _world_and_rank(self.group)
        local_batch = q.shape[0]
        batch_size = world * local_batch
        inv_temperature = _LOG2E / self.temperature
        inv_temperature_orig = (math.sqrt(batch_size / self.temperature)
                                if self.stable else 1.0 / self.temperature)

        pos = (q.detach().float() * d.detach().float()).sum(dim=1)
        d_full, q_full, sum_exp_row = _assemble_ring(
            q.detach(), d.detach(), pos, inv_temperature, self.margin,
            self.use_qq_negatives, self.group)
        offset = rank * local_batch
        return DistributedMemoryEfficientLiTEmbeddingLossNormed.apply(
            q, d, h, d_full, q_full, pos, sum_exp_row,
            offset, local_batch, inv_temperature, inv_temperature_orig, self.margin,
            self.use_qq_negatives, batch_size, self.group)
