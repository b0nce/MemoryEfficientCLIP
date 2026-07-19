"""Qwen3 loss (the Qwen3-Embedding InfoNCE objective), CLIP-style: both towers
receive gradients.

Asymmetric query->document InfoNCE with a false-negative mask (negatives whose
similarity exceeds s(q_i, d_i) + margin are dropped), optional row-specific hard
negatives (batch, K, dim) and optional q-q / d-d in-batch negatives. The softmax is
row-only, so every negative group adds into the same per-row fp32 denominator and
the same kernel pair covers all passes. Includes single-GPU and DDP modules.
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
    hard_negative_exp as _hard_negative_exp,
    launch_qwen3_denom as _launch_denom,
    launch_qwen3_grad as _launch_grad,
    qwen3_assemble_ring as _assemble_ring,
    qwen3_num_negatives as _num_negatives,
    qwen3_smoothing_term as _qwen3_smoothing_term,
    resolve_tau_plus as _resolve_tau_plus,
    validate_features as _validate_features,
    validate_hard_negatives as _validate_hard_negatives,
    validate_label_smoothing as _validate_label_smoothing,
    validate_tau_plus as _validate_tau_plus,
    world_and_rank as _world_and_rank,
)


class MemoryEfficientQwen3LossNormed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, d, h, inv_temperature, inv_temperature_orig, margin, use_qq, use_dd,
                tau_plus, floor):
        batch_size, d_model = q.shape
        pos = (q.float() * d.float()).sum(dim=1)

        sum_exp_row = torch.zeros(batch_size, device=q.device, dtype=torch.float32)
        _launch_denom(q, d, pos, sum_exp_row, inv_temperature, margin)
        if use_qq:
            _launch_denom(q, q, pos, sum_exp_row, inv_temperature, margin, exclude_diag=True)
        if use_dd:
            _launch_denom(d, d, pos, sum_exp_row, inv_temperature, margin, exclude_diag=True)
        if h is not None:
            sum_exp_row += _hard_negative_exp(q, h, pos, inv_temperature, margin).sum(dim=1)

        sv = pos * inv_temperature - inv_temperature
        denom, div, seed = sum_exp_row, sum_exp_row, None
        if tau_plus is not None:
            denom, div, seed = _debias_denominators(
                torch.exp2(sv), sum_exp_row,
                _num_negatives(batch_size, h, use_qq, use_dd), tau_plus, floor)

        saved = ((q, d, pos, div) + (() if seed is None else (seed,))
                 + ((h,) if h is not None else ()))
        ctx.save_for_backward(*saved)
        ctx.debiased = seed is not None
        ctx.has_h = h is not None
        ctx.inv_temperature = inv_temperature
        ctx.inv_temperature_orig = inv_temperature_orig
        ctx.margin = margin
        ctx.use_qq, ctx.use_dd = use_qq, use_dd
        ctx.batch_size = batch_size
        ctx.in_dtype = q.dtype
        return -(sv * _LN2 - torch.log(denom)).mean()

    @staticmethod
    def backward(ctx, grad_output):
        tensors = list(ctx.saved_tensors)
        q, d, pos, div = tensors[:4]
        seed = tensors[4] if ctx.debiased else None
        h = tensors[-1] if ctx.has_h else None
        inv_temperature, margin = ctx.inv_temperature, ctx.margin
        grad_scale = ctx.inv_temperature_orig / ctx.batch_size

        # Seeds carry the positive pair terms, kernels add the p-weighted sums.
        coeff = -grad_scale if seed is None else (seed * grad_scale)[:, None]
        dQ = d.float() * coeff
        dD = q.float() * coeff
        _launch_grad(q, d, pos, div, dQ, dD, inv_temperature, margin, grad_scale,
                     two_sided=True)
        if ctx.use_qq:
            _launch_grad(q, q, pos, div, dQ, dQ, inv_temperature, margin, grad_scale,
                         exclude_diag=True, two_sided=True)
        if ctx.use_dd:
            _launch_grad(d, d, pos, div, dD, dD, inv_temperature, margin, grad_scale,
                         exclude_diag=True, two_sided=True)

        dH = None
        if h is not None:
            p_h = _hard_negative_exp(q, h, pos, inv_temperature, margin) / div[:, None]
            dQ += grad_scale * torch.einsum('bk,bkd->bd', p_h, h.float())
            dH = ((grad_scale * p_h)[:, :, None] * q.float()[:, None, :] * grad_output).to(h.dtype)

        dQ, dD = dQ * grad_output, dD * grad_output
        return (dQ.to(ctx.in_dtype), dD.to(ctx.in_dtype), dH,
                None, None, None, None, None, None, None)


class MemoryEfficientQwen3Loss(nn.Module):
    """forward(query_features, doc_features, hard_negative_features=None).

    Hard negatives have shape (batch, K, dim), each query competes only against its
    own K. margin >= 2 disables the false-negative mask. stable=True rescales the
    gradient by sqrt(batch / temperature) instead of 1 / temperature to avoid fp32
    underflow at very large batches, use lr / sqrt(batch * temperature) to mimic the
    default behaviour.

    tau_plus > 0 switches to the debiased contrastive loss (arXiv 2007.00224): the
    negative sum in the row denominator is replaced by its debiased estimate under a
    class prior of tau_plus. This composes with the false-negative mask: masked
    entries contribute zero to the negative mean but keep their slot in the nominal
    negative count. forward also accepts a per-call tau_plus override -- a float or
    a (batch,) tensor of per-row priors in [0, 1) (e.g. when duplicate rates are
    known per query).

    label_smoothing > 0 smooths the row softmax target with the F.cross_entropy
    convention: (1 - eps) on the positive plus eps/C uniform over the C = N + 1
    nominal candidates (N as in the debiased loss; the false-negative mask does not
    reshape the target). O(batch * dim) eager math; composes with stable and
    tau_plus.
    """
    def __init__(self, temperature=0.07, margin=0.1, use_qq_negatives=False,
                 use_dd_negatives=False, normalized_inputs=False, stable=False,
                 tau_plus=0.0, label_smoothing=0.0):
        super().__init__()
        _validate_tau_plus(tau_plus)
        _validate_label_smoothing(label_smoothing)
        self.temperature = temperature
        self.margin = margin
        self.use_qq_negatives = use_qq_negatives
        self.use_dd_negatives = use_dd_negatives
        self.normalized_inputs = normalized_inputs
        self.stable = stable
        self.tau_plus = tau_plus
        self.label_smoothing = label_smoothing

    def forward(self, query_features, doc_features, hard_negative_features=None,
                tau_plus=None):
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
        tau_plus = _resolve_tau_plus(self.tau_plus, tau_plus, batch_size, q.device)
        inv_temperature = _LOG2E / self.temperature
        inv_temperature_orig = (math.sqrt(batch_size / self.temperature)
                                if self.stable else 1.0 / self.temperature)
        loss = MemoryEfficientQwen3LossNormed.apply(
            q, d, h, inv_temperature, inv_temperature_orig, self.margin,
            self.use_qq_negatives, self.use_dd_negatives,
            tau_plus, math.exp(-2.0 / self.temperature))
        if self.label_smoothing:
            grad_factor = self.temperature * inv_temperature_orig if self.stable else 1.0
            loss = loss + _qwen3_smoothing_term(
                q, d, h, self.label_smoothing,
                _num_negatives(batch_size, h, self.use_qq_negatives, self.use_dd_negatives),
                batch_size, self.temperature, grad_factor,
                self.use_qq_negatives, self.use_dd_negatives)
        return loss


class DistributedMemoryEfficientQwen3LossNormed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_local, d_local, h_local, d_full, q_full, pos, sum_exp_row,
                offset, n, inv_temperature, inv_temperature_orig, margin,
                use_qq, use_dd, tau_plus, floor, batch_size, group):
        if h_local is not None:
            sum_exp_row += _hard_negative_exp(
                q_local, h_local, pos, inv_temperature, margin).sum(dim=1)

        # the row denominators are complete on their home rank, so the debiased
        # transform needs no communication.
        sv = pos * inv_temperature - inv_temperature
        denom, div, seed = sum_exp_row, sum_exp_row, None
        if tau_plus is not None:
            denom, div, seed = _debias_denominators(
                torch.exp2(sv), sum_exp_row,
                _num_negatives(batch_size, h_local, use_qq, use_dd), tau_plus, floor)

        saved = ((q_local, d_local, d_full, pos, div)
                 + (() if seed is None else (seed,))
                 + ((q_full,) if use_qq else ()) + ((h_local,) if h_local is not None else ()))
        ctx.save_for_backward(*saved)
        ctx.debiased = seed is not None
        ctx.has_h = h_local is not None
        ctx.offset, ctx.n, ctx.group = offset, n, group
        ctx.inv_temperature, ctx.inv_temperature_orig = inv_temperature, inv_temperature_orig
        ctx.margin, ctx.use_qq, ctx.use_dd = margin, use_qq, use_dd
        ctx.batch_size, ctx.in_dtype = batch_size, q_local.dtype
        return -(sv * _LN2 - torch.log(denom)).sum() / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        tensors = list(ctx.saved_tensors)
        q_local, d_local, d_full, pos, div = tensors[:5]
        idx = 5
        seed = None
        if ctx.debiased:
            seed, idx = tensors[idx], idx + 1
        q_full = tensors[idx] if ctx.use_qq else None
        h_local = tensors[-1] if ctx.has_h else None
        offset, n, batch_size = ctx.offset, ctx.n, ctx.batch_size
        inv_temperature, margin = ctx.inv_temperature, ctx.margin
        grad_scale = ctx.inv_temperature_orig / batch_size
        device, d_model = q_local.device, q_local.shape[1]
        world, _ = _world_and_rank(ctx.group)

        # dD_partial holds this rank's contribution to every column and is delivered
        # to the owner by reduce-scatter. The d-d row direction goes into this rank's
        # own slice so it comes back through the same reduce-scatter.
        seed_coeff = -grad_scale if seed is None else (seed * grad_scale)[:, None]
        dQ = d_local.float() * seed_coeff
        dD_partial = torch.zeros(batch_size, d_model, device=device, dtype=torch.float32)
        _launch_grad(q_local, d_full, pos, div, dQ, dD_partial,
                     inv_temperature, margin, grad_scale, two_sided=True)
        if ctx.use_dd:
            _launch_grad(d_local, d_full, pos, div,
                         dD_partial.narrow(0, offset, n), dD_partial,
                         inv_temperature, margin, grad_scale,
                         exclude_diag=True, row_offset=offset, two_sided=True)
        dQ_partial = None
        if ctx.use_qq:
            dQ_partial = torch.zeros(batch_size, d_model, device=device, dtype=torch.float32)
            _launch_grad(q_local, q_full, pos, div, dQ, dQ_partial,
                         inv_temperature, margin, grad_scale,
                         exclude_diag=True, row_offset=offset, two_sided=True)

        if world > 1:
            dD = torch.empty(n, d_model, device=device, dtype=torch.float32)
            dist.reduce_scatter_tensor(dD, dD_partial, group=ctx.group)
            if ctx.use_qq:
                dQ_cols = torch.empty(n, d_model, device=device, dtype=torch.float32)
                dist.reduce_scatter_tensor(dQ_cols, dQ_partial, group=ctx.group)
                dQ += dQ_cols
        else:
            dD = dD_partial
            if ctx.use_qq:
                dQ += dQ_partial
        dD += q_local.float() * seed_coeff

        dH = None
        if h_local is not None:
            p_h = (_hard_negative_exp(q_local, h_local, pos, inv_temperature, margin)
                   / div[:, None])
            dQ += grad_scale * torch.einsum('bk,bkd->bd', p_h, h_local.float())
            dH = ((grad_scale * p_h)[:, :, None] * q_local.float()[:, None, :]
                  * grad_output).to(h_local.dtype)

        dQ, dD = dQ * grad_output, dD * grad_output
        return (dQ.to(ctx.in_dtype), dD.to(ctx.in_dtype), dH, None, None, None, None,
                None, None, None, None, None, None, None, None, None, None, None)


class DistributedMemoryEfficientQwen3Loss(nn.Module):
    """DDP counterpart, feed each rank its shard of the global batch. Hard negatives
    stay on their query's rank. forward returns this rank's partial loss (all-reduce
    SUM for the global value), backward fills the shard's gradients. forward accepts
    a per-call tau_plus override -- a float or a (local_batch,) tensor of per-row
    priors in [0, 1) for this rank's shard (the row denominators live on their home
    rank, so per-row priors add no communication). label_smoothing > 0 smooths the
    target over the nominal candidates as in MemoryEfficientQwen3Loss (adds one
    O(dim) all-reduce).
    """
    def __init__(self, temperature=0.07, margin=0.1, use_qq_negatives=False,
                 use_dd_negatives=False, normalized_inputs=False, stable=False,
                 tau_plus=0.0, label_smoothing=0.0, group=None):
        super().__init__()
        _validate_tau_plus(tau_plus)
        _validate_label_smoothing(label_smoothing)
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
        q, d = q.contiguous(), d.contiguous()
        h = h.contiguous() if h is not None else None

        _validate_features(q, d)
        if h is not None:
            _validate_hard_negatives(q, h)

        world, rank = _world_and_rank(self.group)
        local_batch = q.shape[0]
        batch_size = world * local_batch
        tau_plus = _resolve_tau_plus(self.tau_plus, tau_plus, local_batch, q.device)
        inv_temperature = _LOG2E / self.temperature
        inv_temperature_orig = (math.sqrt(batch_size / self.temperature)
                                if self.stable else 1.0 / self.temperature)

        pos = (q.detach().float() * d.detach().float()).sum(dim=1)
        d_full, q_full, sum_exp_row = _assemble_ring(
            q.detach(), d.detach(), pos, inv_temperature, self.margin,
            self.use_qq_negatives, self.use_dd_negatives, self.group)
        offset = rank * local_batch
        loss = DistributedMemoryEfficientQwen3LossNormed.apply(
            q, d, h, d_full, q_full, pos, sum_exp_row,
            offset, local_batch, inv_temperature, inv_temperature_orig, self.margin,
            self.use_qq_negatives, self.use_dd_negatives,
            tau_plus, math.exp(-2.0 / self.temperature), batch_size, self.group)
        if self.label_smoothing:
            grad_factor = self.temperature * inv_temperature_orig if self.stable else 1.0
            loss = loss + _qwen3_smoothing_term(
                q, d, h, self.label_smoothing,
                _num_negatives(batch_size, h, self.use_qq_negatives, self.use_dd_negatives),
                batch_size, self.temperature, grad_factor,
                self.use_qq_negatives, self.use_dd_negatives, self.group, world)
        return loss
