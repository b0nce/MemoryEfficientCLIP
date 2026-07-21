"""Qwen3 loss (the Qwen3-Embedding InfoNCE objective), LiT-style: the document
tower is locked.

Same query->document InfoNCE with false-negative masking, hard negatives and
optional q-q negatives as clip_qwen3_loss.py, but documents and hard negatives
receive no gradient. There is no d-d option: with locked documents its repulsion
gradient has nowhere to land, it only inflates the denominator. The backward emits
dQ only, so the DDP version needs no gradient communication unless q-q is enabled.
The single-GPU backward is FlashAttention-shaped for d_model within the cap in
_common.py (register-tile accumulation, no gradient atomics); larger d_model and
the DDP module use the atomic tile-grid kernel. Includes single-GPU and DDP
modules.
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
    fa_ok as _fa_ok,
    hard_negative_exp as _hard_negative_exp,
    launch_qwen3_denom as _launch_denom,
    launch_qwen3_fa_grad as _launch_fa_grad,
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

# None -> auto (fa_ok picks the FlashAttention-shaped backward for
# d_model <= the cap in _common.py); True/False forces it (test hook).
_FORCE_FA = None


class MemoryEfficientLiTQwen3LossNormed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, d, h, inv_temperature, inv_temperature_orig, margin, use_qq,
                tau_plus, floor):
        batch_size, d_model = q.shape
        pos = (q.float() * d.float()).sum(dim=1)

        sum_exp_row = torch.zeros(batch_size, device=q.device, dtype=torch.float32)
        _launch_denom(q, d, pos, sum_exp_row, inv_temperature, margin)
        if use_qq:
            _launch_denom(q, q, pos, sum_exp_row, inv_temperature, margin, exclude_diag=True)
        if h is not None:
            sum_exp_row += _hard_negative_exp(q, h, pos, inv_temperature, margin).sum(dim=1)

        sv = pos * inv_temperature - inv_temperature
        denom, div, seed = sum_exp_row, sum_exp_row, None
        if tau_plus is not None:
            denom, div, seed = _debias_denominators(
                torch.exp2(sv), sum_exp_row,
                _num_negatives(batch_size, h, use_qq, False), tau_plus, floor)

        saved = ((q, d, pos, div) + (() if seed is None else (seed,))
                 + ((h,) if h is not None else ()))
        ctx.save_for_backward(*saved)
        ctx.debiased = seed is not None
        ctx.has_h = h is not None
        ctx.inv_temperature = inv_temperature
        ctx.inv_temperature_orig = inv_temperature_orig
        ctx.margin = margin
        ctx.use_qq = use_qq
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

        coeff = -grad_scale if seed is None else (seed * grad_scale)[:, None]
        dQ = d.float() * coeff
        use_fa = _FORCE_FA if _FORCE_FA is not None else _fa_ok(q.shape[1], q.device)
        if use_fa:
            _launch_fa_grad(q, d, pos, div, dQ, inv_temperature, margin,
                            grad_scale, row_side=True, col_side=False,
                            exclude_diag=False)
            if ctx.use_qq:
                _launch_fa_grad(q, q, pos, div, dQ, inv_temperature, margin,
                                grad_scale, row_side=True, col_side=True,
                                exclude_diag=True)
        else:
            _launch_grad(q, d, pos, div, dQ, dQ, inv_temperature, margin, grad_scale,
                         two_sided=False)
            if ctx.use_qq:
                _launch_grad(q, q, pos, div, dQ, dQ, inv_temperature, margin, grad_scale,
                             exclude_diag=True, two_sided=True)
        if h is not None:
            p_h = _hard_negative_exp(q, h, pos, inv_temperature, margin) / div[:, None]
            dQ += grad_scale * torch.einsum('bk,bkd->bd', p_h, h.float())

        dQ = dQ * grad_output
        return dQ.to(ctx.in_dtype), None, None, None, None, None, None, None, None


class MemoryEfficientLiTQwen3Loss(nn.Module):
    """forward(query_features, doc_features, hard_negative_features=None).

    Documents and hard negatives are treated as locked and receive no gradient.
    Hard negatives have shape (batch, K, dim). margin >= 2 disables the mask.
    stable=True rescales the gradient by sqrt(batch / temperature), and tau_plus > 0
    switches to the debiased contrastive loss (arXiv 2007.00224); see
    clip_qwen3_loss.py for both. forward also accepts a per-call tau_plus override
    -- a float or a (batch,) tensor of per-row priors in [0, 1). label_smoothing > 0
    smooths the target over the nominal candidates as in MemoryEfficientQwen3Loss
    (the smoothing attraction toward locked documents and hard negatives lands only
    on the queries).
    """
    def __init__(self, temperature=0.07, margin=0.1, use_qq_negatives=False,
                 normalized_inputs=False, stable=True, tau_plus=0.0,
                 label_smoothing=0.0):
        super().__init__()
        _validate_tau_plus(tau_plus)
        _validate_label_smoothing(label_smoothing)
        self.temperature = temperature
        self.margin = margin
        self.use_qq_negatives = use_qq_negatives
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
        loss = MemoryEfficientLiTQwen3LossNormed.apply(
            q, d, h, inv_temperature, inv_temperature_orig, self.margin,
            self.use_qq_negatives, tau_plus, math.exp(-2.0 / self.temperature))
        if self.label_smoothing:
            grad_factor = self.temperature * inv_temperature_orig if self.stable else 1.0
            loss = loss + _qwen3_smoothing_term(
                q, d.detach(), h.detach() if h is not None else None,
                self.label_smoothing,
                _num_negatives(batch_size, h, self.use_qq_negatives, False),
                batch_size, self.temperature, grad_factor,
                self.use_qq_negatives, False)
        return loss


class DistributedMemoryEfficientLiTQwen3LossNormed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_local, d_local, h_local, d_full, q_full, pos, sum_exp_row,
                offset, n, inv_temperature, inv_temperature_orig, margin,
                use_qq, tau_plus, floor, batch_size, group):
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
                _num_negatives(batch_size, h_local, use_qq, False), tau_plus, floor)

        saved = ((q_local, d_local, d_full, pos, div)
                 + (() if seed is None else (seed,))
                 + ((q_full,) if use_qq else ()) + ((h_local,) if h_local is not None else ()))
        ctx.save_for_backward(*saved)
        ctx.debiased = seed is not None
        ctx.has_h = h_local is not None
        ctx.offset, ctx.n, ctx.group = offset, n, group
        ctx.inv_temperature, ctx.inv_temperature_orig = inv_temperature, inv_temperature_orig
        ctx.margin, ctx.use_qq = margin, use_qq
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

        if seed is None:
            dQ = d_local.float() * (-grad_scale)
        else:
            dQ = d_local.float() * (seed * grad_scale)[:, None]
        _launch_grad(q_local, d_full, pos, div, dQ, dQ,
                     inv_temperature, margin, grad_scale, two_sided=False)
        if ctx.use_qq:
            dQ_partial = torch.zeros(batch_size, d_model, device=device, dtype=torch.float32)
            _launch_grad(q_local, q_full, pos, div, dQ, dQ_partial,
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
                   / div[:, None])
            dQ += grad_scale * torch.einsum('bk,bkd->bd', p_h, h_local.float())

        dQ = dQ * grad_output
        return (dQ.to(ctx.in_dtype), None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None, None)


class DistributedMemoryEfficientLiTQwen3Loss(nn.Module):
    """DDP counterpart, feed each rank its shard of the global batch. Only the query
    shard gets a gradient, so no gradient communication happens unless q-q negatives
    are enabled (one reduce-scatter then). forward returns this rank's partial loss
    (all-reduce SUM for the global value). forward accepts a per-call tau_plus
    override -- a float or a (local_batch,) tensor of per-row priors in [0, 1) for
    this rank's shard (adds no communication). label_smoothing > 0 smooths the
    target over the nominal candidates (adds one O(dim) all-reduce).
    """
    def __init__(self, temperature=0.07, margin=0.1, use_qq_negatives=False,
                 normalized_inputs=False, stable=True, tau_plus=0.0,
                 label_smoothing=0.0, group=None):
        super().__init__()
        _validate_tau_plus(tau_plus)
        _validate_label_smoothing(label_smoothing)
        self.temperature = temperature
        self.margin = margin
        self.use_qq_negatives = use_qq_negatives
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
            self.use_qq_negatives, False, self.group)
        offset = rank * local_batch
        loss = DistributedMemoryEfficientLiTQwen3LossNormed.apply(
            q, d, h, d_full, q_full, pos, sum_exp_row,
            offset, local_batch, inv_temperature, inv_temperature_orig, self.margin,
            self.use_qq_negatives, tau_plus, math.exp(-2.0 / self.temperature),
            batch_size, self.group)
        if self.label_smoothing:
            grad_factor = self.temperature * inv_temperature_orig if self.stable else 1.0
            loss = loss + _qwen3_smoothing_term(
                q, d.detach(), h.detach() if h is not None else None,
                self.label_smoothing,
                _num_negatives(batch_size, h, self.use_qq_negatives, False),
                batch_size, self.temperature, grad_factor,
                self.use_qq_negatives, False, self.group, world)
        return loss
