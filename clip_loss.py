"""Single-GPU memory-efficient CLIP loss.

Reuses the block-wise kernels from distributed_clip_loss.py with the whole batch as
one block: a fused sum-exp pass accumulates the row/column denominators without
materializing the B x B similarity matrix, and the backward recomputes each block
once, emitting both gradient directions. Denominators and gradients accumulate in
fp32 regardless of the input precision, so fp16/bf16 features are numerically safe.
For multi-GPU DDP training use DistributedMemoryEfficientCLIPLoss.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ._common import (LOG2E as _LOG2E, debias_denominators as _debias_denominators,
                          validate_features as _validate_features,
                          validate_tau_plus as _validate_tau_plus)
    from .distributed_clip_loss import _launch_denom, _partial_loss, _ring_backward
except ImportError:   # running as a flat module from inside the repo
    from _common import (LOG2E as _LOG2E, debias_denominators as _debias_denominators,
                         validate_features as _validate_features,
                         validate_tau_plus as _validate_tau_plus)
    from distributed_clip_loss import _launch_denom, _partial_loss, _ring_backward


class MemoryEfficientCLIPLossNormed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, inv_temperature, inv_temperature_orig, tau_plus, floor):
        batch_size = x.shape[0]
        sum_exp_row = torch.zeros(batch_size, device=x.device, dtype=torch.float32)
        sum_exp_col = torch.zeros(batch_size, device=x.device, dtype=torch.float32)
        _launch_denom(x, y, sum_exp_row, sum_exp_col, inv_temperature)

        denom_row, denom_col = sum_exp_row, sum_exp_col
        div_row, div_col, seed = sum_exp_row, sum_exp_col, None
        if tau_plus:
            pos_exp = torch.exp2((x.float() * y.float()).sum(dim=1)
                                 * inv_temperature - inv_temperature)
            denom_row, div_row, seed_row = _debias_denominators(
                pos_exp, sum_exp_row, batch_size - 1, tau_plus, floor)
            denom_col, div_col, seed_col = _debias_denominators(
                pos_exp, sum_exp_col, batch_size - 1, tau_plus, floor)
            seed = seed_row + seed_col

        saved = (x, y, div_row, div_col) + (() if seed is None else (seed,))
        ctx.save_for_backward(*saved)
        ctx.debiased = seed is not None
        ctx.inv_temperature = inv_temperature
        ctx.inv_temperature_orig = inv_temperature_orig
        ctx.batch_size = batch_size
        ctx.in_dtype = x.dtype
        return _partial_loss(x, y, denom_row, denom_col, inv_temperature, batch_size)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.debiased:
            x, y, div_row, div_col, seed = ctx.saved_tensors
        else:
            (x, y, div_row, div_col), seed = ctx.saved_tensors, None
        batch_size = ctx.batch_size
        # offset 0 / n = batch: the single-GPU case is the one-shard distributed case.
        dX, dY = _ring_backward(x, y, div_row, div_col, 0, batch_size,
                                ctx.inv_temperature, ctx.inv_temperature_orig,
                                batch_size, seed)
        if seed is None:
            dY -= x.float() * (ctx.inv_temperature_orig / batch_size)  # -x_i positive-pair
        else:
            dY += x.float() * (seed * (ctx.inv_temperature_orig
                                       / (2.0 * batch_size)))[:, None]
        dX, dY = dX * grad_output, dY * grad_output
        return dX.to(ctx.in_dtype), dY.to(ctx.in_dtype), None, None, None, None


class MemoryEfficientCLIPLoss(nn.Module):
    """Memory-efficient bidirectional CLIP loss; both towers receive gradients.

    stable=True rescales the gradient by sqrt(batch / temperature) instead of
    1 / temperature: 1 / (batch * temperature) can nullify small values even in
    fp32, which matters at large batch sizes (300k+ works fine in practice).
    The loss value is unchanged, only the gradient scale differs, so the learning
    rate becomes batch-size dependent -- use lr / sqrt(batch * temperature) to
    mimic the default behaviour, though at large batches standard values like
    1e-4 tend to work well without that correction.

    tau_plus > 0 switches to the debiased contrastive loss (arXiv 2007.00224): the
    negative sum in each softmax denominator is replaced by its debiased estimate
    under a class prior of tau_plus (the probability that an in-batch negative is
    actually a positive). tau_plus = 0 is the standard loss.
    """
    def __init__(self, temperature=0.07, normalized_inputs=False, stable=False,
                 tau_plus=0.0):
        super().__init__()
        _validate_tau_plus(tau_plus)
        self.temperature = temperature
        self.normalized_inputs = normalized_inputs
        self.stable = stable
        self.tau_plus = tau_plus

    def forward(self, image_features, text_features):
        x = image_features if self.normalized_inputs else F.normalize(image_features, dim=1)
        y = text_features if self.normalized_inputs else F.normalize(text_features, dim=1)
        x, y = x.contiguous(), y.contiguous()
        _validate_features(x, y)

        batch_size = x.shape[0]
        inv_temperature = _LOG2E / self.temperature
        inv_temperature_orig = (math.sqrt(batch_size / self.temperature)
                                if self.stable else 1.0 / self.temperature)
        return MemoryEfficientCLIPLossNormed.apply(
            x, y, inv_temperature, inv_temperature_orig,
            self.tau_plus, math.exp(-2.0 / self.temperature))


class StableMemoryEfficientCLIPLoss(MemoryEfficientCLIPLoss):
    """Deprecated alias for MemoryEfficientCLIPLoss(stable=True)."""
    def __init__(self, temperature=0.07, normalized_inputs=False):
        super().__init__(temperature=temperature, normalized_inputs=normalized_inputs,
                         stable=True)
