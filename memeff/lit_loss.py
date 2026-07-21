"""Single-GPU memory-efficient LiT loss.

LiT (Locked-image text Tuning) is a text->image row-softmax with the image tower
locked: only the text features receive a gradient. Reuses the block-wise kernels
from distributed_lit_loss.py with the whole batch as one block; the denominator and
gradient accumulate in fp32 regardless of the input precision, so fp16/bf16
features are numerically safe. For multi-GPU DDP training use
DistributedMemoryEfficientLiTLoss.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._common import (LN2 as _LN2, LOG2E as _LOG2E,
                      clip_smoothing_term as _clip_smoothing_term,
                      debias_denominators as _debias_denominators,
                      resolve_tau_plus as _resolve_tau_plus,
                      validate_features as _validate_features,
                      validate_label_smoothing as _validate_label_smoothing,
                      validate_tau_plus as _validate_tau_plus)
from .clip_loss import fa_backward_one as _fa_backward_one, fa_ok as _fa_ok
from .distributed_lit_loss import _launch_denom, _launch_grad


class MemoryEfficientLiTLossNormed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_text, y_img, inv_temperature, scale, tau_plus, floor):
        batch_size = x_text.shape[0]
        sum_exp_row = torch.zeros(batch_size, device=x_text.device, dtype=torch.float32)
        _launch_denom(x_text, y_img, sum_exp_row, inv_temperature)

        # the log2-domain diagonal, reused for the loss (log(exp2(s)) = ln2 * s) so no
        # exponential is materialized just to be logged, and for the debiased transform.
        sv = (x_text.float() * y_img.float()).sum(dim=1) * inv_temperature - inv_temperature
        denom, div, seed = sum_exp_row, sum_exp_row, None
        if tau_plus is not None:
            denom, div, seed = _debias_denominators(
                torch.exp2(sv), sum_exp_row, batch_size - 1, tau_plus, floor)

        saved = (x_text, y_img, div) + (() if seed is None else (seed,))
        ctx.save_for_backward(*saved)
        ctx.debiased = seed is not None
        ctx.inv_temperature = inv_temperature
        ctx.scale = scale
        ctx.in_dtype = x_text.dtype
        return -(sv * _LN2 - torch.log(denom)).mean()

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.debiased:
            x_text, y_img, div, seed = ctx.saved_tensors
        else:
            (x_text, y_img, div), seed = ctx.saved_tensors, None
        # seed with the positive-pair term, the kernel adds the p-weighted sums.
        if _fa_ok(x_text.shape[1], x_text.device):
            dX = _fa_backward_one(x_text, y_img, div, None, ctx.inv_temperature,
                                  ctx.scale, bidir=False)
            if seed is None:
                dX -= y_img.float() * ctx.scale
            else:
                dX += y_img.float() * (seed * ctx.scale)[:, None]
        else:
            if seed is None:
                dX = y_img.float() * (-ctx.scale)
            else:
                dX = y_img.float() * (seed * ctx.scale)[:, None]
            _launch_grad(x_text, y_img, div, ctx.scale, dX, ctx.inv_temperature)
        dX = dX * grad_output
        return dX.to(ctx.in_dtype), None, None, None, None, None


class MemoryEfficientLiTLoss(nn.Module):
    """Memory-efficient unidirectional LiT loss; only the text tower is trained.

    stable=True rescales the gradient by sqrt(batch / temperature) instead of
    1 / temperature: 1 / (batch * temperature) can nullify small values even in
    fp32, which matters at large batch sizes (300k+ works fine in practice).
    The loss value is unchanged, only the gradient scale differs, so the learning
    rate becomes batch-size dependent -- use lr / sqrt(batch * temperature) to
    mimic the default behaviour, though at large batches standard values like
    1e-4 tend to work well without that correction.

    tau_plus > 0 switches to the debiased contrastive loss (arXiv 2007.00224): the
    negative sum in the row softmax denominator is replaced by its debiased estimate
    under a class prior of tau_plus. tau_plus = 0 is the standard loss. forward also
    accepts a per-call tau_plus override -- a float or a (batch,) tensor of per-row
    priors in [0, 1) (e.g. when duplicate rates are known per sample).

    label_smoothing > 0 smooths the row softmax targets with the F.cross_entropy
    convention: (1 - eps) on the diagonal plus eps/batch uniform. Costs only
    O(batch * dim) eager math; composes with stable and tau_plus.
    """
    def __init__(self, temperature=0.07, normalized_inputs=False, stable=True,
                 tau_plus=0.0, label_smoothing=0.0):
        super().__init__()
        _validate_tau_plus(tau_plus)
        _validate_label_smoothing(label_smoothing)
        self.temperature = temperature
        self.normalized_inputs = normalized_inputs
        self.stable = stable
        self.tau_plus = tau_plus
        self.label_smoothing = label_smoothing

    def forward(self, text_features, image_features, tau_plus=None):
        x = text_features if self.normalized_inputs else F.normalize(text_features, dim=1)
        y = image_features if self.normalized_inputs else F.normalize(image_features, dim=1)
        x, y = x.contiguous(), y.contiguous()
        _validate_features(x, y)

        batch_size = x.shape[0]
        tau_plus = _resolve_tau_plus(self.tau_plus, tau_plus, batch_size, x.device)
        inv_temperature = _LOG2E / self.temperature
        inv_temperature_orig = (math.sqrt(batch_size / self.temperature)
                                if self.stable else 1.0 / self.temperature)
        # image is locked -> detach so no gradient is tracked for it.
        loss = MemoryEfficientLiTLossNormed.apply(
            x, y.detach(), inv_temperature, inv_temperature_orig / batch_size,
            tau_plus, math.exp(-2.0 / self.temperature))
        if self.label_smoothing:
            grad_factor = self.temperature * inv_temperature_orig if self.stable else 1.0
            loss = loss + _clip_smoothing_term(x, y.detach(), self.label_smoothing,
                                               batch_size, self.temperature, grad_factor)
        return loss


class StableMemoryEfficientLiTLoss(MemoryEfficientLiTLoss):
    """Deprecated alias for MemoryEfficientLiTLoss(stable=True)."""
    def __init__(self, temperature=0.07, normalized_inputs=False):
        super().__init__(temperature=temperature, normalized_inputs=normalized_inputs,
                         stable=True)
