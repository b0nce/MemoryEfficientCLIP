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

try:
    from ._common import (LN2 as _LN2, LOG2E as _LOG2E,
                          validate_features as _validate_features)
    from .distributed_lit_loss import _launch_denom, _launch_grad
except ImportError:   # running as a flat module from inside the repo
    from _common import (LN2 as _LN2, LOG2E as _LOG2E,
                         validate_features as _validate_features)
    from distributed_lit_loss import _launch_denom, _launch_grad


class MemoryEfficientLiTLossNormed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_text, y_img, inv_temperature, scale):
        batch_size = x_text.shape[0]
        sum_exp_row = torch.zeros(batch_size, device=x_text.device, dtype=torch.float32)
        _launch_denom(x_text, y_img, sum_exp_row, inv_temperature)

        ctx.save_for_backward(x_text, y_img, sum_exp_row)
        ctx.inv_temperature = inv_temperature
        ctx.scale = scale
        ctx.in_dtype = x_text.dtype

        # loss over the diagonal, formed directly from the log2-domain logit
        # (log(exp2(s)) = ln2 * s) so no exponential is materialized just to be logged.
        sv = (x_text.float() * y_img.float()).sum(dim=1) * inv_temperature - inv_temperature
        return -(sv * _LN2 - torch.log(sum_exp_row)).mean()

    @staticmethod
    def backward(ctx, grad_output):
        x_text, y_img, sum_exp_row = ctx.saved_tensors
        # seed with the -y_i positive-pair term, the kernel adds the p-weighted sums.
        dX = y_img.float() * (-ctx.scale)
        _launch_grad(x_text, y_img, sum_exp_row, ctx.scale, dX, ctx.inv_temperature)
        dX = dX * grad_output
        return dX.to(ctx.in_dtype), None, None, None


class MemoryEfficientLiTLoss(nn.Module):
    """Memory-efficient unidirectional LiT loss; only the text tower is trained.

    stable=True rescales the gradient by sqrt(batch / temperature) instead of
    1 / temperature: 1 / (batch * temperature) can nullify small values even in
    fp32, which matters at large batch sizes (300k+ works fine in practice).
    The loss value is unchanged, only the gradient scale differs, so the learning
    rate becomes batch-size dependent -- use lr / sqrt(batch * temperature) to
    mimic the default behaviour, though at large batches standard values like
    1e-4 tend to work well without that correction.
    """
    def __init__(self, temperature=0.07, normalized_inputs=False, stable=False):
        super().__init__()
        self.temperature = temperature
        self.normalized_inputs = normalized_inputs
        self.stable = stable

    def forward(self, text_features, image_features):
        x = text_features if self.normalized_inputs else F.normalize(text_features, dim=1)
        y = image_features if self.normalized_inputs else F.normalize(image_features, dim=1)
        x, y = x.contiguous(), y.contiguous()
        _validate_features(x, y)

        batch_size = x.shape[0]
        inv_temperature = _LOG2E / self.temperature
        inv_temperature_orig = (math.sqrt(batch_size / self.temperature)
                                if self.stable else 1.0 / self.temperature)
        # image is locked -> detach so no gradient is tracked for it.
        return MemoryEfficientLiTLossNormed.apply(
            x, y.detach(), inv_temperature, inv_temperature_orig / batch_size)


class StableMemoryEfficientLiTLoss(MemoryEfficientLiTLoss):
    """Deprecated alias for MemoryEfficientLiTLoss(stable=True)."""
    def __init__(self, temperature=0.07, normalized_inputs=False):
        super().__init__(temperature=temperature, normalized_inputs=normalized_inputs,
                         stable=True)
