"""Single-GPU memory-efficient CLIP loss.

Reuses the block-wise kernels from distributed_clip_loss.py with the whole batch as
one block: a fused sum-exp pass accumulates the row/column denominators without
materializing the B x B similarity matrix. The backward is FlashAttention-shaped
whenever the feature dimension allows it (d_model <= 512): each program owns a
block of output rows, streams every opposite-tower block, and accumulates its
gradient rows in a (BLOCK, d_model) fp32 register tile written out exactly once
-- no atomics (measured 47% of the tile-grid backward's time at batch 64k) and no
tile-multiplied read-modify-write traffic. Each tower's pass recomputes the
similarities, the same trade FlashAttention-2 makes for dQ vs dK/dV. Larger
d_model falls back to the atomic tile-grid kernel. Denominators and gradients
accumulate in fp32 regardless of the input precision, so bf16/fp16 features are
numerically safe. For multi-GPU DDP training use DistributedMemoryEfficientCLIPLoss.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from ._common import (INPUT_PRECISION as _INPUT_PRECISION, LOG2E as _LOG2E,
                      clip_smoothing_term as _clip_smoothing_term,
                      debias_denominators as _debias_denominators,
                      resolve_tau_plus as _resolve_tau_plus,
                      validate_features as _validate_features,
                      validate_label_smoothing as _validate_label_smoothing,
                      validate_tau_plus as _validate_tau_plus)
from .distributed_clip_loss import _launch_denom, _partial_loss, _ring_backward


@triton.jit
def clip_fa_grad_kernel(
    A_ptr, B_ptr, div_own_ptr, div_other_ptr, dA_ptr,
    inv_temperature, grad_scale, n_own, n_other,
    BIDIR: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr,
    D_POW2: tl.constexpr, D_MODEL: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """One tower's full gradient rows, FlashAttention-style: the program owns
    rows [pid * BLOCK_I, ...) of A, streams every B block, and accumulates
    dA = sum_j prob_ij * B_j in a fp32 register tile stored exactly once.
    prob folds this side's own softmax (div_own, indexed by the owned rows)
    and, iff BIDIR, the opposite direction's (div_other, indexed by the
    streamed columns). Launched twice with the towers swapped for the
    bidirectional loss; the positive-pair seed is added eagerly outside. The
    feature dim is padded to D_POW2 register lanes (masked to D_MODEL), which
    caps this kernel to small d_model -- the launcher falls back to the
    atomic tile-grid kernel above that."""
    pid = tl.program_id(0)
    i_offsets = pid * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    d_offsets = tl.arange(0, D_POW2)
    i_mask = i_offsets < n_own
    d_mask = d_offsets < D_MODEL

    A_own = tl.load(A_ptr + (i_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                    mask=(i_mask[:, None] & d_mask[None, :]), other=0.0)
    # 1D divisor loads use clamped indices instead of mask/other: the padded
    # lanes stay finite (their rows/columns contribute zero through the zeroed
    # feature loads), and the masked-load select trips a triton 3.1 layout-
    # conversion bug when broadcast against the dot layouts.
    div_own = tl.load(div_own_ptr + tl.minimum(i_offsets, n_own - 1))
    acc = tl.zeros([BLOCK_SIZE_I, D_POW2], dtype=tl.float32)

    for j_start in range(0, n_other, BLOCK_SIZE_J):
        j_offsets = j_start + tl.arange(0, BLOCK_SIZE_J)
        j_mask = j_offsets < n_other
        B_block = tl.load(B_ptr + (j_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                          mask=(j_mask[:, None] & d_mask[None, :]), other=0.0)
        S = tl.dot(A_own, tl.trans(B_block), input_precision=INPUT_PRECISION)
        # No padded-lane masking needed: out-of-range B rows load as zeros, so
        # whatever finite prob they carry contributes zero to the acc dot (and
        # a 2D tl.where here trips a triton 3.1 layout-conversion bug anyway).
        exp_S = tl.exp2(S * inv_temperature - inv_temperature)
        prob = tl.math.fdiv(exp_S, div_own[:, None])
        if BIDIR:
            div_other = tl.load(div_other_ptr + tl.minimum(j_offsets, n_other - 1))
            prob += tl.math.fdiv(exp_S, div_other[None, :])
        prob_low = (prob * grad_scale).to(A_ptr.dtype.element_ty)
        acc = tl.dot(prob_low, B_block, acc, input_precision=INPUT_PRECISION)

    tl.store(dA_ptr + (i_offsets[:, None] * D_MODEL + d_offsets[None, :]), acc,
             mask=(i_mask[:, None] & d_mask[None, :]))


# FlashAttention-style backward needs a (BLOCK_I, next_pow2(d_model)) fp32
# register accumulator per program: the row block shrinks as d_model grows to
# hold the accumulator at 128 KB (half the sm80/sm90 register file). Above
# the cap the atomic tile-grid backward takes over. _FORCE_FA: None -> auto;
# True/False forces (test hook). d_model > 512 entries are wired but not yet
# benchmarked, hence the conservative cap.
_FA_MAX_DMODEL = 512
_FORCE_FA = None
_FA_BLOCKS = {                # d_pow2 -> block_i, block_j, num_warps, num_stages
    512: (64, 64, 8, 2),      # serves every d_model <= 512 (padded lanes)
    1024: (32, 64, 8, 2),
    2048: (16, 64, 8, 2),
}


def fa_ok(d_model):
    return triton.next_power_of_2(d_model) <= _FA_MAX_DMODEL


def fa_backward_one(a, b, div_own, div_other, inv_temperature, grad_scale,
                    *, bidir):
    """One tower's gradient rows via the FlashAttention-shaped kernel.
    div_other is ignored when bidir is False (LiT: row softmax only)."""
    n_own, d_model = a.shape
    block_i, block_j, num_warps, num_stages = _FA_BLOCKS[
        max(512, triton.next_power_of_2(d_model))]
    dA = torch.empty(n_own, d_model, device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(n_own, block_i),)
    clip_fa_grad_kernel[grid](
        a, b, div_own, div_own if div_other is None else div_other, dA,
        inv_temperature, grad_scale, n_own, b.shape[0],
        BIDIR=bidir, BLOCK_SIZE_I=block_i, BLOCK_SIZE_J=block_j,
        D_POW2=triton.next_power_of_2(d_model), D_MODEL=d_model,
        INPUT_PRECISION=_INPUT_PRECISION, num_warps=num_warps,
        num_stages=num_stages)
    return dA


def _fa_backward(x, y, div_row, div_col, inv_temperature, grad_scale):
    dX = fa_backward_one(x, y, div_row, div_col, inv_temperature, grad_scale,
                         bidir=True)
    dY = fa_backward_one(y, x, div_col, div_row, inv_temperature, grad_scale,
                         bidir=True)
    return dX, dY


class MemoryEfficientCLIPLossNormed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, inv_temperature, inv_temperature_orig, tau_plus, floor):
        batch_size = x.shape[0]
        sum_exp_row = torch.zeros(batch_size, device=x.device, dtype=torch.float32)
        sum_exp_col = torch.zeros(batch_size, device=x.device, dtype=torch.float32)
        _launch_denom(x, y, sum_exp_row, sum_exp_col, inv_temperature)

        denom_row, denom_col = sum_exp_row, sum_exp_col
        div_row, div_col, seed = sum_exp_row, sum_exp_col, None
        if tau_plus is not None:
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
        use_fa = _FORCE_FA if _FORCE_FA is not None else fa_ok(x.shape[1])
        if use_fa:
            dX, dY = _fa_backward(x, y, div_row, div_col, ctx.inv_temperature,
                                  ctx.inv_temperature_orig / (2.0 * batch_size))
            if seed is None:   # the ring path bakes dX's seed into its buffer
                dX -= y.float() * (ctx.inv_temperature_orig / batch_size)
            else:
                dX += y.float() * (seed * (ctx.inv_temperature_orig
                                           / (2.0 * batch_size)))[:, None]
        else:
            # offset 0 / n = batch: the single-GPU case is the one-shard
            # distributed case.
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
    actually a positive). tau_plus = 0 is the standard loss. forward also accepts a
    per-call tau_plus override -- a float or a (batch,) tensor of per-row priors in
    [0, 1) (e.g. when duplicate rates are known per sample); sample i's prior is
    applied to both its row and its column softmax.

    label_smoothing > 0 smooths the targets of both softmaxes with the
    F.cross_entropy convention: (1 - eps) on the diagonal plus eps/batch uniform.
    Costs only O(batch * dim) eager math; composes with stable and tau_plus.
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

    def forward(self, image_features, text_features, tau_plus=None):
        x = image_features if self.normalized_inputs else F.normalize(image_features, dim=1)
        y = text_features if self.normalized_inputs else F.normalize(text_features, dim=1)
        x, y = x.contiguous(), y.contiguous()
        _validate_features(x, y)

        batch_size = x.shape[0]
        tau_plus = _resolve_tau_plus(self.tau_plus, tau_plus, batch_size, x.device)
        inv_temperature = _LOG2E / self.temperature
        inv_temperature_orig = (math.sqrt(batch_size / self.temperature)
                                if self.stable else 1.0 / self.temperature)
        loss = MemoryEfficientCLIPLossNormed.apply(
            x, y, inv_temperature, inv_temperature_orig,
            tau_plus, math.exp(-2.0 / self.temperature))
        if self.label_smoothing:
            grad_factor = self.temperature * inv_temperature_orig if self.stable else 1.0
            loss = loss + _clip_smoothing_term(x, y, self.label_smoothing, batch_size,
                                               self.temperature, grad_factor)
        return loss


class StableMemoryEfficientCLIPLoss(MemoryEfficientCLIPLoss):
    """Deprecated alias for MemoryEfficientCLIPLoss(stable=True)."""
    def __init__(self, temperature=0.07, normalized_inputs=False):
        super().__init__(temperature=temperature, normalized_inputs=normalized_inputs,
                         stable=True)
