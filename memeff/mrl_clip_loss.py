"""Fused matryoshka (MRL, arXiv 2205.13147) CLIP and LiT losses.

Same design as mrl_qwen3_loss (cumulative prefix dots, per-row scalar
re-normalization, all K denominators snapshotted in one sweep, backward picks
between prefix-emission and telescoping kernels) with the plain CLIP/LiT
semantics: no false-negative mask, no hard negatives, and -- the structural
difference -- the CLIP loss is bidirectional, so the kernels also accumulate
per-dim COLUMN denominators and the gradient coefficients sum the row and
column softmax probabilities. LiT is the row-only, text-tower-only special
case (TWO_SIDED=False prunes every column-side instruction).

Single-GPU modules only; for DDP wrap the distributed CLIP/LiT losses in
memeff.matryoshka.MatryoshkaLoss.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from ._common import (
    INPUT_PRECISION as _INPUT_PRECISION,
    LN2 as _LN2,
    LOG2E as _LOG2E,
    backward_blocks as _backward_blocks,
    check_dims as _check_dims,
    clip_smoothing_term as _clip_smoothing_term,
    debias_denominators as _debias_denominators,
    resolve_tau_plus as _resolve_tau_plus,
    validate_features as _validate_features,
    validate_label_smoothing as _validate_label_smoothing,
    validate_tau_plus as _validate_tau_plus,
)
from .mrl_qwen3_loss import (
    check_mrl_dims,
    _dims_tensor,
    _prefix_inv_norms,
    _prefix_pos,
    _use_prefix_emission,
)


@triton.jit
def mrl_clip_denom_kernel(
    A_ptr, B_ptr, dims_ptr, a_inv_ptr, b_inv_ptr, row_ptr, col_ptr,
    n_i, n_j, stride_row, stride_col, inv_temperature,
    TWO_SIDED: tl.constexpr, NUM_DIMS: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr, D_MODEL: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """Row (and, iff TWO_SIDED, column) sums of one block of exp2(S_k) for
    every prefix dim k, added atomically into the (K, batch) tables. The raw
    dot tile accumulates across feature chunks and is rescaled by each dim's
    per-row inverse norms at the boundaries."""
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    i_offsets = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    j_offsets = pid_j * BLOCK_SIZE_J + tl.arange(0, BLOCK_SIZE_J)
    i_mask = i_offsets < n_i
    j_mask = j_offsets < n_j

    R = tl.zeros([BLOCK_SIZE_I, BLOCK_SIZE_J], dtype=tl.float32)
    lo = 0
    for k in tl.static_range(NUM_DIMS):
        hi = tl.load(dims_ptr + k)
        for d_start in range(lo, hi, BLOCK_SIZE_D):
            d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
            A_block = tl.load(A_ptr + (i_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                              mask=i_mask[:, None], other=0.0)
            B_block = tl.load(B_ptr + (j_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                              mask=j_mask[:, None], other=0.0)
            R = tl.dot(A_block, tl.trans(B_block), R, input_precision=INPUT_PRECISION)
        a_k = tl.load(a_inv_ptr + k * stride_row + i_offsets, mask=i_mask, other=1.0)
        b_k = tl.load(b_inv_ptr + k * stride_col + j_offsets, mask=j_mask, other=1.0)
        S = R * a_k[:, None] * b_k[None, :]
        exp_S = tl.exp2(S * inv_temperature - inv_temperature)
        exp_S = tl.where(i_mask[:, None] & j_mask[None, :], exp_S, 0.0)
        tl.atomic_add(row_ptr + k * stride_row + i_offsets, tl.sum(exp_S, axis=1),
                      mask=i_mask, sem="relaxed")
        if TWO_SIDED:
            tl.atomic_add(col_ptr + k * stride_col + j_offsets, tl.sum(exp_S, axis=0),
                          mask=j_mask, sem="relaxed")
        lo = hi


@triton.jit
def mrl_clip_grad_kernel(
    A_ptr, B_ptr, dims_ptr, a_inv_ptr, b_inv_ptr, div_row_ptr, div_col_ptr, w_ptr,
    dA_ptr, dB_ptr, rho_a_ptr, rho_b_ptr,
    n_i, n_j, stride_row, stride_col, inv_temperature, grad_scale,
    TWO_SIDED: tl.constexpr, NUM_DIMS: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr, D_MODEL: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """Telescoping two-walk backward (see mrl_qwen3_grad_kernel), CLIP
    coefficients: c^k = (P_row + P_col) * w_k * grad_scale * a^k * b^k (the
    column probability only iff TWO_SIDED). Walk-2 dots run on fp32 operands
    -- the triton 3.1 mixed-dtype pipeliner miscompile workaround."""
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    i_offsets = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    j_offsets = pid_j * BLOCK_SIZE_J + tl.arange(0, BLOCK_SIZE_J)
    i_mask = i_offsets < n_i
    j_mask = j_offsets < n_j

    R = tl.zeros([BLOCK_SIZE_I, BLOCK_SIZE_J], dtype=tl.float32)
    C = tl.zeros([BLOCK_SIZE_I, BLOCK_SIZE_J], dtype=tl.float32)
    lo = 0
    for k in tl.static_range(NUM_DIMS):
        hi = tl.load(dims_ptr + k)
        for d_start in range(lo, hi, BLOCK_SIZE_D):
            d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
            A_block = tl.load(A_ptr + (i_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                              mask=i_mask[:, None], other=0.0)
            B_block = tl.load(B_ptr + (j_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                              mask=j_mask[:, None], other=0.0)
            R = tl.dot(A_block, tl.trans(B_block), R, input_precision=INPUT_PRECISION)
        a_k = tl.load(a_inv_ptr + k * stride_row + i_offsets, mask=i_mask, other=1.0)
        b_k = tl.load(b_inv_ptr + k * stride_col + j_offsets, mask=j_mask, other=1.0)
        S = R * a_k[:, None] * b_k[None, :]
        exp_S = tl.exp2(S * inv_temperature - inv_temperature)
        exp_S = tl.where(i_mask[:, None] & j_mask[None, :], exp_S, 0.0)
        dr_k = tl.load(div_row_ptr + k * stride_row + i_offsets, mask=i_mask, other=1.0)
        w_k = tl.load(w_ptr + k)
        prob = tl.math.fdiv(exp_S, dr_k[:, None])
        if TWO_SIDED:
            dc_k = tl.load(div_col_ptr + k * stride_col + j_offsets, mask=j_mask, other=1.0)
            prob += tl.math.fdiv(exp_S, dc_k[None, :])
        c = prob * ((w_k * grad_scale) * a_k[:, None] * b_k[None, :])
        C += c
        cR = c * R
        tl.atomic_add(rho_a_ptr + k * stride_row + i_offsets, tl.sum(cR, axis=1),
                      mask=i_mask, sem="relaxed")
        if TWO_SIDED:
            tl.atomic_add(rho_b_ptr + k * stride_col + j_offsets,
                          tl.sum(cR, axis=0), mask=j_mask, sem="relaxed")
        lo = hi

    R = tl.zeros([BLOCK_SIZE_I, BLOCK_SIZE_J], dtype=tl.float32)
    lo = 0
    for k in tl.static_range(NUM_DIMS):
        hi = tl.load(dims_ptr + k)
        LAST_SEG = k == NUM_DIMS - 1
        for d_start in range(lo, hi, BLOCK_SIZE_D):
            d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
            B_block = tl.load(B_ptr + (j_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                              mask=j_mask[:, None], other=0.0,
                              eviction_policy="evict_first").to(tl.float32)
            dA_contrib = tl.dot(C, B_block, input_precision=INPUT_PRECISION)
            tl.atomic_add(dA_ptr + (i_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                          dA_contrib, mask=i_mask[:, None], sem="relaxed")
            if TWO_SIDED:
                A_block = tl.load(A_ptr + (i_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                                  mask=i_mask[:, None], other=0.0,
                                  eviction_policy="evict_first").to(tl.float32)
                dB_contrib = tl.dot(tl.trans(C), A_block,
                                    input_precision=INPUT_PRECISION)
                tl.atomic_add(dB_ptr + (j_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                              dB_contrib, mask=j_mask[:, None], sem="relaxed")
                if not LAST_SEG:
                    R = tl.dot(A_block, tl.trans(B_block), R, input_precision=INPUT_PRECISION)
            else:
                if not LAST_SEG:
                    A_block = tl.load(A_ptr + (i_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                                      mask=i_mask[:, None], other=0.0,
                                      eviction_policy="evict_first").to(tl.float32)
                    R = tl.dot(A_block, tl.trans(B_block), R, input_precision=INPUT_PRECISION)
        if not LAST_SEG:
            a_k = tl.load(a_inv_ptr + k * stride_row + i_offsets, mask=i_mask, other=1.0)
            b_k = tl.load(b_inv_ptr + k * stride_col + j_offsets, mask=j_mask, other=1.0)
            S = R * a_k[:, None] * b_k[None, :]
            exp_S = tl.exp2(S * inv_temperature - inv_temperature)
            exp_S = tl.where(i_mask[:, None] & j_mask[None, :], exp_S, 0.0)
            dr_k = tl.load(div_row_ptr + k * stride_row + i_offsets, mask=i_mask, other=1.0)
            w_k = tl.load(w_ptr + k)
            prob = tl.math.fdiv(exp_S, dr_k[:, None])
            if TWO_SIDED:
                dc_k = tl.load(div_col_ptr + k * stride_col + j_offsets, mask=j_mask, other=1.0)
                prob += tl.math.fdiv(exp_S, dc_k[None, :])
            C -= prob * ((w_k * grad_scale) * a_k[:, None] * b_k[None, :])
        lo = hi


@triton.jit
def mrl_clip_grad_prefix_kernel(
    A_ptr, B_ptr, dims_ptr, a_inv_ptr, b_inv_ptr, div_row_ptr, div_col_ptr, w_ptr,
    dA_ptr, dB_ptr, rho_a_ptr, rho_b_ptr,
    n_i, n_j, stride_row, stride_col, inv_temperature, grad_scale,
    TWO_SIDED: tl.constexpr, NUM_DIMS: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr, D_MODEL: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """Single-walk prefix-emission backward (see mrl_qwen3_grad_prefix_kernel)
    with the CLIP coefficients. Coefficients are cast to the input dtype
    before emission so every runtime loop keeps uniform-dtype tl.dot
    operands."""
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    i_offsets = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    j_offsets = pid_j * BLOCK_SIZE_J + tl.arange(0, BLOCK_SIZE_J)
    i_mask = i_offsets < n_i
    j_mask = j_offsets < n_j

    R = tl.zeros([BLOCK_SIZE_I, BLOCK_SIZE_J], dtype=tl.float32)
    lo = 0
    for k in tl.static_range(NUM_DIMS):
        hi = tl.load(dims_ptr + k)
        for d_start in range(lo, hi, BLOCK_SIZE_D):
            d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
            A_block = tl.load(A_ptr + (i_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                              mask=i_mask[:, None], other=0.0)
            B_block = tl.load(B_ptr + (j_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                              mask=j_mask[:, None], other=0.0)
            R = tl.dot(A_block, tl.trans(B_block), R, input_precision=INPUT_PRECISION)
        a_k = tl.load(a_inv_ptr + k * stride_row + i_offsets, mask=i_mask, other=1.0)
        b_k = tl.load(b_inv_ptr + k * stride_col + j_offsets, mask=j_mask, other=1.0)
        S = R * a_k[:, None] * b_k[None, :]
        exp_S = tl.exp2(S * inv_temperature - inv_temperature)
        exp_S = tl.where(i_mask[:, None] & j_mask[None, :], exp_S, 0.0)
        dr_k = tl.load(div_row_ptr + k * stride_row + i_offsets, mask=i_mask, other=1.0)
        w_k = tl.load(w_ptr + k)
        prob = tl.math.fdiv(exp_S, dr_k[:, None])
        if TWO_SIDED:
            dc_k = tl.load(div_col_ptr + k * stride_col + j_offsets, mask=j_mask, other=1.0)
            prob += tl.math.fdiv(exp_S, dc_k[None, :])
        c = prob * ((w_k * grad_scale) * a_k[:, None] * b_k[None, :])
        cR = c * R
        tl.atomic_add(rho_a_ptr + k * stride_row + i_offsets, tl.sum(cR, axis=1),
                      mask=i_mask, sem="relaxed")
        if TWO_SIDED:
            tl.atomic_add(rho_b_ptr + k * stride_col + j_offsets,
                          tl.sum(cR, axis=0), mask=j_mask, sem="relaxed")
        c_low = c.to(A_ptr.dtype.element_ty)
        for d_start in range(0, hi, BLOCK_SIZE_D):
            d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
            B_block = tl.load(B_ptr + (j_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                              mask=j_mask[:, None], other=0.0)
            dA_contrib = tl.dot(c_low, B_block, input_precision=INPUT_PRECISION)
            tl.atomic_add(dA_ptr + (i_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                          dA_contrib, mask=i_mask[:, None], sem="relaxed")
            if TWO_SIDED:
                A_block = tl.load(A_ptr + (i_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                                  mask=i_mask[:, None], other=0.0)
                dB_contrib = tl.dot(tl.trans(c_low), A_block,
                                    input_precision=INPUT_PRECISION)
                tl.atomic_add(dB_ptr + (j_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                              dB_contrib, mask=j_mask[:, None], sem="relaxed")
        lo = hi


def launch_mrl_clip_denom(a, b, dims_t, a_inv, b_inv, row, col, inv_temperature,
                          *, two_sided):
    n_i, d_model = a.shape
    n_j = b.shape[0]
    grid = (triton.cdiv(n_i, 128), triton.cdiv(n_j, 128))
    mrl_clip_denom_kernel[grid](
        a, b, dims_t, a_inv, b_inv, row, row if col is None else col,
        n_i, n_j, a_inv.stride(0), b_inv.stride(0), inv_temperature,
        TWO_SIDED=two_sided, NUM_DIMS=dims_t.numel(),
        BLOCK_SIZE_I=128, BLOCK_SIZE_J=128, BLOCK_SIZE_D=_check_dims(d_model),
        D_MODEL=d_model, INPUT_PRECISION=_INPUT_PRECISION,
    )


def launch_mrl_clip_grad(a, b, dims, dims_t, a_inv, b_inv, div_row, div_col, w,
                         dA, dB, rho_a, rho_b, inv_temperature, grad_scale,
                         *, two_sided):
    n_i, d_model = a.shape
    n_j = b.shape[0]
    block_i, block_j, num_warps, num_stages = _backward_blocks(a.device)
    grid = (triton.cdiv(n_i, block_i), triton.cdiv(n_j, block_j))
    block_d = _check_dims(d_model)
    kernel = (mrl_clip_grad_prefix_kernel
              if _use_prefix_emission(dims, d_model, two_sided, block_d)
              else mrl_clip_grad_kernel)
    kernel[grid](
        a, b, dims_t, a_inv, b_inv, div_row,
        div_row if div_col is None else div_col, w,
        dA, dA if dB is None else dB, rho_a, rho_a if rho_b is None else rho_b,
        n_i, n_j, a_inv.stride(0), b_inv.stride(0), inv_temperature, grad_scale,
        TWO_SIDED=two_sided, NUM_DIMS=dims_t.numel(),
        BLOCK_SIZE_I=block_i, BLOCK_SIZE_J=block_j,
        BLOCK_SIZE_D=block_d, D_MODEL=d_model,
        INPUT_PRECISION=_INPUT_PRECISION,
        num_warps=num_warps, num_stages=num_stages,
    )


class MemoryEfficientMRLCLIPLossNormed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, a_inv, b_inv, pos, weights, dims,
                inv_temperature, inv_temperature_orig, tau_plus, floor, lit):
        batch_size, d_model = x.shape
        device = x.device
        K = len(dims)
        w_vec = torch.tensor(weights, device=device, dtype=torch.float32)
        dims_t = _dims_tensor(dims, device)

        row = torch.zeros(K, batch_size, device=device, dtype=torch.float32)
        col = None if lit else torch.zeros_like(row)
        launch_mrl_clip_denom(x, y, dims_t, a_inv, b_inv, row, col,
                              inv_temperature, two_sided=not lit)

        sv = pos * inv_temperature - inv_temperature
        den_row, div_row, seed = row, row, None
        den_col, div_col = col, col
        if tau_plus is not None:
            pos_exp = torch.exp2(sv)
            den_row, div_row, seed = _debias_denominators(
                pos_exp, row, batch_size - 1, tau_plus, floor)
            if not lit:
                den_col, div_col, seed_col = _debias_denominators(
                    pos_exp, col, batch_size - 1, tau_plus, floor)
                seed = seed + seed_col

        saved = ((x, y, a_inv, b_inv, div_row)
                 + (() if lit else (div_col,))
                 + (() if seed is None else (seed,)))
        ctx.save_for_backward(*saved)
        ctx.w_vec, ctx.dims_t = w_vec, dims_t
        ctx.debiased = seed is not None
        ctx.dims, ctx.weights = dims, weights
        ctx.inv_temperature = inv_temperature
        ctx.inv_temperature_orig = inv_temperature_orig
        ctx.lit = lit
        ctx.batch_size = batch_size
        ctx.in_dtype = x.dtype
        if lit:
            return -(((sv * _LN2 - torch.log(den_row)).mean(dim=1)) * w_vec).sum()
        per_dim = (2.0 * sv * _LN2 - torch.log(den_row) - torch.log(den_col)).mean(dim=1)
        return -(per_dim * w_vec).sum() / 2.0

    @staticmethod
    def backward(ctx, grad_output):
        tensors = list(ctx.saved_tensors)
        x, y, a_inv, b_inv, div_row = tensors[:5]
        idx = 5
        div_col = None
        if not ctx.lit:
            div_col, idx = tensors[idx], idx + 1
        seed = tensors[idx] if ctx.debiased else None
        dims, weights = ctx.dims, ctx.weights
        w_vec, dims_t = ctx.w_vec, ctx.dims_t
        K, lit = len(dims), ctx.lit
        batch_size, d_model = ctx.batch_size, x.shape[1]
        inv_temperature = ctx.inv_temperature
        grad_scale = ctx.inv_temperature_orig / (batch_size if lit else 2.0 * batch_size)
        device = x.device

        dX = torch.zeros(batch_size, d_model, device=device, dtype=torch.float32)
        dY = None if lit else torch.zeros_like(dX)
        rho_x = torch.zeros(K, batch_size, device=device, dtype=torch.float32)
        rho_y = None if lit else torch.zeros_like(rho_x)

        launch_mrl_clip_grad(x, y, dims, dims_t, a_inv, b_inv, div_row, div_col,
                             w_vec, dX, dY, rho_x, rho_y,
                             inv_temperature, grad_scale, two_sided=not lit)

        # Eager positive-pair seeds through the prefix-renormalization Jacobian
        # a^k (I - p p^T); the kernels' emissions carry a^k * b^k inside their
        # coefficients and only need the rank-one -a^k * rho^k * p^k correction.
        xf, yf = x.float(), y.float()
        if seed is None:
            base = -grad_scale if lit else -2.0 * grad_scale
            cvec = (w_vec[:, None] * base).expand(K, batch_size)
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
            if not lit:
                vy = px * cvec[k][:, None]
                dY[:, :m] += b * (vy - (vy * py).sum(-1, keepdim=True) * py
                                  - rho_y[k][:, None] * py)

        dX = (dX * grad_output).to(ctx.in_dtype)
        dY = None if lit else (dY * grad_output).to(ctx.in_dtype)
        return (dX, dY, None, None, None, None, None, None, None, None,
                None, None)


class MemoryEfficientMatryoshkaCLIPLoss(nn.Module):
    """Fused matryoshka CLIP loss: the bidirectional softmax loss at every
    nested prefix dim, both towers trained.

    forward(image_features, text_features) as in MemoryEfficientCLIPLoss;
    per-dim symmetric InfoNCE losses on the re-normalized prefixes are summed
    with `weights` (default: 1 each). stable / tau_plus (float, per-row
    tensor, or per-call override) / label_smoothing compose per dim exactly
    as in the non-MRL loss; the debiased estimator uses each dim's own
    re-normalized similarities.

    dims must be strictly increasing multiples of the kernels' feature chunk
    (64 for d_model >= 64) ending exactly at d_model -- checked at forward,
    and intentionally strict; use memeff.MatryoshkaLoss for anything else.
    """
    _lit = False

    def __init__(self, dims, weights=None, temperature=0.07,
                 normalized_inputs=False, stable=True, tau_plus=0.0,
                 label_smoothing=0.0):
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

    def forward(self, image_features, text_features, tau_plus=None):
        x, y = image_features, text_features
        if not self.normalized_inputs:
            x = F.normalize(x, dim=-1)
            y = F.normalize(y, dim=-1)
        if self._lit:
            y = y.detach()
        x, y = x.contiguous(), y.contiguous()
        _validate_features(x, y)

        batch_size, d_model = x.shape
        dims = check_mrl_dims(self.dims, d_model)
        tau_plus = _resolve_tau_plus(self.tau_plus, tau_plus, batch_size, x.device)
        inv_temperature = _LOG2E / self.temperature
        inv_temperature_orig = (math.sqrt(batch_size / self.temperature)
                                if self.stable else 1.0 / self.temperature)

        a_inv = _prefix_inv_norms(x, dims)
        b_inv = _prefix_inv_norms(y, dims)
        pos = _prefix_pos(x, y, a_inv, b_inv, dims)

        loss = MemoryEfficientMRLCLIPLossNormed.apply(
            x, y, a_inv, b_inv, pos, self.weights, dims,
            inv_temperature, inv_temperature_orig, tau_plus,
            math.exp(-2.0 / self.temperature), self._lit)
        if self.label_smoothing:
            grad_factor = self.temperature * inv_temperature_orig if self.stable else 1.0
            for m, w in zip(dims, self.weights):
                loss = loss + w * _clip_smoothing_term(
                    F.normalize(x[:, :m], dim=-1), F.normalize(y[:, :m], dim=-1),
                    self.label_smoothing, batch_size, self.temperature, grad_factor)
        return loss


class MemoryEfficientMatryoshkaLiTLoss(MemoryEfficientMatryoshkaCLIPLoss):
    """Fused matryoshka LiT loss: the row-softmax loss at every nested prefix
    dim, only the text tower trained (the image features are detached).

    forward(text_features, image_features) as in MemoryEfficientLiTLoss.
    Same dims/weights/feature semantics as MemoryEfficientMatryoshkaCLIPLoss.
    """
    _lit = True

    def forward(self, text_features, image_features, tau_plus=None):
        return super().forward(text_features, image_features, tau_plus=tau_plus)
