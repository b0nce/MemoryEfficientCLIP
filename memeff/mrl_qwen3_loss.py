"""Fused matryoshka (MRL, arXiv 2205.13147) Qwen3 losses: every nested prefix
dim trained in (almost) one pass over the similarity blocks.

Raw prefix dots are cumulative across the kernels' feature chunks, and each
prefix's re-normalization is a per-row scalar (a_i^k = 1/||x_i[:m_k]||), so the
denom kernel rescales its running dot tile at every prefix boundary and
accumulates all K denominators in a single sweep. The backward walks the chunks
twice with a telescoping coefficient tile (derivation and cost accounting in
docs/mrl_fused_backward.md): the extra cost over the non-MRL kernels is
m_{K-1}/D of one matmul pass, and the extra state is O(K * batch) scalar tables
(inverse prefix norms, per-dim positives/divisors, renormalization row sums) --
no per-dim feature copies, no per-dim gradient buffers.

Prefix dims must be strictly increasing multiples of the kernels' feature chunk
(64 for d_model >= 64) and end exactly at d_model; anything else raises, by
design -- unaligned dims would force intra-chunk masking on every load. The
dims reach the kernels as a small runtime int32 tensor: only the dim COUNT is a
constexpr (tl.static_range over K instantiates the boundary blocks), while each
segment's chunk loop keeps runtime bounds and stays rolled. Fully unrolling the
chunk walk instead multiplies every tl.dot into the IR and blows LLVM's
scheduling/regalloc up to minutes per variant (measured: ~2 min/kernel at
D=256); the rolled form compiles like the non-MRL kernels and shares one binary
across all same-K dim ladders.

Single-GPU modules only for now; for DDP wrap the distributed Qwen3 losses in
memeff.matryoshka.MatryoshkaLoss (K ring passes instead of one).
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
    debias_denominators as _debias_denominators,
    qwen3_num_negatives as _num_negatives,
    qwen3_smoothing_term as _qwen3_smoothing_term,
    resolve_tau_plus as _resolve_tau_plus,
    validate_features as _validate_features,
    validate_hard_negatives as _validate_hard_negatives,
    validate_label_smoothing as _validate_label_smoothing,
    validate_tau_plus as _validate_tau_plus,
)


def check_mrl_dims(dims, d_model):
    """The kernels walk whole feature chunks and snapshot at chunk boundaries,
    so every prefix dim must land on a chunk edge. Fails loudly instead of
    silently mis-slicing."""
    block_d = _check_dims(d_model)
    dims = tuple(int(m) for m in dims)
    if not dims or list(dims) != sorted(set(dims)):
        raise ValueError(f"mrl dims must be strictly increasing, got {dims}")
    if dims[-1] != d_model:
        raise ValueError(f"the last mrl dim must equal the feature dimension "
                         f"{d_model}, got {dims[-1]} (drop the dim or pad the "
                         f"model, the fused kernels do not support suffixes)")
    misaligned = [m for m in dims if m % block_d]
    if misaligned:
        raise ValueError(f"mrl dims must be multiples of the kernel feature "
                         f"chunk ({block_d} for d_model={d_model}), got "
                         f"{misaligned}; use memeff.MatryoshkaLoss for "
                         f"unaligned dims")
    return dims


def _dims_tensor(dims, device):
    """The prefix dims as a runtime i32 tensor: keeps the kernel binary shared
    across dim ladders of the same length."""
    return torch.tensor(dims, device=device, dtype=torch.int32)


def _prefix_inv_norms(t, dims):
    """(K, ...) fp32 inverse prefix norms 1 / max(||t[..., :m_k]||, 1e-12) --
    the same clamp F.normalize uses. Detached: the norms' gradient is carried
    by the eager (I - p p^T) corrections, not autograd."""
    idx = torch.tensor([m - 1 for m in dims], device=t.device)
    n = t.detach().float().square().cumsum(-1).index_select(-1, idx).sqrt()
    return n.clamp_min(1e-12).reciprocal().movedim(-1, 0).contiguous()


def _prefix_pos(q, d, a_inv, b_inv, dims):
    """(K, batch) re-normalized prefix positive similarities."""
    idx = torch.tensor([m - 1 for m in dims], device=q.device)
    raw = (q.detach().float() * d.detach().float()).cumsum(-1).index_select(-1, idx)
    return (raw.movedim(-1, 0) * a_inv * b_inv).contiguous()


def _mrl_hard_negative_exp(q, h, a_inv, h_inv, pos, inv_temperature, margin, dims):
    """Masked exp2 of every dim's re-normalized (batch, Kh) hard-negative
    similarities, stacked (K, batch, Kh). Eager: O(batch * Kh * sum(dims))."""
    qf, hf = q.float(), h.float()
    out = []
    for k, m in enumerate(dims):
        p = qf[:, :m] * a_inv[k][:, None]
        ph = hf[:, :, :m] * h_inv[k][:, :, None]
        s = torch.einsum('bkd,bd->bk', ph, p)
        keep = s <= (pos[k] + margin)[:, None]
        out.append(torch.exp2(s * inv_temperature - inv_temperature) * keep)
    return torch.stack(out)


@triton.jit
def mrl_qwen3_denom_kernel(
    A_ptr, B_ptr, dims_ptr, a_inv_ptr, b_inv_ptr, pos_ptr, sum_exp_ptr,
    n_i, n_j, stride_row, stride_col, inv_temperature, margin,
    row_offset, col_offset,
    EXCLUDE_DIAG: tl.constexpr, NUM_DIMS: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr, D_MODEL: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """Row sums of one masked block of exp2(S_k) for every prefix dim k, added
    atomically into the (K, n_rows) table. The raw dot tile accumulates across
    feature chunks; at each of the NUM_DIMS boundaries (runtime bounds from
    dims_ptr, rolled inner loops) it is rescaled by that dim's per-row inverse
    norms."""
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
        b_k = tl.load(b_inv_ptr + k * stride_col + col_offset + j_offsets,
                      mask=j_mask, other=1.0)
        S = R * a_k[:, None] * b_k[None, :]
        pos_k = tl.load(pos_ptr + k * stride_row + i_offsets, mask=i_mask, other=0.0)
        keep = (S <= pos_k[:, None] + margin) & i_mask[:, None] & j_mask[None, :]
        if EXCLUDE_DIAG:
            keep = keep & ((i_offsets[:, None] + row_offset) != (j_offsets[None, :] + col_offset))
        exp_S = tl.where(keep, tl.exp2(S * inv_temperature - inv_temperature), 0.0)
        tl.atomic_add(sum_exp_ptr + k * stride_row + i_offsets, tl.sum(exp_S, axis=1),
                      mask=i_mask, sem="relaxed")
        lo = hi


@triton.jit
def mrl_qwen3_grad_kernel(
    A_ptr, B_ptr, dims_ptr, a_inv_ptr, b_inv_ptr, pos_ptr, div_ptr, w_ptr,
    dA_ptr, dB_ptr, rho_a_ptr, rho_b_ptr,
    n_i, n_j, stride_row, stride_col, inv_temperature, margin, grad_scale,
    row_offset, col_offset,
    EXCLUDE_DIAG: tl.constexpr, TWO_SIDED: tl.constexpr, NUM_DIMS: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr, D_MODEL: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """Two chunk walks with a telescoping coefficient tile (see
    docs/mrl_fused_backward.md).

    Walk 1 re-accumulates the prefix dots exactly as the denom kernel did and
    folds every boundary's softmax weights into per-pair scalar coefficients
    c^k = w_k * grad_scale * P^k * a^k * b^k, summed into C. The per-dim
    row/column sums of c^k * R^k -- the re-normalization corrections, applied
    eagerly outside -- are accumulated on the way. Walk 2 emits
    dA[chunk] += C . B[chunk] (and the transposed dB iff TWO_SIDED), then peels
    c^k off C at each boundary; the re-accumulated dots are bit-identical to
    walk 1's, so C telescopes through the exact suffix sums. The last
    boundary's coefficients are never peeled, so walk 2 stops re-accumulating
    once it enters the final segment."""
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
        b_k = tl.load(b_inv_ptr + k * stride_col + col_offset + j_offsets,
                      mask=j_mask, other=1.0)
        S = R * a_k[:, None] * b_k[None, :]
        pos_k = tl.load(pos_ptr + k * stride_row + i_offsets, mask=i_mask, other=0.0)
        keep = (S <= pos_k[:, None] + margin) & i_mask[:, None] & j_mask[None, :]
        if EXCLUDE_DIAG:
            keep = keep & ((i_offsets[:, None] + row_offset) != (j_offsets[None, :] + col_offset))
        exp_S = tl.where(keep, tl.exp2(S * inv_temperature - inv_temperature), 0.0)
        div_k = tl.load(div_ptr + k * stride_row + i_offsets, mask=i_mask, other=1.0)
        w_k = tl.load(w_ptr + k)
        c = tl.math.fdiv(exp_S, div_k[:, None]) * ((w_k * grad_scale)
                                                   * a_k[:, None] * b_k[None, :])
        C += c
        cR = c * R
        tl.atomic_add(rho_a_ptr + k * stride_row + i_offsets, tl.sum(cR, axis=1),
                      mask=i_mask, sem="relaxed")
        if TWO_SIDED:
            tl.atomic_add(rho_b_ptr + k * stride_col + col_offset + j_offsets,
                          tl.sum(cR, axis=0), mask=j_mask, sem="relaxed")
        lo = hi

    R = tl.zeros([BLOCK_SIZE_I, BLOCK_SIZE_J], dtype=tl.float32)
    lo = 0
    for k in tl.static_range(NUM_DIMS):
        hi = tl.load(dims_ptr + k)
        LAST_SEG = k == NUM_DIMS - 1
        for d_start in range(lo, hi, BLOCK_SIZE_D):
            d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
            # All walk-2 dots run on fp32 operands: mixing bf16 and fp32 tl.dot
            # in one runtime loop body trips a triton 3.1 pipeliner miscompile
            # (first segment's emission and the re-accumulated R corrupt; see
            # the num_stages note in launch_mrl_grad). The fp32 R here differs
            # from walk 1's bf16-dot R by ~1e-7 relative, an accepted peel
            # mismatch. Single-condition constexpr ifs only: mixed and/or
            # conditions compile to runtime branches whose assignments do not
            # escape.
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
            b_k = tl.load(b_inv_ptr + k * stride_col + col_offset + j_offsets,
                          mask=j_mask, other=1.0)
            S = R * a_k[:, None] * b_k[None, :]
            pos_k = tl.load(pos_ptr + k * stride_row + i_offsets, mask=i_mask, other=0.0)
            keep = (S <= pos_k[:, None] + margin) & i_mask[:, None] & j_mask[None, :]
            if EXCLUDE_DIAG:
                keep = keep & ((i_offsets[:, None] + row_offset) != (j_offsets[None, :] + col_offset))
            exp_S = tl.where(keep, tl.exp2(S * inv_temperature - inv_temperature), 0.0)
            div_k = tl.load(div_ptr + k * stride_row + i_offsets, mask=i_mask, other=1.0)
            w_k = tl.load(w_ptr + k)
            C -= tl.math.fdiv(exp_S, div_k[:, None]) * ((w_k * grad_scale)
                                                        * a_k[:, None] * b_k[None, :])
        lo = hi


def launch_mrl_denom(a, b, dims_t, a_inv, b_inv, pos, sum_exp, inv_temperature,
                     margin, exclude_diag=False, row_offset=0, col_offset=0):
    n_i, d_model = a.shape
    n_j = b.shape[0]
    grid = (triton.cdiv(n_i, 128), triton.cdiv(n_j, 128))
    mrl_qwen3_denom_kernel[grid](
        a, b, dims_t, a_inv, b_inv, pos, sum_exp, n_i, n_j,
        a_inv.stride(0), b_inv.stride(0), inv_temperature, margin,
        row_offset, col_offset,
        EXCLUDE_DIAG=exclude_diag, NUM_DIMS=dims_t.numel(),
        BLOCK_SIZE_I=128, BLOCK_SIZE_J=128, BLOCK_SIZE_D=_check_dims(d_model),
        D_MODEL=d_model, INPUT_PRECISION=_INPUT_PRECISION,
    )


def launch_mrl_grad(a, b, dims_t, a_inv, b_inv, pos, div, w, dA, dB, rho_a, rho_b,
                    inv_temperature, margin, grad_scale,
                    exclude_diag=False, row_offset=0, col_offset=0, *, two_sided):
    n_i, d_model = a.shape
    n_j = b.shape[0]
    block_i, block_j, num_warps, num_stages = _backward_blocks(a.device)
    grid = (triton.cdiv(n_i, block_i), triton.cdiv(n_j, block_j))
    # Pipelining is safe only because walk 2 casts every dot operand to fp32:
    # with mixed bf16/fp32 dots in its loop body, triton 3.1's software
    # pipeliner corrupts the first segment's emission and the re-accumulated R
    # at any block size (A100 sm80, 2026-07-21; num_stages=1 also fixes it,
    # at ~2x backward cost).
    mrl_qwen3_grad_kernel[grid](
        a, b, dims_t, a_inv, b_inv, pos, div, w, dA, dB, rho_a, rho_b,
        n_i, n_j, a_inv.stride(0), b_inv.stride(0),
        inv_temperature, margin, grad_scale, row_offset, col_offset,
        EXCLUDE_DIAG=exclude_diag, TWO_SIDED=two_sided, NUM_DIMS=dims_t.numel(),
        BLOCK_SIZE_I=block_i, BLOCK_SIZE_J=block_j,
        BLOCK_SIZE_D=_check_dims(d_model), D_MODEL=d_model,
        INPUT_PRECISION=_INPUT_PRECISION,
        num_warps=num_warps, num_stages=num_stages,
    )


class MemoryEfficientMRLQwen3LossNormed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, d, h, a_inv, b_inv, h_inv, pos, weights, dims,
                inv_temperature, inv_temperature_orig, margin, use_qq, use_dd,
                tau_plus, floor, lit):
        batch_size, d_model = q.shape
        device = q.device
        w_vec = torch.tensor(weights, device=device, dtype=torch.float32)
        dims_t = _dims_tensor(dims, device)

        sum_exp = torch.zeros(len(dims), batch_size, device=device, dtype=torch.float32)
        launch_mrl_denom(q, d, dims_t, a_inv, b_inv, pos, sum_exp,
                         inv_temperature, margin)
        if use_qq:
            launch_mrl_denom(q, q, dims_t, a_inv, a_inv, pos, sum_exp,
                             inv_temperature, margin, exclude_diag=True)
        if use_dd:
            launch_mrl_denom(d, d, dims_t, b_inv, b_inv, pos, sum_exp,
                             inv_temperature, margin, exclude_diag=True)
        if h is not None:
            sum_exp += _mrl_hard_negative_exp(
                q, h, a_inv, h_inv, pos, inv_temperature, margin, dims).sum(-1)

        # every per-dim quantity is a (K, batch) row of the same eager transform
        sv = pos * inv_temperature - inv_temperature
        denom, div, seed = sum_exp, sum_exp, None
        if tau_plus is not None:
            denom, div, seed = _debias_denominators(
                torch.exp2(sv), sum_exp,
                _num_negatives(batch_size, h, use_qq, use_dd), tau_plus, floor)

        saved = ((q, d, a_inv, b_inv, pos, div)
                 + (() if seed is None else (seed,))
                 + (() if h is None else (h, h_inv)))
        ctx.save_for_backward(*saved)
        ctx.w_vec, ctx.dims_t = w_vec, dims_t
        ctx.debiased = seed is not None
        ctx.has_h = h is not None
        ctx.dims, ctx.weights = dims, weights
        ctx.inv_temperature = inv_temperature
        ctx.inv_temperature_orig = inv_temperature_orig
        ctx.margin = margin
        ctx.use_qq, ctx.use_dd, ctx.lit = use_qq, use_dd, lit
        ctx.batch_size = batch_size
        ctx.in_dtype = q.dtype
        return -((sv * _LN2 - torch.log(denom)).mean(dim=1) * w_vec).sum()

    @staticmethod
    def backward(ctx, grad_output):
        tensors = list(ctx.saved_tensors)
        q, d, a_inv, b_inv, pos, div = tensors[:6]
        idx = 6
        seed = None
        if ctx.debiased:
            seed, idx = tensors[idx], idx + 1
        h = h_inv = None
        if ctx.has_h:
            h, h_inv = tensors[idx], tensors[idx + 1]
        dims, weights = ctx.dims, ctx.weights
        w_vec, dims_t = ctx.w_vec, ctx.dims_t
        K, lit = len(dims), ctx.lit
        batch_size, d_model = ctx.batch_size, q.shape[1]
        inv_temperature, margin = ctx.inv_temperature, ctx.margin
        grad_scale = ctx.inv_temperature_orig / batch_size
        device = q.device

        dQ = torch.zeros(batch_size, d_model, device=device, dtype=torch.float32)
        dD = None if lit else torch.zeros_like(dQ)
        rho_q = torch.zeros(K, batch_size, device=device, dtype=torch.float32)
        rho_d = None if lit else torch.zeros_like(rho_q)

        launch_mrl_grad(q, d, dims_t, a_inv, b_inv, pos, div, w_vec,
                        dQ, dQ if lit else dD, rho_q, rho_q if lit else rho_d,
                        inv_temperature, margin, grad_scale,
                        two_sided=not lit)
        if ctx.use_qq:
            launch_mrl_grad(q, q, dims_t, a_inv, a_inv, pos, div, w_vec,
                            dQ, dQ, rho_q, rho_q,
                            inv_temperature, margin, grad_scale,
                            exclude_diag=True, two_sided=True)
        if ctx.use_dd:
            launch_mrl_grad(d, d, dims_t, b_inv, b_inv, pos, div, w_vec,
                            dD, dD, rho_d, rho_d,
                            inv_temperature, margin, grad_scale,
                            exclude_diag=True, two_sided=True)

        # Eager per-dim terms, all O(batch * sum(dims)): the positive-pair
        # seeds and hard-negative pulls go through the prefix-renormalization
        # Jacobian a^k (I - p p^T) directly; the kernels' emissions already
        # carry a^k inside their coefficients and only need the rank-one
        # correction -a^k * rho^k * p^k with the row sums accumulated above.
        qf, df = q.float(), d.float()
        hf = h.float() if h is not None else None
        if seed is None:
            cvec = (w_vec[:, None] * (-grad_scale)).expand(K, batch_size)
        else:
            cvec = seed * (w_vec[:, None] * grad_scale)
        e_h = (_mrl_hard_negative_exp(q, h, a_inv, h_inv, pos, inv_temperature,
                                      margin, dims)
               if h is not None else None)
        dH = None
        if h is not None and not lit:
            dH = torch.zeros(batch_size, h.shape[1], d_model, device=device,
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
        return (dQ, dD, dH, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None)


class MemoryEfficientMatryoshkaQwen3Loss(nn.Module):
    """Fused matryoshka Qwen3 loss, CLIP-style: both towers receive gradients
    at every prefix dim.

    forward(query_features, doc_features, hard_negative_features=None) as in
    MemoryEfficientQwen3Loss; per-dim InfoNCE losses on the re-normalized
    prefixes are summed with `weights` (default: 1 each). All of margin /
    stable / tau_plus (float, per-row tensor, or per-call override) /
    label_smoothing compose per dim exactly as in the non-MRL loss; the
    false-negative mask and the debiased estimator use each dim's own
    re-normalized similarities.

    dims must be strictly increasing multiples of the kernels' feature chunk
    (64 for d_model >= 64) ending exactly at d_model -- checked at forward, and
    intentionally strict; use memeff.MatryoshkaLoss for anything else.
    """
    def __init__(self, dims, weights=None, temperature=0.07, margin=0.1,
                 use_qq_negatives=False, use_dd_negatives=False,
                 normalized_inputs=False, stable=False, tau_plus=0.0,
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
        self.margin = margin
        self.use_qq_negatives = use_qq_negatives
        self.use_dd_negatives = use_dd_negatives
        self.normalized_inputs = normalized_inputs
        self.stable = stable
        self.tau_plus = tau_plus
        self.label_smoothing = label_smoothing
        self._lit = False

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

        batch_size, d_model = q.shape
        dims = check_mrl_dims(self.dims, d_model)
        tau_plus = _resolve_tau_plus(self.tau_plus, tau_plus, batch_size, q.device)
        inv_temperature = _LOG2E / self.temperature
        inv_temperature_orig = (math.sqrt(batch_size / self.temperature)
                                if self.stable else 1.0 / self.temperature)

        a_inv = _prefix_inv_norms(q, dims)
        b_inv = _prefix_inv_norms(d, dims)
        h_inv = _prefix_inv_norms(h, dims) if h is not None else None
        pos = _prefix_pos(q, d, a_inv, b_inv, dims)

        use_dd = (not self._lit) and self.use_dd_negatives
        loss = MemoryEfficientMRLQwen3LossNormed.apply(
            q, d, h, a_inv, b_inv, h_inv, pos, self.weights, dims,
            inv_temperature, inv_temperature_orig, self.margin,
            self.use_qq_negatives, use_dd, tau_plus,
            math.exp(-2.0 / self.temperature), self._lit)
        if self.label_smoothing:
            grad_factor = self.temperature * inv_temperature_orig if self.stable else 1.0
            n_neg = _num_negatives(batch_size, h, self.use_qq_negatives, use_dd)
            d_s = d.detach() if self._lit else d
            h_s = h.detach() if (h is not None and self._lit) else h
            for m, w in zip(dims, self.weights):
                loss = loss + w * _qwen3_smoothing_term(
                    F.normalize(q[:, :m], dim=-1),
                    F.normalize(d_s[:, :m], dim=-1),
                    (F.normalize(h_s[:, :, :m], dim=-1) if h is not None else None),
                    self.label_smoothing, n_neg, batch_size, self.temperature,
                    grad_factor, self.use_qq_negatives, use_dd)
        return loss


class MemoryEfficientMatryoshkaLiTQwen3Loss(MemoryEfficientMatryoshkaQwen3Loss):
    """Fused matryoshka Qwen3 loss, LiT-style: documents and hard negatives are
    locked and receive no gradient at any prefix dim. No d-d negatives (locked
    documents repel nothing). Same dims/weights/feature semantics as
    MemoryEfficientMatryoshkaQwen3Loss.
    """
    def __init__(self, dims, weights=None, temperature=0.07, margin=0.1,
                 use_qq_negatives=False, normalized_inputs=False, stable=False,
                 tau_plus=0.0, label_smoothing=0.0):
        super().__init__(dims, weights=weights, temperature=temperature,
                         margin=margin, use_qq_negatives=use_qq_negatives,
                         use_dd_negatives=False,
                         normalized_inputs=normalized_inputs, stable=stable,
                         tau_plus=tau_plus, label_smoothing=label_smoothing)
        self._lit = True
