"""Fused matryoshka (MRL, arXiv 2205.13147) Qwen3 losses: every nested prefix
dim trained in (almost) one pass over the similarity blocks.

Raw prefix dots are cumulative across the kernels' feature chunks, and each
prefix's re-normalization is a per-row scalar (a_i^k = 1/||x_i[:m_k]||), so the
denom kernel rescales its running dot tile at every prefix boundary and
accumulates all K denominators in a single sweep. For d_model within the FA
cap (see _common.fa_ok) the backward runs FlashAttention-shaped launches --
one per (output tower, similarity block), gradient and rho rows accumulated
in registers, zero atomics (mrl_fa_grad_kernel). Above the cap it picks
between two atomic tile kernels per ladder (see _use_prefix_emission): a
single walk that emits each boundary's prefix gradient immediately (extra
prefix matmuls on cache-hot chunks, every exp2/mask sweep run once -- wins at
small d_model, where the sweeps dominate), or two walks with a telescoping
coefficient tile (minimal matmuls, K - 1 extra peel sweeps -- wins at large
d_model). Either way the extra state is O(K * batch) scalar tables (inverse
prefix norms, per-dim positives/divisors, renormalization row sums) -- no
per-dim feature copies, no per-dim gradient buffers.

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

Single-GPU modules; the DDP counterparts (one ring pass for all dims, same
tile kernels launched rectangularly) live in distributed_mrl_qwen3_loss.py.
For unaligned dims wrap the plain losses in memeff.matryoshka.MatryoshkaLoss
(K passes instead of one).
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
    fa_blocks as _fa_blocks,
    fa_ok as _fa_ok,
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
    """Two chunk walks with a telescoping coefficient tile.

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


@triton.jit
def mrl_qwen3_grad_prefix_kernel(
    A_ptr, B_ptr, dims_ptr, a_inv_ptr, b_inv_ptr, pos_ptr, div_ptr, w_ptr,
    dA_ptr, dB_ptr, rho_a_ptr, rho_b_ptr,
    n_i, n_j, stride_row, stride_col, inv_temperature, margin, grad_scale,
    row_offset, col_offset,
    EXCLUDE_DIAG: tl.constexpr, TWO_SIDED: tl.constexpr, NUM_DIMS: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr, D_MODEL: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """Single chunk walk with per-boundary prefix emission.

    dX[:, :m_k] receives w_k's contribution the moment c^k exists:
    dA[:, :m_k] += c^k . B[:, :m_k] (and the transposed dB iff TWO_SIDED).
    Summed over boundaries this equals the telescoping kernel's suffix-sum
    emissions, but with no second walk, no peel re-computations and no
    re-accumulated dot tile -- each boundary's exp2/mask sweep runs exactly
    once. The price is prefix matmul redundancy (sum(dims) instead of D_MODEL
    chunk passes per side, re-reading early chunks that stay cache-hot), the
    right trade below the launcher's sweep-cost crossover. The coefficients are
    cast to the input dtype before emission, so every runtime loop body keeps
    uniform-dtype tl.dot operands (bf16 tensor-core emissions, and no triton
    3.1 mixed-dtype pipeliner miscompile; see launch_mrl_grad)."""
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
        div_k = tl.load(div_ptr + k * stride_row + i_offsets, mask=i_mask, other=1.0)
        w_k = tl.load(w_ptr + k)
        c = tl.math.fdiv(exp_S, div_k[:, None]) * ((w_k * grad_scale)
                                                   * a_k[:, None] * b_k[None, :])
        cR = c * R
        tl.atomic_add(rho_a_ptr + k * stride_row + i_offsets, tl.sum(cR, axis=1),
                      mask=i_mask, sem="relaxed")
        if TWO_SIDED:
            tl.atomic_add(rho_b_ptr + k * stride_col + col_offset + j_offsets,
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


@triton.jit
def mrl_fa_grad_kernel(
    A_ptr, B_ptr, a_inv_ptr, b_inv_ptr, pos_ptr, div_own_ptr, div_other_ptr,
    dA_ptr, rho_ptr,
    inv_temperature, margin, grad_scale, n_own, n_other,
    HAS_MASK: tl.constexpr, OWN_SIDE: tl.constexpr, OTHER_SIDE: tl.constexpr,
    EXCLUDE_DIAG: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr,
    D_POW2: tl.constexpr, D_PREFIX: tl.constexpr, D_STRIDE: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """FlashAttention-shaped MRL gradient, ONE prefix boundary per launch: the
    program owns rows [pid * BLOCK_I, ...) of tower A, streams every B block
    restricted to this boundary's prefix (D_PREFIX = m_k features at row
    stride D_STRIDE = d_model, dots sized D_POW2 = next_pow2(m_k)) and
    accumulates dA[:, :m_k] += sum_j c_ij * B_j[:m_k] in registers -- no
    gradient atomics, one read-modify-write per owned row, and the same RMW
    for this boundary's rho row (the c . R row sums the eager
    re-normalization correction needs). A register-masked full-width variant
    that fused all K boundaries into one launch ran 23x SLOWER than the tile
    kernels (A100, 2026-07-22): the prefix-masked B tile became
    register-resident and every boundary's two dots round-tripped it through
    SMEM, so keep B feeding the dots straight from its load and pay the
    K-fold restream instead -- boundary launches also size their dots by
    next_pow2(m_k) rather than the full width, and per-boundary w_k *
    grad_scale folds into the scalar.

    One kernel serves both loss families: HAS_MASK enables the Qwen3
    false-negative keep mask (pos/margin), OWN_SIDE / OTHER_SIDE add the
    softmax terms whose divisors (and pos rows, iff HAS_MASK) are indexed by
    the owned / streamed rows -- Qwen3 passes this boundary's div row twice,
    CLIP its row/col rows per launch orientation; the self blocks (q-q / d-d)
    enable both sides plus EXCLUDE_DIAG. All 0/1 masks multiply as floats
    (tl.where trips the triton 3.1 select-layout bug) and 1D loads use
    clamped indices; out-of-range streamed rows load zero features, so their
    finite prob dies in both the emission dot and the c * R rho product."""
    pid = tl.program_id(0)
    i_offsets = pid * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    d_offsets = tl.arange(0, D_POW2)
    i_mask = i_offsets < n_own
    d_mask = d_offsets < D_PREFIX
    i_clamped = tl.minimum(i_offsets, n_own - 1)
    out_offsets = i_offsets[:, None] * D_STRIDE + d_offsets[None, :]
    out_mask = i_mask[:, None] & d_mask[None, :]

    A_own = tl.load(A_ptr + out_offsets, mask=out_mask, other=0.0)
    a_own = tl.load(a_inv_ptr + i_clamped)
    if OWN_SIDE:
        div_own = tl.load(div_own_ptr + i_clamped)
        if HAS_MASK:
            pos_own = tl.load(pos_ptr + i_clamped)
    acc = tl.zeros([BLOCK_SIZE_I, D_POW2], dtype=tl.float32)
    rho = tl.zeros([BLOCK_SIZE_I], dtype=tl.float32)

    for j_start in range(0, n_other, BLOCK_SIZE_J):
        j_offsets = j_start + tl.arange(0, BLOCK_SIZE_J)
        j_clamped = tl.minimum(j_offsets, n_other - 1)
        B_block = tl.load(B_ptr + (j_offsets[:, None] * D_STRIDE + d_offsets[None, :]),
                          mask=((j_offsets[:, None] < n_other) & d_mask[None, :]),
                          other=0.0)
        R = tl.dot(A_own, tl.trans(B_block), input_precision=INPUT_PRECISION)
        b_str = tl.load(b_inv_ptr + j_clamped)
        S = R * a_own[:, None] * b_str[None, :]
        exp_S = tl.exp2(S * inv_temperature - inv_temperature)
        prob = tl.zeros([BLOCK_SIZE_I, BLOCK_SIZE_J], dtype=tl.float32)
        if OWN_SIDE:
            term = tl.math.fdiv(exp_S, div_own[:, None])
            if HAS_MASK:
                term *= (S <= pos_own[:, None] + margin).to(tl.float32)
            prob += term
        if OTHER_SIDE:
            div_str = tl.load(div_other_ptr + j_clamped)
            term = tl.math.fdiv(exp_S, div_str[None, :])
            if HAS_MASK:
                pos_str = tl.load(pos_ptr + j_clamped)
                term *= (S <= pos_str[None, :] + margin).to(tl.float32)
            prob += term
        if EXCLUDE_DIAG:
            prob *= (i_offsets[:, None] != j_offsets[None, :]).to(tl.float32)
        c = prob * (grad_scale * a_own[:, None] * b_str[None, :])
        rho += tl.sum(c * R, axis=1)
        acc = tl.dot(c.to(A_ptr.dtype.element_ty), B_block, acc,
                     input_precision=INPUT_PRECISION)

    acc += tl.load(dA_ptr + out_offsets, mask=out_mask, other=0.0)
    tl.store(dA_ptr + out_offsets, acc, mask=out_mask)
    rho += tl.load(rho_ptr + i_clamped)
    tl.store(rho_ptr + i_offsets, rho, mask=i_mask)


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


# One boundary sweep (exp2/mask/atomics over a B^2 tile) costs about as much
# as this many 64-wide matmul chunk passes (A100, bf16, D=384, 2026-07).
_SWEEP_CHUNK_COST = 5
_FORCE_BACKWARD = None  # 'prefix' | 'telescope' | 'fa' -- test hook

# Per-boundary block overrides, keyed (d_pow2, wide) where d_pow2 =
# next_pow2(m_k) (min 512) and wide = 4-byte inputs. bf16/fp16 boundaries
# fall back to the plain per-arch FA tables (the kernel has the same SMEM /
# register shape as the plain FA kernels); fp32 needs its own entries because
# its staged loads are twice the size (sm80 budget: stages * BJ * d_pow2 *
# 4B + the BI-row A tile in registers).
_MRL_FA_BLOCKS = {
    (512, True): (32, 32, 8, 2),      # 128 KB SMEM
    (1024, True): (16, 16, 8, 2),     # 128 KB
}


def _mrl_fa_blocks(device, m, dtype):
    d_pow2 = max(512, triton.next_power_of_2(m))
    wide = dtype.itemsize >= 4
    cfg = _MRL_FA_BLOCKS.get((d_pow2, wide))
    if cfg is None and not wide:
        cfg = _fa_blocks(device, m)
    return cfg


def use_fa_backward(d_model, device, dtype):
    if _FORCE_BACKWARD is not None:
        if _FORCE_BACKWARD != 'fa':
            return False
    elif not _fa_ok(d_model, device):
        return False
    return _mrl_fa_blocks(device, d_model, dtype) is not None


def launch_mrl_fa_grad(a, b, dims, weights, a_inv, b_inv, pos, div_own,
                       div_other, dA, rho, inv_temperature, margin, grad_scale,
                       *, own_side, other_side, has_mask, exclude_diag):
    """One tower's gradient rows (and rho rows) accumulated into the
    pre-seeded fp32 buffers: one plain-FA-shaped launch per prefix boundary,
    each sized next_pow2(m_k) with that boundary's rows of the (K, batch)
    tables and w_k folded into the scalar. pos is ignored when has_mask is
    False (pass any same-shape table)."""
    n_own, d_model = a.shape
    for k, m in enumerate(dims):
        block_i, block_j, num_warps, num_stages = _mrl_fa_blocks(a.device, m, a.dtype)
        grid = (triton.cdiv(n_own, block_i),)
        mrl_fa_grad_kernel[grid](
            a, b, a_inv[k], b_inv[k], pos[k], div_own[k], div_other[k],
            dA, rho[k],
            inv_temperature, margin, weights[k] * grad_scale, n_own, b.shape[0],
            HAS_MASK=has_mask, OWN_SIDE=own_side, OTHER_SIDE=other_side,
            EXCLUDE_DIAG=exclude_diag,
            BLOCK_SIZE_I=block_i, BLOCK_SIZE_J=block_j,
            D_POW2=triton.next_power_of_2(m), D_PREFIX=m, D_STRIDE=d_model,
            INPUT_PRECISION=_INPUT_PRECISION,
            num_warps=num_warps, num_stages=num_stages)


def _use_prefix_emission(dims, d_model, two_sided, block_d):
    """The single-walk prefix kernel beats the telescoping two-walk kernel
    when its extra prefix matmul chunks cost less than the K - 1 peel sweeps
    (plus the walk-2 re-accumulation) they remove -- i.e. at small
    d_model / dense ladders, where the sweeps dominate."""
    if _FORCE_BACKWARD is not None:
        return _FORCE_BACKWARD == 'prefix'
    if len(dims) == 1:
        return False  # identical work either way; keep the older path
    sides = 2 if two_sided else 1
    extra_chunks = (sides * (sum(dims) - d_model) - dims[-2]) // block_d
    return extra_chunks < _SWEEP_CHUNK_COST * (len(dims) - 1)


def launch_mrl_grad(a, b, dims, dims_t, a_inv, b_inv, pos, div, w, dA, dB,
                    rho_a, rho_b, inv_temperature, margin, grad_scale,
                    exclude_diag=False, row_offset=0, col_offset=0, *, two_sided):
    n_i, d_model = a.shape
    n_j = b.shape[0]
    block_i, block_j, num_warps, num_stages = _backward_blocks(a.device)
    grid = (triton.cdiv(n_i, block_i), triton.cdiv(n_j, block_j))
    block_d = _check_dims(d_model)
    # Pipelining is safe only with uniform-dtype tl.dot operands per loop body:
    # with mixed bf16/fp32 dots in one loop, triton 3.1's software pipeliner
    # corrupts the first segment's emission and the re-accumulated R at any
    # block size (A100 sm80, 2026-07-21; num_stages=1 also fixes it, at ~2x
    # backward cost). The telescoping kernel casts its walk-2 operands to
    # fp32; the prefix kernel casts its coefficients down to the input dtype.
    kernel = (mrl_qwen3_grad_prefix_kernel
              if _use_prefix_emission(dims, d_model, two_sided, block_d)
              else mrl_qwen3_grad_kernel)
    kernel[grid](
        a, b, dims_t, a_inv, b_inv, pos, div, w, dA, dB, rho_a, rho_b,
        n_i, n_j, a_inv.stride(0), b_inv.stride(0),
        inv_temperature, margin, grad_scale, row_offset, col_offset,
        EXCLUDE_DIAG=exclude_diag, TWO_SIDED=two_sided, NUM_DIMS=dims_t.numel(),
        BLOCK_SIZE_I=block_i, BLOCK_SIZE_J=block_j,
        BLOCK_SIZE_D=block_d, D_MODEL=d_model,
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

        if use_fa_backward(d_model, device, q.dtype):
            # One FA launch per (output tower, similarity block): q.d row side
            # feeds dQ, its transposed role feeds dD, the self blocks feed
            # their tower from both roles at once (rows and columns index the
            # same batch, so one pos/div table serves both).
            launch_mrl_fa_grad(q, d, dims, weights, a_inv, b_inv, pos, div,
                               div, dQ, rho_q, inv_temperature, margin,
                               grad_scale, own_side=True, other_side=False,
                               has_mask=True, exclude_diag=False)
            if ctx.use_qq:
                launch_mrl_fa_grad(q, q, dims, weights, a_inv, a_inv, pos, div,
                                   div, dQ, rho_q, inv_temperature, margin,
                                   grad_scale, own_side=True, other_side=True,
                                   has_mask=True, exclude_diag=True)
            if not lit:
                launch_mrl_fa_grad(d, q, dims, weights, b_inv, a_inv, pos, div,
                                   div, dD, rho_d, inv_temperature, margin,
                                   grad_scale, own_side=False, other_side=True,
                                   has_mask=True, exclude_diag=False)
                if ctx.use_dd:
                    launch_mrl_fa_grad(d, d, dims, weights, b_inv, b_inv, pos,
                                       div, div, dD, rho_d, inv_temperature,
                                       margin, grad_scale, own_side=True,
                                       other_side=True, has_mask=True,
                                       exclude_diag=True)
        else:
            launch_mrl_grad(q, d, dims, dims_t, a_inv, b_inv, pos, div, w_vec,
                            dQ, dQ if lit else dD, rho_q, rho_q if lit else rho_d,
                            inv_temperature, margin, grad_scale,
                            two_sided=not lit)
            if ctx.use_qq:
                launch_mrl_grad(q, q, dims, dims_t, a_inv, a_inv, pos, div, w_vec,
                                dQ, dQ, rho_q, rho_q,
                                inv_temperature, margin, grad_scale,
                                exclude_diag=True, two_sided=True)
            if ctx.use_dd:
                launch_mrl_grad(d, d, dims, dims_t, b_inv, b_inv, pos, div, w_vec,
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
                 use_qq_negatives=False, normalized_inputs=False, stable=True,
                 tau_plus=0.0, label_smoothing=0.0):
        super().__init__(dims, weights=weights, temperature=temperature,
                         margin=margin, use_qq_negatives=use_qq_negatives,
                         use_dd_negatives=False,
                         normalized_inputs=normalized_inputs, stable=stable,
                         tau_plus=tau_plus, label_smoothing=label_smoothing)
        self._lit = True
