"""Shared helpers and Triton kernels for the memory-efficient contrastive losses.

Holds everything that was previously copy-pasted across the loss modules: numeric
constants, the per-architecture backward tile table, distributed helpers, and the
masked embedding (Qwen3-style) kernel pair used by clip_embedding_loss.py and
lit_embedding_loss.py (one kernel pair covers both: the gradient kernel takes a
TWO_SIDED flag for whether the column direction is emitted).
"""
import os
import torch
import torch.distributed as dist
import triton
import triton.language as tl


# "ieee" override exists for tests: under tf32 the false-negative mask can disagree
# with an fp32 reference near the threshold, and such entries carry large softmax
# weight. Must be set in the environment before this module is imported.
INPUT_PRECISION = os.environ.get("MEMEFF_INPUT_PRECISION", "tf32")
LN2 = 0.6931471805599453     # ln(2): converts exp2 (log2-domain) logits back to nats
LOG2E = 1.4426950408889634

# Backward tile (BLOCK_SIZE_I, BLOCK_SIZE_J, num_warps, num_stages), keyed by compute
# capability (major * 10 + minor). A 128x128 tile at num_stages >= 3 exceeds shared
# memory on sm80/sm100, so those architectures use narrower tiles.
_BACKWARD_BLOCKS = {
    80: (128, 64, 8, 2),    # A100
    90: (128, 128, 8, 2),   # H100
    100: (64, 128, 8, 2),   # B200
}
_DEFAULT_BLOCKS = (128, 64, 8, 2)


def backward_blocks(device):
    cap = torch.cuda.get_device_capability(device)
    return _BACKWARD_BLOCKS.get(cap[0] * 10 + cap[1], _DEFAULT_BLOCKS)


def world_and_rank(group):
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(group), dist.get_rank(group)
    return 1, 0


def check_dims(d_model):
    """The feature-dimension block size the kernels can use for `d_model`."""
    block_d = min(64, d_model)
    if triton.next_power_of_2(block_d) != block_d:
        raise ValueError(f"feature dimension {d_model} is not supported: dimensions "
                         "below 64 must be a power of two")
    if block_d < 16:
        raise ValueError(f"feature dimension {d_model} is not supported: "
                         "dimensions must be >= 16")
    return block_d


def validate_features(x, y):
    if not (x.is_cuda and y.is_cuda):
        raise ValueError("features must be CUDA tensors")
    if x.shape != y.shape:
        raise ValueError(f"feature shapes must match, got {tuple(x.shape)} "
                         f"and {tuple(y.shape)}")


def validate_hard_negatives(q, h):
    if h.dim() != 3 or h.shape[0] != q.shape[0] or h.shape[2] != q.shape[1]:
        raise ValueError("hard negatives must have shape (batch, K, dim) matching "
                         f"the queries, got {tuple(h.shape)} for queries "
                         f"{tuple(q.shape)}")


def ring_post(tensor, send_to, recv_from, group):
    """Post the send of `tensor` to (rank + 1) and a matching recv from (rank - 1),
    returning the recv buffer and the work handles. Posting before the block's kernel
    lets the transfer (on the NCCL stream) overlap the matmul (on the compute stream).
    Grouped batch_isend_irecv is required: ungrouped ring P2P deadlocks under NCCL."""
    recv = torch.empty_like(tensor)
    ops = [dist.P2POp(dist.isend, tensor.contiguous(), send_to, group=group),
           dist.P2POp(dist.irecv, recv, recv_from, group=group)]
    return recv, dist.batch_isend_irecv(ops)


def hard_negative_exp(q, h, pos, inv_temperature, margin):
    """Masked exp2 of the (batch, K) row-specific hard-negative similarities."""
    s_h = torch.einsum('bkd,bd->bk', h.float(), q.float())
    keep = s_h <= (pos + margin)[:, None]
    return torch.exp2(s_h * inv_temperature - inv_temperature) * keep


@triton.jit
def emb_denom_kernel(
    A_ptr, B_ptr, pos_ptr, sum_exp_row_ptr,
    n_i, n_j, inv_temperature, margin, row_offset, col_offset,
    EXCLUDE_DIAG: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr, D_MODEL: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """Row sums of one masked block of exp2(A.B^T/t - 1/t), added atomically."""
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)

    i_offsets = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    j_offsets = pid_j * BLOCK_SIZE_J + tl.arange(0, BLOCK_SIZE_J)
    i_mask = i_offsets < n_i
    j_mask = j_offsets < n_j

    S_partial = tl.zeros([BLOCK_SIZE_I, BLOCK_SIZE_J], dtype=tl.float32)
    for d_start in range(0, D_MODEL, BLOCK_SIZE_D):
        d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
        d_mask = d_offsets < D_MODEL
        A_block = tl.load(A_ptr + (i_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                          mask=(i_mask[:, None] & d_mask[None, :]), other=0.0)
        B_block = tl.load(B_ptr + (j_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                          mask=(j_mask[:, None] & d_mask[None, :]), other=0.0)
        S_partial = tl.dot(A_block, tl.trans(B_block), S_partial,
                           input_precision=INPUT_PRECISION)

    pos = tl.load(pos_ptr + i_offsets, mask=i_mask, other=0.0)
    keep = (S_partial <= pos[:, None] + margin) & i_mask[:, None] & j_mask[None, :]
    if EXCLUDE_DIAG:
        keep = keep & ((i_offsets[:, None] + row_offset) != (j_offsets[None, :] + col_offset))
    exp_S = tl.where(keep, tl.exp2(S_partial * inv_temperature - inv_temperature), 0.0)
    tl.atomic_add(sum_exp_row_ptr + i_offsets, tl.sum(exp_S, axis=1), mask=i_mask, sem="relaxed")


@triton.jit
def emb_grad_kernel(
    A_ptr, B_ptr, pos_ptr, sum_exp_row_ptr, dA_ptr, dB_ptr,
    n_i, n_j, inv_temperature, margin, grad_scale, row_offset, col_offset,
    EXCLUDE_DIAG: tl.constexpr, TWO_SIDED: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr, D_MODEL: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """Recomputes a masked block, then dA += p.B (and dB += p^T.A iff TWO_SIDED)."""
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)

    i_offsets = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    j_offsets = pid_j * BLOCK_SIZE_J + tl.arange(0, BLOCK_SIZE_J)
    i_mask = i_offsets < n_i
    j_mask = j_offsets < n_j

    S_partial = tl.zeros([BLOCK_SIZE_I, BLOCK_SIZE_J], dtype=tl.float32)
    for d_start in range(0, D_MODEL, BLOCK_SIZE_D):
        d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
        d_mask = d_offsets < D_MODEL
        A_block = tl.load(A_ptr + (i_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                          mask=(i_mask[:, None] & d_mask[None, :]), other=0.0)
        B_block = tl.load(B_ptr + (j_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                          mask=(j_mask[:, None] & d_mask[None, :]), other=0.0)
        S_partial = tl.dot(A_block, tl.trans(B_block), S_partial,
                           input_precision=INPUT_PRECISION)

    pos = tl.load(pos_ptr + i_offsets, mask=i_mask, other=0.0)
    keep = (S_partial <= pos[:, None] + margin) & i_mask[:, None] & j_mask[None, :]
    if EXCLUDE_DIAG:
        keep = keep & ((i_offsets[:, None] + row_offset) != (j_offsets[None, :] + col_offset))
    exp_S = tl.where(keep, tl.exp2(S_partial * inv_temperature - inv_temperature), 0.0)

    sum_exp_row = tl.load(sum_exp_row_ptr + i_offsets, mask=i_mask, other=1.0)
    grad = tl.math.fdiv(exp_S, sum_exp_row[:, None]) * grad_scale

    for d_start in range(0, D_MODEL, BLOCK_SIZE_D):
        d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
        d_mask = d_offsets < D_MODEL
        B_block = tl.load(B_ptr + (j_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                          mask=(j_mask[:, None] & d_mask[None, :]), other=0.0,
                          eviction_policy="evict_first").to(tl.float32)
        dA_contrib = tl.dot(grad, B_block, input_precision=INPUT_PRECISION)
        tl.atomic_add(dA_ptr + (i_offsets[:, None] * D_MODEL + d_offsets[None, :]), dA_contrib,
                      mask=(i_mask[:, None] & d_mask[None, :]), sem="relaxed")
        if TWO_SIDED:
            A_block = tl.load(A_ptr + (i_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                              mask=(i_mask[:, None] & d_mask[None, :]), other=0.0,
                              eviction_policy="evict_first").to(tl.float32)
            dB_contrib = tl.dot(tl.trans(grad), A_block, input_precision=INPUT_PRECISION)
            tl.atomic_add(dB_ptr + (j_offsets[:, None] * D_MODEL + d_offsets[None, :]), dB_contrib,
                          mask=(j_mask[:, None] & d_mask[None, :]), sem="relaxed")


def launch_emb_denom(a, b, pos, sum_exp_row, inv_temperature, margin,
                     exclude_diag=False, row_offset=0, col_offset=0):
    n_i, d_model = a.shape
    n_j = b.shape[0]
    grid = (triton.cdiv(n_i, 128), triton.cdiv(n_j, 128))
    emb_denom_kernel[grid](
        a, b, pos, sum_exp_row, n_i, n_j, inv_temperature, margin,
        row_offset, col_offset, EXCLUDE_DIAG=exclude_diag,
        BLOCK_SIZE_I=128, BLOCK_SIZE_J=128, BLOCK_SIZE_D=check_dims(d_model),
        D_MODEL=d_model, INPUT_PRECISION=INPUT_PRECISION,
    )


def launch_emb_grad(a, b, pos, sum_exp_row, dA, dB, inv_temperature, margin, grad_scale,
                    exclude_diag=False, row_offset=0, col_offset=0, *, two_sided):
    n_i, d_model = a.shape
    n_j = b.shape[0]
    block_i, block_j, num_warps, num_stages = backward_blocks(a.device)
    grid = (triton.cdiv(n_i, block_i), triton.cdiv(n_j, block_j))
    emb_grad_kernel[grid](
        a, b, pos, sum_exp_row, dA, dB, n_i, n_j, inv_temperature, margin, grad_scale,
        row_offset, col_offset, EXCLUDE_DIAG=exclude_diag, TWO_SIDED=two_sided,
        BLOCK_SIZE_I=block_i, BLOCK_SIZE_J=block_j, BLOCK_SIZE_D=check_dims(d_model),
        D_MODEL=d_model, INPUT_PRECISION=INPUT_PRECISION,
        num_warps=num_warps, num_stages=num_stages,
    )
