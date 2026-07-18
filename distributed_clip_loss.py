"""Multi-GPU (DDP) memory-efficient CLIP loss.

Each rank owns a contiguous shard of the global batch and never materializes the full
B x B similarity matrix. Only the Y (column) tower travels: the local X rows stay home,
so the ring carries and assembles Y alone while the denominators accumulate block-wise.
Each Y block is prefetched one hop ahead so its transfer overlaps the current block's
matmul (SigLIP-style). The backward computes each similarity block once, emitting dX for
the local rows directly and reduce-scattering the column gradient to its owning shard.

`forward` returns THIS rank's contribution to the loss (all-reduce SUM for the global
scalar). Calling `.backward()` fills the complete gradient of the *global* loss for the
local shard. Reduction buffers (denominators and gradients) are kept in fp32 regardless
of the input dtype, so bf16/fp16 features stay numerically safe. See README for the
derivation of the gradient formula.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import triton
import triton.language as tl


# tf32 tensor-core matmuls (Triton's default for an fp32 dot), kept explicit so the
# precision is easy to change; ~1e-3 relative error against ieee.
INPUT_PRECISION = "tf32"
_LN2 = 0.6931471805599453   # ln(2): converts the exp2 (log2-domain) logits back to nats

# Backward tile (BLOCK_SIZE_I, BLOCK_SIZE_J, num_warps, num_stages) for clip_grad_both_kernel,
# keyed by compute capability (major * 10 + minor). A 128x128 tile at num_stages >= 3 exceeds
# shared memory on sm80/sm100, so those architectures use narrower tiles.
_BACKWARD_BLOCKS = {
    80: (128, 64, 8, 2),    # A100
    90: (128, 128, 8, 2),   # H100
    100: (64, 128, 8, 2),   # B200
}
_DEFAULT_BLOCKS = (128, 64, 8, 2)


def _backward_blocks(device):
    cap = torch.cuda.get_device_capability(device)
    return _BACKWARD_BLOCKS.get(cap[0] * 10 + cap[1], _DEFAULT_BLOCKS)


def _world_and_rank(group):
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(group), dist.get_rank(group)
    return 1, 0


@triton.jit
def clip_denom_kernel(
    X_ptr, Y_ptr, sum_exp_row_ptr, sum_exp_col_ptr,
    n_i, n_j, inv_temperature,
    BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr, D_MODEL: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """One (n_i x n_j) block of exp2(X . Y^T / temp - 1/temp). Adds its row sums to
    sum_exp_row (the home rows) and its column sums to sum_exp_col (this Y block's
    columns). Atomic adds let successive ring blocks accumulate into the fp32 buffers."""
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
        X_block = tl.load(X_ptr + (i_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                          mask=(i_mask[:, None] & d_mask[None, :]), other=0.0)
        Y_block = tl.load(Y_ptr + (j_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                          mask=(j_mask[:, None] & d_mask[None, :]), other=0.0)
        S_partial = tl.dot(X_block, tl.trans(Y_block), S_partial,
                           input_precision=INPUT_PRECISION)

    exp_S = tl.exp2(S_partial * inv_temperature - inv_temperature)
    exp_S = tl.where(i_mask[:, None] & j_mask[None, :], exp_S, 0.0)   # zero padded lanes
    tl.atomic_add(sum_exp_row_ptr + i_offsets, tl.sum(exp_S, axis=1), mask=i_mask, sem="relaxed")
    tl.atomic_add(sum_exp_col_ptr + j_offsets, tl.sum(exp_S, axis=0), mask=j_mask, sem="relaxed")


@triton.jit
def clip_grad_both_kernel(
    X_ptr, Y_ptr, sum_exp_row_ptr, sum_exp_col_ptr, dX_ptr, dY_partial_ptr,
    inv_temperature, inv_temperature_orig, n_i, batch_size,
    BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr, D_MODEL: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """One similarity block over the local rows [0 : n_i] (X is the home shard) x all
    columns, emitting BOTH gradient directions from a single pass: dX for the local rows
    (final) and dY_partial for the global columns (this rank's contribution, reduce-
    scattered later). Feature blocks are upcast to fp32 for the output GEMMs so bf16/fp16
    inputs still accumulate into the fp32 gradient buffers cleanly."""
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)

    i_offsets = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)   # local dX row
    j_offsets = pid_j * BLOCK_SIZE_J + tl.arange(0, BLOCK_SIZE_J)   # global column
    i_mask = i_offsets < n_i
    j_mask = j_offsets < batch_size

    S_partial = tl.zeros([BLOCK_SIZE_I, BLOCK_SIZE_J], dtype=tl.float32)
    for d_start in range(0, D_MODEL, BLOCK_SIZE_D):
        d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
        d_mask = d_offsets < D_MODEL
        X_block = tl.load(X_ptr + (i_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                          mask=(i_mask[:, None] & d_mask[None, :]), other=0.0)
        Y_block = tl.load(Y_ptr + (j_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                          mask=(j_mask[:, None] & d_mask[None, :]), other=0.0)
        S_partial = tl.dot(X_block, tl.trans(Y_block), S_partial,
                           input_precision=INPUT_PRECISION)

    exp_S = tl.exp2(S_partial * inv_temperature - inv_temperature)
    sum_exp_row = tl.load(sum_exp_row_ptr + i_offsets, mask=i_mask, other=1.0)
    sum_exp_col = tl.load(sum_exp_col_ptr + j_offsets, mask=j_mask, other=1.0)
    grad = (tl.math.fdiv(exp_S, sum_exp_row[:, None]) +
            tl.math.fdiv(exp_S, sum_exp_col[None, :])) * (inv_temperature_orig / (2 * batch_size))

    for d_start in range(0, D_MODEL, BLOCK_SIZE_D):
        d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
        d_mask = d_offsets < D_MODEL
        Y_block = tl.load(Y_ptr + (j_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                          mask=(j_mask[:, None] & d_mask[None, :]), other=0.0,
                          eviction_policy="evict_first").to(tl.float32)
        X_block = tl.load(X_ptr + (i_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                          mask=(i_mask[:, None] & d_mask[None, :]), other=0.0,
                          eviction_policy="evict_first").to(tl.float32)
        dX_contrib = tl.dot(grad, Y_block, input_precision=INPUT_PRECISION)
        dY_contrib = tl.dot(tl.trans(grad), X_block, input_precision=INPUT_PRECISION)
        tl.atomic_add(dX_ptr + (i_offsets[:, None] * D_MODEL + d_offsets[None, :]), dX_contrib,
                      mask=(i_mask[:, None] & d_mask[None, :]), sem="relaxed")
        tl.atomic_add(dY_partial_ptr + (j_offsets[:, None] * D_MODEL + d_offsets[None, :]), dY_contrib,
                      mask=(j_mask[:, None] & d_mask[None, :]), sem="relaxed")


def _launch_denom(x_block, y_block, sum_exp_row, sum_exp_col, inv_temperature):
    """Accumulate one block's row/column sum-exp into the given fp32 buffers, in place."""
    n_i, d_model = x_block.shape
    n_j = y_block.shape[0]
    grid = (triton.cdiv(n_i, 128), triton.cdiv(n_j, 128))
    clip_denom_kernel[grid](
        x_block, y_block, sum_exp_row, sum_exp_col, n_i, n_j, inv_temperature,
        BLOCK_SIZE_I=128, BLOCK_SIZE_J=128, BLOCK_SIZE_D=min(64, d_model),
        D_MODEL=d_model, INPUT_PRECISION=INPUT_PRECISION,
    )


def _assemble_ring(x_local, y_local, inv_temperature, group):
    """Assemble the full Y tower and the global denominators. X never leaves its rank:
    the ring rotates only Y, each block prefetched one hop ahead so its transfer overlaps
    the current block's denom matmul. sum_exp_row is complete locally (home rows see every
    column); sum_exp_col holds this rank's column contributions and is all-reduced at the
    end (an O(batch) vector, versus O(batch*dim) for gathering the tower again)."""
    world, rank = _world_and_rank(group)
    local_batch, d_model = x_local.shape
    device = x_local.device
    sum_exp_row = torch.zeros(local_batch, device=device, dtype=torch.float32)  # home rows

    if world == 1:
        sum_exp_col = torch.zeros(local_batch, device=device, dtype=torch.float32)
        _launch_denom(x_local, y_local, sum_exp_row, sum_exp_col, inv_temperature)
        return y_local.contiguous(), sum_exp_row, sum_exp_col

    batch_size = world * local_batch
    y_full = torch.empty(batch_size, d_model, device=device, dtype=y_local.dtype)
    sum_exp_col = torch.zeros(batch_size, device=device, dtype=torch.float32)   # this rank's part
    cur_col = torch.zeros(local_batch, device=device, dtype=torch.float32)      # one block's cols
    send_to, recv_from = (rank + 1) % world, (rank - 1) % world

    # The travelling block rides a standalone buffer (NCCL P2P wants distinct, zero-offset
    # send/recv tensors) and is copied into its slot in y_full as it arrives. The next block
    # is prefetched into a second buffer, posted before the denom launch so its transfer (on
    # the NCCL stream) overlaps the current block's matmul (on the compute stream).
    cur_y = y_local.contiguous().clone()
    for hop in range(world):
        src = (rank - hop) % world   # global position of the block currently held
        reqs = None
        if hop + 1 < world:
            recv_y = torch.empty_like(cur_y)
            ops = [dist.P2POp(dist.isend, cur_y, send_to, group=group),
                   dist.P2POp(dist.irecv, recv_y, recv_from, group=group)]
            reqs = dist.batch_isend_irecv(ops)
        y_full.narrow(0, src * local_batch, local_batch).copy_(cur_y)
        cur_col.zero_()
        _launch_denom(x_local, cur_y, sum_exp_row, cur_col, inv_temperature)
        sum_exp_col.narrow(0, src * local_batch, local_batch).copy_(cur_col)
        if reqs is not None:
            for req in reqs:
                req.wait()
            cur_y = recv_y
    dist.all_reduce(sum_exp_col, group=group)   # sum column contributions across ranks
    return y_full, sum_exp_row, sum_exp_col


def _partial_loss(x_local, y_local, sum_exp_row, sum_exp_col_home, inv_temperature, batch_size):
    """This rank's additive contribution to the mean symmetric loss over its rows. The
    logits are formed directly from the log2-domain diagonal (log(exp2(s)) = ln2 * s), so
    no exponential is materialized just to be logged."""
    sv = (x_local.float() * y_local.float()).sum(dim=1) * inv_temperature - inv_temperature
    logits = (sv * _LN2 - torch.log(sum_exp_row)) + (sv * _LN2 - torch.log(sum_exp_col_home))
    return -logits.sum() / (2.0 * batch_size)


def _ring_backward(x_local, y_full, sum_exp_row, sum_exp_col, offset, n,
                   inv_temperature, inv_temperature_orig, batch_size):
    """dX for the local rows (final) and dY_partial for all columns (this rank's part).
    dX is seeded with its -y_i/(batch*temp) positive-pair term; dY gets its -x_i term
    after the caller reduce-scatters dY_partial to the owning shard. Buffers are fp32."""
    device, d_model = x_local.device, x_local.shape[1]
    dX = y_full.narrow(0, offset, n).float() * (-inv_temperature_orig / batch_size)
    dY_partial = torch.zeros(batch_size, d_model, device=device, dtype=torch.float32)
    block_i, block_j, num_warps, num_stages = _backward_blocks(device)
    grid = (triton.cdiv(n, block_i), triton.cdiv(batch_size, block_j))
    clip_grad_both_kernel[grid](
        x_local, y_full, sum_exp_row, sum_exp_col, dX, dY_partial,
        inv_temperature, inv_temperature_orig, n, batch_size,
        BLOCK_SIZE_I=block_i, BLOCK_SIZE_J=block_j, BLOCK_SIZE_D=min(64, d_model),
        D_MODEL=d_model, INPUT_PRECISION=INPUT_PRECISION,
        num_warps=num_warps, num_stages=num_stages,
    )
    return dX, dY_partial


class DistributedMemoryEfficientCLIPLossNormed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_local, y_local, y_full, sum_exp_row, sum_exp_col,
                offset, n, inv_temperature, inv_temperature_orig, batch_size, group):
        ctx.save_for_backward(x_local, y_full, sum_exp_row, sum_exp_col)
        ctx.offset, ctx.n, ctx.group = offset, n, group
        ctx.inv_temperature, ctx.inv_temperature_orig = inv_temperature, inv_temperature_orig
        ctx.batch_size, ctx.in_dtype = batch_size, x_local.dtype
        # x_local / y_local carry the grad path only; the value uses the assembled tower.
        return _partial_loss(x_local, y_local, sum_exp_row,
                             sum_exp_col[offset:offset + n], inv_temperature, batch_size)

    @staticmethod
    def backward(ctx, grad_output):
        x_local, y_full, sum_exp_row, sum_exp_col = ctx.saved_tensors
        offset, n, batch_size = ctx.offset, ctx.n, ctx.batch_size
        inv_temperature_orig = ctx.inv_temperature_orig
        dX, dY_partial = _ring_backward(x_local, y_full, sum_exp_row, sum_exp_col, offset, n,
                                        ctx.inv_temperature, inv_temperature_orig, batch_size)
        world, _ = _world_and_rank(ctx.group)
        if world > 1:
            # reduce-scatter delivers exactly this rank's summed column shard at half the
            # bandwidth of an all-reduce (which also gathers columns nobody here reads).
            dY = torch.empty(n, x_local.shape[1], device=x_local.device, dtype=torch.float32)
            dist.reduce_scatter_tensor(dY, dY_partial, group=ctx.group)
        else:
            dY = dY_partial
        dY = dY - x_local.float() * (inv_temperature_orig / batch_size)   # -x_i/(batch*temp) term
        dX, dY = dX * grad_output, dY * grad_output
        return (dX.to(ctx.in_dtype), dY.to(ctx.in_dtype),
                None, None, None, None, None, None, None, None, None)


class DistributedMemoryEfficientCLIPLoss(nn.Module):
    """Distributed counterpart of MemoryEfficientCLIPLoss. Feed each rank its own shard
    of the batch (detached image/text features); forward returns this rank's partial
    loss (all-reduce SUM for the global value) and backward fills the shard's gradient.

    stable=True rescales the gradient by sqrt(global_batch / temperature) instead of
    1 / temperature, which keeps values out of fp32 underflow at very large batches;
    use lr / sqrt(global_batch * temperature) to mimic the default behaviour.
    """
    def __init__(self, temperature=0.07, normalized_inputs=False, stable=False, group=None):
        super().__init__()
        self.temperature = temperature
        self.normalized_inputs = normalized_inputs
        self.stable = stable
        self.group = group

    def forward(self, image_features, text_features):
        x = image_features if self.normalized_inputs else F.normalize(image_features, dim=1)
        y = text_features if self.normalized_inputs else F.normalize(text_features, dim=1)
        x, y = x.contiguous(), y.contiguous()   # the kernels index the shards row-major

        assert x.is_cuda and y.is_cuda
        assert x.shape == y.shape

        world, rank = _world_and_rank(self.group)
        local_batch = x.shape[0]
        batch_size = world * local_batch
        inv_temperature = 1.4426950408889634 / self.temperature
        inv_temperature_orig = (math.sqrt(batch_size / self.temperature)
                                if self.stable else 1.0 / self.temperature)

        y_full, sum_exp_row, sum_exp_col = _assemble_ring(
            x.detach(), y.detach(), inv_temperature, self.group)
        offset = rank * local_batch
        return DistributedMemoryEfficientCLIPLossNormed.apply(
            x, y, y_full, sum_exp_row, sum_exp_col,
            offset, local_batch, inv_temperature, inv_temperature_orig, batch_size, self.group)
