"""Multi-GPU (DDP) memory-efficient LiT loss.

LiT is a text->image row-softmax with the image tower LOCKED: only the text tower
receives a gradient. That asymmetry removes all gradient communication. The trainable
text is the row side and d(text_i) sums over the locked image columns, so each rank
keeps its texts HOME, streams the locked image features past them around a SigLIP-style
ring (batched point-to-point), and accumulates its shard's COMPLETE text gradient
locally -- no gradient all-reduce, and the full towers are never assembled (peak memory
stays O(local_batch * dim), independent of world size). Only image features travel.

Two ring laps rotate only the image block:
  * Lap 1 (denominator): accumulate row_sum on home texts; complete after `world` hops.
  * Lap 2 (gradient):    with complete row_sum, accumulate dX on home texts; done after
                         the lap, seeded with the -y_i/(B*temp) positive-pair term.

`forward` returns THIS rank's contribution to the loss (all-reduce SUM for the global
scalar); calling `.backward()` fills the complete text gradient for the local shard.
The image argument is locked and receives no gradient. Mirrors lit_loss.py.
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

# Backward tile (BLOCK_SIZE_I, BLOCK_SIZE_J, num_warps, num_stages) for lit_grad_kernel.
# Its single output GEMM (the image tower is locked) leaves enough shared memory for a
# 128x128 tile at num_stages=3 from Ampere through Blackwell, so one tile serves every arch.
_BACKWARD_TILE = (128, 128, 8, 3)


def _world_and_rank(group):
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(group), dist.get_rank(group)
    return 1, 0


@triton.jit
def lit_denom_kernel(
    X_ptr, Y_ptr, sum_exp_row_ptr,
    n_i, n_j, inv_temperature,
    BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr, D_MODEL: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """One (n_i x n_j) block of exp2(text . image^T / temp - 1/temp), adding its row sums
    to the home texts' sum_exp_row. Atomic adds let successive ring hops (one image block
    each) accumulate into the same buffer. Unidirectional -> no column sum (image locked)."""
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


@triton.jit
def lit_grad_kernel(
    X_ptr, Y_ptr, sum_exp_row_ptr, dX_ptr,
    inv_temperature, scale, n_i, n_j,
    BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr, D_MODEL: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """One similarity block over home texts [0:n_i] x one image block [0:n_j], adding this
    block's contribution to the home text gradient: dX += (E/row_sum) @ image * scale.
    A single output GEMM (image locked -> no image gradient); the image block is upcast to
    fp32 so bf16/fp16 inputs still accumulate into the fp32 dX buffer cleanly."""
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
    sum_exp_row = tl.load(sum_exp_row_ptr + i_offsets, mask=i_mask, other=1.0)
    grad = tl.math.fdiv(exp_S, sum_exp_row[:, None]) * scale

    for d_start in range(0, D_MODEL, BLOCK_SIZE_D):
        d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
        d_mask = d_offsets < D_MODEL
        Y_block = tl.load(Y_ptr + (j_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                          mask=(j_mask[:, None] & d_mask[None, :]), other=0.0,
                          eviction_policy="evict_first").to(tl.float32)
        dX_contrib = tl.dot(grad, Y_block, input_precision=INPUT_PRECISION)
        tl.atomic_add(dX_ptr + (i_offsets[:, None] * D_MODEL + d_offsets[None, :]), dX_contrib,
                      mask=(i_mask[:, None] & d_mask[None, :]), sem="relaxed")


def _launch_denom(x_text, y_block, sum_exp_row, inv_temperature):
    """Accumulate one image block's row sum-exp into the home texts' buffer, in place."""
    n_i, d_model = x_text.shape
    n_j = y_block.shape[0]
    grid = (triton.cdiv(n_i, 128), triton.cdiv(n_j, 128))
    lit_denom_kernel[grid](
        x_text, y_block, sum_exp_row, n_i, n_j, inv_temperature,
        BLOCK_SIZE_I=128, BLOCK_SIZE_J=128, BLOCK_SIZE_D=min(64, d_model),
        D_MODEL=d_model, INPUT_PRECISION=INPUT_PRECISION,
    )


def _launch_grad(x_text, y_block, sum_exp_row, scale, dX, inv_temperature):
    """Accumulate one image block's contribution to the home text gradient, in place."""
    n_i, d_model = x_text.shape
    n_j = y_block.shape[0]
    block_i, block_j, num_warps, num_stages = _BACKWARD_TILE
    grid = (triton.cdiv(n_i, block_i), triton.cdiv(n_j, block_j))
    lit_grad_kernel[grid](
        x_text, y_block, sum_exp_row, dX, inv_temperature, scale, n_i, n_j,
        BLOCK_SIZE_I=block_i, BLOCK_SIZE_J=block_j, BLOCK_SIZE_D=min(64, d_model),
        D_MODEL=d_model, INPUT_PRECISION=INPUT_PRECISION,
        num_warps=num_warps, num_stages=num_stages,
    )


def _ring_post(tensor, send_to, recv_from, group):
    """Post the send of `tensor` to (rank + 1) and a matching recv from (rank - 1), returning
    the recv buffer and the work handles. Posting before the block's kernel lets the transfer
    (on the NCCL stream) overlap the matmul (on the compute stream). Grouped batch_isend_irecv
    is required: ungrouped ring P2P deadlocks under NCCL."""
    recv = torch.empty_like(tensor)
    ops = [dist.P2POp(dist.isend, tensor.contiguous(), send_to, group=group),
           dist.P2POp(dist.irecv, recv, recv_from, group=group)]
    return recv, dist.batch_isend_irecv(ops)


class DistributedMemoryEfficientLiTLossNormed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_text, y_img, inv_temperature, scale, batch_size, group):
        world, rank = _world_and_rank(group)
        send_to, recv_from = (rank + 1) % world, (rank - 1) % world

        # Lap 1: stream image blocks past home texts -> complete row_sum for home texts.
        # row_sum is fp32 regardless of input dtype (it sums up to `batch_size` exponentials).
        sum_exp_row = torch.zeros(x_text.shape[0], device=x_text.device, dtype=torch.float32)
        cur_y = y_img.contiguous().clone()
        for hop in range(world):
            reqs = None
            if hop + 1 < world:   # prefetch the next block (its recv overlaps this matmul)
                recv_y, reqs = _ring_post(cur_y, send_to, recv_from, group)
            _launch_denom(x_text, cur_y, sum_exp_row, inv_temperature)
            if reqs is not None:
                for req in reqs:
                    req.wait()
                cur_y = recv_y

        ctx.save_for_backward(x_text, y_img, sum_exp_row)
        ctx.inv_temperature, ctx.scale, ctx.group = inv_temperature, scale, group
        ctx.batch_size, ctx.in_dtype = batch_size, x_text.dtype

        # partial loss over the HOME diagonal (home text . home image), formed directly from
        # the log2-domain logit (log(exp2(s)) = ln2 * s) so no exponential is materialized.
        sv = (x_text.float() * y_img.float()).sum(dim=1) * inv_temperature - inv_temperature
        return -(sv * _LN2 - torch.log(sum_exp_row)).sum() / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        x_text, y_img, sum_exp_row = ctx.saved_tensors
        scale, group = ctx.scale, ctx.group
        world, rank = _world_and_rank(group)
        send_to, recv_from = (rank + 1) % world, (rank - 1) % world

        # seed the -y_i/(B*temp) positive-pair term from the HOME image (fp32 accumulator),
        # then stream images past home texts again -> complete dX locally, no gradient comm.
        dX = y_img.float() * (-scale)
        cur_y = y_img.contiguous().clone()
        for hop in range(world):
            reqs = None
            if hop + 1 < world:
                recv_y, reqs = _ring_post(cur_y, send_to, recv_from, group)
            _launch_grad(x_text, cur_y, sum_exp_row, scale, dX, ctx.inv_temperature)
            if reqs is not None:
                for req in reqs:
                    req.wait()
                cur_y = recv_y
        dX = dX * grad_output
        return dX.to(ctx.in_dtype), None, None, None, None, None


class DistributedMemoryEfficientLiTLoss(nn.Module):
    """Distributed counterpart of MemoryEfficientLiTLoss. Feed each rank its own shard of
    the batch; forward returns this rank's partial loss (all-reduce SUM for the global
    value) and backward fills the shard's text gradient. The image tower is locked.

    stable=True rescales the gradient by sqrt(global_batch / temperature) instead of
    1 / temperature, keeping values out of fp32 underflow at very large batches; use
    lr / sqrt(global_batch * temperature) to mimic the default behaviour (see lit_loss.py).
    """
    def __init__(self, temperature=0.07, normalized_inputs=False, stable=False, group=None):
        super().__init__()
        self.temperature = temperature
        self.normalized_inputs = normalized_inputs
        self.stable = stable
        self.group = group

    def forward(self, text_features, image_features):
        x = text_features if self.normalized_inputs else F.normalize(text_features, dim=1)
        y = image_features if self.normalized_inputs else F.normalize(image_features, dim=1)
        x, y = x.contiguous(), y.contiguous()   # the kernels index the shards row-major

        assert x.is_cuda and y.is_cuda
        assert x.shape == y.shape

        world, _ = _world_and_rank(self.group)
        local_batch = x.shape[0]
        batch_size = world * local_batch
        inv_temperature = 1.4426950408889634 / self.temperature
        inv_temperature_orig = (math.sqrt(batch_size / self.temperature)
                                if self.stable else 1.0 / self.temperature)
        scale = inv_temperature_orig / batch_size

        # image is locked -> detach so no gradient is tracked for it.
        return DistributedMemoryEfficientLiTLossNormed.apply(
            x, y.detach(), inv_temperature, scale, batch_size, self.group)
