"""Perf probe: the plain-CLIP backward emission with atomic_add vs plain
tl.store (stores race, results are garbage -- timing only). Bounds the gain
available to a FlashAttention-style no-atomics backward.

    python probe_atomics.py [batch] [d_model]
"""
import sys
import time
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

import memeff._common as c


@triton.jit
def probe_kernel(
    X_ptr, Y_ptr, sum_exp_row_ptr, sum_exp_col_ptr, dX_ptr, dY_ptr,
    inv_temperature, inv_temperature_orig, n_i, batch_size,
    ATOMIC: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr, D_MODEL: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    i_offsets = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    j_offsets = pid_j * BLOCK_SIZE_J + tl.arange(0, BLOCK_SIZE_J)
    i_mask = i_offsets < n_i
    j_mask = j_offsets < batch_size
    S = tl.zeros([BLOCK_SIZE_I, BLOCK_SIZE_J], dtype=tl.float32)
    for d_start in range(0, D_MODEL, BLOCK_SIZE_D):
        d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
        Xb = tl.load(X_ptr + (i_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                     mask=i_mask[:, None], other=0.0)
        Yb = tl.load(Y_ptr + (j_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                     mask=j_mask[:, None], other=0.0)
        S = tl.dot(Xb, tl.trans(Yb), S, input_precision=INPUT_PRECISION)
    exp_S = tl.exp2(S * inv_temperature - inv_temperature)
    sr = tl.load(sum_exp_row_ptr + i_offsets, mask=i_mask, other=1.0)
    sc = tl.load(sum_exp_col_ptr + j_offsets, mask=j_mask, other=1.0)
    grad = (tl.math.fdiv(exp_S, sr[:, None]) +
            tl.math.fdiv(exp_S, sc[None, :])) * (inv_temperature_orig / (2 * batch_size))
    for d_start in range(0, D_MODEL, BLOCK_SIZE_D):
        d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
        Yb = tl.load(Y_ptr + (j_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                     mask=j_mask[:, None], other=0.0).to(tl.float32)
        Xb = tl.load(X_ptr + (i_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                     mask=i_mask[:, None], other=0.0).to(tl.float32)
        dXc = tl.dot(grad, Yb, input_precision=INPUT_PRECISION)
        dYc = tl.dot(tl.trans(grad), Xb, input_precision=INPUT_PRECISION)
        if ATOMIC:
            tl.atomic_add(dX_ptr + (i_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                          dXc, mask=i_mask[:, None], sem="relaxed")
            tl.atomic_add(dY_ptr + (j_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                          dYc, mask=j_mask[:, None], sem="relaxed")
        else:
            tl.store(dX_ptr + (i_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                     dXc, mask=i_mask[:, None])
            tl.store(dY_ptr + (j_offsets[:, None] * D_MODEL + d_offsets[None, :]),
                     dYc, mask=j_mask[:, None])


if __name__ == "__main__":
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 65536
    D = int(sys.argv[2]) if len(sys.argv) > 2 else 384
    x = F.normalize(torch.randn(B, D, device="cuda", dtype=torch.bfloat16), dim=1)
    y = F.normalize(torch.randn(B, D, device="cuda", dtype=torch.bfloat16), dim=1)
    sr = torch.rand(B, device="cuda") + 1.0
    sc = torch.rand(B, device="cuda") + 1.0
    dX = torch.zeros(B, D, device="cuda", dtype=torch.float32)
    dY = torch.zeros(B, D, device="cuda", dtype=torch.float32)
    bi, bj, nw, nst = 128, 64, 8, 2
    grid = (triton.cdiv(B, bi), triton.cdiv(B, bj))

    def run(atomic):
        probe_kernel[grid](x, y, sr, sc, dX, dY, 20.0, 1.0, B, B, ATOMIC=atomic,
                           BLOCK_SIZE_I=bi, BLOCK_SIZE_J=bj, BLOCK_SIZE_D=64,
                           D_MODEL=D, INPUT_PRECISION=c.INPUT_PRECISION,
                           num_warps=nw, num_stages=nst)

    print(f"B={B} D={D} blocks=({bi},{bj})")
    for atomic in (True, False):
        for _ in range(2):
            run(atomic)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            run(atomic)
        torch.cuda.synchronize()
        print(f"  atomic={atomic}: {(time.perf_counter() - t0) / 5 * 1e3:8.1f} ms")
