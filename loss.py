import triton
import triton.language as tl
import torch

@triton.jit
def clip_sum_exp_kernel(
    X_ptr, Y_ptr, sum_exp_row_ptr, sum_exp_col_ptr,
    batch_size, inv_temperature,
    BLOCK_SIZE_I: tl.constexpr, 
    BLOCK_SIZE_J: tl.constexpr, 
    BLOCK_SIZE_D: tl.constexpr,
    D_MODEL: tl.constexpr,
):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_start = pid_i * BLOCK_SIZE_I
    j_start = pid_j * BLOCK_SIZE_J
    
    i_offsets = i_start + tl.arange(0, BLOCK_SIZE_I)
    j_offsets = j_start + tl.arange(0, BLOCK_SIZE_J)
    
    # Validity masks for batch dimensions
    i_mask = i_offsets < batch_size
    j_mask = j_offsets < batch_size
    
    # Initialize accumulators for exp(S) sums
    exp_S_row_acc = tl.zeros([BLOCK_SIZE_I], dtype=tl.float32)
    exp_S_col_acc = tl.zeros([BLOCK_SIZE_J], dtype=tl.float32)
    
    # Initialize partial similarity matrix
    S_partial = tl.zeros([BLOCK_SIZE_I, BLOCK_SIZE_J], dtype=tl.float32)
    
    # Process the full feature dimension in chunks
    for d_start in range(0, D_MODEL, BLOCK_SIZE_D):
        d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
        d_mask = d_offsets < D_MODEL
        
        # Load feature blocks with boundary checks
        X_block = tl.load(
            X_ptr + (i_offsets[:, None] * D_MODEL + d_offsets[None, :]),
            mask=(i_mask[:, None] & d_mask[None, :]),
            other=0.0
        )
        Y_block = tl.load(
            Y_ptr + (j_offsets[:, None] * D_MODEL + d_offsets[None, :]),
            mask=(j_mask[:, None] & d_mask[None, :]),
            other=0.0
        )
        
        # Accumulate partial dot products for similarity matrix
        S_partial += tl.dot(X_block, tl.trans(Y_block))
    
    # Apply temperature scaling
    S = S_partial * inv_temperature
    
    # Compute exponentials (inv_temperature is stable enough as maximum for softmax)
    exp_S = tl.exp(S - inv_temperature)
    
    # Compute sum over axis
    rows_sum_exp_S = tl.sum(exp_S, axis=1)
    cols_sum_exp_S = tl.sum(exp_S, axis=0)
    
    # Atomic adds for row and column sums
    tl.atomic_add(sum_exp_row_ptr + i_offsets, rows_sum_exp_S, mask=i_mask, sem="relaxed")
    tl.atomic_add(sum_exp_col_ptr + j_offsets, cols_sum_exp_S, mask=j_mask, sem="relaxed")


@triton.jit
def clip_grad_kernel(
    X_ptr, Y_ptr, sum_exp_row_ptr, sum_exp_col_ptr,
    dX_ptr, dY_ptr,
    batch_size, inv_temperature,
    BLOCK_SIZE_I: tl.constexpr,
    BLOCK_SIZE_J: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    D_MODEL: tl.constexpr,
):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_start = pid_i * BLOCK_SIZE_I
    j_start = pid_j * BLOCK_SIZE_J
    
    # Generate offsets
    i_offsets = i_start + tl.arange(0, BLOCK_SIZE_I)
    j_offsets = j_start + tl.arange(0, BLOCK_SIZE_J)
    
    # Validity masks for batch dimensions
    i_mask = i_offsets < batch_size
    j_mask = j_offsets < batch_size
    
    # Initialize partial dot products
    S_partial = tl.zeros([BLOCK_SIZE_I, BLOCK_SIZE_J], dtype=tl.float32)
    
    # Process the full feature dimension in chunks
    for d_start in range(0, D_MODEL, BLOCK_SIZE_D):
        d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
        d_mask = d_offsets < D_MODEL
        
        # Load feature blocks with boundary checks
        X_block = tl.load(
            X_ptr + (i_offsets[:, None] * D_MODEL + d_offsets[None, :]),
            mask=(i_mask[:, None] & d_mask[None, :]),
            other=0.0
        )
        Y_block = tl.load(
            Y_ptr + (j_offsets[:, None] * D_MODEL + d_offsets[None, :]),
            mask=(j_mask[:, None] & d_mask[None, :]),
            other=0.0
        )
        
        # Accumulate partial dot products for similarity matrix
        S_partial += tl.dot(X_block, tl.trans(Y_block))
    
    # Apply temperature scaling to get full similarity matrix
    S = tl.exp(S_partial * inv_temperature - inv_temperature)
    
    # Load precomputed sums
    sum_exp_row = tl.load(sum_exp_row_ptr + i_offsets, mask=i_mask, other=1.0)
    sum_exp_col = tl.load(sum_exp_col_ptr + j_offsets, mask=j_mask, other=1.0)
    
    # Compute probabilities
    probs = S / sum_exp_row[:, None]
    probs += S / sum_exp_col[None, :]
    
    # Compute combined gradient
    tl.static_assert(BLOCK_SIZE_I == BLOCK_SIZE_J)
    if pid_i == pid_j:
        # Create diagonal mask for identity elements
        diag_mask = (i_offsets[:, None] == j_offsets[None, :])
        diag_mask = diag_mask.to(tl.float32)
        grad = (probs - 2 * diag_mask) * (inv_temperature / (2 * batch_size))
    else:
        grad = probs * (inv_temperature / (2 * batch_size))
    
    # Process gradients for all feature dimensions
    for d_start in range(0, D_MODEL, BLOCK_SIZE_D):
        d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
        d_mask = d_offsets < D_MODEL
        
        # Load feature blocks for computing gradient contributions
        Y_block = tl.load(
            Y_ptr + (j_offsets[:, None] * D_MODEL + d_offsets[None, :]),
            mask=(j_mask[:, None] & d_mask[None, :]),
            other=0.0
        )
        X_block = tl.load(
            X_ptr + (i_offsets[:, None] * D_MODEL + d_offsets[None, :]),
            mask=(i_mask[:, None] & d_mask[None, :]),
            other=0.0
        )
        
        # Calculate gradient contributions for this feature chunk
        dX_contrib = tl.dot(grad, Y_block)
        dY_contrib = tl.dot(tl.trans(grad), X_block)
        
        # Atomic updates to gradients
        X_addrs = i_offsets[:, None] * D_MODEL + d_offsets[None, :]
        Y_addrs = j_offsets[:, None] * D_MODEL + d_offsets[None, :]
        
        tl.atomic_add(dX_ptr + X_addrs, dX_contrib, mask=i_mask[:, None] & d_mask[None, :], sem="relaxed")
        tl.atomic_add(dY_ptr + Y_addrs, dY_contrib, mask=j_mask[:, None] & d_mask[None, :], sem="relaxed")


def clip_loss_gradients_triton(X, Y, temperature=0.07, block_size=64):
    assert X.is_cuda and Y.is_cuda
    assert X.shape == Y.shape
    batch_size, d_model = X.shape
    device = X.device
    
    inv_temp = 1.0 / temperature
    BLOCK_SIZE_I = block_size
    BLOCK_SIZE_J = block_size
    BLOCK_SIZE_D = min(block_size, d_model)
    
    # Initialize sum buffers
    sum_exp_row = torch.zeros(batch_size, device=device)
    sum_exp_col = torch.zeros(batch_size, device=device)
    
    # Compute sum_exp
    grid = (triton.cdiv(batch_size, 128), triton.cdiv(batch_size, 128))
    clip_sum_exp_kernel[grid](
        X, Y, sum_exp_row, sum_exp_col,
        batch_size, inv_temp,
        BLOCK_SIZE_I=128,
        BLOCK_SIZE_J=128,
        BLOCK_SIZE_D=64,
        D_MODEL=d_model,
    )
    
    # Initialize gradients
    dX = torch.zeros_like(X)
    dY = torch.zeros_like(Y)
    
    # Compute gradients
    grid = (triton.cdiv(batch_size, BLOCK_SIZE_I), triton.cdiv(batch_size, BLOCK_SIZE_J))
    clip_grad_kernel[grid](
        X, Y, sum_exp_row, sum_exp_col,
        dX, dY,
        batch_size, inv_temp,
        BLOCK_SIZE_I=BLOCK_SIZE_I,
        BLOCK_SIZE_J=BLOCK_SIZE_J,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        D_MODEL=d_model,
    )
    
    return dX, dY
