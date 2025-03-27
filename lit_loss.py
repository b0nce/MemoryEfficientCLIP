import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def lit_sum_exp_kernel(
    X_ptr, Y_ptr, sum_exp_row_ptr,
    inv_temperature, batch_size,
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
    exp_S = tl.exp2(S_partial * inv_temperature - inv_temperature)
    
    # Atomic adds for row and column sums
    tl.atomic_add(sum_exp_row_ptr + i_offsets, tl.sum(exp_S, axis=1), mask=i_mask, sem="relaxed")


@triton.jit
def lit_grad_kernel(
    X_ptr, Y_ptr, sum_exp_row_ptr,
    dX_ptr,
    inv_temperature, inv_temperature_orig,
    batch_size, 
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
            other=0.0,
        )
        Y_block = tl.load(
            Y_ptr + (j_offsets[:, None] * D_MODEL + d_offsets[None, :]),
            mask=(j_mask[:, None] & d_mask[None, :]),
            other=0.0,
        )
        
        # Accumulate partial dot products for similarity matrix
        S_partial = tl.dot(X_block, tl.trans(Y_block), S_partial)

    exp_S = tl.exp2(S_partial * inv_temperature - inv_temperature)
    
    # Load precomputed sums
    sum_exp_row = tl.load(sum_exp_row_ptr + i_offsets, mask=i_mask, other=1.0)
    
    probs = tl.math.fdiv(exp_S, sum_exp_row[:, None])
    
    grad = probs * (inv_temperature_orig / batch_size)
    
    # Process gradients for all feature dimensions
    for d_start in range(0, D_MODEL, BLOCK_SIZE_D):
        d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
        d_mask = d_offsets < D_MODEL
        
        # Load feature blocks for computing gradient contributions
        Y_block = tl.load(
            Y_ptr + (j_offsets[:, None] * D_MODEL + d_offsets[None, :]),
            mask=(j_mask[:, None] & d_mask[None, :]),
            other=0.0,
            cache_modifier='.cg',
            eviction_policy="evict_first",
        )
        X_block = tl.load(
            X_ptr + (i_offsets[:, None] * D_MODEL + d_offsets[None, :]),
            mask=(i_mask[:, None] & d_mask[None, :]),
            other=0.0,
            cache_modifier='.cg',
            eviction_policy="evict_first",
        )
        
        tl.atomic_add(
            dX_ptr + i_offsets[:, None] * D_MODEL + d_offsets[None, :], 
            tl.dot(grad, Y_block), 
            mask=i_mask[:, None] & d_mask[None, :], 
            sem="relaxed"
        )


class MemoryEfficientLiTLossNormed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_normalized, y_normalized, temperature=0.07):
        """Forward pass for LiT loss."""
        X = x_normalized
        Y = y_normalized
        
        assert X.is_cuda and Y.is_cuda
        assert X.shape == Y.shape

        batch_size, d_model = X.shape
        device = X.device
        
        # Hardcoded for now
        BLOCK_SIZE_D = min(64, d_model)
        
        err_msg_pow2 = "Triton kernel cant work with tensors with non power of two dimensions"
        err_msg_lt16 = "Triton kernel cant work with tensors with dimensions <16"
        
        assert triton.next_power_of_2(BLOCK_SIZE_D) == BLOCK_SIZE_D, err_msg_pow2
        assert BLOCK_SIZE_D >= 16, err_msg_lt16
        
        inv_temp = 1.4426950408889634 / temperature
        inv_temp_orig = 1 / temperature

        # Initialize sum buffers
        sum_exp_row = torch.zeros(batch_size, device=device)
        
        # Compute sum_exp
        grid = (triton.cdiv(batch_size, 128), triton.cdiv(batch_size, 128))
        lit_sum_exp_kernel[grid](
            X, Y, 
            sum_exp_row,
            inv_temp,
            batch_size,
            BLOCK_SIZE_I=128,
            BLOCK_SIZE_J=128,
            BLOCK_SIZE_D=64,
            D_MODEL=d_model,
        )

        ctx.save_for_backward(X, Y, sum_exp_row)
        ctx.inv_temp = inv_temp
        ctx.inv_temp_orig = inv_temp_orig
        ctx.batch_size = batch_size
        ctx.d_model = d_model  # Save d_model for backward pass

        # TODO: save loss during sum computation, or at least create a separate fused kernel
        Sv = torch.einsum('bi,bi->b', X, Y) * inv_temp - inv_temp
        
        logits = torch.log(torch.exp(Sv) / sum_exp_row.flatten())
        
        loss = -logits.mean()
        
        return loss
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for LiT loss."""
        X, Y, sum_exp_row = ctx.saved_tensors
        inv_temp = ctx.inv_temp
        inv_temp_orig = ctx.inv_temp_orig
        batch_size = ctx.batch_size
        d_model = ctx.d_model
        
        # Hardcoded block sizes for now
        BLOCK_SIZE_I = 64
        BLOCK_SIZE_J = 64
        BLOCK_SIZE_D = min(64, d_model)

        err_msg_pow2 = "Triton kernel cant work with tensors with non power of two dimensions"
        err_msg_lt16 = "Triton kernel cant work with tensors with dimensions <16"
        assert triton.next_power_of_2(BLOCK_SIZE_D) == BLOCK_SIZE_D, err_msg_pow2
        assert BLOCK_SIZE_D >= 16, err_msg_lt16
        
        # Initialize gradients
        dX = Y.clone().mul_(-inv_temp_orig / batch_size)
        
        # Compute gradients
        grid = (triton.cdiv(batch_size, BLOCK_SIZE_I), triton.cdiv(batch_size, BLOCK_SIZE_J))
        lit_grad_kernel[grid](
            X, Y, 
            sum_exp_row,
            dX,
            inv_temp, inv_temp_orig,
            batch_size,
            BLOCK_SIZE_I=BLOCK_SIZE_I,
            BLOCK_SIZE_J=BLOCK_SIZE_J,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
            D_MODEL=d_model,
        )
        
        # Scale gradients by grad_output (which is typically 1.0)
        if grad_output != 1.0:
            dX = dX * grad_output
            
        return dX, None, None


class MemoryEfficientLiTLoss(nn.Module):
    def __init__(self, temperature=0.07, normalized_inputs=False):
        super().__init__()
        self.temperature = temperature
        self.normalized_inputs = normalized_inputs
        
    def forward(self, text_features, image_features):
        if not self.normalized_inputs:
            image_features = F.normalize(image_features, dim=1)
            text_features = F.normalize(text_features, dim=1)
        
        # Apply custom LiT loss function
        loss = MemoryEfficientLiTLossNormed.apply(text_features, image_features, self.temperature)
        
        return loss
