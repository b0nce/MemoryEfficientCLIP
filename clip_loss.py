import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


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
    sum_exp_col = tl.load(sum_exp_col_ptr + j_offsets, mask=j_mask, other=1.0)

    probs = tl.math.fdiv(exp_S, sum_exp_row[:, None]) + \
            tl.math.fdiv(exp_S, sum_exp_col[None, :])
    
    grad = probs * (inv_temperature_orig / (2 * batch_size))
    
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
        
        # Calculate gradient contributions for this feature chunk
        dX_contrib = tl.dot(grad, Y_block)
        dY_contrib = tl.dot(tl.trans(grad), X_block)
        
        # Atomic updates to gradients
        X_addrs = i_offsets[:, None] * D_MODEL + d_offsets[None, :]
        Y_addrs = j_offsets[:, None] * D_MODEL + d_offsets[None, :]
        
        tl.atomic_add(dX_ptr + X_addrs, dX_contrib, mask=i_mask[:, None] & d_mask[None, :], sem="relaxed")
        tl.atomic_add(dY_ptr + Y_addrs, dY_contrib, mask=j_mask[:, None] & d_mask[None, :], sem="relaxed")


class MemoryEfficientCLIPLossNormed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_normalized, y_normalized, temperature=0.07):
        """Forward pass for CLIP loss."""
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

        ctx.save_for_backward(X, Y, sum_exp_row, sum_exp_col)
        ctx.inv_temp = inv_temp
        ctx.inv_temp_orig = inv_temp_orig
        ctx.batch_size = batch_size
        ctx.d_model = d_model  # Save d_model for backward pass

        # TODO: save loss during sum computation, or at least create a separate fused kernel
        Sv = torch.einsum('bi,bi->b', X, Y) * inv_temp - inv_temp
        
        logits = torch.log(torch.exp(Sv) / sum_exp_row.flatten())
        logits += torch.log(torch.exp(Sv) / sum_exp_col.flatten())
        
        loss = -logits.mean() / 2.0
        
        return loss
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for CLIP loss."""
        X, Y, sum_exp_row, sum_exp_col = ctx.saved_tensors
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
        dY = X.clone().mul_(-inv_temp_orig / batch_size)
        
        # Compute gradients
        grid = (triton.cdiv(batch_size, BLOCK_SIZE_I), triton.cdiv(batch_size, BLOCK_SIZE_J))
        clip_grad_kernel[grid](
            X, Y, sum_exp_row, sum_exp_col,
            dX, dY,
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
            dY = dY * grad_output
            
        return dX, dY, None


class MemoryEfficientCLIPLoss(nn.Module):
    def __init__(self, temperature=0.07, normalized_inputs=False):
        super().__init__()
        self.temperature = temperature
        self.normalized_inputs = normalized_inputs
        
    def forward(self, image_features, text_features):
        if not self.normalized_inputs:
            image_features = F.normalize(image_features, dim=1)
            text_features = F.normalize(text_features, dim=1)
        
        # Apply custom CLIP loss function
        loss = MemoryEfficientCLIPLossNormed.apply(image_features, text_features, self.temperature)
        
        return loss


class StableMemoryEfficientCLIPLossNormed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_normalized, y_normalized, temperature=0.07):
        """Forward pass for CLIP loss."""
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
        inv_temp_orig = math.sqrt(1 / temperature)

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

        ctx.save_for_backward(X, Y, sum_exp_row, sum_exp_col)
        ctx.inv_temp = inv_temp
        ctx.inv_temp_orig = inv_temp_orig
        ctx.batch_size = batch_size
        ctx.d_model = d_model  # Save d_model for backward pass

        # TODO: save loss during sum computation, or at least create a separate fused kernel
        Sv = torch.einsum('bi,bi->b', X, Y) * inv_temp - inv_temp
        
        logits = torch.log(torch.exp(Sv) / sum_exp_row.flatten())
        logits += torch.log(torch.exp(Sv) / sum_exp_col.flatten())
        
        loss = -logits.mean() / 2.0
        
        return loss
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for CLIP loss."""
        X, Y, sum_exp_row, sum_exp_col = ctx.saved_tensors
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
        dY = X.clone().mul_(-inv_temp_orig / batch_size)
        
        # Compute gradients
        grid = (triton.cdiv(batch_size, BLOCK_SIZE_I), triton.cdiv(batch_size, BLOCK_SIZE_J))
        clip_grad_kernel[grid](
            X, Y, sum_exp_row, sum_exp_col,
            dX, dY,
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
            dY = dY * grad_output
            
        return dX, dY, None


class StableMemoryEfficientCLIPLoss(nn.Module):
    """
    This one uses math.sqrt of the scaling coefficient in gradient computation.
    1 / (batch_sz * temperature) can nullify small values even in fp32, which is
    crucial for larger batch sizes. In my tests, 300k+ works just fine. Please 
    note that you still have to choose proper scale for lr in your optimizer, 
    and with such a loss it would become batch size dependent. To mimic default 
    behavior of CLIP, use lr / sqrt(batch_sz * temperature), where lr is the 
    learning rate for default CLIP loss. Though for larger batch sizes, it 
    worked best for me to just use some standard values like 1e-4, ignoring 
    proper scaling to mimic default loss behavior.
    """
    def __init__(self, temperature=0.07, normalized_inputs=False):
        super().__init__()
        self.temperature = temperature
        self.normalized_inputs = normalized_inputs
        
    def forward(self, image_features, text_features):
        if not self.normalized_inputs:
            image_features = F.normalize(image_features, dim=1)
            text_features = F.normalize(text_features, dim=1)
        
        # Apply custom CLIP loss function
        loss = StableMemoryEfficientCLIPLossNormed.apply(image_features, text_features, self.temperature)
        
        return loss
