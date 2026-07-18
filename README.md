# Memory Efficient CLIP and LiT Loss

A memory-efficient implementation of CLIP (Contrastive Language-Image Pre-training) and LiT (Locked-image text Tuning) contrastive loss functions using the Triton compiler. This repository provides high-performance CUDA kernels that optimize the computation of contrastive loss functions for large batch sizes and embedding dimensions. Tested only on A100 80Gb.

## Overview

This implementation offers significant memory savings compared to standard PyTorch implementations by:

1. Using custom Triton kernels to avoid materializing the full similarity matrix
2. Computing row and column sums directly during the forward pass
3. Efficiently calculating gradients with relaxed atomic operations during backpropagation
4. Uses fixed maximum trick of 1/temperature (which is numerically stable enough for CLIP task)

## Features

- Memory-efficient computation with Triton
- Support for normalized and unnormalized input features
- Customizable temperature scaling
- Fully differentiable with optimized gradient computation
- Handles large batch sizes that would cause OOM errors with naive implementations
- Implementations for both CLIP (bidirectional) and LiT (unidirectional) loss functions

## Usage

### CLIP Loss

```python
import torch
from clip_loss import MemoryEfficientCLIPLoss

# Initialize the loss function
clip_loss = MemoryEfficientCLIPLoss(temperature=0.07)

# For pre-normalized features
clip_loss_normed = MemoryEfficientCLIPLoss(temperature=0.07, normalized_inputs=True)

# Forward pass
batch_size, dim = 2 ** 15, 1152
image_features = torch.randn(batch_size, dim, device="cuda")
text_features = torch.randn(batch_size, dim, device="cuda")

# Compute loss
loss = clip_loss(image_features, text_features)
```

### LiT Loss

```python
import torch
from lit_loss import MemoryEfficientLiTLoss

# Initialize the loss function
lit_loss = MemoryEfficientLiTLoss(temperature=0.07)

# For pre-normalized features
lit_loss_normed = MemoryEfficientLiTLoss(temperature=0.07, normalized_inputs=True)

# Forward pass
batch_size, dim = 2 ** 15, 1152
text_features = torch.randn(batch_size, dim, device="cuda")
image_features = torch.randn(batch_size, dim, device="cuda")

# Compute loss
loss = lit_loss(text_features, image_features)
```

### Distributed CLIP Loss (multi-GPU DDP)

```python
import torch
import torch.distributed as dist
from distributed_clip_loss import DistributedMemoryEfficientCLIPLoss

# One process per GPU, e.g. launched with torchrun.
dist.init_process_group("nccl")
torch.cuda.set_device(dist.get_rank())

clip_loss = DistributedMemoryEfficientCLIPLoss(temperature=0.07)

# Each rank passes ONLY its shard of the global batch.
local_batch, dim = 2 ** 15, 1152
image_features = torch.randn(local_batch, dim, device="cuda", requires_grad=True)
text_features = torch.randn(local_batch, dim, device="cuda", requires_grad=True)

# forward returns this rank's contribution; backward fills the shard's full gradient.
partial_loss = clip_loss(image_features, text_features)
partial_loss.backward()

# Sum the partial losses across ranks for the global scalar (logging only).
global_loss = partial_loss.detach().clone()
dist.all_reduce(global_loss)
```

### Distributed LiT Loss (multi-GPU DDP)

```python
import torch
import torch.distributed as dist
from distributed_lit_loss import DistributedMemoryEfficientLiTLoss

# One process per GPU, e.g. launched with torchrun.
dist.init_process_group("nccl")
torch.cuda.set_device(dist.get_rank())

lit_loss = DistributedMemoryEfficientLiTLoss(temperature=0.07)

# Each rank passes ONLY its shard of the global batch. The image tower is locked,
# so only the text features receive a gradient.
local_batch, dim = 2 ** 15, 1152
text_features = torch.randn(local_batch, dim, device="cuda", requires_grad=True)
image_features = torch.randn(local_batch, dim, device="cuda")

partial_loss = lit_loss(text_features, image_features)
partial_loss.backward()

global_loss = partial_loss.detach().clone()
dist.all_reduce(global_loss)
```

Both distributed modules accept `normalized_inputs=True` (skip the internal L2 normalize) and `stable=True` (the large-batch gradient rescaling described below), and take fp32, fp16, or bf16 features.

## Requirements

- PyTorch >= 1.10
- Triton
- CUDA-capable GPU

## Performance

This implementation is designed for large batch sizes and embedding dimensions where standard implementations would exceed GPU memory. The Triton kernels are automatically tuned for the specific hardware they run on.

## Implementation Details

### CLIP Loss

The CLIP implementation consists of two main Triton kernels:

1. `clip_sum_exp_kernel`: Computes partial similarity matrices and accumulates both row and column sums without materializing the full matrix
2. `clip_grad_kernel`: Computes gradients efficiently for both image and text features during backpropagation

### LiT Loss

The LiT implementation also uses two main Triton kernels:

1. `lit_sum_exp_kernel`: Computes partial similarity matrices and accumulates row sums only (unidirectional)
2. `lit_grad_kernel`: Computes gradients efficiently for text features only during backpropagation, as LiT (Locked-image text Tuning) is designed to train text encoders while keeping image encoders fixed

Both implementations are wrapped by PyTorch autograd Functions and nn.Modules for easy integration into PyTorch workflows.

### Distributed CLIP Loss

`distributed_clip_loss.py` shards the global batch one contiguous slice per rank and never materializes the full similarity matrix or gathers per-sample gradients. Each rank keeps its own rows at home; only the column (text) tower travels, rotating around a SigLIP-style ring of point-to-point transfers. Every block is prefetched one hop ahead so its transfer overlaps the current block's matmul.

1. `clip_denom_kernel`: a fused rectangular sum-exp for one ring block, accumulating the row and column denominators as the column tower rotates. Row denominators complete locally; the column denominators are a single O(batch) all-reduce.
2. `clip_grad_both_kernel`: computes each similarity block once over the local rows and emits both gradient directions from that pass. The local-row gradient is final; the column gradient is delivered to its owning shard with a reduce-scatter. Backward tile sizes are selected per GPU architecture.

`forward` returns the local rank's contribution to the loss (all-reduce SUM for the global value); `backward` produces the complete gradient of the global loss for the local shard. Communication is O(batch) against O(batch^2) compute. Denominators and gradients accumulate in fp32 regardless of the input precision, keeping fp16/bf16 features numerically safe.

### Distributed LiT Loss

`distributed_lit_loss.py` uses the same sharded ring, specialized to LiT's locked image tower. Because only the text tower trains and its gradient sums over the locked image columns, each rank keeps its text rows at home and streams only the image features past them. Every rank therefore accumulates its shard's complete text gradient locally, with no gradient communication and without ever assembling the full towers, so peak memory stays proportional to the per-rank shard.

1. `lit_denom_kernel`: the row-only sum-exp for one image block (the loss is unidirectional, so there is no column denominator).
2. `lit_grad_kernel`: a single output GEMM that accumulates the text gradient for one image block; the image features receive no gradient.

The image features stream past the home texts twice, once to complete the row denominators and once for the gradient. As with the distributed CLIP loss, `forward` returns the local partial loss and `backward` fills the local shard's text gradient, with fp32 accumulation for fp16/bf16 inputs.

## Differences between CLIP and LiT

- **CLIP**: Bidirectional loss that computes gradients for both image and text features
- **LiT**: Unidirectional loss that computes gradients only for text features (by design)

## Citation

If you use this implementation in your research, please cite:

```
@misc{memory-efficient-clip-loss,
  author = {Mikhail Kindulov},
  title = {Memory Efficient CLIP Loss},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/b0nce/MemoryEfficientCLIP}
}
```

## License

Apache License 2.0