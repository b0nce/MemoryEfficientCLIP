# Triton Optimized CLIP Loss

A memory-efficient implementation of CLIP contrastive loss using the Triton compiler. This repository provides high-performance CUDA kernels that optimize the computation of CLIP (Contrastive Language-Image Pre-training) loss functions for large batch sizes and embedding dimensions. Tested only on A100 80Gb.

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

## Usage

```python
import torch
from loss import MemoryEfficientCLIPLoss

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

## Requirements

- PyTorch >= 1.10
- Triton
- CUDA-capable GPU

## Performance

This implementation is designed for large batch sizes and embedding dimensions where standard implementations would exceed GPU memory. The Triton kernels are automatically tuned for the specific hardware they run on.

## Implementation Details

The implementation consists of two main Triton kernels:

1. `clip_sum_exp_kernel`: Computes partial similarity matrices and accumulates sums without materializing the full matrix
2. `clip_grad_kernel`: Computes gradients efficiently during backpropagation

These are wrapped by a PyTorch autograd Function (`MemoryEfficientCLIPLossNormed`) and a nn.Module (`MemoryEfficientCLIPLoss`) for easy integration into PyTorch workflows.

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
