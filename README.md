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