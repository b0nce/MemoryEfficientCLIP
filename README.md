# Memory Efficient CLIP and LiT Loss

A memory-efficient implementation of CLIP (Contrastive Language-Image Pre-training) and LiT (Locked-image text Tuning) contrastive loss functions using the Triton compiler. This repository provides high-performance CUDA kernels that optimize the computation of contrastive loss functions for large batch sizes and embedding dimensions. Numerically validated on A100 (sm80), H100 (sm90), and B200 (sm100).

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
- Qwen3 loss (the Qwen3-Embedding InfoNCE objective): query->document InfoNCE with false-negative masking, hard negatives, and optional q-q / d-d in-batch negatives, in a CLIP-style (both towers trained) and a LiT-style (locked document tower) variant
- Debiased contrastive loss (arXiv 2007.00224) on every module via `tau_plus`, at no extra kernel cost

## Usage

The modules work both flat (run scripts from inside the repo, `from clip_loss import
MemoryEfficientCLIPLoss`) and as a package (clone the repo next to your project and
`from MemoryEfficientCLIP import MemoryEfficientCLIPLoss` — every public class is
re-exported from `__init__.py`).

### CLIP Loss

```python
import torch
from clip_loss import MemoryEfficientCLIPLoss

# Initialize the loss function
clip_loss = MemoryEfficientCLIPLoss(temperature=0.07)

# For pre-normalized features
clip_loss_normed = MemoryEfficientCLIPLoss(temperature=0.07, normalized_inputs=True)

# For very large batches: rescale the gradient by sqrt(batch / temperature) instead
# of 1 / temperature so small values stay out of fp32 underflow (see "Stable
# gradient rescaling" below).
clip_loss_stable = MemoryEfficientCLIPLoss(temperature=0.07, stable=True)

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

### Qwen3 Loss

```python
import torch
from clip_qwen3_loss import MemoryEfficientQwen3Loss

# Asymmetric query->document InfoNCE with a false-negative mask: any negative whose
# similarity exceeds s(q_i, d_i) + margin is dropped as a presumed unlabeled positive.
qwen3_loss = MemoryEfficientQwen3Loss(
    temperature=0.05,
    margin=0.1,
    use_qq_negatives=True,   # queries repel other queries      (optional)
    use_dd_negatives=True,   # documents repel other documents  (optional)
)

batch_size, num_hard_negatives, dim = 2 ** 15, 4, 1152
query_features = torch.randn(batch_size, dim, device="cuda")
doc_features = torch.randn(batch_size, dim, device="cuda")
# Row-specific hard negatives: each query only competes against its own K.
hard_negatives = torch.randn(batch_size, num_hard_negatives, dim, device="cuda")

loss = qwen3_loss(query_features, doc_features, hard_negatives)  # hard_negatives optional
```

For a locked document tower (precomputed corpus embeddings, frozen doc encoder) use
the LiT-style variant. Only the queries receive gradients and there is no d-d option,
since with locked documents its repulsion gradient has nowhere to land:

```python
from lit_qwen3_loss import MemoryEfficientLiTQwen3Loss

lit_qwen3_loss = MemoryEfficientLiTQwen3Loss(
    temperature=0.05, margin=0.1, use_qq_negatives=True)
loss = lit_qwen3_loss(query_features, doc_features, hard_negatives)
```

Both files also contain the corresponding DDP modules
(`DistributedMemoryEfficientQwen3Loss`, `DistributedMemoryEfficientLiTQwen3Loss`).

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

### Distributed Qwen3 Loss (multi-GPU DDP)

```python
import torch
import torch.distributed as dist
from clip_qwen3_loss import DistributedMemoryEfficientQwen3Loss

dist.init_process_group("nccl")
torch.cuda.set_device(dist.get_rank())

qwen3_loss = DistributedMemoryEfficientQwen3Loss(
    temperature=0.05, margin=0.1, use_qq_negatives=True, use_dd_negatives=True)

# Each rank passes ONLY its shard; hard negatives stay on their query's rank.
local_batch, K, dim = 2 ** 15, 4, 1152
query_features = torch.randn(local_batch, dim, device="cuda", requires_grad=True)
doc_features = torch.randn(local_batch, dim, device="cuda", requires_grad=True)
hard_negatives = torch.randn(local_batch, K, dim, device="cuda", requires_grad=True)

partial_loss = qwen3_loss(query_features, doc_features, hard_negatives)
partial_loss.backward()

global_loss = partial_loss.detach().clone()
dist.all_reduce(global_loss)
```

All modules (single-GPU and distributed) accept `normalized_inputs=True` (skip the internal L2 normalize), `stable=True` (the large-batch gradient rescaling described below), and `tau_plus` (the debiased contrastive loss described below), and take fp32, fp16, or bf16 features.

### Stable gradient rescaling

`stable=True` rescales the gradient by `sqrt(batch / temperature)` instead of `1 / temperature`: the default `1 / (batch * temperature)` factor can nullify small values even in fp32, which matters at large batch sizes (300k+ works fine in practice). The loss value is unchanged, only the gradient scale differs, so the learning rate becomes batch-size dependent — use `lr / sqrt(batch * temperature)` to mimic the default behaviour, though at large batches standard values like 1e-4 tend to work well without that correction. The old `StableMemoryEfficientCLIPLoss` / `StableMemoryEfficientLiTLoss` classes remain as deprecated aliases for `stable=True`.

### Debiased contrastive loss

Every module accepts `tau_plus` (default 0 = the standard loss). `tau_plus > 0` switches to the debiased contrastive objective of [Chuang et al., 2020](https://arxiv.org/abs/2007.00224): with probability `tau_plus` an in-batch "negative" is actually an unlabeled positive, so each softmax denominator's negative sum `sum_neg` is replaced by `N * g` with

```
g = max((sum_neg / N - tau_plus * pos) / (1 - tau_plus), e^(-1/temperature))
```

where `N` is the negative count and `pos` the positive exponential (the paper's estimator with M = 1). The clamp keeps the estimate at its theoretical minimum; rows where it fires push no gradient into their negatives.

```python
clip_loss = MemoryEfficientCLIPLoss(temperature=0.07, tau_plus=0.1)
```

`tau_plus` composes with `stable` and, on the Qwen3 losses, with the false-negative mask (masked entries contribute zero to the negative mean but keep their slot in the nominal count `N`). See the implementation notes below for why this costs no new kernels.

## Requirements

- PyTorch >= 2.0 (the distributed modules use `dist.reduce_scatter_tensor` and `dist.batch_isend_irecv`)
- Triton >= 3.0 (the kernels use `tl.dot(..., input_precision=...)`)
- CUDA-capable GPU. Backward tile sizes are tuned for A100 / H100 / B200; other architectures fall back to a safe default. Numerically validated on A100 (sm80), H100 (sm90), and B200 (sm100).

## Performance

This implementation is designed for large batch sizes and embedding dimensions where standard implementations would exceed GPU memory. The Triton kernels are automatically tuned for the specific hardware they run on.

## Implementation Details

### Code layout

Shared constants, distributed helpers, and the masked Qwen3-loss kernel pair live in `_common.py`. The single-GPU CLIP and LiT losses (`clip_loss.py`, `lit_loss.py`) reuse the kernels of their distributed counterparts with the whole batch as one block, so there is exactly one implementation of each kernel.

### CLIP Loss

Two Triton kernels (defined in `distributed_clip_loss.py`, shared with the single-GPU module):

1. `clip_denom_kernel`: Computes partial similarity blocks and accumulates both row and column sum-exp without materializing the full matrix
2. `clip_grad_both_kernel`: Recomputes each similarity block once during backpropagation and emits both gradient directions from that pass

### LiT Loss

Two Triton kernels (defined in `distributed_lit_loss.py`, shared with the single-GPU module):

1. `lit_denom_kernel`: Computes partial similarity blocks and accumulates row sum-exp only (unidirectional)
2. `lit_grad_kernel`: Computes gradients for the text features only, as LiT (Locked-image text Tuning) trains the text encoder while keeping the image encoder fixed

All implementations are wrapped by PyTorch autograd Functions and nn.Modules, accumulate denominators and gradients in fp32 regardless of the input precision, and compute the loss value directly in the log2 domain (no exp/log round trip).

### Distributed CLIP Loss

`distributed_clip_loss.py` shards the global batch one contiguous slice per rank and never materializes the full similarity matrix or gathers per-sample gradients. Each rank keeps its own rows at home; only the column (text) tower travels, rotating around a SigLIP-style ring of point-to-point transfers. Every block is prefetched one hop ahead so its transfer overlaps the current block's matmul.

1. `clip_denom_kernel`: a fused rectangular sum-exp for one ring block, accumulating the row and column denominators as the column tower rotates. Row denominators complete locally; the column denominators are a single O(batch) all-reduce.
2. `clip_grad_both_kernel`: computes each similarity block once over the local rows and emits both gradient directions from that pass. The local-row gradient is final; the column gradient is delivered to its owning shard with a reduce-scatter. Backward tile sizes are selected per GPU architecture.

`forward` returns the local rank's contribution to the loss (all-reduce SUM for the global value); `backward` produces the complete gradient of the global loss for the local shard. Communication is O(batch) against O(batch^2) compute. Denominators and gradients accumulate in fp32 regardless of the input precision, keeping fp16/bf16 features numerically safe.

Note on memory: the backward needs every column against the home rows, so the assembled text tower (`y_full`) is kept between forward and backward — peak memory is O(global_batch x dim) per rank and grows with world size. The full B x B similarity matrix is still never materialized. The distributed LiT loss below avoids even the assembled tower (its locked image side lets it re-stream the ring in backward), keeping peak memory at O(local_batch x dim).

### Distributed LiT Loss

`distributed_lit_loss.py` uses the same sharded ring, specialized to LiT's locked image tower. Because only the text tower trains and its gradient sums over the locked image columns, each rank keeps its text rows at home and streams only the image features past them. Every rank therefore accumulates its shard's complete text gradient locally, with no gradient communication and without ever assembling the full towers, so peak memory stays proportional to the per-rank shard.

1. `lit_denom_kernel`: the row-only sum-exp for one image block (the loss is unidirectional, so there is no column denominator).
2. `lit_grad_kernel`: a single output GEMM that accumulates the text gradient for one image block; the image features receive no gradient.

The image features stream past the home texts twice, once to complete the row denominators and once for the gradient. As with the distributed CLIP loss, `forward` returns the local partial loss and `backward` fills the local shard's text gradient, with fp32 accumulation for fp16/bf16 inputs.

### Qwen3 Loss

`clip_qwen3_loss.py` implements the improved InfoNCE objective of the Qwen3
Embedding report: a row-only (query -> document) softmax whose denominator adds, per
query, the positive, the in-batch documents, K row-specific hard negatives, and
optionally query-query and document-document in-batch negatives. Every negative is
gated by a false-negative mask that zeroes similarities above `s(q_i, d_i) + margin`
(the mask is a stop-gradient, the positive itself always survives it).

Because the softmax is row-only, each negative group is just another additive
contribution to the same per-row fp32 denominator vector, so one pair of kernels
covers everything:

1. `qwen3_denom_kernel`: a masked rectangular sum-exp block pass with an optional global
   diagonal exclusion, launched once per enabled group ((Q, D), (Q, Q), (D, D)).
2. `qwen3_grad_kernel`: recomputes a masked block and emits both gradient directions
   (`dA += p @ B`, `dB += p^T @ A`) in a single pass; for the q-q / d-d passes both
   outputs point at the same buffer. The mask recomputes identically to the forward,
   so no B x B state is ever stored.

The B x K hard-negative block is handled eagerly in fp32 (it is tiny next to B x B).
Queries, documents, and hard negatives all receive gradients.

The DDP module in the same file runs these kernels on the sharded ring of the
distributed CLIP loss. The per-row denominators live entirely on the row's home rank,
so the forward needs no denominator communication: the document tower travels the ring
(prefetched one hop ahead), and the d-d pass rides the same travelling blocks for
free. Enabling q-q negatives makes the query tower travel too (doubling the ring
payload) and adds a second reduce-scatter in the backward; hard negatives never leave
their rank. As with the distributed CLIP loss, the assembled document tower (and query
tower iff q-q is enabled) is kept for the backward, so peak memory is
O(global_batch x dim) per rank. As elsewhere, `forward` returns the local partial loss and `backward`
fills the shard's gradients, with fp32 accumulation for fp16/bf16 inputs.

`lit_qwen3_loss.py` is the locked-document variant. Documents and hard negatives
receive no gradient, the q-d backward pass emits only the query direction, and the
d-d option is dropped (with locked documents its repulsion gradient has nowhere to
land, it would only inflate the denominator). In the DDP module this removes all
gradient communication; enabling q-q negatives adds back one reduce-scatter for the
query columns.

### Debiased contrastive loss

Debiasing (`tau_plus > 0`) reuses every kernel untouched, because the debiased
denominator is a per-row transform of exactly the quantities the kernels already
produce. In the kernels' shifted exp domain (all exponentials carry a fixed
`e^(-1/t)` factor, which cancels in the loss) the forward computes, per row,
`D = pos + N * max((sum_neg/N - tau_plus * pos)/(1 - tau_plus), e^(-2/t))` from the
accumulated sum-exp vector and the O(batch) diagonal — the paper's floor `e^(-1/t)`
lands at `e^(-2/t)` after the shift. The loss then just logs `D` instead of the raw
sum.

The backward exploits the fact that the grad kernels divide each recomputed
exponential by a per-row divisor they load from a vector: the debiased gradient of
every off-diagonal pair is `exp / ((1 - tau_plus) * D)`, so that product is passed
in place of the sum-exp vector (`+inf` on clamped rows, whose negatives get zero
gradient, matching the clamp's zero derivative). The diagonal term the kernel then
gets wrong is absorbed into the eager positive-pair seed, whose coefficient becomes
a per-row value instead of the plain `-1`; `tau_plus = 0` reduces to `-1` and the
standard loss exactly. Everything outside the kernels is O(batch) eager fp32 math
(`debias_denominators` in `_common.py`).

Distributed: the CLIP column transform needs every column's positive exponential,
one O(batch) all-gather next to the existing column-denominator all-reduce. The
row-only losses (LiT and both Qwen3 variants) debias entirely on the row's home
rank with no extra communication. In the Qwen3 losses the false-negative mask
composes with debiasing: masked entries contribute zero to the negative mean but
keep their slot in the nominal negative count
`N = (B-1) * (1 + qq + dd) + K`.

## Tests

Both test scripts compare losses and all gradients against dense autograd references
(they need a CUDA GPU):

- `test_clip_lit_loss.py`: the CLIP and LiT losses, single-GPU and distributed,
  including the `stable=True` rescaling, the deprecated `Stable*` aliases, and the
  debiased (`tau_plus > 0`) variants with clamp-firing rows.
- `test_qwen3_loss.py`: the Qwen3 losses, single-GPU and distributed, including
  the debiased variants.

Run `python <script>.py` on one GPU, or `torchrun --nproc-per-node=N <script>.py`
for the distributed versions. `bench_qwen3_loss.py` reports timings and peak
memory and exports torch.profiler traces to `./traces/`.

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