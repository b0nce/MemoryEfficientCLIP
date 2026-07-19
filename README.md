# memeff — Memory-Efficient Contrastive Losses

Triton-kernel implementations of CLIP, LiT, and Qwen3-Embedding (InfoNCE) contrastive losses that never materialize the B×B similarity matrix, on a single GPU or sharded across a DDP ring. Batch sizes of 300k+ work in practice. Numerically validated against dense autograd references on A100 (sm80), H100 (sm90), and B200 (sm100).

## Installation

```bash
pip install memeff
```

Requires PyTorch ≥ 2.0, Triton ≥ 3.0 (ships with the Linux torch wheels), and a CUDA GPU. Inputs may be fp32, fp16, or bf16 — denominators and gradients always accumulate in fp32.

## Quickstart

```python
import torch
from memeff import MemoryEfficientCLIPLoss

clip_loss = MemoryEfficientCLIPLoss(temperature=0.07)

batch_size, dim = 2 ** 15, 1152
image_features = torch.randn(batch_size, dim, device="cuda", requires_grad=True)
text_features = torch.randn(batch_size, dim, device="cuda", requires_grad=True)

loss = clip_loss(image_features, text_features)
loss.backward()
```

All losses come in a single-GPU and a `Distributed*` (multi-GPU DDP) variant:

| Loss | Single GPU | Multi-GPU DDP |
|---|---|---|
| CLIP (bidirectional) | `MemoryEfficientCLIPLoss` | `DistributedMemoryEfficientCLIPLoss` |
| LiT (locked image tower) | `MemoryEfficientLiTLoss` | `DistributedMemoryEfficientLiTLoss` |
| Qwen3 InfoNCE (both towers) | `MemoryEfficientQwen3Loss` | `DistributedMemoryEfficientQwen3Loss` |
| Qwen3 InfoNCE (locked docs) | `MemoryEfficientLiTQwen3Loss` | `DistributedMemoryEfficientLiTQwen3Loss` |

## Options (all modules)

- `temperature` — softmax temperature (default 0.07).
- `normalized_inputs=True` — skip the internal L2 normalization.
- `stable=True` — rescale gradients by `sqrt(batch / temperature)` instead of `1 / temperature` to avoid fp32 underflow at very large batches (loss value unchanged; see details below).
- `tau_plus > 0` — the debiased contrastive loss of [Chuang et al., 2020](https://arxiv.org/abs/2007.00224). `forward` also accepts a per-call override: a float or a **(batch,) tensor of per-row priors** — useful when some samples are known to have approximate copies in the dataset. Costs no extra kernels.
- `label_smoothing > 0` — smoothed softmax targets with the `F.cross_entropy` convention (`(1-eps)` on the positive plus `eps/C` uniform). Costs only O(batch·dim) eager math.

All options compose with each other and with the distributed variants.

<details>
<summary><b>LiT loss (locked image tower)</b></summary>

```python
from memeff import MemoryEfficientLiTLoss

lit_loss = MemoryEfficientLiTLoss(temperature=0.07)
# text first; the image tower is locked and receives no gradient.
loss = lit_loss(text_features, image_features)
```

</details>

<details>
<summary><b>Qwen3 loss (embedding-model InfoNCE)</b></summary>

The improved InfoNCE objective of the Qwen3-Embedding report: an asymmetric query→document row softmax with a false-negative mask (any negative whose similarity exceeds `s(q_i, d_i) + margin` is dropped as a presumed unlabeled positive), row-specific hard negatives, and optional q-q / d-d in-batch negatives.

```python
from memeff import MemoryEfficientQwen3Loss

qwen3_loss = MemoryEfficientQwen3Loss(
    temperature=0.05,
    margin=0.1,               # margin >= 2 disables the false-negative mask
    use_qq_negatives=True,    # queries repel other queries      (optional)
    use_dd_negatives=True,    # documents repel other documents  (optional)
)

# Row-specific hard negatives (batch, K, dim): each query competes only against its own K.
loss = qwen3_loss(query_features, doc_features, hard_negatives)  # hard_negatives optional
```

For a locked document tower (precomputed corpus embeddings) use `MemoryEfficientLiTQwen3Loss`: only the queries receive gradients, and there is no d-d option (with locked documents its repulsion gradient has nowhere to land).

</details>

<details>
<summary><b>Distributed (multi-GPU DDP) usage</b></summary>

One process per GPU (e.g. `torchrun`). Each rank passes **only its shard** of the global batch; `forward` returns this rank's contribution to the loss, and `backward` fills the shard's full gradient of the *global* loss. Hard negatives (Qwen3) stay on their query's rank; per-row `tau_plus` tensors cover the local shard.

```python
import torch.distributed as dist
from memeff import DistributedMemoryEfficientCLIPLoss

dist.init_process_group("nccl")
torch.cuda.set_device(dist.get_rank())

clip_loss = DistributedMemoryEfficientCLIPLoss(temperature=0.07)
partial_loss = clip_loss(image_shard, text_shard)
partial_loss.backward()

global_loss = partial_loss.detach().clone()
dist.all_reduce(global_loss)   # logging only
```

Peak memory: the distributed CLIP and Qwen3 losses keep the assembled travelling tower(s) for the backward — O(global_batch × dim) per rank. The distributed LiT losses re-stream the ring in backward instead, staying at O(local_batch × dim). The full B×B similarity matrix is never materialized anywhere; communication is O(batch) against O(batch²) compute.

</details>

<details>
<summary><b>Stable gradient rescaling (<code>stable=True</code>)</b></summary>

`stable=True` rescales the gradient by `sqrt(batch / temperature)` instead of `1 / temperature`: the default `1 / (batch * temperature)` factor can nullify small values even in fp32, which matters at large batch sizes (300k+ works fine in practice). The loss value is unchanged, only the gradient scale differs, so the learning rate becomes batch-size dependent — use `lr / sqrt(batch * temperature)` to mimic the default behaviour, though at large batches standard values like 1e-4 tend to work well without that correction. The old `StableMemoryEfficientCLIPLoss` / `StableMemoryEfficientLiTLoss` classes remain as deprecated aliases.

</details>

<details>
<summary><b>Debiased contrastive loss (<code>tau_plus</code>)</b></summary>

`tau_plus > 0` switches to the debiased objective of [Chuang et al., 2020](https://arxiv.org/abs/2007.00224): with probability `tau_plus` an in-batch "negative" is actually an unlabeled positive, so each softmax denominator's negative sum `sum_neg` is replaced by `N * g` with

```
g = max((sum_neg / N - tau_plus * pos) / (1 - tau_plus), e^(-1/temperature))
```

where `N` is the nominal negative count and `pos` the positive exponential (the paper's estimator with M = 1). The clamp keeps the estimate at its theoretical minimum; rows where it fires push no gradient into their negatives.

```python
clip_loss = MemoryEfficientCLIPLoss(temperature=0.07, tau_plus=0.1)

# per-row priors: rows with known approximate copies get a higher prior
tau_row = duplicate_rates            # (batch,) tensor, values in [0, 1)
loss = clip_loss(image_features, text_features, tau_plus=tau_row)
```

Each row's denominator is debiased independently, so per-row priors slot straight into the estimator; for CLIP, sample i's prior applies to both its row and its column softmax. On the Qwen3 losses debiasing composes with the false-negative mask: masked entries contribute zero to the negative mean but keep their slot in the nominal count `N = (B-1)(1 + qq + dd) + K`.

Debiasing reuses every kernel untouched — the debiased denominator is a per-row transform of quantities the kernels already produce, and the gradient change rides the per-row divisor vector plus the eager positive-pair seed (`debias_denominators` in `_common.py`). Distributed: the CLIP column transform all-gathers the positive exponentials (and the prior vector, if per-row) — O(batch); the row-only losses debias with no communication.

</details>

<details>
<summary><b>Label smoothing</b></summary>

`label_smoothing=eps` smooths the softmax targets with the `F.cross_entropy` convention: `(1-eps)` on the positive plus `eps/C` uniform over the C candidates. For the Qwen3 losses the candidate set is the nominal one (positive + all nominal negatives); the false-negative mask does not reshape the target — masked entries are presumed positives, and a sliver of attraction toward them is the point of smoothing.

```python
clip_loss = MemoryEfficientCLIPLoss(temperature=0.07, label_smoothing=0.1)
```

Because the smoothed targets still sum to 1, the log-denominator cancels in the difference between the smoothed and unsmoothed loss, which collapses to batch-level scalars (the diagonal trace and dot products of summed embeddings). The correction is therefore a plain differentiable eager term added outside the kernels — O(batch·dim) math, one O(dim) all-reduce in DDP — and composes mechanically with `stable` (the term's gradient is rescaled to match) and `tau_plus`.

Note: `label_smoothing` and `tau_plus` push in related directions (both address "some negatives aren't really negatives") — smoothing adds uniform attraction to all negatives, debiasing removes estimated false-negative repulsion. They compose, but don't stack both at full strength blindly; if the motivation is known duplicates, per-row `tau_plus` is the more targeted tool.

</details>

<details>
<summary><b>Implementation details</b></summary>

**Code layout.** Shared constants, distributed helpers, the debias transform, the smoothing terms, and the masked Qwen3 kernel pair with its ring assembly live in `memeff/_common.py`. The single-GPU CLIP and LiT losses reuse the kernels of their distributed counterparts with the whole batch as one block, so there is exactly one implementation of each kernel.

**CLIP.** Two kernels in `distributed_clip_loss.py`: `clip_denom_kernel` accumulates row and column sum-exp block-wise without materializing the similarity matrix; `clip_grad_both_kernel` recomputes each block once in backward and emits both gradient directions from that pass. The loss value is formed directly in the log2 domain (no exp/log round trip), with a fixed maximum trick of `1/temperature` (numerically stable enough for the CLIP task).

**LiT.** `lit_denom_kernel` / `lit_grad_kernel`: row-only sum-exp and a single output GEMM — the image tower is locked, so no image gradient and no column denominator.

**Distributed CLIP.** The global batch is sharded one contiguous slice per rank. Only the column (text) tower travels, rotating around a SigLIP-style ring of batched point-to-point transfers, each block prefetched one hop ahead so its transfer overlaps the current block's matmul. Row denominators complete locally; column denominators are one O(batch) all-reduce. In backward the local-row gradient is final and the column gradient reaches its owner via reduce-scatter. Backward tile sizes are selected per GPU architecture (A100 / H100 / B200, with a safe default elsewhere).

**Distributed LiT.** Same ring, specialized: each rank keeps its text rows home and streams the locked image features past them twice (denominator lap, gradient lap). No gradient communication, no assembled towers — peak memory stays O(local_batch × dim).

**Qwen3.** The softmax is row-only, so every negative group ((Q,D), (Q,Q), (D,D)) is another additive contribution to the same per-row fp32 denominator, and one masked kernel pair covers all passes (`qwen3_denom_kernel` / `qwen3_grad_kernel`, with an optional global diagonal exclusion). The mask recomputes identically in backward, so no B×B state is stored. The B×K hard-negative block is handled eagerly in fp32. The DDP variant runs on the same ring as the distributed CLIP loss: row denominators live entirely on the row's home rank, d-d rides the travelling document blocks for free, q-q makes the query tower travel too (doubling ring payload, one extra reduce-scatter), hard negatives never leave their rank.

</details>

<details>
<summary><b>Tests and benchmarks</b></summary>

Both test scripts compare losses and all gradients against dense autograd references (they need a CUDA GPU) and cover the `stable`, `tau_plus` (scalar and per-row), and `label_smoothing` variants and their combinations:

```bash
python test_clip_lit_loss.py                        # CLIP + LiT, single GPU
python test_qwen3_loss.py                           # Qwen3, single GPU
torchrun --nproc-per-node=2 test_clip_lit_loss.py   # distributed variants
torchrun --nproc-per-node=2 test_qwen3_loss.py
```

The kernels run with ieee fp32 matmuls in the tests (`MEMEFF_INPUT_PRECISION=ieee`) so the comparison is not drowned in tensor-core rounding noise; production runs default to tf32.

`bench_qwen3_loss.py` reports timings and peak memory and exports torch.profiler traces to `./traces/`.

</details>

## Citation

```
@misc{memory-efficient-clip-loss,
  author = {Mikhail Kindulov},
  title = {memeff: Memory Efficient CLIP Loss},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/b0nce/MemoryEfficientCLIP}
}
```

## License

Apache License 2.0
