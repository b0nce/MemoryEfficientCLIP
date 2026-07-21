# Examples

## GradCache + Qwen3 loss: 65k-batch fine-tuning on a few GPUs

`run_train_gradcache.sh` installs dependencies and runs
`train_gradcache_qwen3.py` end to end: a real encoder
(`sentence-transformers/all-MiniLM-L6-v2`) fine-tuned on all-nli triplets with
`DistributedMemoryEfficientQwen3Loss` — debiasing (`tau_plus=0.1`),
`stable=True`, label-smoothing-ready — under a minimal vendored gradient cache
(arXiv 2101.06983), so the contrastive batch is bounded by the loss
(O(B·dim)), not by encoder activations.

```bash
bash examples/run_train_gradcache.sh            # uses all visible GPUs
GLOBAL_BATCH=32768 bash examples/run_train_gradcache.sh
```

It runs two stages:

1. **`--mode equiv`** — correctness gate: parameter gradients of the
   grad-cached step must match a single full-batch forward/backward (fp32,
   ieee matmuls, dropout off). Attention key biases are compared with an
   absolute tolerance — their gradients are theoretical zeros (softmax shift
   invariance), so relative error there is float noise.
2. **`--mode train`** — bf16 smoke train at a 65,536 global contrastive batch.

Measured on 2 × A100-PCIE-40GB (vast.ai, torch 2.5.1+cu124, transformers
5.14.1, 2026-07-19):

| stage | result |
|---|---|
| equiv gate | worst grad rel err 2.8e-6, loss bit-identical |
| train, 65k global batch | ~9–11 s/step, loss 0.2521 → 0.2001 over 30 steps (~3.7 epochs) |
| peak GPU memory | 25.9 GiB per GPU (32,768 rows/rank) |

Batch/lr/temperature findings from the 2×A100 sweeps (2026-07-20):

- Peak memory barely moves with batch (25.9 → 27.4 → 29.5 GiB at 1×/4×/8× of
  65k global): the loss is O(B·dim) as designed. Step time is the constraint —
  the pairwise-logits FLOPs grow as B², ~70 s of the 163 s step at 524k.
- Loss offsets between batch sizes match the c·log(negatives) InfoNCE floor;
  per-step descent was identical — until the lr was scaled. **lr must scale
  with batch**: at 262k global and τ=0.02, lr=8e-5 reached loss 0.72 by step
  20 where the default 2e-5 was still at 1.04. The launcher linear-scales lr
  from that point (8e-5 · batch/262144).
- Raw loss values are not comparable across temperatures (τ rescales logits) —
  use the downstream proxies, printed every `--eval-every` steps: all-nli dev
  triplet accuracy and STS-B dev Spearman.

Feature validation in the from-scratch regime (2026-07-20, 2×A100): with an
MLM-only backbone (MiniLM-L12-H384-uncased) on gooaq's 3M Q-A pairs
(`--columns question,answer`, in-batch negatives only), 65k global batch,
τ=0.05, lr 8e-5, WSD, 100 steps (≈2.2 epochs), from a 0.640 nli / 0.605 stsb
floor — the loss features **win**, and `tau_plus` must be ~1/dataset-diversity,
not the paper's CIFAR-10 value of 0.1 (the correction scales with τ⁺·N):

| arm | nli-dev acc | stsb-dev spearman |
|---|---|---|
| baseline (`--no-stable --tau-plus 0`) | 0.7772 | 0.7991 |
| stable + `--tau-plus 1e-4` | 0.7822 | **0.8032** |
| stable + `--tau-plus 1e-3` | **0.7860** | 0.8008 |

On an already-contrastively-trained model (all-MiniLM-L6-v2 on its own
training data) the same features cannot win: there is no headroom, τ≠0.05
fights the model's native temperature (stsb collapses at τ=0.02), and
τ⁺=0.1 is ~1000× too large a prior.

Training knobs added after those sweeps:

- `--schedule wsd|cosine|const` (+ `--warmup-frac`, `--decay-frac`) — WSD
  (warmup / stable / linear tail) is the default; `const` reproduces the old
  fixed-lr behaviour.
- `--wd-anchor` — AdamW weight decay toward the **pretrained** weights instead
  of 0 (decoupled L2-SP, arXiv 1802.01483): `w ← (1−lr·wd)·w + lr·wd·w₀`.
  A cheap trust region for full-parameter fine-tuning at aggressive lr. Pull
  rate is lr·wd per step, so `--wd` must be large to matter (wd=25 at lr 4e-5
  ≈ 1e-3/step ≈ 9% cumulative over 100 steps).
- `--dataset` / `--dataset-config` / `--columns` — any HF dataset; 2 columns =
  pairs (in-batch negatives only), 3 = triplets with a hard negative.
- `--pretokenize --num-workers N` — tokenize the whole dataset once
  (datasets.map cache) so collate only pads. At ≥65k rows/rank the per-step
  tokenization otherwise becomes the bottleneck (the 524k run was
  tokenization-bound).

Environment notes baked into the launcher (learned the hard way):

- `NCCL_P2P_DISABLE=1` — GPU P2P over PCIe hangs NCCL's first collective on
  some virtualized hosts (600 s watchdog SIGABRT). Costs little here since the
  loss collectives are O(B·dim).
- `TOKENIZERS_PARALLELISM=false` + `persistent_workers=True` — the Rust fast
  tokenizer's thread pool is fork-unsafe and segfaults DataLoader workers
  otherwise.
