#!/usr/bin/env bash
# End-to-end GradCache + DistributedMemoryEfficientQwen3Loss example.
#
# Installs everything it needs (memeff from the repo root, transformers,
# datasets; torch + triton are expected from your CUDA environment), runs the
# gradient-equivalence gate first, then a real smoke train at a 65k global
# contrastive batch. Validated on 2 x A100-PCIE-40GB: ~9-11 s/step, loss
# 0.25 -> 0.20 over 30 steps (~3.7 epochs of all-nli), peak 25.9 GiB/GPU.
#
#   bash examples/run_train_gradcache.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- dependencies -----------------------------------------------------------
# torch >= 2.0 with CUDA (the Linux wheels bundle the required triton).
python -c "import torch; assert torch.cuda.is_available(), 'CUDA torch required'"
pip install -q "$REPO_ROOT" transformers datasets

# --- environment ------------------------------------------------------------
# GPU P2P over PCIe hangs NCCL on some virtualized multi-GPU hosts (the first
# collective stalls until the 600 s watchdog SIGABRTs). Disabling it costs
# little here: the loss collectives are O(batch * dim), not activation-sized.
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
# The Rust fast-tokenizer thread pool is fork-unsafe and segfaults DataLoader
# workers (also set in the script; exported here for good measure).
export TOKENIZERS_PARALLELISM=false

NGPUS=$(python -c "import torch; print(torch.cuda.device_count())")
GLOBAL_BATCH=${GLOBAL_BATCH:-65536}
PER_RANK=$((GLOBAL_BATCH / NGPUS))
# lr: linear scaling from the validated 8e-5 @ 262144 global batch
LR=${LR:-$(python -c "print(8e-5 * $GLOBAL_BATCH / 262144)")}
cd "$REPO_ROOT/examples"
echo "gpus=$NGPUS global_batch=$GLOBAL_BATCH per_rank=$PER_RANK lr=$LR"

# --- 1. correctness gate: grad-cached grads == full-batch grads --------------
torchrun --nproc-per-node="$NGPUS" train_gradcache_qwen3.py \
    --mode equiv --tau-plus 0.1

# --- 2. real train: debiased (tau_plus=0.1), stable=True, bf16, WSD lr -------
# Downstream proxies (all-nli dev accuracy, STS-B dev Spearman) print every
# --eval-every steps. Add --wd-anchor to decay toward the pretrained weights
# (decoupled L2-SP) instead of 0 — useful at aggressive lr.
# Note: 65536 on a single 40 GB GPU will not fit (peak was ~26 GiB at
# 32768/rank); scale GLOBAL_BATCH with your GPU count and memory.
torchrun --nproc-per-node="$NGPUS" train_gradcache_qwen3.py \
    --mode train --per-rank-batch "$PER_RANK" --chunk 4096 \
    --steps 30 --log-every 5 --eval-every 10 --tau-plus 0.1 \
    --temperature 0.02 --lr "$LR" --schedule wsd
