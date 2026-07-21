"""GradCache + memeff Qwen3-loss integration test / smoke train (2 x A100 box).

Trains a small text encoder on sentence-transformers/all-nli triplets: anchors are
queries, positives are documents, negatives ride as K=1 row-specific hard negatives
of DistributedMemoryEfficientQwen3Loss. A minimal gradient-cache (arXiv 2101.06983)
is vendored below -- chunked no-grad forward, loss backward on the cached reps,
RNG-replayed re-forward per chunk with a surrogate backward -- so the contrastive
batch is bounded by the loss (O(B * dim)), not by encoder activations.

All three texts of a triplet go through ONE encoder as a single (3B, L) batch, so
DDP gradient sync happens once, on the last chunk. The loss_fn splits the reps
back into q / d / h.

Modes (run under torchrun, one process per GPU; world=1 also works):
  equiv  -- correctness gate: param grads of the grad-cached step must match a
            single full-batch forward/backward through the same loss (dropout off,
            fp32, ieee matmuls). Run this FIRST.
  train  -- smoke train: bf16 autocast, large batch, logs loss / lr / step time /
            peak memory, plus downstream proxies (all-nli dev triplet accuracy,
            STS-B dev Spearman) every --eval-every steps. LR follows --schedule
            (wsd | cosine | const; const reproduces the old fixed-lr runs).
            --wd-anchor switches AdamW's decay target from 0 to the pretrained
            weights (decoupled L2-SP, arXiv 1802.01483). Scale --lr with the
            global batch: 8e-5 was right at 262k, so ~2e-5 at 65k.

  torchrun --nproc-per-node=2 train_gradcache_qwen3.py --mode equiv
  torchrun --nproc-per-node=2 train_gradcache_qwen3.py --mode train \
      --per-rank-batch 32768 --chunk 4096 --steps 30 --tau-plus 0.1

Or just `bash examples/run_train_gradcache.sh`, which installs the deps and
runs both modes. Validated on 2 x A100-PCIE-40GB (torch 2.5.1, transformers
5.14): equiv worst grad rel err 2.8e-6; 65k global batch trains at ~9-11
s/step, loss 0.25 -> 0.20 over 30 steps, peak 25.9 GiB/GPU.

Needs: pip install <repo root> transformers datasets  (plus torch + triton).
"""
import argparse
import contextlib
import math
import os
import sys
import time

if "equiv" in sys.argv:  # ieee matmuls for the grad comparison; must precede memeff import
    os.environ.setdefault("MEMEFF_INPUT_PRECISION", "ieee")

import torch
import torch.distributed as dist
import torch.nn as nn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["equiv", "train"], default="train")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--dataset", default="sentence-transformers/all-nli")
    p.add_argument("--dataset-config", default=None,
                   help="dataset config name (all-nli needs 'triplet')")
    p.add_argument("--columns", default="anchor,positive,negative",
                   help="text columns: 2 = pairs (in-batch negatives only), "
                        "3 = triplets with one hard negative per row")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--pretokenize", action="store_true",
                   help="tokenize the whole dataset once up front (datasets.map "
                        "cache); collate only pads. Removes the per-step "
                        "tokenization bottleneck at large per-rank batches")
    p.add_argument("--per-rank-batch", type=int, default=16384)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--margin", type=float, default=0.1)
    p.add_argument("--tau-plus", type=float, default=0.1)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--no-stable", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--schedule", choices=["const", "wsd", "cosine"], default="wsd",
                   help="const reproduces the old fixed-lr behaviour; wsd = "
                        "warmup / stable / linear-decay tail; cosine = warmup + "
                        "cosine to 0")
    p.add_argument("--warmup-frac", type=float, default=0.1)
    p.add_argument("--decay-frac", type=float, default=0.2,
                   help="fraction of steps in the WSD decay tail")
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--wd-anchor", action="store_true",
                   help="decay weights toward the PRETRAINED values instead of 0 "
                        "(decoupled L2-SP, arXiv 1802.01483): "
                        "w <- (1 - lr*wd) * w + lr*wd * w_pretrained")
    p.add_argument("--eval-every", type=int, default=10,
                   help="run downstream proxies every N steps (0 = start/end only)")
    return p.parse_args()


def lr_lambda(args):
    """Multiplier on args.lr per optimizer step (LambdaLR)."""
    warm = max(1, round(args.steps * args.warmup_frac))
    decay_start = args.steps - max(1, round(args.steps * args.decay_frac))

    def f(step):
        if args.schedule == "const":
            return 1.0
        if step < warm:
            return (step + 1) / warm
        if args.schedule == "cosine":
            t = (step - warm) / max(1, args.steps - warm)
            return 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))
        if step >= decay_start:  # wsd tail: linear to 0
            return max(0.0, (args.steps - step) / max(1, args.steps - decay_start))
        return 1.0

    return f


class Encoder(nn.Module):
    """Backbone + masked mean pooling. Returns UNnormalized reps (the loss
    normalizes). bf16 autocast in train mode; pooling always accumulates in fp32."""

    def __init__(self, backbone, use_bf16):
        super().__init__()
        self.backbone = backbone
        self.use_bf16 = use_bf16

    def forward(self, input_ids=None, attention_mask=None):
        with torch.autocast("cuda", torch.bfloat16, enabled=self.use_bf16):
            hidden = self.backbone(input_ids=input_ids,
                                   attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden.float() * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
        return pooled.to(hidden.dtype)


def split_dict(inputs, chunk_size):
    keys = list(inputs.keys())
    chunked = [inputs[k].split(chunk_size, dim=0) for k in keys]
    return [dict(zip(keys, vals)) for vals in zip(*chunked)]


class RandContext:
    """Fork-and-replay RNG state so a chunk's dropout pattern matches between the
    rep pass and the gradient pass."""

    def __init__(self, device):
        self.cpu_state = torch.get_rng_state()
        self.device = device
        self.gpu_state = torch.cuda.get_rng_state(device)

    def __enter__(self):
        self._fork = torch.random.fork_rng(devices=[self.device], enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.cpu_state)
        torch.cuda.set_rng_state(self.gpu_state, self.device)

    def __exit__(self, *exc):
        self._fork.__exit__(*exc)
        self._fork = None


class GradCache:
    """Minimal gradient cache (arXiv 2101.06983) for one encoder. The loss sees
    the full batch of reps as a leaf tensor; its backward (which, for the memeff
    distributed losses, runs the ring + reduce-scatter collectives) produces the
    cached rep grads, which the chunked re-forwards then push through the encoder
    via a surrogate dot-product backward."""

    def __init__(self, model, chunk_size, loss_fn):
        self.model = model
        self.chunk_size = chunk_size
        self.loss_fn = loss_fn

    def __call__(self, inputs, no_sync_except_last=False):
        chunks = split_dict(inputs, self.chunk_size)
        reps, states = [], []
        for c in chunks:
            states.append(RandContext(c["input_ids"].device))
            with torch.no_grad():
                reps.append(self.model(**c))
        full = torch.cat(reps).detach().requires_grad_(True)
        loss = self.loss_fn(full)
        loss.backward()
        grads = full.grad.split([r.shape[0] for r in reps])
        for i, (c, state, g) in enumerate(zip(chunks, states, grads)):
            sync = (not no_sync_except_last) or i == len(chunks) - 1
            ctx = contextlib.nullcontext() if sync else self.model.no_sync()
            with ctx, state:
                out = self.model(**c)
                (out.float() * g.float()).sum().backward()
        return loss.detach()


def make_loss_fn(args):
    from memeff import DistributedMemoryEfficientQwen3Loss
    loss_mod = DistributedMemoryEfficientQwen3Loss(
        temperature=args.temperature, margin=args.margin,
        stable=not args.no_stable, tau_plus=args.tau_plus,
        label_smoothing=args.label_smoothing)
    n_parts = len(args.columns.split(","))
    assert n_parts in (2, 3), "--columns must name 2 (pair) or 3 (triplet) columns"

    def loss_fn(reps):
        b = reps.shape[0] // n_parts
        q, d = reps[:b], reps[b:2 * b]
        h = reps[2 * b:].unsqueeze(1) if n_parts == 3 else None
        return loss_mod(q, d, h)

    return loss_fn


def make_loader(args, world, rank, tokenizer):
    from datasets import load_dataset
    from torch.utils.data import DataLoader, DistributedSampler

    cols = args.columns.split(",")
    config = args.dataset_config or (
        "triplet" if args.dataset == "sentence-transformers/all-nli" else None)
    ds = load_dataset(args.dataset, config, split="train")

    if args.pretokenize:
        def tok_fn(batch):
            return {f"{c}_ids": tokenizer(batch[c], truncation=True,
                                          max_length=args.max_len)["input_ids"]
                    for c in cols}
        ds = ds.map(tok_fn, batched=True, num_proc=max(1, args.num_workers),
                    remove_columns=ds.column_names, desc="pretokenize")

        def collate(rows):
            ids = [r[f"{c}_ids"] for c in cols for r in rows]
            enc = tokenizer.pad({"input_ids": ids}, padding=True,
                                return_tensors="pt")
            return {"input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"]}
    else:
        def collate(rows):
            texts = [r[c] for c in cols for r in rows]
            return dict(tokenizer(texts, padding=True, truncation=True,
                                  max_length=args.max_len, return_tensors="pt",
                                  return_token_type_ids=False))

    sampler = (DistributedSampler(ds, world, rank, shuffle=True, seed=args.seed,
                                  drop_last=True) if world > 1 else None)
    # fork-safety: the Rust tokenizer's thread pool segfaults forked workers
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return DataLoader(ds, batch_size=args.per_rank_batch, sampler=sampler,
                      shuffle=sampler is None, drop_last=True, collate_fn=collate,
                      num_workers=args.num_workers, pin_memory=True,
                      persistent_workers=args.num_workers > 0)


def build_eval(args, tokenizer):
    """Pre-tokenize the two downstream proxies: all-nli dev triplet accuracy
    (in-domain, saturates quickly) and STS-B dev Spearman (the standard
    sentence-embedding proxy). Small enough to run redundantly on every rank,
    which avoids any cross-rank synchronization."""
    from datasets import load_dataset

    def tok(texts):
        return dict(tokenizer(list(texts), padding=True, truncation=True,
                              max_length=args.max_len, return_tensors="pt",
                              return_token_type_ids=False))

    nli = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")
    sts = load_dataset("sentence-transformers/stsb", split="validation")
    return {"nli": (tok(nli["anchor"]), tok(nli["positive"]), tok(nli["negative"])),
            "sts": (tok(sts["sentence1"]), tok(sts["sentence2"]),
                    torch.tensor(sts["score"], dtype=torch.float32))}


def encode_eval(mod, enc, device, chunk):
    outs = []
    n = enc["input_ids"].shape[0]
    with torch.no_grad():
        for i in range(0, n, chunk):
            c = {k: v[i:i + chunk].to(device) for k, v in enc.items()}
            outs.append(nn.functional.normalize(mod(**c).float(), dim=-1))
    return torch.cat(outs)


def spearman(a, b):
    """Spearman rho, ties broken by order (fine for a training proxy)."""
    def rank(x):
        r = torch.empty_like(x)
        r[x.argsort()] = torch.arange(len(x), dtype=x.dtype)
        return r
    ra, rb = rank(a) - (len(a) - 1) / 2, rank(b) - (len(b) - 1) / 2
    return ((ra * rb).sum() / (ra.norm() * rb.norm()).clamp_min(1e-12)).item()


def run_eval(model, evalpack, device, chunk, rank, step):
    mod = model.module if hasattr(model, "module") else model
    was_training = mod.training
    mod.eval()
    a, p, n = (encode_eval(mod, e, device, chunk) for e in evalpack["nli"])
    acc = ((a * p).sum(-1) > (a * n).sum(-1)).float().mean().item()
    s1, s2, gold = evalpack["sts"]
    cos = (encode_eval(mod, s1, device, chunk)
           * encode_eval(mod, s2, device, chunk)).sum(-1).cpu()
    rho = spearman(cos, gold)
    if was_training:
        mod.train()
    if rank == 0:
        print(f"eval  step {step:5d}  nli-dev acc {acc:.4f}  "
              f"stsb-dev spearman {rho:.4f}", flush=True)


def run_equiv(model, gc_wrap, loss_fn, batch, world, rank):
    """Param grads: grad-cached step vs one full-batch forward/backward."""
    model.eval()  # dropout off: the two paths must see identical stochasticity

    model.zero_grad(set_to_none=True)
    loss_gc = gc_wrap(batch, no_sync_except_last=world > 1)
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()
             if p.grad is not None}

    model.zero_grad(set_to_none=True)
    loss_fn(model(**batch)).backward()

    worst, worst_name, tiny_bad = 0.0, "?", []
    for n, p in model.named_parameters():
        if p.grad is None:
            assert n not in grads, f"{n}: grad only in the grad-cache path"
            continue
        ref = p.grad.detach()
        diff = (grads[n] - ref).norm().item()
        if ref.norm().item() < 1e-6:
            # theoretically-zero grads (softmax shift invariance makes the
            # attention key biases' grads exact zeros): compare absolutely
            if diff > 1e-6:
                tiny_bad.append(n)
            continue
        err = diff / ref.norm().item()
        if err > worst:
            worst, worst_name = err, n
    dl = (loss_gc - loss_fn(model(**batch)).detach()).abs().item()
    ok = worst < 1e-3 and not tiny_bad
    print(f"[rank {rank}] equiv: {'OK ' if ok else 'FAIL'} "
          f"worst grad rel err {worst:.2e} ({worst_name}), |dloss| {dl:.2e}"
          + (f", zero-grad params off: {tiny_bad}" if tiny_bad else ""))
    assert ok


def main():
    args = parse_args()
    from transformers import AutoModel, AutoTokenizer

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(rank)
    device = torch.device("cuda", rank if world > 1 else 0)
    torch.manual_seed(args.seed + rank)

    equiv = args.mode == "equiv"
    if equiv:
        torch.backends.cuda.matmul.allow_tf32 = False
        args.per_rank_batch = min(args.per_rank_batch, 256)
        args.chunk = min(args.chunk, 64)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # no pooling head: it would receive no grad and trip DDP's reducer
    try:
        backbone = AutoModel.from_pretrained(args.model, add_pooling_layer=False)
    except TypeError:  # architectures without a pooler (e.g. distilbert)
        backbone = AutoModel.from_pretrained(args.model)
    model = Encoder(backbone, use_bf16=not equiv).to(device)
    if world > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    loss_fn = make_loss_fn(args)
    gc_wrap = GradCache(model, args.chunk, loss_fn)
    loader = make_loader(args, world, rank, tokenizer)
    if rank == 0:
        print(f"{args.mode}: world={world} global_batch={world * args.per_rank_batch} "
              f"chunk={args.chunk} stable={not args.no_stable} "
              f"tau_plus={args.tau_plus} ls={args.label_smoothing} "
              f"temp={args.temperature} lr={args.lr} sched={args.schedule} "
              f"wd={args.wd}{'->pretrained' if args.wd_anchor else '->0'} "
              f"model={args.model} data={args.dataset}[{args.columns}]")

    if equiv:
        batch = {k: v.to(device) for k, v in next(iter(loader)).items()}
        run_equiv(model, gc_wrap, loss_fn, batch, world, rank)
        if world > 1:
            dist.destroy_process_group()
        return

    # anchors capture the PRETRAINED weights before any update; with
    # --wd-anchor AdamW's own decay (toward 0) is disabled and replaced by a
    # decoupled pull toward the anchor after each step (L2-SP style).
    anchors = None
    if args.wd_anchor:
        anchors = [(p, p.detach().clone()) for p in model.parameters()
                   if p.requires_grad]
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=0.0 if args.wd_anchor else args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda(args))
    evalpack = build_eval(args, tokenizer)
    run_eval(model, evalpack, device, args.chunk, rank, step=0)

    torch.cuda.reset_peak_memory_stats(device)
    step, t0 = 0, time.perf_counter()
    epoch = 0
    while step < args.steps:
        if world > 1:
            loader.sampler.set_epoch(epoch)  # reshuffle if the data wraps around
        epoch += 1
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optim.zero_grad(set_to_none=True)
            loss = gc_wrap(batch, no_sync_except_last=world > 1)
            optim.step()
            if anchors is not None:
                rate = optim.param_groups[0]["lr"] * args.wd
                with torch.no_grad():
                    for p, p0 in anchors:
                        p.lerp_(p0, rate)  # w <- (1-rate)*w + rate*w_pretrained
            sched.step()
            step += 1
            if step % args.log_every == 0:
                total = loss.clone()
                if world > 1:
                    dist.all_reduce(total)  # partial losses sum to the global loss
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / args.log_every
                peak = torch.cuda.max_memory_allocated(device) / 2**30
                if rank == 0:
                    print(f"step {step:5d}  loss {total.item():.4f}  "
                          f"lr {sched.get_last_lr()[0]:.2e}  "
                          f"{dt:6.2f} s/step  peak {peak:5.2f} GiB")
                t0 = time.perf_counter()
            if (args.eval_every and step % args.eval_every == 0) \
                    or step >= args.steps:
                run_eval(model, evalpack, device, args.chunk, rank, step)
                t0 = time.perf_counter()  # keep eval time out of s/step
            if step >= args.steps:
                break
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
