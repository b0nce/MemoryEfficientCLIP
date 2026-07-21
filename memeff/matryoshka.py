"""Eager matryoshka (MRL) wrapper: nested-prefix training for any memeff loss."""
import torch
import torch.nn as nn


class MatryoshkaLoss(nn.Module):
    """Matryoshka representation learning (arXiv 2205.13147) over any memeff
    loss: every feature tensor is sliced to each prefix dim, re-normalized by
    the wrapped loss, and the per-dim losses are summed with `weights`.

    Runs K separate loss passes, so memory stays O(batch * dim) but compute
    (and, for the distributed losses, communication) scales with sum(dims).
    The fused single-pass variant for the Qwen3 losses lives in
    mrl_qwen3_loss.py; this wrapper is its correctness oracle and covers every
    other loss class, the DDP modules, and any dims the kernels accept
    (>= 16, powers of two below 64 -- no chunk alignment needed).

    The wrapped loss must have normalized_inputs=False: prefix slices must be
    re-normalized, and a loss that trusts its inputs' norms would silently
    skip that.
    """
    def __init__(self, base_loss, dims, weights=None):
        super().__init__()
        if getattr(base_loss, "normalized_inputs", False):
            raise ValueError("MatryoshkaLoss requires base_loss.normalized_inputs="
                             "False so that every prefix slice is re-normalized")
        dims = tuple(int(m) for m in dims)
        if not dims or list(dims) != sorted(set(dims)):
            raise ValueError(f"dims must be strictly increasing, got {dims}")
        if weights is None:
            weights = (1.0,) * len(dims)
        weights = tuple(float(w) for w in weights)
        if len(weights) != len(dims):
            raise ValueError(f"got {len(weights)} weights for {len(dims)} dims")
        self.base_loss = base_loss
        self.dims = dims
        self.weights = weights

    def forward(self, *features, **kwargs):
        d_model = features[0].shape[-1]
        if self.dims[-1] > d_model:
            raise ValueError(f"largest mrl dim {self.dims[-1]} exceeds the "
                             f"feature dimension {d_model}")
        total = None
        for m, w in zip(self.dims, self.weights):
            sliced = tuple(f[..., :m].contiguous() if torch.is_tensor(f) else f
                           for f in features)
            part = w * self.base_loss(*sliced, **kwargs)
            total = part if total is None else total + part
        return total
