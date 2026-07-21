# Fused MRL backward: the data dependency, and why it is (mostly) not a problem

Setting: the fused matryoshka variant of the memory-efficient losses. One kernel
pass over the $N \times N$ similarity blocks serves all $K$ nested dims; the only
extra state is $O(K \cdot N)$ per-row scalars. The question is whether the
*backward* can also run in a single chunk-wise walk over the feature dimension,
"renormalizing on the fly", with the same register budget as the existing grad
kernel.

Short answer: **yes** — but not with a single forward walk. There is a genuine
forward-in-$d$ / backward-in-$k$ data dependency; it is resolved by a
telescoping (running total minus running prefix) schedule that keeps exactly
**two** register tiles live, at the cost of one extra recompute matmul. The
"K coefficient tiles in registers" concern from the earlier discussion was
wrong.

## 1. Notation

- $X, Y \in \mathbb{R}^{N \times D}$: row-normalized full-dim features
  (queries / documents). Kernels tile rows $i$ and columns $j$.
- MRL dims $m_1 < m_2 < \dots < m_K = D$, defining **segments**
  $\text{seg}_s = [m_{s-1}, m_s)$ with $m_0 = 0$. Segments are the kernel's
  $d$-chunks (so each $m_k$ must be a multiple of `BLOCK_SIZE_D`).
- Prefix inverse norms (the $2KN$ precomputed scalars):

$$
a_i^k = \left\lVert X_i[{:}m_k] \right\rVert^{-1}, \qquad
b_j^k = \left\lVert Y_j[{:}m_k] \right\rVert^{-1}
$$

- Normalized prefixes: $p_i^k = a_i^k\, X_i[{:}m_k]$, $\;q_j^k = b_j^k\, Y_j[{:}m_k]$.
- Raw prefix dots, **cumulative across segments** — this is the whole trick:

$$
R^k_{ij} = \sum_{d < m_k} X_{id} Y_{jd},
\qquad
R^s = R^{s-1} + X[\text{seg}_s]\, Y[\text{seg}_s]^\top
$$

- Prefix similarities: $S^k_{ij} = a_i^k\, b_j^k\, R^k_{ij}$.
- Per-dim softmax weights, with denominators $Z_i^k$ accumulated by the fused
  forward (a $K{\times}N$ table, debias transform applied per $k$ eagerly):

$$
P^k_{ij} = \frac{\exp\!\left(S^k_{ij}/\tau\right)}{Z_i^k}
$$

- Total loss $\mathcal{L} = \sum_k w_k\, \mathcal{L}^k$ with
  $\mathcal{L}^k$ the InfoNCE loss on the dim-$m_k$ prefixes.

## 2. What the backward must produce

Gradient w.r.t. the normalized prefix (kernel-side negative term; positive-pair
seeds stay eager as in the existing code):

$$
g_i^k \;=\; \frac{\partial \mathcal{L}}{\partial p_i^k}
\;=\; \frac{w_k}{N\tau} \sum_j P^k_{ij}\, q_j^k \;+\; (\text{eager seed terms})
$$

Chain through the prefix renormalization $p_i^k = a_i^k X_i[{:}m_k]$ (Jacobian
$a_i^k (I - p_i^k {p_i^k}^\top)$):

$$
\frac{\partial \mathcal{L}}{\partial X_i[{:}m_k]}
\;\mathrel{+}=\; a_i^k \Big( g_i^k \;-\; \big(g_i^k \cdot p_i^k\big)\, p_i^k \Big)
$$

The rank-one correction needs only the per-row scalar
$\rho_i^k = g_i^k \cdot p_i^k$; expanding,

$$
\rho_i^k \;=\; \frac{w_k}{N\tau} \sum_j P^k_{ij}\, S^k_{ij} \;+\; (\text{eager})
\;=\; \sum_j c^k_{ij}\, R^k_{ij} \;+\; (\text{eager})
$$

which is structurally another row-sum: the forward/backward pass accumulates it
into a $K{\times}N$ buffer next to the denominators, and the correction
$-a_i^k \rho_i^k p_i^k$ is applied eagerly per dim, $O(N \cdot \textstyle\sum_k m_k)$
flops, **no big buffers**. So the renorm chain is a non-issue. The interesting
term is the first one.

Define the **per-(i,j,k) scalar coefficient**

$$
c^k_{ij} \;=\; \frac{w_k}{N\tau}\; P^k_{ij}\; a_i^k\, b_j^k
$$

Substituting $q_j^k = b_j^k Y_j[{:}m_k]$ and summing over dims, the kernel-side
output for **segment** $s$ (a slice of the single $(N, D)$ gradient buffer) is

$$
\boxed{\;
dX_i[\text{seg}_s] \;=\; \sum_j \underbrace{\Big(\sum_{k \ge s} c^k_{ij}\Big)}_{T^s_{ij}} \; Y_j[\text{seg}_s]
\;}
$$

because segment $s$ belongs to prefix $m_k$ **iff** $k \ge s$. The two-sided
direction is the transpose with the same tile:
$dY_j[\text{seg}_s] \mathrel{+}= \sum_i T^s_{ij}\, X_i[\text{seg}_s]$.

## 3. The data dependency (the "why it can't be one walk" part)

Availability order, walking segments forward ($s = 1, \dots, K$):

- After finishing segment $s$ we know $R^s$, hence $S^s$, hence $c^s$.
- But **emitting** segment $s$ requires $T^s = c^s + c^{s+1} + \dots + c^K$ —
  coefficients of *later* prefixes, which depend on raw dots not yet
  accumulated.

Concretely with $K = 3$:

| after segment | coefficients known | coefficient needed to emit it |
|---|---|---|
| 1 | $c^1$ | $c^1 + c^2 + c^3$ |
| 2 | $c^1, c^2$ | $c^2 + c^3$ |
| 3 | $c^1, c^2, c^3$ | $c^3$ |

Every segment except the last needs future information. A single forward walk
that both computes and consumes is impossible — *this* dependency is real.
Walking backward doesn't help either: $c^K$ needs the full-prefix dot $R^K$,
which itself takes a complete pass over $d$.

The naive resolution is to hold all $K$ coefficient tiles
($K \times$ `BLOCK_I` ${\times}$ `BLOCK_J` fp32) until the emit loop — that was
the register-pressure claim. It is unnecessary:

## 4. The fix: telescoping suffix sums

$$
T^s \;=\; \underbrace{\sum_{k=1}^{K} c^k}_{C \;(=\,T^1)} \;-\; \sum_{k < s} c^k,
\qquad\text{i.e.}\qquad
T^{s+1} = T^s - c^s
$$

The suffix sum is a running total minus a running prefix — and both are
maintainable with **one** tile. Note the existing grad kernel already has a
two-loop structure (loop 1 recomputes $S$, loop 2 emits $dA$/$dB$); the fused
schedule keeps that shape:

**Loop 1** ($d$ forward over segments): accumulate $R$; at each prefix boundary
$k$: load the per-row scalars $a^k, b^k, Z^k$ (and per-dim `pos` for the
false-negative mask), form $c^k$, and

$$
C \mathrel{+}= c^k, \qquad \rho^k_i \mathrel{+}= \textstyle\sum_j c^k_{ij} R^k_{ij} \;(\text{row-sum, atomic})
$$

End of loop 1: $T \leftarrow C$, discard $R$.

**Loop 2** ($d$ forward over segments, $R \leftarrow 0$): at segment $s$:

1. emit $dX[\text{seg}_s] \mathrel{+}= T \cdot Y[\text{seg}_s]$ (and the
   transposed $dY$ if two-sided) — `tl.dot`, exactly like today;
2. re-accumulate $R$ over $\text{seg}_s$;
3. at the boundary, recompute $c^s$ from the now-complete $R^s$ and update
   $T \leftarrow T - c^s$.

**Live register tiles: $R$ and $T$. Two.** The same budget as the current
kernel's `S_partial` + `grad` pair, independent of $K$. Per-boundary work adds
only $O(K)$ per-row scalar loads and elementwise tile ops.

(Variant: after loop 1, keep $R = R^K$ and walk segments in *reverse*,
computing $c^s$ from the current $R$, using $T \mathrel{+}= c^s$, then
$R \mathrel{-}= X[\text{seg}_s]Y[\text{seg}_s]^\top$. Same two-tile budget and
same flops, but the subtractive $R$ updates accumulate rounding error against
the forward-accumulated forward pass — the forward-recompute schedule is
numerically cleaner and keeps loop 2's $R$ bit-identical to loop 1's.)

## 5. What it actually costs

The price of the fusion is loop 2's re-accumulation of $R$ (step 2 above),
which the non-MRL kernel does not do. But it is **not** a full extra
$B^2 D$ matmul: the last coefficient loop 2 ever subtracts is $c^{K-1}$
(producing $T^K$, which emits the final segment — $T^{K+1}$ is never used), so
the re-accumulation only needs to reach $m_{K-1}$. The extra cost is
$m_{K-1}/D$ matmul units. In units per block, with $\sigma_{-} = m_{K-1}/D$:

| | one-sided (LiT) | two-sided (CLIP / Qwen3) |
|---|---|---|
| loop 1: recompute dots | 1 | 1 |
| loop 2: emit $dX$ (and $dY$) | 1 | 2 |
| loop 2: re-accumulate $R$ to $m_{K-1}$ | $\sigma_{-}$ | $\sigma_{-}$ |
| **fused total** | $2 + \sigma_{-}$ | $3 + \sigma_{-}$ |
| **non-MRL baseline** | 2 | 3 |
| dyadic ladder 64/128/256/384 ($\sigma_{-} = \tfrac{2}{3}$) | 2.67 **(+33%)** | 3.67 **(+22%)** |
| sparse ladder 64/384 ($\sigma_{-} = \tfrac{1}{6}$) | 2.17 **(+8%)** | 3.17 **(+6%)** |

The extra cost is the *same absolute* $\sigma_{-}$ for both sidedness — the
one-sided loss pays a higher *relative* overhead only because its baseline is
2 units instead of 3, and it is capped at +50% (dense ladders where
$m_{K-1} \to D$).

**Alternative schedule, for comparison** — reorganize by $k$ instead of by
segment: a single walk over $d$; at each boundary $k$, form $c^k$ and
immediately emit $dX[:, {:}m_k] \mathrel{+}= c^k\, Y[:, {:}m_k]$ (walking back
over segments $1..k$). No telescoping, no $T$ tile, no re-accumulation. Cost:
$1 + \sum_k m_k/D$ one-sided, $1 + 2\sum_k m_k/D$ two-sided. Since
$\sum_k m_k/D = 1 + \sum_{k<K} m_k/D \ge 1 + \sigma_{-}$ (equality iff
$K = 2$), this is **never better** than telescoping-with-skip — for the dyadic
ladder it costs 3.17 / 5.33. Telescoping remains the schedule of choice; at
$K = 2$ the two coincide.

Closing the remaining $\sigma_{-}$ gap would require keeping the boundary
coefficient tiles instead of recomputing them — $O(K)$ register/shared tiles
per CTA or $O(B^2)$ cache — i.e. exactly the costs this design exists to
avoid. $\sigma_{-}$ is the price of $O(1)$-tile state.

All of this is versus the eager wrapper's $K$ full backward passes
($\approx \text{base} \cdot \sum_k m_k / D \approx 2\times$ the non-MRL
backward for dyadic dims, on top of $K$ ring transfers). Forward is unchanged:
one pass, all $K$ denominators.

Remaining true constraints (unchanged from the earlier discussion):

1. **Alignment**: every $m_k$ a multiple of `BLOCK_SIZE_D` (= `min(64, D)`),
   so e.g. 64/128/256/384 for $D = 384$; a 32-dim prefix needs a smaller
   $d$-block for the whole kernel.
2. **Per-dim eager plumbing**: $K$ debias transforms, $K$ seed vectors, $K$
   `pos`/mask tables, $K$ label-smoothing terms — all $O(KN)$ or
   $O(N \sum_k m_k)$, mechanical.

## 6. Empirical postscript (2026-07-21, A100-PCIE-40GB, triton 3.1)

Implemented in `memeff/mrl_qwen3_loss.py` and validated against the per-dim
dense reference (fp32 ≤1e-4, bf16 ≤6e-3, full feature matrix). Three findings
the derivation above did not predict:

1. **Never fully unroll the chunk walk.** A `tl.static_range` over all
   $D/\text{BLOCK}_D$ chunks multiplies every `tl.dot` into the IR; LLVM's
   scheduling/regalloc went superlinear — ~2 min compile per kernel variant at
   $D=256$. Rolled runtime-bound chunk loops (static only over the $K$
   boundary blocks) compile in seconds and share one binary across same-$K$
   ladders.
2. **Triton 3.1 pipeliner miscompile.** Mixing bf16 and fp32 `tl.dot` in walk
   2's loop body corrupts the first segment's emission and the re-accumulated
   $R$ (any block size; `num_stages=1` or uniform fp32 operands both fix it —
   the shipped kernel casts all walk-2 dot operands to fp32, which costs the
   bit-exact telescoping in bf16: the peel mismatch is ~1e-7 relative, far
   below tolerance).
3. **The matmul-only cost model under-counts.** Each boundary does a full
   $B^2$ of exp2/compare/atomic work per walk; at $D=384$, $K=4$ that rivals
   the matmuls. Measured (bf16, $B=65536$, fwd+bwd): plain 505 ms, fused
   1046 ms, eager wrapper 1148 ms — i.e. fused ≈ 2× plain and only ~10% ahead
   of the wrapper at this $D/K$; the gap widens with $D/(64K)$. The memory
   claim holds as derived: $O(KN)$ scalar state, no feature copies, no
   per-dim gradient buffers.

## 7. Conclusion

The chunk-wise, renormalize-on-the-fly backward **works**. The forward-in-$d$ /
suffix-in-$k$ dependency is real but telescopes away; register pressure does
not grow with $K$; total extra memory over the non-MRL loss is
$O(K N)$ scalars ($a, b, Z, \rho$ tables), no feature copies, no per-dim
gradient buffers, and backward pays $m_{K-1}/D$ extra matmul units — +22%
(two-sided) / +33% (one-sided LiT) for a dense dyadic ladder, single-digit
percent for sparse ladders — instead of the wrapper's $\sim\!2\times$.
