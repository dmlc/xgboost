# Exact multinomial leaf objective, gain, and the K−1 → K gauge

Design note for XGBoost issue #12278. This fixes the mathematics that Milestones 2–4 must
implement. It is a *derivation*, not a proposal: every constant below is taken from the
repository's existing code rather than assumed.

## 1. XGBoost's existing convention (measured, not assumed)

From `src/tree/param.h:245-248`:

```cpp
CalcGainGivenWeight(p, G, H, w) = -(2*G*w + (H + lambda)*w^2 + 2*alpha*|w|)
```

So if the node's second-order model of the loss change is

```
ΔL(w) = G·w + ½(H+λ)w² + α|w|
```

then **XGBoost's "gain" is −2·ΔL(w)** — twice the loss reduction. Checking against the
closed form at `param.h:272`: with α=0 the optimum is `w* = −G/(H+λ)`, giving
`ΔL(w*) = −½G²/(H+λ)` and `−2ΔL(w*) = G²/(H+λ)`, which is exactly what `CalcGain` returns.

Two further conventions matter:

- **Degenerate curvature** (`param.h:254, 267`): `sum_hess <= 0` ⟹ weight 0 **and** gain 0.
  The repository's policy is *return zero*, not *reject the candidate*.
- **`min_split_loss` / gamma** (`param.h:187`) is compared against `loss_chg`, i.e. it lives
  in these same doubled units.
- **`min_child_weight`** (`param.h:222-223`) is compared against each child's `sum_hess`,
  with the additional hard requirement `left_hess > 0 && right_hess > 0`.

Any dense generalisation must reproduce all of this, or it silently changes the meaning of
`gamma` and `min_child_weight` for every user.

## 2. The exact leaf objective

At a leaf, accumulate over its rows in the K−1 free coordinates:

```
G_i  = Σ_rows w·(p_i − y_i)                    G ∈ ℝ^{K−1}
H_ij = Σ_rows w·(p_i·δ_ij − p_i·p_j)           H ∈ ℝ^{(K−1)×(K−1)}, symmetric PSD
```

The second-order model of the loss change for a leaf output `w ∈ ℝ^{K−1}` is

```
ΔL(w) = Gᵀw + ½·wᵀ(H + R)w
```

where `R` is the regularization matrix derived in §4. Stationarity gives the Newton system
already implemented in `leaf_solver.h`:

```
(H + R)·w* = −G
```

## 3. Exact gain

Substituting `w*` into `ΔL` and applying the repository's −2 convention:

```
Gain = −2·ΔL(w*) = Gᵀ(H + R)⁻¹G
```

**This needs no second solve.** Since `w* = −(H+R)⁻¹G`, we have `Gain = −Gᵀw*`. Better
still, with the LDLᵀ factorization the solver already computes, `(H+R) = LDLᵀ`, and with
`z = L⁻¹G` from the forward substitution:

```
Gain = Σ_i z_i² / d_i
```

This form is **provably non-negative** whenever `H+R` is positive definite (all `d_i > 0`),
costs nothing beyond the solve already being performed, and never forms an inverse.

**Reduction check (K = 2).** Then `n_free = 1`, `H = Σ w·p₀(1−p₀)` is a scalar, `R = λ`, and
`Gain = G²/(H+λ)` — *identical* to `CalcGain` at `param.h:272`. The dense path therefore
degenerates exactly onto the existing scalar path, which is the cross-check Milestone 3B
requires.

**Degenerate curvature.** If the LDLᵀ factorization encounters a non-positive pivot, the
matrix is not positive definite. Matching `param.h:254/267`, the policy is **weight 0,
gain 0** — not a rejected split, not a regularised fudge. The existing solver already
returns `false` in exactly this case, so the policy maps onto it directly.

## 4. Regularization and the gauge — the decisive derivation

The model stores **K** outputs per leaf; the solve produces **K−1** coordinates. The
embedding is a gauge choice, because `softmax(z + c·1) = softmax(z)`. Two candidates:

- raw: `δ = [w; 0]`
- centered: `δ = [w; 0] − mean([w; 0])·1`

They give *identical probabilities*. They are **not** equivalent under L2 regularization,
and that is what settles the choice.

Let `E: ℝ^{K−1} → ℝ^K` be the raw embedding `Ew = [w; 0]`, and `C = I_K − 11ᵀ/K` the
centering projection (symmetric, idempotent). Penalising the **centered K-output** leaf:

```
‖δ‖² = ‖CEw‖² = wᵀEᵀCᵀCEw = wᵀ(EᵀCE)w
```

Now compute `M = EᵀCE = EᵀE − (1/K)·EᵀJE` where `J = 11ᵀ` (all ones, K×K):

- `EᵀE = I_{K−1}` (the embedding is an isometry onto the first K−1 coordinates)
- `(JEw)_m = Σ_n (Ew)_n = Σ_i w_i` for every m, so `EᵀJE = 1_{K−1}1_{K−1}ᵀ`

Therefore

```
M = I_{K−1} − 11ᵀ/K
```

giving the two regularization matrices:

| Gauge | L2 lives in | `R` | Eigenvalues of `R/λ` | Reference-class invariant |
|---|---|---|---|---|
| raw | K−1 free space | `λ·I` | 1 (×K−1) | **No** |
| centered | K output space | `λ(I − 11ᵀ/K)` | 1 (×K−2), 1/K | **Yes** |

Both are SPD, so `H + R` is SPD and the existing LDLᵀ solver applies unchanged.

**The choice: centered, `R = λ(I − 11ᵀ/K)`.** Three independent reasons:

1. **Model semantics.** XGBoost's L2 penalises *leaf output values*. Today's multiclass
   builds one tree per class, so λ penalises each class's output symmetrically, in the
   K-output space. The centered form is the one that preserves that meaning.
2. **Reference-class invariance.** With `R = λI` in free space, the penalty is on logits
   *relative to class K−1*, so relabelling which class is the reference changes the fitted
   model whenever λ > 0. That is an arbitrary implementation detail leaking into results.
   With the centered form it does not. **This is directly testable**: permute the class
   labels, refit, and require identical probabilities.
3. **Consistency with the existing intercept.** `MulticlassInitEstimationCpu` already
   centers the intercept (`MulticlassCenter{mean}`), so the centered gauge matches how the
   model's base margin is defined.

The centered embedding is also the minimum-norm representative of its gauge class, since
centering is the orthogonal projection onto `1⊥`.

**Learning rate.** `η` scales the leaf value; scaling commutes with the centering
projection, so applying `η` after centering preserves centering. No interaction.

## 5. min_child_weight

Today: `sum_hess` per child, plus `> 0`. For dense `H` the candidates are

| Candidate | Cost | K=2 reduction | Issue |
|---|---|---|---|
| `trace(H)` | free (diagonal already packed) | ✓ | Scale grows ~(K−1)×, silently changing the meaning of a user's existing value |
| `min_i H_ii` | free | ✓ | Per-coordinate curvature floor; does **not** bound `λ_min(H)` |
| `λ_min(H)` | eigendecomposition per candidate | ✓ | Correct conditioning guarantee, far too expensive |

**Proposed: `min_i H_ii`**, because (a) it reduces exactly to today's `sum_hess` when K=2,
(b) it keeps the same numeric scale as today's per-class value so existing user settings
keep their meaning, and (c) it directly expresses the property the parameter exists for —
every class coordinate carries enough curvature to be estimated. Note honestly that
`λ_min(H) ≤ min_i H_ii`, so this is a necessary but not sufficient condition for good
conditioning; `λ + λ_min(H) > 0` is what actually guarantees the solve, and that is enforced
by the factorization itself.

**This is a proposal, finalised in Milestone 3D, not a settled decision.**

## 6. What is still open for Milestone 4

- Whether the centered gauge should be applied at leaf-write time or folded into `R` only.
  (Folding into `R` changes `w*`; centering at write time does not. These are *different
  models* — `R = λ(I−11ᵀ/K)` is the one that makes them agree.)
- `reg_alpha` (L1): soft-thresholding has no closed form for a coupled system; it becomes a
  lasso subproblem requiring coordinate descent. Restrict initially.
- `max_delta_step`: a scalar clip on `|w|` has no unique vector analogue (per-coordinate
  clip vs norm ball). Restrict initially.
