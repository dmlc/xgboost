# Exact multinomial-logistic Hessian — research notes

Supporting material for XGBoost issue
[#12278](https://github.com/dmlc/xgboost/issues/12278), *"Using exact Hessian for
multinomial logistic"*.

These scripts are independent of XGBoost's C++ implementation. They exist so the
mathematics behind the implementation work can be checked against something that
does not share its assumptions, and so a reviewer can re-derive the claims
without trusting either the code or this write-up.

## 1. Why this research exists

XGBoost trains multiclass models by treating each class as a separate output with
a **diagonal** second-order approximation. The true multinomial Hessian is dense:
raising one class logit necessarily lowers the other probabilities, and a
diagonal approximation discards exactly that coupling. The question in #12278 is
whether using the exact dense Hessian is worth the cost.

These scripts establish the mathematical groundwork: what the exact Hessian is,
why it cannot be used naively, and how much the approximation actually costs in
convergence terms on problems where the answer is known in closed form.

## 2. The mathematical problem

For `K` classes with probabilities `p = softmax(z)` and a one-hot label `y`, the
negative log-likelihood has

```
g = p - y
H = diag(p) - p pᵀ
```

`H` is dense. Its off-diagonal entries `-p_i p_j` are precisely the coupling a
diagonal approximation throws away.

## 3. Why the K-dimensional Hessian is singular

`H` is **always** singular, for every `p`, independent of the data:

```
H · 1 = p - p·Σp = p - p = 0
```

The all-ones vector is a null direction. This is the gauge freedom of the
softmax — `softmax(z + c·1) = softmax(z)`, so shifting every logit by a constant
changes nothing observable, and the loss is flat along that line.

The consequence is practical, not cosmetic: the Newton system `H w = -g` has no
unique solution. Worse, it fails *silently*. `exact_multiclass_hessian.py`
reports what `numpy.linalg.solve` actually does with the singular system, and on
the tested environment it returns a solution rather than raising, because
round-off leaves the matrix numerically just barely invertible. Adding a small
constant to the diagonal would hide the symptom while changing the problem being
solved.

## 4. The K-1 reference-class parameterization

Pin the last logit to zero, `z_{K-1} = 0`, and optimize only `z_0 … z_{K-2}`.

This removes exactly the redundant direction and nothing else. Every softmax
distribution is still reachable — just by a unique parameter vector instead of a
line of equivalent ones. Over the free coordinates `i, j < K-1`:

```
g_i  = p_i - y_i
H_ij = p_i·δ_ij - p_i·p_j
```

This `H` is the leading `(K-1)×(K-1)` block of the full matrix, and it is
positive definite whenever every `p_i > 0`. The Newton system is then well posed.
Both scripts verify the block identity and the definiteness rather than assuming
them.

This choice is the reason the C++ side stores `K-1` coordinates. It is a
deliberate design decision, documented here rather than buried.

## 5. Why the dense Hessian matters for convergence

Newton's method with the exact Hessian converges quadratically; with a diagonal
approximation it converges linearly. The visible consequence is that the exact
method's iteration count is almost independent of how much accuracy you demand,
while the diagonal method's count grows roughly in proportion to the number of
decades demanded. Both scripts report this as a *sweep*, not a single number,
because a single iteration count is not interpretable without the tolerance that
produced it.

## 6. `exact_multiclass_hessian.py` — canonical mathematics

Self-contained (numpy only). Validates:

| | Check |
|---|---|
| A | Stable softmax: normalisation, shift invariance, ±1000 logits |
| B | Analytic gradient vs central-difference of the objective |
| C | Analytic Hessian vs central-difference of the gradient |
| D | Hessian symmetry |
| E | Singularity of the K-dim Hessian: `H·1 = 0`, rank `K-1`, and what `solve` does |
| F | Reference-class block identity and positive definiteness |
| G | Newton solve |
| H | Convergence: exact dense vs exact diagonal vs diagonal upper bound |

It also checks the hand-verifiable case `p = [0.1, 0.3, 0.6]` with class 2 as
reference, whose free Hessian must be

```
[[ 0.09, -0.03],
 [-0.03,  0.21]]
```

Every validation raises on failure; the script fails loudly rather than printing
a bad number.

## 7. `reproduce_issue_12278.py` — faithful upstream reproduction

Covertype, first 5,000 rows in natural order (no shuffle, so no seed is needed),
7 classes, intercept-only. Requires scikit-learn for the dataset fetcher only.

Because the model has no features, the optimum is available in closed form:

```
z_i = log(q_i / q_{K-1})        q = empirical class proportions
```

which gives a correctness check that does not depend on the optimiser at all.

## 8. Canonical experiment vs faithful reproduction

These are kept deliberately separate.

- `exact_multiclass_hessian.py` asks **"what is mathematically true?"** and is
  free to use the cleanest correct formulation.
- `reproduce_issue_12278.py` asks **"can the reported behaviour be
  reproduced?"** and is *not* free to quietly improve the upstream setup.

`multinomial.py` holds only what must not differ between them — a stable softmax,
the reference-class probability map, a finite-difference helper, and one Newton
driver with a single explicit stopping criterion. The objective, gradient and
Hessian are defined separately in each script, so the reproduction is never
forced to inherit the canonical script's opinion of what the objective should be.

Sharing the stopping criterion is the important part: it guarantees no method is
judged by an easier test than another.

## 9. The known objective-expression mismatch

**This is preserved on purpose and must not be "fixed" in the reproduction.**

The upstream experiment writes its objective in a binary, one-vs-rest style:

```
L = -Σ_{n,k} [ Y_nk·log(p_nk) + (1 - Y_nk)·log(1 - p_nk) ]
```

but uses the standard **multinomial** gradient and Hessian (`p - y`,
`diag(p) - p pᵀ`). The multinomial gradient is not the derivative of that
expression. The consistent multinomial objective would be

```
L = -Σ_{n,k} Y_nk·log(p_nk)
```

The script reports **both values, separately labelled**, and substitutes neither.
The reported optimum is unaffected, because the optimiser is driven by the
multinomial gradient whose root is the closed form in §7 — the objective acts as
a diagnostic readout rather than as the thing being minimised. That is itself
part of what is being documented.

Reproducing the upstream objective *value* is only possible using the upstream
expression, which is why it is kept.

## 10. Expected qualitative findings

On these intercept-only problems the exact dense Hessian converges in a handful
of Newton steps, while the diagonal approximations need one to three orders of
magnitude more iterations, and the gap **widens** as the convergence tolerance
tightens.

That is the qualitative claim. See §13 for what this does not establish.

## 11. How to run

```bash
python research/exact_multiclass_hessian.py     # numpy only
python research/reproduce_issue_12278.py        # + scikit-learn (Covertype)
```

Both are deterministic, take no arguments, and exit non-zero if any check fails.
Covertype is downloaded and cached by scikit-learn on first use.

Versions used for the numbers quoted below: Python 3.10.0, NumPy 2.0.2,
scikit-learn 1.7.2.

## 12. What the reported iteration counts mean

An iteration is one Newton **update** actually applied. Convergence is always
measured on the **exact** gradient, whichever curvature model the method used, so
all methods are held to an identical standard and the only thing that varies is
the quality of the curvature.

The stopping criterion is `max|g| ≤ tol` (infinity norm). Two conventions matter
when comparing against previously reported numbers:

- **Norm.** Since `‖g‖₂ ≥ ‖g‖∞`, an L2 criterion at the same numeric tolerance is
  slightly stricter. On the toy problem this is the whole difference between 54
  iterations (infinity norm) and 55 (L2) for the upper-bound method — verified,
  not assumed.
- **Scaling.** In the Covertype script the gradient is a **sum** over rows, so it
  carries a factor of `n = 5000`. A given numeric tolerance is therefore 5000×
  stricter there than the same number applied to a per-sample mean gradient.

This is why the reproduction prints a tolerance sweep. Earlier informal runs
reporting roughly 4 and 404 iterations correspond to a looser or per-sample
criterion; the sweep reproduces that region (4 and ~366 at `1e-2`) and shows how
the counts move from there. No constant was adjusted to recover those figures.

## 13. What this research does **not** claim

- **It is not a benchmark.** Both experiments are intercept-only models with a
  closed-form optimum. They isolate the convergence behaviour of the curvature
  model; they say nothing about wall-clock performance.
- **It does not claim a speedup for XGBoost training.** Iteration counts for a
  dense Newton solve on `K-1` coordinates do not translate into boosting rounds,
  and the exact method's per-iteration cost is higher — `O(K²)` storage and an
  `O(K³)` solve per leaf, versus `O(K)` for the diagonal.
- **It does not evaluate tree construction, histogram accumulation, or
  regularization**, all of which are where the real implementation cost lives.
- **It does not settle the `K-1 → K` gauge question.** Mapping a `K-1`
  dimensional Newton step back to a `K`-output model is a choice of gauge; raw
  and centered embeddings give identical probabilities but different L2 norms,
  so the choice interacts with regularization and must be made alongside the
  regularized leaf objective, not here. It *was* subsequently settled, in
  `design_exact_leaf_objective.md` and in the implementation, by penalising the
  centered `K`-output leaf — `R = lambda (I - 11^T / K)` — which is what makes the
  fitted model invariant to the choice of reference class. That derivation is not
  a result of these experiments and should not be attributed to them.
- **It does not claim the upstream experiment is wrong.** §9 documents an
  internal inconsistency in the objective expression; the conclusion drawn from
  it about dense-vs-diagonal curvature is unaffected.

## 14. Files

| File | Role |
|---|---|
| `multinomial.py` | Shared primitives only: stable softmax, reference-class map, finite differences, Newton driver |
| `exact_multiclass_hessian.py` | Canonical mathematical validation (numpy only) |
| `reproduce_issue_12278.py` | Faithful Covertype reproduction (numpy + scikit-learn) |
| `validate_xgboost_exact.py` | Validation against the **built XGBoost**, not against numpy: trains `multi_hessian=exact` and checks the production path |
| `design_exact_leaf_objective.md` | Derivation of the regularized leaf objective and the centered gauge that the implementation uses |
| `EXACT_HESSIAN_STATUS.md` | Authoritative engineering status: support matrix, numerical policy, environment, performance baseline |
| `README.md` | This document |

The three roles are deliberately distinct and must not be conflated:

| Role | File | What a result there means |
|---|---|---|
| Mathematical experiment | `exact_multiclass_hessian.py` | A property of the mathematics, in an intercept-only setting with a closed-form optimum |
| Faithful upstream reproduction | `reproduce_issue_12278.py` | What the issue's own setup produces when run as written, caveats in §9 included |
| Production validation | `validate_xgboost_exact.py` | What the compiled XGBoost in this tree actually does |

Only the third says anything about XGBoost. The first two are Newton-solver
experiments on a fixed design matrix and say nothing about boosting.
