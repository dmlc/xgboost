# Exact multinomial Hessian — engineering status

Authoritative record of what `multi_hessian=exact` does, what it refuses, how it
behaves numerically, and what has not been verified. Where this document and the
code disagree, the code is the defect — every row below was produced by running
the thing it describes, not by reading the source.

Last verified against the working tree on 2026-09-23.

---

## 1. What the feature is

`multi_hessian` selects the curvature model for multi-class objectives.

| Value | Curvature | Leaf value |
|---|---|---|
| `diagonal` (default) | One scalar second-order statistic per class; the coupling between classes is ignored | `K` independent scalar Newton steps |
| `exact` | The full multinomial Hessian `H = diag(p) - p pᵀ` over `K-1` free logits | One joint `(K-1)`-dimensional Newton solve per leaf |

`diagonal` is the long-standing behaviour and is untouched. Exact mode is opt-in,
and every configuration it cannot honour is **rejected**, never silently
downgraded — a silent downgrade would hand back a different model than the one
that was asked for.

`multi_hessian` is a **training** parameter. It is not written to the model, and a
model trained with it loads into a default learner and predicts identically.

---

## 2. Support matrix

Status vocabulary, used strictly:

- **Supported** — implemented, and covered by a test that would fail if it broke.
- **Rejected** — refused with an actionable error before any training happens.
- **Unverified** — not known to be broken, but nothing tests it. Do not rely on it.
- **Deferred** — deliberately out of scope.

### 2.1 Supported

Every row was exercised end-to-end through the Python API in the M12 regression
matrix (36 cells, 0 unexpected).

| Capability | Detail |
|---|---|
| `multi:softprob`, `multi:softmax` | `K = 2, 3, 7, 10` |
| Multiple boosting rounds | 1–50 verified; loss monotone in the round count |
| Sample weights | Including zero weights |
| `subsample` | Uniform only; the sidecar replays the identical row selection |
| `colsample_bytree` / `bylevel` / `bynode` | |
| Sparse input and missing values | Both enumeration directions; default direction selected per split |
| `QuantileDMatrix` | |
| `interaction_constraints` | Applied before enumeration, consistent under distribution |
| `base_score`, `boost_from_average` | |
| Early stopping | |
| Continued training | Verified equal to training straight through |
| Save / load | Bit-identical predictions; no sidecar and no `multi_hessian` in the model |
| Distributed CPU | Histogram all-reduce; 2, 3, 4 workers; striped and contiguous partitions |
| External memory / multi-page | Verified against **ground truth**: with 4 real `GHistIndexMatrix` pages the root node total equals the sum of every row's statistics, and the full updater path produces valid centred leaves (`MultiPageNodeTotalMatchesGroundTruth`). Note the quantile *sketch* differs between a one-page and a four-page matrix — for the diagonal path too — so trees are not identical across paging; that is upstream sketcher behaviour, not exact mode. |

### 2.2 Rejected

Each fires before any tree is built. The "Raised by" column matters: only the
first group is exact-mode-specific code.

| Configuration | Raised by | Message names the alternative |
|---|---|---|
| Categorical features | `exact_builder.h` | yes — `multi_hessian=diagonal` |
| `monotone_constraints` | `exact_evaluator.h` | yes |
| `reg_alpha != 0` | `exact_evaluator.h` | yes |
| `max_delta_step != 0` | `exact_evaluator.h` | yes |
| `sampling_method=gradient_based` | `sampler.cc` | yes — uniform |
| Reduced value gradient | `updater_quantile_hist.cc` | yes |
| Non-multinomial objective | `learner.cc` | yes — lists the supported objectives |
| Custom objective / `XGBoosterTrainOneIter` | `learner.cc` (`BoostOneIter`) | yes — that path supplies the gradient directly and bypasses the sidecar producer, so exact mode would otherwise downgrade to diagonal in silence |
| `device=cuda` | `learner.cc` | yes — reads the *requested* device, so the answer is the same on a machine with no GPU |
| `multi_strategy=one_output_per_tree` | `learner.cc` | yes |
| `num_class < 2` | `learner.cc` | yes |
| `tree_method=approx` / `exact` | `gbm/gbtree.cc` — **upstream's** pre-existing multi-output-tree constraint, inherited because exact mode requires `multi_output_tree` | yes — names `hist` |

### 2.3 Unverified — do not claim

| Item | Why it is listed here |
|---|---|
| Multi-host distributed | Validated only in-process, via the test tracker on one machine. Nothing has been run across real hosts or a real network. |
| `K > 20` | The packed layout and solver are dimension-generic and tested to `K = 20` in unit tests, `K = 10` end-to-end. Beyond that, the `O(K³)` solve cost is the practical limit, not a correctness claim. |

### 2.4 Deferred

| Item | Reason |
|---|---|
| GPU | The GPU histogram is fixed-point (`GradientPairInt64` + `GradientQuantiser`), not double. A port is a quantisation design problem, not a translation. Explicitly rejected rather than approximated. |
| `reg_alpha` under exact curvature | L1 soft thresholding has no closed form for a coupled system. |
| Categorical splits | Needs a partition search over category sets against a vector-valued gain. |

---

## 2.5 Public API and compatibility

XGBoost makes **no C++ ABI promise**, and `include/xgboost/task.h` says so explicitly of
`ObjInfo` ("should not be serialized since it can be recovered from objective function,
hence it doesn't need to be stable"). Downstream C++ users are expected to rebuild against
the headers they link. Nothing here claims otherwise.

Within that, the additions are arranged to minimise disruption:

| Change | Placement | Why it matters |
|---|---|---|
| `ObjFunction::GetGradientAndExactHessian` | **Last** in the virtual interface | Every pre-existing virtual keeps its original vtable slot index. `plugin/example/custom_obj.cc` shows subclassing `ObjFunction` is a supported extension point; inserting earlier would shift later slots, and a stale object file would then misdispatch **silently** rather than fail to link. That failure actually occurred during this project's development, which is why the placement is deliberate rather than incidental. |
| `ObjInfo::exact_hess` | Appended after `const_hess`, with a default member initialiser | The 1- and 2-argument constructors are untouched and both leave `exact_hess == false`, so no existing construction can accidentally advertise exact support. |
| `ExactHessian` | New type in `gradient.h` | Needed publicly because `GradientContainer` holds it and the new virtual takes it. It carries layout only — the packed *index* convention stays in the internal `common/exact_multinomial/packed_stats.h`, which forwards to `ExactHessian::PackedSize` so the two cannot disagree. |
| `MultiHessian` enum | `learner.h`, beside `MultiStrategy` | Mirrors the existing parameter enum pattern. |

Source compatibility is preserved: no existing signature, default or behaviour changes.

---

## 3. Numerical policy

### 3.1 Precision

| Stage | Type | Reason |
|---|---|---|
| Per-row sidecar transport | `float` | Matches the precision of the objective's own probability computation; nothing is lost that was not already lost |
| Histogram bin accumulation | `double` | Mirrors the existing `GradientPair` → `GradientPairPrecise` split |
| Leaf solve | `double` | |

Measured transport error over 4096 rows, worst relative:
`K=3 → 1.25e-09`, `K=7 → 2.25e-09`, `K=10 → 5.34e-09`. Accumulating in double
keeps the error at a single row's float rounding rather than growing it with the
row count.

### 3.2 The solve

`(H + R) w = -G`, factored as `L D Lᵀ` (square-root-free Cholesky) over the same
packed lower triangle the statistics use. The inverse is never formed.

Gain is `Gᵀ (H + R)⁻¹ G`, which is XGBoost's convention (twice the loss
reduction), and it falls out of the forward substitution at no extra cost:
with `z = L⁻¹G` it is `Σ zᵢ²/dᵢ`. Split enumeration therefore stops after the
forward pass.

### 3.3 Invalid curvature

A non-positive or non-finite pivot means `H + R` is not positive definite. The
solve **refuses**: it returns false and leaves its output buffers untouched, so
nothing partial and nothing non-finite escapes. The builder then writes a zero
leaf, which is the same policy the scalar path applies to `sum_hess <= 0`. This
is a refusal, not a recovery strategy, and is not presented as one.

### 3.4 Regularization and the gauge

`R = λ (I_d − 11ᵀ/K)` with `d = K−1`: the penalty on the *centered* `K`-output
leaf. Its eigenvalues are `λ` and `λ/K`, both positive, so `H + R` is positive
definite for `λ > 0` even where `H` alone is only semi-definite.

The centered form is used because it makes the fitted model invariant to which
class is the reference. `R = λI` does not: the measured drift under relabeling is
`3.05e-01` for a representative distribution, which is recorded as a test
(`RawGaugeIsNotReferenceClassInvariant`) so the centered form cannot quietly stop
being load-bearing.

### 3.5 `min_child_weight` — deliberately NOT the same number

Compared against `trace(H_full)/K`, recovered exactly from the stored free block
via `trace(H_full) = 2 · Σ(packed triangle)`, and independent of the reference
class.

This matches the **shape** of the existing vector-leaf gate
(`split_evaluator.h:234`, `IsValidSplit(param, left_hess/k, right_hess/k)`) but
**not its value**, and that is deliberate. The diagonal multiclass objective does
not store `p_k(1−p_k)`; it stores the absolute-residual pseudo-Hessian
`|p_k − y_k|·w`. So:

```
diagonal:  mean_k |p_k − y_k| · w
exact:     w · (1 − Σ_k p_k²) / K
```

At a uniform prediction the diagonal quantity is **exactly 2×** the exact one for
every `K`; the ratio falls towards 1 as predictions sharpen. End to end this is
4–10% fewer nodes in exact mode at the same `min_child_weight`.

Exact mode gates on true curvature because that is the matrix its Newton system
inverts — gating on the pseudo-Hessian would admit nodes whose real curvature
cannot support a leaf value. Rescaling either side to force agreement would make
the gate mean something other than the matrix being factorized. The difference is
pinned by `MinChildWeightGateDiffersFromDiagonalByDesign` and documented in
`doc/parameter.rst`, so code and docs cannot drift.

### 3.6 Missing-value enumeration

The backward enumeration pass is skipped when the dataset has no missing values,
read from `DMatrix::IsDense()`. This is a property of the data, not a tolerance.
An earlier version compared floating-point sums against a relative epsilon; that
made a split decision depend on a magic constant, and a feature with a genuinely
tiny missing mass would have silently lost its default-left candidate.

The dataset-level predicate is used rather than the per-page one because a
dataset can span several pages and only some need be dense.

### 3.7 Thresholds

The exact implementation introduces **no epsilon of its own**. Two predicates
exist in the whole path:

| Predicate | Where | Nature |
|---|---|---|
| `pivot > 0 && isfinite(pivot)` | `leaf_solver.h` | The definition of positive-definiteness, not a tolerance |
| `!DMatrix::IsDense()` | `exact_builder.h` | Structural, exact |

Two thresholds are **inherited** from upstream and apply identically to the
scalar path: `Driver::Push`'s `loss_chg > kRtEps` expansion gate, and the
`1e-16f` floor on the diagonal pseudo-Hessian in `multiclass_obj.cc` (which the
exact path does not read — it is preserved only so the *gradient* stays
bit-identical).

---

## 3.9 Retracted benchmark work

An earlier benchmark and all conclusions drawn from it were **retracted**, for two
independent reasons:

1. **Test-set selection.** The script chose the winning configuration with
   `r["test_mlogloss"] < best["test_mlogloss"]`, and its time-to-target metric measured
   reachability on test predictions against a test-derived target. Every headline number it
   produced was contaminated by test-set peeking.
2. **Concurrent process contamination.** A second benchmark process believed dead was running
   the same workload simultaneously; the identical fit was recorded as 1.7s and 1.2s by the
   two processes. Those timings were discarded, not rescaled -- correcting them by an
   inferred contention factor would have been a fabricated measurement.

The scripts and log are quarantined under `research/retracted/` with a README explaining
why. Nothing from them appears in any table. `benchmark_v2.py` replaces them under a protocol
that passed an eight-point source audit before it was run.

A separate observation from that episode is worth keeping: the quality metrics were
bit-identical across the two concurrent processes, which independently confirms that training
is deterministic and that the contamination was purely a timing effect.

---

## 4. Performance baseline

**Environment-dependent baseline, not a universal benchmark.** These numbers
describe one machine, one dataset shape, and one build. They are recorded so that
a future change can be compared against them, not to characterise the feature.

Measured 2026-09-22, Windows 11, Intel64 Family 6 Model 183 (Raptor Lake),
OpenMP on, Release build; 20 000 rows × 32 columns, 20 rounds, `max_depth=6`;
best of 3 runs after a warm-up.

| K | `diagonal` | `exact` | ratio |
|---:|---:|---:|---:|
| 3 | 0.441 s | 1.992 s | 4.5× |
| 7 | 1.000 s | 7.348 s | 7.4× |
| 10 | 1.015 s | 13.493 s | 13.3× |

This is the cost of exactness, and it is expected: storage is `O(K²)` per bin
against `O(K)`, and the leaf solve is `O(K³)` against `O(K)`. Exact mode is not
faster than diagonal mode and is not offered as such — it computes a different,
more accurate Newton step.

Memory, per histogram bin, against the **multi-target** baseline (not the scalar
one): exact stores `(K-1) + (K-1)K/2` doubles versus `2K`; at `K = 7` that is
1.93×.

The M10 work reduced `UpdateTree` wall time by 2.87× relative to the first
working implementation, mainly by hoisting the parent-gain factorization out of
the enumeration loop and by adding a gain-only solve path. Those are improvements
against an earlier state of this branch, not against upstream XGBoost.

---

## 5. Reproducing the validation

### 5.1 Build

```
cmake -S . -B build -DGOOGLE_TEST=ON
cmake --build build --target xgboost testxgboost --config Release
```

On Windows with MSBuild, pass `-- /m` for parallel builds. **After changing any
header that affects a vtable** (`objective.h`, `gradient.h`, `task.h`), delete
`build/src/objxgboost.dir` and `build/testxgboost.dir` before rebuilding: MSBuild
has been observed to miss that dependency, and mixed-vintage objects dispatch a
virtual call into the wrong slot, which presents as an unrelated objective
claiming exact-Hessian support.

### 5.2 Tests

```
build/testxgboost.exe                                   # full C++ suite
build/testxgboost.exe --gtest_filter='Exact*:*Multiclass*'   # exact mode only
build/testxgboost.exe --gtest_filter='ExactDistributed.*'    # distributed

# Order independence: every exact test is seeded, and none may depend on run order.
build/testxgboost.exe --gtest_filter='Exact*:*Multiclass*'     --gtest_shuffle --gtest_random_seed=1 --gtest_repeat=2

PYTHONPATH=python-package python -m pytest \
    tests/python/test_exact_multinomial.py \
    tests/python/test_exact_multinomial_compat.py \
    tests/python/test_exact_multinomial_stability.py
```

The distributed tests spawn in-process workers through the test tracker; no
external launcher is needed.

### 5.3 Research validation

```
PYTHONPATH=python-package python research/validate_xgboost_exact.py
python research/exact_multiclass_hessian.py
python research/reproduce_issue_12278.py      # needs scikit-learn
```

### 5.4 Environment requirements and known limitations

| Requirement | Status in the environment this was developed in |
|---|---|
| C++17 toolchain, CMake | present (MSVC 2022 build tools) |
| OpenMP | present |
| `numpy`, `pytest` | present |
| `pandas` | present — the categorical rejection test skips without it |
| `scikit-learn` | needed only by `reproduce_issue_12278.py` |
| `hypothesis` | **absent** — `tests/python/test_updaters.py` and `tests/python/test_multi_target.py` cannot be collected. Those suites were therefore not run; the C++ suite covers the same updaters. |
| CUDA toolkit | **absent** — no GPU code was compiled. GPU support is rejected at configuration time, which is verified, but nothing GPU-side was built or run. |
| Multiple hosts | **absent** — distributed testing is in-process on one machine only. |

None of these gaps are worked around in the tests. Where something could not be
verified, it is listed as unverified in §2.3 rather than assumed to work.

---

## 5.5 What belongs in the upstream PR

`research/` is development material, not production code. Shipping all of it would enlarge
the patch a reviewer has to read without helping them judge the change. The split:

**Include in the PR** — these are evidence a reviewer needs to check the claims:

| File | Why |
|---|---|
| `benchmark_v2.py` | The PR's central claim is a cost/benefit trade-off. A reviewer must be able to re-run it. Selection, early stopping and all targets use validation; the test split is evaluated once, after selection; the search phase never constructs a test DMatrix, so test leakage is structurally impossible. |
| `summarize_v2.py` | Derives every table from the raw JSONL alone, re-deriving the selection rather than trusting a stored label. No number is typed by hand. |
| `benchmark_k_scaling.py` | Backs the documented `O(K^2)` / `O(K^3)` cost statement with measurements. |
| `EXACT_HESSIAN_STATUS.md` | The support/rejection/unverified matrix, condensed into `doc/` for users but kept in full here for reviewers. |

**Keep out of the PR** — link from the issue instead:

| File | Why |
|---|---|
| `multinomial.py`, `exact_multiclass_hessian.py` | Standalone numpy Newton experiments. They justify the mathematics but are not XGBoost code and would need their own maintenance story. |
| `reproduce_issue_12278.py` | A faithful reproduction of the issue's own setup, including the objective-expression inconsistency documented in `README.md` §9. Valuable in the issue thread; out of place in the tree. |
| `validate_xgboost_exact.py` | Useful during development; its content is now covered by the committed C++ and Python tests. |
| `design_exact_leaf_objective.md`, `README.md` | Derivations. The load-bearing parts are already in the header comments where the code is. |

If maintainers ask for the research directory, it can be added; the default is the smaller
patch.

---

## 6. Invariants the tests hold

Each of these would fail loudly if broken; they are the contract, not
incidental behaviour.

| Invariant | Test |
|---|---|
| Exact-mode gradients are bit-identical to diagonal-mode gradients | golden bit patterns in `test_multiclass_exact_obj.cc` |
| `Gain()` and `Solve()` agree, and both equal `-2ΔL` | `GainMatchesSolveDerivedGain` |
| The node total counts each row exactly once, for any feature count and sparsity | `NodeTotalCountsEachRowOnce` |
| The centered gauge is reference-class invariant; the raw gauge is not | `ReferenceClassInvarianceUnderStress`, `RawGaugeIsNotReferenceClassInvariant` |
| Leaf outputs stay centered over 50 rounds | `test_leaf_outputs_stay_centered_over_fifty_rounds` |
| Gradient and Hessian sampling select the same rows | `test_exact_sampling.cc` |
| The sidecar always matches the gradient it accompanies | `ModeTransitionsKeepTheSidecarConsistent`, `RowCountChangeResizesTheSidecar` |
| Skipping the backward pass on dense data changes nothing; on sparse data it matters | `DenseMatrixMakesTheBackwardPassRedundant`, `SparseMatrixNeedsTheBackwardPass` |
| Distributed reduction reproduces single-process statistics | `RootHistogramReducesToSingleProcessTotal` |
| `multi_hessian` never reaches the model | `ModeIsTrainingConfigNotModelState` |
