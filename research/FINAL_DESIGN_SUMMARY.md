# Exact multinomial second-order curvature for multi-output trees — final design summary

Status: ready for a fresh upstream PR, superseding the closed `#12619`. Addresses
[issue #12278](https://github.com/dmlc/xgboost/issues/12278). Branch
`feat/exact-multinomial-hessian`, rebased onto current `origin/master`.

## 1. What this adds

For `multi:softprob` / `multi:softmax` with `multi_strategy=multi_output_tree`, XGBoost's
existing multi-output trees use a **diagonal** curvature approximation: each class's second
derivative is treated independently, discarding the multinomial objective's cross-class
Hessian terms. The true per-sample Hessian, `H_ij = p_i(delta_ij - p_j)`, is dense — coupling
every pair of classes. This PR adds `multi_hessian=exact`, an opt-in path that carries that
full dense Hessian through histogram building, split evaluation and leaf solving, so the tree
is grown against the real second-order model instead of a diagonal surrogate. `diagonal`
remains the default; nothing about default behavior changes.

## 2. Design

- **Transport.** The exact Hessian travels as a packed, symmetric sidecar
  (`GradientContainer::exact_hessian`, `include/xgboost/gradient.h`) alongside the ordinary
  gradient pair, not inside `GradientPair` itself — the scalar/diagonal path's data layout and
  hot paths are untouched.
- **Reduction.** The solve happens in `K-1` free coordinates (one class is the reference), via
  a **centered** regularizer `R = lambda*(I - 11^T/K)` rather than the naively simpler
  `R = lambda*I`. The centered form is what makes the fitted model invariant to which class is
  picked as the reference — tested directly
  (`ExactEvaluator.CenteredGaugeIsReferenceClassInvariant`,
  `ExactNumerics.ReferenceClassInvarianceUnderStress`, with
  `ExactNumerics.RawGaugeIsNotReferenceClassInvariant` as the negative control proving the
  property isn't vacuous).
- **Histogram.** Packed symmetric per-bin statistics (`src/common/exact_multinomial/packed_stats.h`),
  `O(K^2)` storage per bin versus `O(K)` for the diagonal path — an inherent cost of carrying
  cross-class terms, not an implementation inefficiency.
- **Leaf solve.** A square-root-free `LDL^T` Cholesky factorization
  (`src/common/exact_multinomial/leaf_solver.h`) solves `(H+R)w* = -G` without ever forming an
  inverse; gain falls out of the same factorization at no extra cost
  (`Gain = sum_i z_i^2/d_i`). Non-positive-definite systems return `false`, mapped to XGBoost's
  existing "weight 0, gain 0" degenerate-curvature policy — not a rejected candidate, not a
  regularized fudge.
- **Builder.** `ExactMultiTargetHistBuilder` (`src/tree/hist/exact_builder.h`) is a separate
  builder class from the diagonal multi-target path, instantiated only when exact mode is
  requested, sharing the generic driver/expand-entry machinery but not the diagonal path's
  histogram code.
- **Scope, enforced not just documented.** CPU/`hist`-only, rejects gradient-based sampling,
  categorical/monotone constraints, `reg_alpha`, `max_delta_step`, and custom objectives —
  each rejection is a `CHECK` with an actionable message, not a silent fallback
  (`ExactMultinomialLearner.RejectsUnsupportedConfigurations`,
  `.RejectsCategoricalFeatures`, `.ErrorMessagesAreActionable`).
- **Not in the model.** `multi_hessian` is a training-only parameter (excluded from
  `SaveModel`, the same way `multi_strategy` already is); the Hessian sidecar is
  training-time-only state, never serialized
  (`ExactMultinomialLearner.SaveLoadRoundTrip` greps the saved model for both leaking in).

## 3. What this recovery effort changed, and why

The prior PR (`#12619`) was closed same-day after an automated review surfaced real gaps.
Issue `#12278`'s own thread (comments from `RAMitchell`, `trivialfis`) had already flagged the
two underlying risks months earlier: numerical fragility of a dense solve, and the need for a
compelling, honestly-reported cost/benefit case. This recovery addressed both, plus every
concrete review finding, in reviewable chunks:

1. **Rebase onto current `origin/master`.** Clean except one real API migration: upstream's
   `c2ca8c99a` ("Fix and unify base weight handling") replaced `ExpandBatch::Push`/
   `MultiTargetTree::SetLeaves` with `ExpandData`/`FinalizeLeaves`, moving learning-rate
   scaling out of `Expand()` into leaf finalization. Ported `exact_builder.h` to match
   (`f0a8df131`), verified eta is applied exactly once end-to-end, and added a dedicated
   regression (`ExactBuilder.LearningRateAppliedExactlyOnce`) since the existing
   centering/finiteness checks couldn't have caught a double- or zero-application.

2. **Distributed zero-row worker correctness** (the review's most important functional
   finding). A distributed worker can legitimately own zero rows while `multi_hessian=exact`
   is still requested; mode selection used to infer from `HasExactHessian()` (does the sidecar
   have data), so an empty worker would silently fall back to the diagonal builder while its
   peers used the exact one — a collective-sequence mismatch. Fixed by tracking the request
   explicitly (`GradientContainer::HasExactHessianRequested()`), independent of data presence
   (`6d89ea4dc`). Added the regression the closed PR never got to write: a 2-worker test, one
   rank empty, asserting both take the same exact path for two full rounds
   (`ExactMultinomialLearner.DistributedExactWithEmptyWorker`, `54ccbc2b8`). Fixing it then
   surfaced that several existing low-level builder tests constructed a `GradientContainer` by
   hand without setting the new flag, silently exercising the diagonal path while believing
   they tested exact mode — fixed (`a4c5a51dd`), and a second, unrelated instance of the same
   stale data-presence check was found and aligned during the Phase 3A audit
   (`src/gbm/gbtree.cc`, `59edda650`).

3. **K=2 regularizer.** Not a production bug — `exact_builder.h`/`exact_evaluator.h` already
   used the centered gauge exclusively — but the design doc's and a test's "K=2 reduces to the
   scalar path" cross-check used the *raw* gauge (`R=lambda`), which production never selects.
   The real, shipped K=2 result is `Gain = G^2/(H+lambda/2)`, since
   `Centered(lambda,2).Diagonal() == lambda/2`. Fixed the claim and added the test that
   actually validates the shipped default
   (`ExactEvaluator.ReducesToScalarGainForTwoClassesUnderCenteredRegularization`, `46fa6ab11`).

4. **Histogram cache budget.** `ExactHistCollection` already had the scalar path's
   `CanHost`/`Clear`/`HasExceeded` cache-eviction primitives, even unit-tested in isolation —
   but `ExactMultiTargetHistBuilder` never called them, so `max_cached_hist_node` was silently
   inert for exact mode regardless of tree depth. Ported the scalar path's
   `HistogramBuilder::AddHistRows` eviction-and-rebuild logic (`c70070804`); added a
   correctness regression training the same problem under a near-unusable cache budget versus
   an unbounded one and requiring identical results, since eviction changes *how* a histogram
   is computed, never what it computes (`ExactBuilder.HonoursMaxCachedHistNode`).

5. **Thread scratch buffer.** The per-thread histogram accumulation buffer zeroed its full
   `n_threads * stride` extent with a single thread, and reduced all threads' blocks back into
   the final histogram with a single serial loop — both structurally different from (slower
   than) the scalar path's lazy-zero, skip-single-thread-nodes, parallel-reduce design.
   Investigation found the original review's 350MB/node estimate assumed ~100 features at
   `max_bin=256`; a realistic 8-16 feature dataset is closer to 7-54MB/node — real, but
   overstated 6-50x. Rather than replicate the scalar path's larger touched-bin-tracking
   redesign, parallelized both the zero and the reduction with the same thread count already
   splitting row accumulation (`731dec3a7`) — a minimal, upstream-consistent fix matched to the
   measured scale.

6. **PR scope.** `research/EXACT_HESSIAN_STATUS.md` already specified, before this recovery
   touched it, which research files belong in the tree versus the issue thread. Reconciled the
   tree with that plan: kept `benchmark_v2.py` (authoritative benchmark),
   `summarize_v2.py`, `benchmark_k_scaling.py`, this status doc, and a new small sanity sweep
   (`benchmark_phase3d_evidence.py`); removed four standalone validation scripts and two
   derivation docs whose content is now covered by the committed test suite or lives in header
   comments next to the code (`8ca0f5471`).

## 4. Evidence

- **C++ suite**: `build/testxgboost.exe --gtest_filter='*Exact*'` — 132 tests, all passing,
  spanning numerics (finite-difference cross-checks, reference-class invariance, degenerate
  curvature), the solver (closed-form 2x2, near-singular stability), histogram construction
  and distributed reduction (bit-identical across worker partitions), split evaluation, sampling,
  serialization across every model format (JSON, UBJSON, legacy binary), and the full learner.
- **Phase 3D sanity sweep** (`research/benchmark_phase3d_evidence.py`, small/bounded,
  illustrative not rigorous): exact/diagonal wall-time ratio grows with K (1.3x at K=2 to 2.6x
  at K=10) and with `max_bin`, consistent with the documented `O(K^2)` per-bin cost, not a
  structural blowup; the ratio *shrinks* from 3.9x at 1 thread to 3.0x at 4 threads, consistent
  with the thread-buffer parallelization actually paying off; a tiny `max_cached_hist_node`
  budget grows peak working set by ~1.6GB on a deliberately wide/deep synthetic problem versus
  ~4.1GB unbounded, consistent with the cache fix actually bounding memory.
- **`research/benchmark_v2.py`** (the authoritative, validation-only, multi-seed protocol)
  remains the source for any dataset-level accuracy/convergence claim; this PR makes none —
  every result is dataset- and configuration-dependent by design, reported, not asserted.

## 5. Known limitations

- CPU-only; no GPU implementation. `device=cuda` with `multi_hessian=exact` is rejected with
  an actionable error, not silently run on CPU or silently ignored.
- No cap on `num_class`; cost grows as `O(K^2)` per bin and `O(K^3)` per leaf solve by
  construction. Documented, not hidden.
- Incompatible with gradient-based sampling, categorical splits, monotone constraints,
  `reg_alpha`, `max_delta_step`, and custom objectives — each enforced, not just documented.
- The Phase 3D sweep is illustrative (small datasets, 2-3 points per dimension, single machine,
  single seed per configuration); it is not a substitute for `benchmark_v2.py`'s rigor and
  should not be quoted as a headline number.

## 6. What's left before opening a PR

A later hardening pass addressed every item this section used to list as deferred: the
`versionadded` directive now reads `3.5.0`, the broken LaTeX `\top` rendering in
`doc/tutorials/multioutput.rst` is fixed, and the source-tree-only Python test
(`test_library_is_from_this_source_tree`) is removed rather than replaced with another
filesystem-layout assumption. That same pass also fixed two real production bugs found by
review (a histogram-cache-eviction crash under `grow_policy=lossguide` with a small
`max_cached_hist_node`, and a missing-value split candidate the forward enumeration never
reached), corrected an imprecise curvature-ratio claim, and fixed two `research/summarize_v2.py`
reporting bugs and a benchmark-timing-order confound across all three benchmark scripts.

Deliberately still open: `src/tree/hist/exact_histogram.h`'s sparse-page row loop calls
`GetGindex()` once per feature rather than iterating the page's own stored (non-missing) row
indices directly. A reusable, row-subset-aware version of the existing
`AssignColumnBinIndex` machinery (or an equivalent) would be needed to do this without either
inventing a new abstraction or hand-replicating `GHistIndexMatrix`'s compressed-index dispatch
inline -- both bigger than a cleanup-pass nit should cost. Existing sparse/dense correctness
coverage already passes either way; this is a deferred performance optimization, not a
correctness gap.

No branch has been pushed and no PR has been opened. Confirm before either.
