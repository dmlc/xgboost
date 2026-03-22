# Worker Report: issue #11845 (`initial`)

## Status
- Completed

## Context loaded
- `AGENTS.md`
- `.bot/retrieved_memory.md`
- `.bot/memory/worker_retrieved_memory.md`
- `.bot/coordination/worker_handoff.md`
- `.bot/coordination/coordinator_plan.md`

## Outcome selected
- **Outcome B: Narrow, coherent code patch**

## What was implemented
### Parameter
- Added `depth_decay` to `TrainParam` in `src/tree/param.h`.
  - default: `1.0`
  - range: `[0, 1]`
  - semantics documented as depth-wise effective LR decay.
- Added helpers:
  - `DepthScale(depth)`
  - `LearningRate(depth)`
- Added equivalent helpers/field to `GPUTrainingParam` for CUDA paths.

### Semantics (explicit)
- Implemented **leaf-output scaling only** using 0-based internal node depth:
  - `effective_lr(depth) = learning_rate * depth_decay^depth`
- Did **not** implement split-search-aware depth regularization (gain/objective perturbation by depth).

### Updater coherence
Applied depth-aware LR consistently where leaf values are written:

- Exact:
  - `src/tree/updater_colmaker.cc`
- CPU hist/approx:
  - `src/tree/hist/evaluate_splits.h`
  - `src/tree/updater_quantile_hist.cc`
  - `src/tree/updater_approx.cc`
- GPU hist/approx:
  - `src/tree/updater_gpu_hist.cu`
  - `src/tree/gpu_hist/multi_evaluate_splits.cu`
  - `src/tree/updater_gpu_hist.cuh`
  - `src/tree/updater_gpu_common.cuh`
- Related leaf rewrite paths:
  - `src/tree/updater_refresh.cc`
  - `src/tree/updater_prune.cc`

### Tests added
- `tests/cpp/tree/test_tree_stat.cc`
  - Added `TestDepthDecay` suite.
  - Verifies `depth_decay=1.0` default invariance.
  - Verifies `depth_decay<1` shrinks deeper leaf outputs by depth.
  - Verifies split structure remains unchanged vs baseline in this leaf-scaling implementation.
  - Mirrors updater coverage style of existing `TestSplitWithEta` tests (CPU + CUDA guards).

## Validation
- Targeted lint on touched files:
  - `pre-commit run --files ... --show-diff-on-failure`
  - Result: Pass
- Full repo lint attempt:
  - `pre-commit run --all-files --show-diff-on-failure`
  - Result: Fails on pre-existing repo-wide issues outside patch scope; unrelated auto-edits from hooks were reverted.
- Targeted C++ execution:
  - Blocked in this environment (`cmake` missing, no `build/testxgboost` present).

## Artifact outputs
- `.bot/pr_body.md`
- `.bot/validation.md`
- `.bot/coordination/worker_report.md`

## Constraints respected
- No PRs opened
- No pushes
- No dangerous commands used
