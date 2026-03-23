## Summary
Implement a narrow `depth_decay` hyperparameter for tree boosters, with default-compatible behavior and targeted tests.

- Added `depth_decay` to `TrainParam` (`default=1.0`, range `[0, 1]`).
- Added depth-aware learning-rate helpers:
  - `TrainParam::DepthScale(depth)`
  - `TrainParam::LearningRate(depth)`
- Mirrored the helper behavior in `GPUTrainingParam` for CUDA code paths.

## Semantics
This patch implements **leaf-output scaling only**:

- Effective per-node learning rate is:
  - `learning_rate * depth_decay^depth`
- Depth is **0-based internal node depth** (`root=0`, children at `1`, ...).
- With `depth_decay=1.0`, behavior remains equivalent to existing behavior.

This patch does **not** alter split-search objective/gain ranking to explicitly penalize deeper splits.

## Implementation Scope
Leaf-writing paths were updated coherently across major tree updaters:

- Exact: `grow_colmaker`
- CPU hist/approx: `grow_quantile_histmaker`, `grow_histmaker`
- GPU hist/approx: `grow_gpu_hist`, `grow_gpu_approx`
- Related leaf rewrite paths: `refresh`, `prune`

Also updated multi-target leaf refresh/recompute paths so depth scaling is preserved there.

## Tests
Added targeted C++ coverage in `tests/cpp/tree/test_tree_stat.cc`:

- `TestDepthDecay` verifies `depth_decay=1.0` matches baseline behavior.
- `TestDepthDecay` verifies `depth_decay<1` shrinks deeper leaf outputs by depth while preserving split structure.
- Coverage mirrors existing updater matrix used by `TestSplitWithEta` (CPU and CUDA tests under CUDA guard).

## Notes
- No PRs were opened.
