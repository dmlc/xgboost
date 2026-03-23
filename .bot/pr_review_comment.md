Addressed the CI/root-cause feedback for PR #12120 with minimal scope:

1. Removed accidental assistant/workspace files from the PR diff by adding a remediation commit that drops:
   - `.bot/**` tracked artifacts
   - `.openclaw/workspace-state.json`
   - persona/workspace files (`AGENTS.md`, `SOUL.md`, `USER.md`, etc.)
2. Fixed C++ compile failure in `tests/cpp/tree/test_tree_stat.cc`:
   - `CHECK_NEAR(expected, actual, tol);` -> `ASSERT_NEAR(expected, actual, tol);`

Validation run:
- `pre-commit run --files src/tree/gpu_hist/multi_evaluate_splits.cu src/tree/hist/evaluate_splits.h src/tree/param.h src/tree/updater_approx.cc src/tree/updater_colmaker.cc src/tree/updater_gpu_common.cuh src/tree/updater_gpu_hist.cu src/tree/updater_gpu_hist.cuh src/tree/updater_prune.cc src/tree/updater_quantile_hist.cc src/tree/updater_refresh.cc tests/cpp/tree/test_tree_stat.cc --show-diff-on-failure` (pass)
- `./build/testxgboost --gtest_filter='TestDepthDecay.*:TestSplitWithEta.*'` (not run: `build/testxgboost` unavailable in this environment)

Final `git diff --name-only origin/master...HEAD` now contains only intended depth-decay files:
- `src/tree/gpu_hist/multi_evaluate_splits.cu`
- `src/tree/hist/evaluate_splits.h`
- `src/tree/param.h`
- `src/tree/updater_approx.cc`
- `src/tree/updater_colmaker.cc`
- `src/tree/updater_gpu_common.cuh`
- `src/tree/updater_gpu_hist.cu`
- `src/tree/updater_gpu_hist.cuh`
- `src/tree/updater_prune.cc`
- `src/tree/updater_quantile_hist.cc`
- `src/tree/updater_refresh.cc`
- `tests/cpp/tree/test_tree_stat.cc`
