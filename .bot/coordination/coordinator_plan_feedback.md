# Coordinator Plan — PR #12120 Feedback Remediation

## Mission
Prepare a worker to remediate PR **#12120** (issue #11845) with the **smallest review-safe patch** that addresses the actual blockers from CI and branch hygiene.

This is a feedback/remediation pass, not a greenfield implementation plan for the whole feature.

## Inputs reviewed
- `AGENTS.md`
- `.bot/maintainer_feedback.md`
- `.bot/retrieved_memory.md`
- PR metadata/checks for `dmlc/xgboost#12120`
- failed CI logs for representative failing jobs
- current PR changed-file list via `gh pr diff 12120 --name-only`

## Maintainer feedback summary
Loaded feedback is narrow and CI-driven:
- identify failing suites
- reproduce locally where possible
- patch only root-cause issues

No loaded feedback asks for redesigning the `depth_decay` feature itself beyond fixing what is currently broken.

## Actual blockers
### 1) PR diff still contains accidental assistant/workspace files
`gh pr diff 12120 --name-only` shows the PR currently includes many files that do not belong in an upstream XGBoost PR:

- `.bot/coordination/acceptance_criteria.md`
- `.bot/coordination/coordinator_plan.md`
- `.bot/coordination/worker_handoff.md`
- `.bot/coordination/worker_report.md`
- `.bot/memory/coordinator_retrieved_memory.md`
- `.bot/memory/worker_retrieved_memory.md`
- `.bot/pr_body.md`
- `.bot/retrieved_memory.md`
- `.bot/validation.md`
- `.bot/validation_matrix.json`
- `.openclaw/workspace-state.json`
- `AGENTS.md`
- `HEARTBEAT.md`
- `IDENTITY.md`
- `SOUL.md`
- `TOOLS.md`
- `USER.md`

These directly explain the failing **Run pre-commit** job:
- trailing whitespace fixer modified `.bot/memory/coordinator_retrieved_memory.md`
- end-of-file fixer modified `.bot/validation_matrix.json`, `.bot/memory/worker_retrieved_memory.md`, `.bot/retrieved_memory.md`

**Conclusion:** these files must be removed from the PR diff.

### 2) Real compile failure in the new C++ test
Multiple C++ build jobs fail on the same root cause, e.g.:
- Build CPU (default)
- Build CUDA 13 (x86_64)
- Build CPU (nogpu, non-omp)
- likely downstream test jobs blocked by the same compile failure

Concrete compiler error from CI:

```text
/__w/xgboost/xgboost/tests/cpp/tree/test_tree_stat.cc:222:7: error: 'CHECK_NEAR' was not declared in this scope; did you mean 'CHECK_NE'?
  222 |       CHECK_NEAR(expected, actual, tol);
```

This is inside:
- `xgboost::TestDepthDecay::CheckLeaf(float, float, float)`
- file: `tests/cpp/tree/test_tree_stat.cc`

**Conclusion:** the test helper uses a non-existent assertion macro in this test environment. This is a small, concrete compile fix.

## Important split between hygiene and product code
There are two separate remediation tracks here:
1. **PR diff hygiene** — remove accidental assistant/workspace files from the branch diff
2. **Product-code/test fix** — correct the `CHECK_NEAR` test assertion usage so builds compile again

Do not mix these into broader refactors.

## Expected final PR diff
After cleanup, the PR diff should contain only intended product/test files for the `depth_decay` patch:
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

No `.bot/**`, persona, or workspace files should remain in the diff.

## Recommended concrete fixes
### A) Branch cleanup
Remove the accidental files from the diff.
- if a path exists on `origin/master`, restore it from base
- if it does not exist on `origin/master`, remove it from the index so it can remain only as a local untracked workspace file

### B) Test compile fix
In `tests/cpp/tree/test_tree_stat.cc`, replace the invalid `CHECK_NEAR` usage with a gtest assertion macro that is actually available in this file/test environment.

Most likely correct fix:
- `ASSERT_NEAR(expected, actual, tol);`
or
- `EXPECT_NEAR(expected, actual, tol);`

Prefer whichever matches surrounding test style, but keep the change minimal.

## Validation plan
### Required
1. Confirm diff cleanup:
   - `git diff --name-only origin/master...HEAD`
2. Run targeted pre-commit on the intended changed files (or at least the touched ones if cleanup is staged incrementally)
3. Run the narrowest feasible compile/test validation around `test_tree_stat.cc`

### Strongly recommended if environment supports it
If a test binary/build environment exists, run a narrow gtest filter around the new depth-decay test(s), for example:
- `./build/testxgboost --gtest_filter=TestDepthDecay.*:TestSplitWithEta.*`

If no local build environment exists, the worker should still at minimum ensure the assertion macro change is syntactically and semantically correct and note the CI evidence.

## Non-goals
- Do not redesign the `depth_decay` algorithm in this pass
- Do not expand feature scope
- Do not add docs/tutorial work
- Do not touch unrelated product code outside the intended changed files
- Do not open PRs

## Success definition
This remediation succeeds when:
1. accidental assistant/workspace files are no longer in the PR diff
2. `tests/cpp/tree/test_tree_stat.cc` no longer uses invalid `CHECK_NEAR`
3. targeted validation is run at the strongest feasible level
4. the PR is left as a focused `depth_decay` patch with only intended source/test files in the diff
