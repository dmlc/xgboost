# Worker Handoff — PR #12120 Feedback Remediation

## Task
Remediate PR **#12120** for issue **#11845** with the **smallest possible scoped fix**.

This is a feedback/CI pass, not a fresh feature implementation.

## Read first
Before editing, read:
- `AGENTS.md`
- `.bot/maintainer_feedback.md`
- `.bot/retrieved_memory.md`
- `.bot/coordination/coordinator_plan_feedback.md`

## Hard guardrails
- **Do not open PRs**
- **Do not push**
- **Do not broaden scope**
- **Do not perform repo-wide cleanup as product work**
- **Do not redesign the `depth_decay` feature in this pass**

## Actual blockers to fix
### 1) Remove accidental assistant/workspace files from the PR diff
The live PR diff currently includes many files that should not be upstreamed:
- `.bot/**`
- `.openclaw/**`
- `AGENTS.md`
- `HEARTBEAT.md`
- `IDENTITY.md`
- `SOUL.md`
- `TOOLS.md`
- `USER.md`

These are directly causing the failing **Run pre-commit** job because the hooks rewrite those text/json files.

### 2) Fix the compile error in `tests/cpp/tree/test_tree_stat.cc`
Representative CI failure:

```text
/__w/xgboost/xgboost/tests/cpp/tree/test_tree_stat.cc:222:7: error: 'CHECK_NEAR' was not declared in this scope; did you mean 'CHECK_NE'?
```

This occurs in:
- `xgboost::TestDepthDecay::CheckLeaf(float, float, float)`

This is a real product/test issue and blocks multiple build jobs.

## Intended final product files in the PR diff
After cleanup, the branch diff should contain only the intended `depth_decay` patch files:
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

No assistant/workspace files should remain in the diff.

## What to do

### 1) Inspect current branch diff
Run:
```bash
git diff --name-only origin/master...HEAD
```

If assistant/workspace files are present, remove them from the branch diff before touching product code.

### 2) Remove accidental files from the diff
For each accidental path:
- if it exists on `origin/master`, restore it from base
- otherwise remove it from the index so it remains only local/untracked

Suggested shell pattern:
```bash
for p in \
  .bot/coordination/acceptance_criteria.md \
  .bot/coordination/coordinator_plan.md \
  .bot/coordination/worker_handoff.md \
  .bot/coordination/worker_report.md \
  .bot/memory/coordinator_retrieved_memory.md \
  .bot/memory/worker_retrieved_memory.md \
  .bot/pr_body.md \
  .bot/retrieved_memory.md \
  .bot/validation.md \
  .bot/validation_matrix.json \
  .openclaw/workspace-state.json \
  AGENTS.md HEARTBEAT.md IDENTITY.md SOUL.md TOOLS.md USER.md
 do
  if git cat-file -e origin/master:"$p" 2>/dev/null; then
    git restore --source origin/master -- "$p"
  else
    git rm --cached -- "$p"
  fi
done
```

If some of those files remain untracked locally afterward, that is fine.
The goal is that they disappear from the branch diff.

### 3) Fix the compile error in `tests/cpp/tree/test_tree_stat.cc`
Find the helper using:
```cpp
CHECK_NEAR(expected, actual, tol);
```

Replace it with a valid gtest assertion macro.
Preferred minimal fix:
```cpp
ASSERT_NEAR(expected, actual, tol);
```

`EXPECT_NEAR` is also acceptable if better aligned with surrounding style, but keep the change minimal.

Do **not** refactor the whole test suite.
Do **not** redesign the feature implementation in this pass.

### 4) Re-check final diff
Run:
```bash
git diff --name-only origin/master...HEAD
```

Expected output: only the intended source/test files listed above.

### 5) Run targeted validation
#### Required
At minimum run a focused pre-commit pass on the intended changed files:
```bash
pre-commit run --files \
  src/tree/gpu_hist/multi_evaluate_splits.cu \
  src/tree/hist/evaluate_splits.h \
  src/tree/param.h \
  src/tree/updater_approx.cc \
  src/tree/updater_colmaker.cc \
  src/tree/updater_gpu_common.cuh \
  src/tree/updater_gpu_hist.cu \
  src/tree/updater_gpu_hist.cuh \
  src/tree/updater_prune.cc \
  src/tree/updater_quantile_hist.cc \
  src/tree/updater_refresh.cc \
  tests/cpp/tree/test_tree_stat.cc \
  --show-diff-on-failure
```

#### Strongly recommended if build environment exists
Run the narrowest feasible gtest/build validation around `test_tree_stat.cc`, for example:
```bash
./build/testxgboost --gtest_filter=TestDepthDecay.*:TestSplitWithEta.*
```

If `build/testxgboost` is not available locally, say so clearly instead of inventing a run.

## What NOT to do
- Do not leave `.bot/**`, `.openclaw/**`, or persona files in the PR diff
- Do not run broad repo-wide cleanup just because hooks complain about unrelated files
- Do not redesign `depth_decay`
- Do not add docs/tutorials in this pass
- Do not touch unrelated files outside the intended patch unless a reproduced failure proves they are needed

## Done criteria
You are done when:
1. accidental assistant/workspace files are no longer in the branch diff
2. `CHECK_NEAR` is replaced with a valid available test assertion macro
3. targeted validation is run and passes at the strongest feasible level
4. the PR diff contains only intended `depth_decay` source/test files
