# Acceptance Criteria — PR #12120 Feedback Remediation

## Scope control
- [ ] This pass is limited to PR feedback / failing-check remediation
- [ ] No PRs are opened
- [ ] No pushes are performed
- [ ] No unrelated feature work or refactors are added

## PR diff hygiene
The branch diff against `origin/master...HEAD` must no longer include assistant/workspace files.

### Must NOT appear in the branch diff
- [ ] `.bot/**`
- [ ] `.openclaw/**`
- [ ] `AGENTS.md`
- [ ] `HEARTBEAT.md`
- [ ] `IDENTITY.md`
- [ ] `SOUL.md`
- [ ] `TOOLS.md`
- [ ] `USER.md`

## Expected final branch diff contents
- [ ] `src/tree/gpu_hist/multi_evaluate_splits.cu`
- [ ] `src/tree/hist/evaluate_splits.h`
- [ ] `src/tree/param.h`
- [ ] `src/tree/updater_approx.cc`
- [ ] `src/tree/updater_colmaker.cc`
- [ ] `src/tree/updater_gpu_common.cuh`
- [ ] `src/tree/updater_gpu_hist.cu`
- [ ] `src/tree/updater_gpu_hist.cuh`
- [ ] `src/tree/updater_prune.cc`
- [ ] `src/tree/updater_quantile_hist.cc`
- [ ] `src/tree/updater_refresh.cc`
- [ ] `tests/cpp/tree/test_tree_stat.cc`
- [ ] No other files remain in `git diff --name-only origin/master...HEAD`

## Compile-fix remediation
- [ ] `tests/cpp/tree/test_tree_stat.cc` no longer uses `CHECK_NEAR`
- [ ] The invalid assertion is replaced with a valid available gtest assertion macro
- [ ] The change is minimal and limited to fixing the compile failure
- [ ] No broad test refactor is introduced

## Validation
### Required
- [ ] `git diff --name-only origin/master...HEAD` shows only intended `depth_decay` source/test files
- [ ] focused `pre-commit run --files ... --show-diff-on-failure` passes for the intended changed files

### Strongly recommended
- [ ] targeted local build/gtest validation around `tests/cpp/tree/test_tree_stat.cc` passes if environment supports it
- [ ] if local build/test execution is unavailable, the worker clearly states that limitation

## If fixes were needed
- [ ] Any code change is directly tied to a reproduced failure
- [ ] Any code change is minimal and root-cause oriented
- [ ] No unrelated files are edited as part of the fix

## Final quality bar
- [ ] PR remains a focused `depth_decay` patch
- [ ] Assistant/workspace files are gone from the diff
- [ ] The concrete C++ compile blocker is fixed
- [ ] The worker can summarize both remediation tracks separately: diff cleanup and compile fix
