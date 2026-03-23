# Coordinator Plan — PR #12119 Feedback Remediation

## Mission
Coordinate a minimal remediation pass for PR #12119 (issue #11947) in `dmlc/xgboost`.

This is not a fresh implementation task. The intended `xgboost.interpret` feature patch already exists. The current remediation goal is to clean the branch diff, preserve the intended product changes, and revalidate only what actually belongs in the PR.

## Inputs reviewed
- `AGENTS.md`
- `.bot/maintainer_feedback.md`
- `.bot/retrieved_memory.md`
- current branch diff vs `origin/master...HEAD`
- local worktree status
- targeted pre-commit status for intended product files

## Maintainer feedback summary
Loaded feedback is CI-driven and narrow:
- required checks are failing
- identify failing suites
- reproduce locally where possible
- patch only root-cause issues

No loaded reviewer feedback asks for API redesign, feature expansion, or broader refactors.

## Current factual state
### Intended product files
The intended feature patch should be limited to:
- `python-package/xgboost/__init__.py`
- `python-package/xgboost/interpret.py`
- `tests/python/test_shap.py`

### Current branch diff is broader than intended
`git diff --name-only origin/master...HEAD` currently includes many accidental assistant/workspace files, including:
- `.bot/coordination/*.md`
- `.bot/memory/*.md`
- `.bot/pr_body.md`
- `.bot/pr_review_comment.md`
- `.bot/retrieved_memory.md`
- `.bot/revalidation_failures.md`
- `.bot/validation*.md/json`
- `.openclaw/workspace-state.json`
- `AGENTS.md`
- `HEARTBEAT.md`
- `IDENTITY.md`
- `SOUL.md`
- `TOOLS.md`
- `USER.md`

These do not belong in an upstream XGBoost PR.

### Local worktree is also dirty
`git status --short` currently shows modified tracked files under:
- `.bot/memory/coordinator_retrieved_memory.md`
- `.bot/memory/worker_retrieved_memory.md`
- `.bot/retrieved_memory.md`

This is additional workspace noise and should not be treated as product work.

### Targeted validation on intended product files passes
The intended product files already pass targeted pre-commit:
- `pre-commit run --files python-package/xgboost/__init__.py python-package/xgboost/interpret.py tests/python/test_shap.py --show-diff-on-failure`
- result: pass

## Working diagnosis
At the moment, there is no evidence that the intended three-file Python patch needs additional product-code changes.

The main remediation task is **branch hygiene**:
1. remove accidental assistant/workspace files from the branch diff
2. restore stray tracked `.bot/*` modifications in the local worktree
3. preserve the intended three-file product diff
4. re-run targeted validation only on the intended product files

## Scope
### In scope
- cleaning accidental assistant/workspace files out of the branch diff
- restoring local tracked `.bot/*` noise
- confirming the branch diff is limited to the intended three files
- re-running targeted validation on the intended three files
- making only the smallest product fix if targeted validation unexpectedly fails

### Out of scope
- repo-wide formatting/lint cleanup
- editing demos, docs, R package files, C++, JVM, plugins, or unrelated tests
- redesigning `xgboost.interpret`
- moving tests unless a reproduced failure forces it
- opening PRs

## Required remediation order
1. Clean the branch diff first.
2. Restore any dirty tracked `.bot/*` files in the local worktree.
3. Confirm `git diff --name-only origin/master...HEAD` shows only the intended three files.
4. Re-run targeted pre-commit on the intended three files.
5. Only if that fails, patch the exact root cause in those files.

## Required validation
- `git diff --name-only origin/master...HEAD`
- `pre-commit run --files python-package/xgboost/__init__.py python-package/xgboost/interpret.py tests/python/test_shap.py --show-diff-on-failure`

## Recommended lightweight check
- `python -m py_compile python-package/xgboost/__init__.py python-package/xgboost/interpret.py tests/python/test_shap.py`

## Success definition
This remediation succeeds when:
1. branch diff is reduced to only the intended three product files
2. stray tracked `.bot/*` changes are restored
3. targeted pre-commit on the intended files passes
4. no unrelated files are edited as part of remediation
