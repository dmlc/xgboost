# Worker Handoff — PR #12119 Feedback Retry

## Task
Recover cleanly from the failed revalidation recorded in `.bot/revalidation_failures.md`.

This is a retry focused on validation scope and worktree cleanup, not a fresh feature edit.

## Read first
Before changing anything, read:
- `AGENTS.md`
- `.bot/maintainer_feedback.md`
- `.bot/retrieved_memory.md`
- `.bot/revalidation_failures.md`
- `.bot/coordination/coordinator_plan_feedback.md`

## Guardrails
- Do not open PRs
- Do not push
- Do not run `pre-commit run --all-files` again
- Do not edit unrelated files because a prior hook run touched them
- Do not broaden scope beyond the intended product files unless a reproduced failure forces it

## Recorded failure
The only recorded retry failure is:

```bash
pre-commit run --all-files --show-diff-on-failure
```

That command is too broad for this PR. It is not evidence that the intended three-file product patch is failing.

## Current verified state
Coordinator verified all three of these facts:

### 1) Branch diff vs base
`git diff --name-only origin/master...HEAD` shows only:
- `python-package/xgboost/__init__.py`
- `python-package/xgboost/interpret.py`
- `tests/python/test_shap.py`

### 2) Targeted validation passes
This command passes:

```bash
pre-commit run --files python-package/xgboost/__init__.py python-package/xgboost/interpret.py tests/python/test_shap.py --show-diff-on-failure
```

### 3) Local worktree is dirty from the bad revalidation run
`git status --short` shows many unrelated tracked files modified outside the branch diff. Treat these as local hook fallout, not intended patch work.

## Required outcome
Restore the unrelated tracked modifications while preserving the intended three-file patch.

## Files that must remain as the intended branch diff
Preserve these:
- `python-package/xgboost/__init__.py`
- `python-package/xgboost/interpret.py`
- `tests/python/test_shap.py`

## Cleanup method
Do not use `git reset --hard`.
Use targeted restore only.

A suitable cleanup pattern for tracked modified files is:

```bash
git status --porcelain \
  | awk '/^ M / || /^M  / {print substr($0,4)}' \
  | grep -vE '^(python-package/xgboost/__init__.py|python-package/xgboost/interpret.py|tests/python/test_shap.py)$' \
  | xargs -r git restore --
```

Notes:
- This is for tracked modified files only.
- Untracked local files under `.bot/`, `.openclaw/`, and workspace/persona files may stay untracked locally.
- If any unintended tracked file remains modified afterward, restore it explicitly.

## Verification steps
### 1) Confirm cleanup
Run:

```bash
git status --short
git diff --name-only origin/master...HEAD
```

Expected branch diff output:
- `python-package/xgboost/__init__.py`
- `python-package/xgboost/interpret.py`
- `tests/python/test_shap.py`

No other files should appear in `git diff --name-only origin/master...HEAD`.

### 2) Re-run targeted validation
Required:

```bash
pre-commit run --files python-package/xgboost/__init__.py python-package/xgboost/interpret.py tests/python/test_shap.py --show-diff-on-failure
```

Recommended:

```bash
python -m py_compile python-package/xgboost/__init__.py python-package/xgboost/interpret.py tests/python/test_shap.py
```

## What not to do
- Do not re-run the broad command from `.bot/revalidation_failures.md`
- Do not try to fix repo-wide lint fallout in demos, docs, R code, C++, JVM, plugins, or unrelated tests
- Do not change product code unless targeted validation unexpectedly fails after cleanup
- Do not create new files in the product diff

## If targeted validation unexpectedly fails
Only then make the smallest root-cause fix, and only in:
- `python-package/xgboost/__init__.py`
- `python-package/xgboost/interpret.py`
- `tests/python/test_shap.py`

Then re-run the same targeted validation.

## Done criteria
You are done when:
1. unrelated tracked modifications are restored
2. `git diff --name-only origin/master...HEAD` shows only the intended three files
3. targeted pre-commit passes on those three files
4. no unrelated files were edited during the retry
