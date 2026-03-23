# Worker Handoff — PR #12119 Feedback Remediation

## Task
Remediate PR #12119 for issue #11947 with the smallest possible scoped fix.

This is a feedback/CI pass, not a fresh feature implementation.

## Read first
Before editing, read:
- `AGENTS.md`
- `.bot/maintainer_feedback.md`
- `.bot/retrieved_memory.md`
- `.bot/coordination/coordinator_plan_feedback.md`

## Guardrails
- Do not open PRs
- Do not push
- Do not broaden scope
- Do not perform repo-wide cleanup as product work
- Do not edit unrelated files

## Intended product files
Keep the patch constrained to these files unless a reproduced failure proves otherwise:
- `python-package/xgboost/__init__.py`
- `python-package/xgboost/interpret.py`
- `tests/python/test_shap.py`

## Current verified state
Coordinator verified:

### 1) Branch diff vs base
`git diff --name-only origin/master...HEAD` shows only:
- `python-package/xgboost/__init__.py`
- `python-package/xgboost/interpret.py`
- `tests/python/test_shap.py`

### 2) Targeted validation passes
This command currently passes:
```bash
pre-commit run --files python-package/xgboost/__init__.py python-package/xgboost/interpret.py tests/python/test_shap.py --show-diff-on-failure
```

### 3) Local worktree may be dirty outside branch diff
`git status --short` shows many unrelated tracked files modified locally. Treat these as local worktree fallout, not intended PR edits.

## What to do
### 1) Confirm branch diff scope
Run:
```bash
git diff --name-only origin/master...HEAD
```

Expected output:
- `python-package/xgboost/__init__.py`
- `python-package/xgboost/interpret.py`
- `tests/python/test_shap.py`

If any other file appears in the branch diff, remove it from the branch diff before doing anything else.

### 2) Re-run targeted validation only
Run:
```bash
pre-commit run --files python-package/xgboost/__init__.py python-package/xgboost/interpret.py tests/python/test_shap.py --show-diff-on-failure
```

Recommended sanity check:
```bash
python -m py_compile python-package/xgboost/__init__.py python-package/xgboost/interpret.py tests/python/test_shap.py
```

### 3) Only if targeted validation fails, make the smallest exact fix
If targeted validation fails, patch only the root cause and only in the intended files.

Likely acceptable examples:
- import layout/order stabilization in `__init__.py`
- narrow lint suppression in `interpret.py` if required by the reproduced failure
- formatting/import cleanup in `tests/python/test_shap.py`

### 4) If the local worktree is dirty, restore unrelated tracked files
Do not use `git reset --hard`.
Use targeted restore for tracked modified files while preserving the intended three files.

A suitable pattern is:
```bash
git status --porcelain \
  | awk '/^ M / || /^M  / {print substr($0,4)}' \
  | grep -vE '^(python-package/xgboost/__init__.py|python-package/xgboost/interpret.py|tests/python/test_shap.py)$' \
  | xargs -r git restore --
```

Untracked local `.bot/`, `.openclaw/`, and workspace/persona files may remain untracked locally.

## What NOT to do
- Do not run `pre-commit run --all-files`
- Do not try to fix repo-wide lint fallout in demos, docs, R code, C++, JVM, plugins, or unrelated tests
- Do not redesign `xgboost.interpret`
- Do not move tests to a new file unless a concrete failure forces it
- Do not create new files in the product diff

## Done criteria
You are done when:
1. `git diff --name-only origin/master...HEAD` shows only the intended three files
2. targeted pre-commit on those three files passes
3. optional `py_compile` sanity check passes
4. no unrelated files were modified during remediation
