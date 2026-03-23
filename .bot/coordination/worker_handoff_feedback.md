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

### 1) Branch diff vs base is currently polluted
`git diff --name-only origin/master...HEAD` currently includes many files that should not be in the PR diff, including:
- `.bot/**`
- `.openclaw/workspace-state.json`
- `AGENTS.md`
- `HEARTBEAT.md`
- `IDENTITY.md`
- `SOUL.md`
- `TOOLS.md`
- `USER.md`

These must be removed from the branch diff.

### 2) Local tracked `.bot/*` files are dirty
`git status --short` shows tracked modifications in:
- `.bot/memory/coordinator_retrieved_memory.md`
- `.bot/memory/worker_retrieved_memory.md`
- `.bot/retrieved_memory.md`

Restore them. Do not treat them as product changes.

### 3) Targeted validation on intended product files already passes
This command passes:
```bash
pre-commit run --files python-package/xgboost/__init__.py python-package/xgboost/interpret.py tests/python/test_shap.py --show-diff-on-failure
```

That means cleanup is the primary task unless validation regresses after cleanup.

## What to do

### 1) Inspect current branch diff
Run:
```bash
git diff --name-only origin/master...HEAD
```

Your target end state is that this shows only:
- `python-package/xgboost/__init__.py`
- `python-package/xgboost/interpret.py`
- `tests/python/test_shap.py`

### 2) Remove accidental assistant/workspace files from the branch diff
Remove all non-product paths from the branch diff.
This includes:
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

#### Recommended cleanup approach
For each accidental path:
- if it exists on `origin/master`, restore it from base
- otherwise remove it from the index so it can remain only as a local untracked workspace file

Suggested shell pattern:
```bash
for p in \
  .bot/coordination/acceptance_criteria.md \
  .bot/coordination/acceptance_feedback.md \
  .bot/coordination/coordinator_plan.md \
  .bot/coordination/coordinator_plan_feedback.md \
  .bot/coordination/worker_handoff.md \
  .bot/coordination/worker_handoff_feedback.md \
  .bot/coordination/worker_handoff_feedback_retry.md \
  .bot/coordination/worker_handoff_retry.md \
  .bot/coordination/worker_report.md \
  .bot/memory/coordinator_retrieved_memory.md \
  .bot/memory/worker_retrieved_memory.md \
  .bot/pr_body.md \
  .bot/pr_review_comment.md \
  .bot/retrieved_memory.md \
  .bot/revalidation_failures.md \
  .bot/validation.md \
  .bot/validation_failures.md \
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

If some of these remain untracked locally afterward, that is fine.
The goal is that they disappear from the branch diff.

### 3) Restore currently modified tracked `.bot/*` files in the local worktree
After the branch diff cleanup, ensure these no longer appear as modified tracked files:
- `.bot/memory/coordinator_retrieved_memory.md`
- `.bot/memory/worker_retrieved_memory.md`
- `.bot/retrieved_memory.md`

If still modified, restore them explicitly:
```bash
git restore -- .bot/memory/coordinator_retrieved_memory.md .bot/memory/worker_retrieved_memory.md .bot/retrieved_memory.md
```

### 4) Re-check final branch diff
Run:
```bash
git diff --name-only origin/master...HEAD
```

Expected output:
- `python-package/xgboost/__init__.py`
- `python-package/xgboost/interpret.py`
- `tests/python/test_shap.py`

No other files should appear.

### 5) Re-run targeted validation
Required:
```bash
pre-commit run --files python-package/xgboost/__init__.py python-package/xgboost/interpret.py tests/python/test_shap.py --show-diff-on-failure
```

Recommended:
```bash
python -m py_compile python-package/xgboost/__init__.py python-package/xgboost/interpret.py tests/python/test_shap.py
```

### 6) Only if targeted validation now fails, make the smallest exact fix
If cleanup somehow exposes a real product failure, patch only the root cause and only in the intended three files.

Likely acceptable examples:
- import layout/order stabilization in `__init__.py`
- narrow lint suppression in `interpret.py` if required by the reproduced failure
- formatting/import cleanup in `tests/python/test_shap.py`

## What NOT to do
- Do not run `pre-commit run --all-files`
- Do not try to fix repo-wide lint fallout in demos, docs, R code, C++, JVM, plugins, or unrelated tests
- Do not redesign `xgboost.interpret`
- Do not move tests to a new file unless a concrete failure forces it
- Do not leave assistant/workspace files in the branch diff

## Done criteria
You are done when:
1. `git diff --name-only origin/master...HEAD` shows only the intended three files
2. tracked `.bot/*` noise is restored
3. targeted pre-commit on those three files passes
4. optional `py_compile` sanity check passes
5. no unrelated files were modified during remediation
