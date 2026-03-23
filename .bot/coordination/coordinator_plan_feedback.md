# Coordinator Plan — PR #12119 Feedback Remediation

## Mission
Coordinate a minimal remediation pass for PR #12119 (issue #11947) in `dmlc/xgboost`.

This is not a fresh implementation task. The intended feature patch already exists. The remediation goal is to keep the PR tightly scoped, reproduce only relevant failures, and avoid turning local validation fallout into product work.

## Inputs reviewed
- `AGENTS.md`
- `.bot/maintainer_feedback.md`
- `.bot/retrieved_memory.md`
- current branch diff and local worktree status
- targeted pre-commit status for intended product files

## Maintainer feedback summary
Loaded feedback is CI-driven and narrow:
- required checks are failing
- identify failing suites
- reproduce locally where possible
- patch only root-cause issues

No loaded reviewer feedback asks for API redesign, feature expansion, or broader refactors.

## Current factual state
### Intended branch diff
`git diff --name-only origin/master...HEAD` currently shows only:
- `python-package/xgboost/__init__.py`
- `python-package/xgboost/interpret.py`
- `tests/python/test_shap.py`

That is the correct intended PR scope.

### Targeted validation
The intended files pass targeted pre-commit:
- `pre-commit run --files python-package/xgboost/__init__.py python-package/xgboost/interpret.py tests/python/test_shap.py --show-diff-on-failure`
- result: pass

### Local worktree noise
`git status --short` currently shows a very large number of unrelated tracked files modified locally across demos, docs, R, C++, JVM, plugins, and tests.

Treat that as local validation fallout / worktree noise, not as PR scope.

## Working diagnosis
At the moment there is no evidence that the intended three-file Python patch needs additional product-code changes.

The likely remediation task is:
1. keep the branch diff limited to the intended three files
2. avoid broad validation commands
3. if the local worktree is dirty from an earlier broad lint run, restore unrelated tracked files
4. re-run targeted validation only on the intended files

## Scope
### In scope
- confirm branch diff scope
- restore unrelated tracked worktree modifications if present
- re-run targeted validation on the intended files
- make only the smallest root-cause fix if targeted validation actually fails

### Out of scope
- repo-wide formatting/lint cleanup
- editing demos, docs, R package files, C++, JVM, plugins, or unrelated tests
- redesigning `xgboost.interpret`
- moving tests unless a reproduced failure forces it
- opening PRs

## Required validation
- `git diff --name-only origin/master...HEAD`
- `pre-commit run --files python-package/xgboost/__init__.py python-package/xgboost/interpret.py tests/python/test_shap.py --show-diff-on-failure`

## Recommended lightweight check
- `python -m py_compile python-package/xgboost/__init__.py python-package/xgboost/interpret.py tests/python/test_shap.py`

## Success definition
This remediation succeeds when:
1. branch diff remains limited to the intended three files
2. targeted pre-commit on those files passes
3. unrelated tracked worktree noise is not mistaken for PR work
4. no unrelated files are edited during remediation
