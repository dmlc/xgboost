# Coordinator Retry Plan — dmlc/xgboost issue #11947

## Goal of this retry
This retry is **not** a fresh implementation plan for the whole RFC. It is a **targeted retry** based strictly on the recorded validation failure in `.bot/validation_failures.md`.

## Failure summary
Only one required check failed:
- suite: `lint`
- command:
  - `pre-commit run --files python-package/xgboost/__init__.py tests/python/test_shap.py --show-diff-on-failure`

The failure log shows that **ruff auto-fixed formatting/import layout** in exactly two files:
1. `python-package/xgboost/__init__.py`
2. `tests/python/test_shap.py`

No other required validation failures are recorded.

## What the failure means
The previous attempt already had the intended retry scope loaded into the linted files, but the patch did not land in a pre-commit-clean form.

### Exact hook-driven fixes from the log
#### `python-package/xgboost/__init__.py`
Ruff changed:
- from:
  - `from . import collective`
  - plus implicit/combined import arrangement
- to separate import lines:
  - `from . import collective`
  - `from . import interpret`

This means the retry must preserve **separate first-party import lines**, not collapse them.

#### `tests/python/test_shap.py`
Ruff changed:
- removed the extra blank line between:
  - `import scipy.special`
  - `import xgboost as xgb`

The file must keep normal import grouping with no stray blank line there.

## Retry scope
### Required edits only
Limit this retry to:
- `python-package/xgboost/__init__.py`
- `tests/python/test_shap.py`

### Explicitly avoid in this retry
Do **not** expand scope unless the exact failing command forces it.
In particular, do not spend this retry on:
- new docs
- new result classes
- new public methods on booster/sklearn models
- unrelated refactors
- new test files unless absolutely necessary

## Important repo state note
There is already a pending implementation file in the worktree:
- `python-package/xgboost/interpret.py`

That file is **not mentioned in the recorded failure**. The retry worker should treat it as already part of the patch and **avoid churning it** unless a follow-up validation step shows a concrete issue.

## Required retry actions
1. Ensure `python-package/xgboost/__init__.py` uses the ruff-compatible separate import line for `interpret`.
2. Ensure `tests/python/test_shap.py` matches the ruff-compatible import spacing.
3. Keep the new interpret wrapper tests in `tests/python/test_shap.py`; do not move them elsewhere during this retry unless unavoidable.
4. Re-run the exact failing command:
   - `pre-commit run --files python-package/xgboost/__init__.py tests/python/test_shap.py --show-diff-on-failure`

## Optional follow-up validation (only if inexpensive and available)
If the environment supports Python/native test execution, run a narrow targeted test for the newly added interpret wrapper coverage inside `tests/python/test_shap.py`.
Suggested example:
- `python -m pytest tests/python/test_shap.py -k "TestInterpret" -q`

If that is not feasible, the exact pre-commit pass is sufficient for this retry plan.

## Success definition for this retry
This retry succeeds when:
- the exact failing pre-commit command passes cleanly
- no additional unrelated edits are introduced
- the worker leaves the patch focused on the already-implemented interpretability slice

## Main risks to avoid
- Re-expanding scope beyond the logged failure
- Moving tests into a different file during the retry
- Collapsing imports in `__init__.py` back into a form ruff rewrites
- Touching `interpret.py` without a concrete failure signal
