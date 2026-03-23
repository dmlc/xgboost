# Acceptance Criteria — Retry for issue #11947

## Source of truth for this retry
- [ ] Retry scope is derived from `.bot/validation_failures.md`
- [ ] Only required fixes tied to the recorded failure are included

## Files in scope
- [ ] `python-package/xgboost/__init__.py`
- [ ] `tests/python/test_shap.py`

## `python-package/xgboost/__init__.py`
- [ ] `interpret` is imported on its own line:
  - `from . import interpret`
- [ ] `collective` remains on its own line:
  - `from . import collective`
- [ ] `"interpret"` remains present in `__all__`
- [ ] Import layout is stable under ruff/pre-commit (no auto-rewrite on rerun)

## `tests/python/test_shap.py`
- [ ] No extra blank line remains between `import scipy.special` and `import xgboost as xgb`
- [ ] The interpret wrapper tests remain present in this file
- [ ] File formatting is stable under ruff/pre-commit (no auto-rewrite on rerun)

## Validation
### Required
- [ ] Exact failed command now passes:
  - `pre-commit run --files python-package/xgboost/__init__.py tests/python/test_shap.py --show-diff-on-failure`

### Optional if environment supports it
- [ ] Narrow targeted test run for interpret wrapper coverage passes:
  - `python -m pytest tests/python/test_shap.py -k "TestInterpret" -q`

## Scope discipline
- [ ] No unrelated files were edited for this retry
- [ ] `python-package/xgboost/interpret.py` was left unchanged unless a concrete follow-up error required touching it
- [ ] No docs, refactors, or broader API work were added in this retry
- [ ] No PRs were opened and no dangerous commands were run
