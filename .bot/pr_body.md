## Summary
Deliver the minimal Python interpretability surface for issue #11947 and complete retry pass `repair-2` by fixing the recorded lint instability.

## What changed
- Added module-level interpretability API in `python-package/xgboost/interpret.py`:
  - `shap_values(...)`
  - `shap_interactions(...)`
- Exported `interpret` in `python-package/xgboost/__init__.py` and kept it in `__all__`.
- Added targeted wrapper tests in `tests/python/test_shap.py` (`TestInterpret`).

## Repair-2 remediation (required failure only)
- Scoped to the lint failure from `.bot/validation_failures.md`.
- Ensured `python-package/xgboost/__init__.py` uses separate import lines:
  - `from . import collective`
  - `from . import interpret`
- Ensured `tests/python/test_shap.py` has no extra blank line between:
  - `import scipy.special`
  - `import xgboost as xgb`
- Re-ran the exact failing command successfully.

## Validation
- Passed:
  - `pre-commit run --files python-package/xgboost/__init__.py tests/python/test_shap.py --show-diff-on-failure`
- Passed:
  - `python -m py_compile python-package/xgboost/interpret.py tests/python/test_shap.py python-package/xgboost/__init__.py`
- Attempted but blocked by missing runtime deps:
  - `PYTHONPATH=python-package python -m pytest tests/python/test_shap.py -k "TestInterpret" -q`
  - error: `ModuleNotFoundError: No module named 'numpy'`

## Scope discipline
- Retry changes were kept focused on the recorded lint failure and existing targeted tests.
- No PRs were opened.

## Branch Policy Compatibility
- Base branch: `master`
- Base branch protected: yes
- Merge queue required: no
- Push strategy: `fork`
- Fork repo: `t7r0n/xgboost`
