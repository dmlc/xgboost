# Validation Failures (attempt 1)

Only required failing checks are listed.

## Failure 1
- suite: lint
- command: pre-commit run --files python-package/xgboost/__init__.py tests/python/test_shap.py --show-diff-on-failure
- exit_code: 1
- stdout_log: /workspace/state/run-artifacts/run-44/013.stdout.log
- stderr_log: /workspace/state/run-artifacts/run-44/013.stderr.log
