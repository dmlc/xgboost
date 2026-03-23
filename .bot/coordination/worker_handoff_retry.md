# Worker Handoff Retry — GPT-5.3-Codex

## Context
This is a **retry handoff**, not a greenfield implementation brief.
You must scope your work to the recorded validation failure in:
- `.bot/validation_failures.md`

Read that file first if you need to confirm the failure context.

## Failure to fix
Only one required check failed:

```bash
pre-commit run --files python-package/xgboost/__init__.py tests/python/test_shap.py --show-diff-on-failure
```

The failure log shows `ruff check` auto-modified these two files.

## Exact required changes from the failure log
### 1) `python-package/xgboost/__init__.py`
Ruff wanted this import layout:

```python
from . import tracker  # noqa
from . import collective
from . import interpret
```

Do **not** collapse `collective` and `interpret` into one import line.
Keep them as separate `from . import ...` statements.

Also keep `"interpret"` in `__all__`.

### 2) `tests/python/test_shap.py`
Ruff removed one extra blank line in the import section.
The top of the file should look like:

```python
import itertools
import re

import numpy as np
import scipy.special
import xgboost as xgb
from xgboost import testing as tm
```

Do not leave a stray blank line between `scipy.special` and `import xgboost as xgb`.

## Scope constraints
### Only touch what is required by this retry
Focus on:
- `python-package/xgboost/__init__.py`
- `tests/python/test_shap.py`

### Do not expand scope
Do **not** use this retry to:
- redesign the API
- move tests into a new file unless absolutely forced
- add docs
- add new methods to booster/sklearn wrappers
- refactor unrelated code
- touch unrelated files because they “look nicer”

## Important current state
There is already a pending implementation file in the worktree:
- `python-package/xgboost/interpret.py`

That file was **not** part of the failing lint command. Assume it is already part of the intended patch and **leave it alone unless a concrete error requires a change**.

## What to preserve
The retry should preserve the already-added interpret wrapper coverage in `tests/python/test_shap.py`.
Do not remove those tests. The retry is about making the existing patch pass the exact failed validation.

## Recommended step-by-step
1. Inspect current diffs/status.
2. Ensure `python-package/xgboost/__init__.py` matches the ruff-preferred import layout.
3. Ensure `tests/python/test_shap.py` import spacing matches the ruff-preferred layout.
4. Do not make unrelated edits.
5. Run the exact failed validation command:
   ```bash
   pre-commit run --files python-package/xgboost/__init__.py tests/python/test_shap.py --show-diff-on-failure
   ```
6. If that passes and the environment is healthy enough, optionally run a narrow targeted test:
   ```bash
   python -m pytest tests/python/test_shap.py -k "TestInterpret" -q
   ```

## Expected end state
- `python-package/xgboost/__init__.py`:
  - separate import line for `interpret`
  - `"interpret"` remains in `__all__`
- `tests/python/test_shap.py`:
  - clean import spacing
  - interpret wrapper tests still present
- exact pre-commit command passes without rewriting files

## Non-goals
- No PR creation
- No pushes
- No dangerous commands
- No broad cleanup pass

## Deliverable quality bar
This retry is successful only if it is:
- minimal
- directly tied to the logged failure
- pre-commit clean for the exact failing command
