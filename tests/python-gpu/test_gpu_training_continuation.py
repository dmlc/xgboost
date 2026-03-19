from typing import Any

import pytest
from hypothesis import given, settings
from xgboost import testing as tm
from xgboost.testing.continuation import (
    make_determinism_strategy,
    run_training_continuation_determinism,
    run_training_continuation_model_output,
)


@pytest.mark.parametrize("tree_method", ["hist", "approx"])
def test_model_output(tree_method: str) -> None:
    run_training_continuation_model_output("cuda", tree_method)


@given(make_determinism_strategy(["hist", "approx"]))
@settings(deadline=None, print_blob=True, max_examples=10)
@pytest.mark.skipif(**tm.no_sklearn())
def test_continuation_determinism(
    kwargs: Any,
) -> None:
    run_training_continuation_determinism(device="cuda", **kwargs)
