import pytest
from hypothesis import given, settings, strategies
from xgboost import testing as tm
from xgboost.testing.continuation import (
    run_training_continuation_determinism,
    run_training_continuation_model_output,
)


@pytest.mark.parametrize("tree_method", ["hist", "approx"])
def test_model_output(tree_method: str) -> None:
    run_training_continuation_model_output("cuda", tree_method)


@given(
    subsample=strategies.floats(0.5, 1.0),
    sampling_method=strategies.sampled_from(["uniform", "gradient_based"]),
    colsample_bytree=strategies.floats(0.5, 1.0),
    colsample_bylevel=strategies.floats(0.5, 1.0),
    colsample_bynode=strategies.floats(0.5, 1.0),
    booster=strategies.sampled_from(["gbtree", "dart"]),
    num_class=strategies.sampled_from([1, 3]),
)
@settings(deadline=None, print_blob=True, max_examples=20)
@pytest.mark.skipif(**tm.no_sklearn())
def test_continuation_determinism(
    subsample: float,
    sampling_method: str,
    colsample_bytree: float,
    colsample_bylevel: float,
    colsample_bynode: float,
    booster: str,
    num_class: int,
) -> None:
    run_training_continuation_determinism(
        device="cuda",
        booster=booster,
        subsample=subsample,
        sampling_method=sampling_method,
        colsample_bytree=colsample_bytree,
        colsample_bylevel=colsample_bylevel,
        colsample_bynode=colsample_bynode,
        num_class=num_class,
    )
