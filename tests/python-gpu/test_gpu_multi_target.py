from typing import Any, Dict, Optional

import pytest
from hypothesis import given, note, settings, strategies

from xgboost import config_context
from xgboost import testing as tm
from xgboost.testing.multi_target import (
    run_absolute_error,
    run_column_sampling,
    run_deterministic,
    run_eta,
    run_feature_importance_strategy_compare,
    run_grow_policy,
    run_mixed_strategy,
    run_multiclass,
    run_multilabel,
    run_quantile_loss,
    run_reduced_grad,
    run_with_iter,
)
from xgboost.testing.params import hist_parameter_strategy
from xgboost.testing.updater import check_quantile_loss_rf, train_result


@pytest.mark.parametrize("learning_rate", [1.0, None])
def test_multiclass(learning_rate: Optional[float]) -> None:
    run_multiclass("cuda", learning_rate)


@pytest.mark.parametrize("learning_rate", [1.0, None])
def test_multilabel(learning_rate: Optional[float]) -> None:
    run_multilabel("cuda", learning_rate)


@pytest.mark.parametrize("weighted", [True, False])
def test_quantile_loss(weighted: bool) -> None:
    run_quantile_loss("cuda", weighted)


@pytest.mark.parametrize("multi_strategy", ["multi_output_tree", "one_output_per_tree"])
def test_quantile_loss_rf(multi_strategy: str) -> None:
    check_quantile_loss_rf("cuda", "hist", multi_strategy)
    if multi_strategy == "one_output_per_tree":
        check_quantile_loss_rf("cuda", "approx", multi_strategy)


def test_absolute_error() -> None:
    run_absolute_error("cuda")


def test_reduced_grad() -> None:
    run_reduced_grad("cuda")


def test_with_iter() -> None:
    with config_context(use_rmm=True):
        run_with_iter("cuda")


def test_eta() -> None:
    run_eta("cuda")


def test_deterministic() -> None:
    run_deterministic("cuda")


def test_column_sampling() -> None:
    run_column_sampling("cuda")


@pytest.mark.parametrize("grow_policy", ["depthwise", "lossguide"])
def test_grow_policy(grow_policy: str) -> None:
    run_grow_policy("cuda", grow_policy)


def test_mixed_strategy() -> None:
    run_mixed_strategy("cuda")


def test_feature_importance_strategy_compare() -> None:
    run_feature_importance_strategy_compare("cuda")


@given(hist_parameter_strategy, strategies.integers(1, 20), tm.multi_dataset_strategy)
@settings(deadline=None, max_examples=50, print_blob=True)
def test_hist(param: Dict[str, Any], num_rounds: int, dataset: tm.TestDataset) -> None:
    param["tree_method"] = "hist"
    param["device"] = "cuda"
    param = dataset.set_params(param)
    result = train_result(param, dataset.get_dmat(), num_rounds)
    note(str(result))
    assert tm.non_increasing(result["train"][dataset.metric])
