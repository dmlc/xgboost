from typing import Any, Callable, Dict

import pytest
from hypothesis import given, note, settings, strategies

from xgboost import testing as tm
from xgboost.testing.multi_target import (
    all_reg_objectives,
    run_absolute_error,
    run_column_sampling,
    run_eta,
    run_feature_importance_strategy_compare,
    run_grow_policy,
    run_mixed_strategy,
    run_multiclass,
    run_multilabel,
    run_quantile_loss,
    run_reduced_grad,
    run_subsample,
    run_with_iter,
)
from xgboost.testing.params import (
    exact_parameter_strategy,
    hist_cache_strategy,
    hist_multi_parameter_strategy,
    hist_parameter_strategy,
)
from xgboost.testing.updater import check_quantile_loss_rf, train_result
from xgboost.testing.utils import Device


@pytest.mark.parametrize("multi_strategy", ["multi_output_tree", "one_output_per_tree"])
def test_quantile_loss_rf(multi_strategy: str) -> None:
    check_quantile_loss_rf("cpu", "hist", multi_strategy)
    if multi_strategy == "one_output_per_tree":
        check_quantile_loss_rf("cpu", "approx", multi_strategy)


class TestTreeMethodMulti:
    @given(
        exact_parameter_strategy, strategies.integers(1, 20), tm.multi_dataset_strategy
    )
    @settings(deadline=None, print_blob=True)
    def test_exact(self, param: dict, num_rounds: int, dataset: tm.TestDataset) -> None:
        if dataset.name.endswith("-l1"):
            return
        param["tree_method"] = "exact"
        param = dataset.set_params(param)
        result = train_result(param, dataset.get_dmat(), num_rounds)
        assert tm.non_increasing(result["train"][dataset.metric])

    @given(
        exact_parameter_strategy,
        hist_parameter_strategy,
        hist_cache_strategy,
        strategies.integers(1, 20),
        tm.multi_dataset_strategy,
    )
    @settings(deadline=None, print_blob=True)
    def test_approx(
        self,
        param: Dict[str, Any],
        hist_param: Dict[str, Any],
        cache_param: Dict[str, Any],
        num_rounds: int,
        dataset: tm.TestDataset,
    ) -> None:
        param["tree_method"] = "approx"
        param = dataset.set_params(param)
        param.update(hist_param)
        param.update(cache_param)
        result = train_result(param, dataset.get_dmat(), num_rounds)
        note(str(result))
        assert tm.non_increasing(result["train"][dataset.metric])

    @given(
        exact_parameter_strategy,
        hist_multi_parameter_strategy,
        hist_cache_strategy,
        strategies.integers(1, 20),
        tm.multi_dataset_strategy,
    )
    @settings(deadline=None, print_blob=True)
    def test_hist(
        self,
        param: Dict[str, Any],
        hist_param: Dict[str, Any],
        cache_param: Dict[str, Any],
        num_rounds: int,
        dataset: tm.TestDataset,
    ) -> None:
        if dataset.name.endswith("-l1"):
            return
        param["tree_method"] = "hist"
        param = dataset.set_params(param)
        param.update(hist_param)
        param.update(cache_param)
        result = train_result(param, dataset.get_dmat(), num_rounds)
        note(str(result))
        assert tm.non_increasing(result["train"][dataset.metric])


def test_multiclass() -> None:
    run_multiclass("cpu", None)


def test_multilabel() -> None:
    run_multilabel("cpu", None)


@pytest.mark.parametrize("weighted", [True, False])
def test_quantile_loss(weighted: bool) -> None:
    run_quantile_loss("cpu", weighted)


def test_absolute_error() -> None:
    run_absolute_error("cpu")


def test_reduced_grad() -> None:
    run_reduced_grad("cpu")


def test_with_iter() -> None:
    run_with_iter("cpu")


def test_eta() -> None:
    run_eta("cpu")


def test_column_sampling() -> None:
    run_column_sampling("cpu")


@pytest.mark.parametrize("grow_policy", ["depthwise", "lossguide"])
def test_grow_policy(grow_policy: str) -> None:
    run_grow_policy("cpu", grow_policy)


def test_mixed_strategy() -> None:
    run_mixed_strategy("cpu")


def test_feature_importance_strategy_compare() -> None:
    run_feature_importance_strategy_compare("cpu")


@pytest.mark.parametrize("obj_fn", all_reg_objectives())
def test_reg_objective(obj_fn: Callable[[Device], None]) -> None:
    obj_fn("cpu")


def test_subsample() -> None:
    run_subsample("cpu", "uniform")
