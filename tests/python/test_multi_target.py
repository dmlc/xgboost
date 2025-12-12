from typing import Any, Dict

from hypothesis import given, note, settings, strategies

from xgboost import testing as tm
from xgboost.testing.multi_target import run_multiclass, run_multilabel
from xgboost.testing.params import (
    exact_parameter_strategy,
    hist_cache_strategy,
    hist_multi_parameter_strategy,
    hist_parameter_strategy,
)
from xgboost.testing.updater import train_result


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
