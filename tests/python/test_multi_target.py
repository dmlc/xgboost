from typing import Any, Dict

from hypothesis import given, note, settings, strategies

import xgboost as xgb
from xgboost import testing as tm
from xgboost.testing.params import (
    exact_parameter_strategy,
    hist_cache_strategy,
    hist_multi_parameter_strategy,
    hist_parameter_strategy,
)
from xgboost.testing.updater import ResetStrategy, train_result


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
    X, y = tm.datasets.make_classification(
        128, n_features=12, n_informative=10, n_classes=4
    )
    clf = xgb.XGBClassifier(
        multi_strategy="multi_output_tree", callbacks=[ResetStrategy()], n_estimators=10
    )
    clf.fit(X, y, eval_set=[(X, y)])
    assert clf.objective == "multi:softprob"
    assert tm.non_increasing(clf.evals_result()["validation_0"]["mlogloss"])

    proba = clf.predict_proba(X)
    assert proba.shape == (y.shape[0], 4)


def test_multilabel() -> None:
    X, y = tm.datasets.make_multilabel_classification(128)
    clf = xgb.XGBClassifier(
        multi_strategy="multi_output_tree", callbacks=[ResetStrategy()], n_estimators=10
    )
    clf.fit(X, y, eval_set=[(X, y)])
    assert clf.objective == "binary:logistic"
    assert tm.non_increasing(clf.evals_result()["validation_0"]["logloss"])

    proba = clf.predict_proba(X)
    assert proba.shape == y.shape
