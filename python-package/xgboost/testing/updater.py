"""Tests for updaters."""
import numpy as np

import xgboost as xgb


def check_init_estimation(tree_method: str) -> None:
    """Test for init estimation."""
    from sklearn.datasets import (
        make_classification,
        make_multilabel_classification,
        make_regression,
    )

    def run_reg(X: np.ndarray, y: np.ndarray) -> None:  # pylint: disable=invalid-name
        reg = xgb.XGBRegressor(tree_method=tree_method, max_depth=1, n_estimators=1)
        reg.fit(X, y, eval_set=[(X, y)])
        base_score_0 = reg.get_params()["base_score"]
        score_0 = reg.evals_result()["validation_0"]["rmse"][0]

        reg = xgb.XGBRegressor(
            tree_method=tree_method, max_depth=1, n_estimators=1, boost_from_average=0
        )
        reg.fit(X, y, eval_set=[(X, y)])
        base_score_1 = reg.get_params()["base_score"]
        score_1 = reg.evals_result()["validation_0"]["rmse"][0]
        assert not np.isclose(base_score_0, base_score_1)
        assert score_0 < score_1  # must be better

    X, y = make_regression(n_samples=4096)  # pylint: disable=unbalanced-tuple-unpacking
    run_reg(X, y)
    # pylint: disable=unbalanced-tuple-unpacking
    X, y = make_regression(n_samples=4096, n_targets=3)
    run_reg(X, y)

    def run_clf(X: np.ndarray, y: np.ndarray) -> None:  # pylint: disable=invalid-name
        clf = xgb.XGBClassifier(tree_method=tree_method, max_depth=1, n_estimators=1)
        clf.fit(X, y, eval_set=[(X, y)])
        base_score_0 = clf.get_params()["base_score"]
        score_0 = clf.evals_result()["validation_0"]["logloss"][0]

        clf = xgb.XGBClassifier(
            tree_method=tree_method, max_depth=1, n_estimators=1, boost_from_average=0
        )
        clf.fit(X, y, eval_set=[(X, y)])
        base_score_1 = clf.get_params()["base_score"]
        score_1 = clf.evals_result()["validation_0"]["logloss"][0]
        assert not np.isclose(base_score_0, base_score_1)
        assert score_0 < score_1  # must be better

    # pylint: disable=unbalanced-tuple-unpacking
    X, y = make_classification(n_samples=4096)
    run_clf(X, y)
    X, y = make_multilabel_classification(n_samples=4096, n_labels=3, n_classes=5)
    run_clf(X, y)
