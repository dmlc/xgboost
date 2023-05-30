"""Tests for evaluation metrics."""
from typing import Dict, List

import numpy as np
import pytest

import xgboost as xgb
from xgboost.compat import concat
from xgboost.core import _parse_eval_str


def check_precision_score(tree_method: str) -> None:
    """Test for precision with ranking and classification."""
    datasets = pytest.importorskip("sklearn.datasets")
    model_selection = pytest.importorskip("sklearn.model_selection")

    X, y = datasets.make_classification(
        n_samples=1024, n_features=4, n_classes=2, random_state=2023
    )
    qid = np.zeros(shape=y.shape)  # same group

    ltr = xgb.XGBRanker(n_estimators=2, tree_method=tree_method)
    ltr.fit(X, y, qid=qid)

    # re-generate so that XGBoost doesn't evaluate the result to 1.0
    X, y = datasets.make_classification(
        n_samples=512, n_features=4, n_classes=2, random_state=1994
    )

    ltr.set_params(eval_metric="pre@32")
    result = _parse_eval_str(
        ltr.get_booster().eval_set(evals=[(xgb.DMatrix(X, y), "Xy")])
    )
    score_0 = result[1][1]

    X_list = []
    y_list = []
    n_query_groups = 3
    q_list: List[np.ndarray] = []
    for i in range(n_query_groups):
        # same for all groups
        X, y = datasets.make_classification(
            n_samples=512, n_features=4, n_classes=2, random_state=1994
        )
        X_list.append(X)
        y_list.append(y)
        q = np.full(shape=y.shape, fill_value=i, dtype=np.uint64)
        q_list.append(q)

    qid = concat(q_list)
    X = concat(X_list)
    y = concat(y_list)

    result = _parse_eval_str(
        ltr.get_booster().eval_set(evals=[(xgb.DMatrix(X, y, qid=qid), "Xy")])
    )
    assert result[1][0].endswith("pre@32")
    score_1 = result[1][1]
    assert score_1 == score_0

    X, y = datasets.make_classification(n_samples=128, n_features=4, random_state=1994)
    clf = xgb.XGBClassifier(
        tree_method=tree_method, eval_metric="pre@32", n_estimators=3
    )
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, random_state=1994
    )
    w_train = np.ones(y_train.shape)
    w_test = np.ones(y_test.shape)
    clf.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_test, y_test)],
        sample_weight_eval_set=[w_test],
    )

    clf_1 = xgb.XGBClassifier(
        tree_method=tree_method, eval_metric="pre@32", n_estimators=3
    )
    clf_1.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
    )
    print(clf.evals_result())
    print(clf_1.evals_result())

    from sklearn.metrics import precision_score


def check_quantile_error(tree_method: str) -> None:
    """Test for the `quantile` loss."""
    from sklearn.datasets import make_regression
    from sklearn.metrics import mean_pinball_loss

    rng = np.random.RandomState(19)
    # pylint: disable=unbalanced-tuple-unpacking
    X, y = make_regression(128, 3, random_state=rng)
    Xy = xgb.QuantileDMatrix(X, y)
    evals_result: Dict[str, Dict] = {}
    booster = xgb.train(
        {"tree_method": tree_method, "eval_metric": "quantile", "quantile_alpha": 0.3},
        Xy,
        evals=[(Xy, "Train")],
        evals_result=evals_result,
    )
    predt = booster.inplace_predict(X)
    loss = mean_pinball_loss(y, predt, alpha=0.3)
    np.testing.assert_allclose(evals_result["Train"]["quantile"][-1], loss)
