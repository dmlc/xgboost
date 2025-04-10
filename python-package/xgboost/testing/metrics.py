"""Tests for evaluation metrics."""

from typing import Dict, List

import numpy as np
import pytest

from ..compat import concat
from ..core import DMatrix, QuantileDMatrix, _parse_eval_str
from ..sklearn import XGBClassifier, XGBRanker
from ..training import train
from .utils import Device


def check_precision_score(  # pylint: disable=too-many-locals
    tree_method: str, device: Device
) -> None:
    """Test for precision with ranking and classification."""
    datasets = pytest.importorskip("sklearn.datasets")

    X, y = datasets.make_classification(
        n_samples=1024, n_features=4, n_classes=2, random_state=2023
    )
    qid = np.zeros(shape=y.shape)  # same group

    ltr = XGBRanker(n_estimators=2, tree_method=tree_method, device=device)
    ltr.fit(X, y, qid=qid)

    # re-generate so that XGBoost doesn't evaluate the result to 1.0
    X, y = datasets.make_classification(
        n_samples=512, n_features=4, n_classes=2, random_state=1994
    )

    ltr.set_params(eval_metric="pre@32")
    result = _parse_eval_str(ltr.get_booster().eval_set(evals=[(DMatrix(X, y), "Xy")]))
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
        ltr.get_booster().eval_set(evals=[(DMatrix(X, y, qid=qid), "Xy")])
    )
    assert result[1][0].endswith("pre@32")
    score_1 = result[1][1]
    assert score_1 == score_0


def check_quantile_error(tree_method: str, device: Device) -> None:
    """Test for the `quantile` loss."""
    from sklearn.datasets import make_regression
    from sklearn.metrics import mean_pinball_loss

    rng = np.random.RandomState(19)
    # pylint: disable=unbalanced-tuple-unpacking
    X, y = make_regression(128, 3, random_state=rng)
    Xy = QuantileDMatrix(X, y)
    evals_result: Dict[str, Dict] = {}
    booster = train(
        {
            "tree_method": tree_method,
            "eval_metric": "quantile",
            "quantile_alpha": 0.3,
            "device": device,
        },
        Xy,
        evals=[(Xy, "Train")],
        evals_result=evals_result,
    )
    predt = booster.inplace_predict(X)
    loss = mean_pinball_loss(y, predt, alpha=0.3)
    np.testing.assert_allclose(evals_result["Train"]["quantile"][-1], loss)

    alpha = [0.25, 0.5, 0.75]
    booster = train(
        {
            "tree_method": tree_method,
            "eval_metric": "quantile",
            "quantile_alpha": alpha,
            "objective": "reg:quantileerror",
            "device": device,
        },
        Xy,
        evals=[(Xy, "Train")],
        evals_result=evals_result,
    )
    predt = booster.inplace_predict(X)
    loss = np.mean(
        [mean_pinball_loss(y, predt[:, i], alpha=alpha[i]) for i in range(3)]
    )
    np.testing.assert_allclose(evals_result["Train"]["quantile"][-1], loss)


def run_roc_auc_binary(tree_method: str, n_samples: int, device: Device) -> None:
    """TestROC AUC metric on a binary classification problem."""
    from sklearn.datasets import make_classification
    from sklearn.metrics import roc_auc_score

    rng = np.random.RandomState(1994)
    n_features = 10

    X, y = make_classification(
        n_samples,
        n_features,
        n_informative=n_features,
        n_redundant=0,
        random_state=rng,
    )
    Xy = DMatrix(X, y)
    booster = train(
        {
            "tree_method": tree_method,
            "device": device,
            "eval_metric": "auc",
            "objective": "binary:logistic",
        },
        Xy,
        num_boost_round=1,
    )
    score = booster.predict(Xy)
    skl_auc = roc_auc_score(y, score)
    auc = float(booster.eval(Xy).split(":")[1])
    np.testing.assert_allclose(skl_auc, auc, rtol=1e-6)

    X = rng.randn(*X.shape)
    score = booster.predict(DMatrix(X))
    skl_auc = roc_auc_score(y, score)
    auc = float(booster.eval(DMatrix(X, y)).split(":")[1])
    np.testing.assert_allclose(skl_auc, auc, rtol=1e-6)


def run_pr_auc_multi(tree_method: str, device: Device) -> None:
    """Test for PR AUC metric on a multi-class classification problem."""
    from sklearn.datasets import make_classification

    X, y = make_classification(64, 16, n_informative=8, n_classes=3, random_state=1994)
    clf = XGBClassifier(
        tree_method=tree_method, n_estimators=1, eval_metric="aucpr", device=device
    )
    clf.fit(X, y, eval_set=[(X, y)])
    evals_result = clf.evals_result()["validation_0"]["aucpr"][-1]
    # No available implementation for comparison, just check that XGBoost converges
    # to 1.0
    clf = XGBClassifier(
        tree_method=tree_method, n_estimators=10, eval_metric="aucpr", device=device
    )
    clf.fit(X, y, eval_set=[(X, y)])
    evals_result = clf.evals_result()["validation_0"]["aucpr"][-1]
    np.testing.assert_allclose(1.0, evals_result, rtol=1e-2)


def run_roc_auc_multi(  # pylint: disable=too-many-locals
    tree_method: str, n_samples: int, weighted: bool, device: Device
) -> None:
    """Test for ROC AUC metric on a multi-class classification problem."""
    from sklearn.datasets import make_classification
    from sklearn.metrics import roc_auc_score

    rng = np.random.RandomState(1994)
    n_features = 10
    n_classes = 4

    X, y = make_classification(
        n_samples,
        n_features,
        n_informative=n_features,
        n_redundant=0,
        n_classes=n_classes,
        random_state=rng,
    )
    if weighted:
        weights = rng.randn(n_samples)
        weights -= weights.min()
        weights /= weights.max()
    else:
        weights = None

    Xy = DMatrix(X, y, weight=weights)
    booster = train(
        {
            "tree_method": tree_method,
            "eval_metric": "auc",
            "objective": "multi:softprob",
            "num_class": n_classes,
            "device": device,
        },
        Xy,
        num_boost_round=1,
    )
    score = booster.predict(Xy)
    skl_auc = roc_auc_score(
        y, score, average="weighted", sample_weight=weights, multi_class="ovr"
    )
    auc = float(booster.eval(Xy).split(":")[1])
    np.testing.assert_allclose(skl_auc, auc, rtol=1e-6)

    X = rng.randn(*X.shape)

    score = booster.predict(DMatrix(X, weight=weights))
    skl_auc = roc_auc_score(
        y, score, average="weighted", sample_weight=weights, multi_class="ovr"
    )
    auc = float(booster.eval(DMatrix(X, y, weight=weights)).split(":")[1])
    np.testing.assert_allclose(skl_auc, auc, rtol=1e-5)


def run_pr_auc_ltr(tree_method: str, device: Device) -> None:
    """Test for PR AUC metric on a ranking problem."""
    from sklearn.datasets import make_classification

    X, y = make_classification(128, 4, n_classes=2, random_state=1994)
    ltr = XGBRanker(
        tree_method=tree_method,
        n_estimators=16,
        objective="rank:pairwise",
        eval_metric="aucpr",
        device=device,
    )
    groups = np.array([32, 32, 64])
    ltr.fit(
        X,
        y,
        group=groups,
        eval_set=[(X, y)],
        eval_group=[groups],
    )
    results = ltr.evals_result()["validation_0"]["aucpr"]
    assert results[-1] >= 0.99


def run_pr_auc_binary(tree_method: str, device: Device) -> None:
    """Test for PR AUC metric on a binary classification problem."""
    from sklearn.datasets import make_classification
    from sklearn.metrics import auc, precision_recall_curve

    X, y = make_classification(128, 4, n_classes=2, random_state=1994)
    clf = XGBClassifier(
        tree_method=tree_method, n_estimators=1, eval_metric="aucpr", device=device
    )
    clf.fit(X, y, eval_set=[(X, y)])
    evals_result = clf.evals_result()["validation_0"]["aucpr"][-1]

    y_score = clf.predict_proba(X)[:, 1]  # get the positive column
    precision, recall, _ = precision_recall_curve(y, y_score)
    prauc = auc(recall, precision)
    # Interpolation results are slightly different from sklearn, but overall should
    # be similar.
    np.testing.assert_allclose(prauc, evals_result, rtol=1e-2)

    clf = XGBClassifier(
        tree_method=tree_method, n_estimators=10, eval_metric="aucpr", device=device
    )
    clf.fit(X, y, eval_set=[(X, y)])
    evals_result = clf.evals_result()["validation_0"]["aucpr"][-1]
    np.testing.assert_allclose(0.99, evals_result, rtol=1e-2)
