"""Tests for estimating the intercept."""

import json
from typing import Dict, List, Optional

import numpy as np
from scipy.special import softmax
from sklearn.datasets import (
    make_classification,
    make_multilabel_classification,
    make_regression,
)

from ..core import Booster, DMatrix, QuantileDMatrix
from ..sklearn import XGBClassifier, XGBRegressor
from ..training import train
from .updater import get_basescore
from .utils import Device, non_increasing


# pylint: disable=too-many-statements
def run_init_estimation(tree_method: str, device: Device) -> None:
    """Test for init estimation."""

    def run_reg(X: np.ndarray, y: np.ndarray) -> None:  # pylint: disable=invalid-name
        reg = XGBRegressor(
            tree_method=tree_method, max_depth=1, n_estimators=1, device=device
        )
        reg.fit(X, y, eval_set=[(X, y)])
        base_score_0 = get_basescore(reg)
        score_0 = reg.evals_result()["validation_0"]["rmse"][0]

        n_targets = 1 if y.ndim == 1 else y.shape[1]
        intercept = np.full(shape=(n_targets,), fill_value=0.5, dtype=np.float32)
        reg = XGBRegressor(
            tree_method=tree_method,
            device=device,
            max_depth=1,
            n_estimators=1,
            base_score=intercept,
        )
        reg.fit(X, y, eval_set=[(X, y)])
        base_score_1 = get_basescore(reg)
        score_1 = reg.evals_result()["validation_0"]["rmse"][0]
        assert not np.isclose(base_score_0, base_score_1).any()
        assert score_0 < score_1  # should be better

    # pylint: disable=unbalanced-tuple-unpacking
    X, y = make_regression(n_samples=4096, random_state=17)
    run_reg(X, y)
    # pylint: disable=unbalanced-tuple-unpacking
    X, y = make_regression(n_samples=4096, n_targets=3, random_state=17)
    run_reg(X, y)

    # pylint: disable=invalid-name
    def run_clf(
        X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None
    ) -> List[float]:
        clf = XGBClassifier(
            tree_method=tree_method, max_depth=1, n_estimators=1, device=device
        )
        if w is not None:
            clf.fit(
                X, y, sample_weight=w, eval_set=[(X, y)], sample_weight_eval_set=[w]
            )
        else:
            clf.fit(X, y, eval_set=[(X, y)])
        base_score_0 = get_basescore(clf)
        if clf.n_classes_ == 2:
            score_0 = clf.evals_result()["validation_0"]["logloss"][0]
        else:
            score_0 = clf.evals_result()["validation_0"]["mlogloss"][0]

        n_targets = 1 if y.ndim == 1 else y.shape[1]
        intercept = np.full(shape=(n_targets,), fill_value=0.5, dtype=np.float32)
        clf = XGBClassifier(
            tree_method=tree_method,
            max_depth=1,
            n_estimators=1,
            device=device,
            base_score=intercept,
        )
        if w is not None:
            clf.fit(
                X, y, sample_weight=w, eval_set=[(X, y)], sample_weight_eval_set=[w]
            )
        else:
            clf.fit(X, y, eval_set=[(X, y)])
        base_score_1 = get_basescore(clf)
        if clf.n_classes_ == 2:
            score_1 = clf.evals_result()["validation_0"]["logloss"][0]
        else:
            score_1 = clf.evals_result()["validation_0"]["mlogloss"][0]
        assert not np.isclose(base_score_0, base_score_1).any()
        assert score_0 < score_1 + 1e-4  # should be better

        return base_score_0

    # pylint: disable=unbalanced-tuple-unpacking
    X, y = make_classification(n_samples=4096, random_state=17)
    run_clf(X, y)
    X, y = make_multilabel_classification(
        n_samples=4096, n_labels=3, n_classes=5, random_state=17
    )
    run_clf(X, y)

    # Extra tests for the classifier.
    X, y = make_classification(
        n_samples=4096, random_state=17, n_classes=5, n_informative=20, n_redundant=0
    )
    intercept = run_clf(X, y)
    # un-transformed intercept sums to 0, as a convention.
    np.testing.assert_allclose(np.sum(softmax(intercept)), 1.0)
    np.testing.assert_allclose(np.sum(intercept), 0.0, atol=1e-6)

    assert np.all(softmax(intercept) > 0)
    np_int = (
        np.histogram(
            y, bins=np.concatenate([np.unique(y), np.array([np.finfo(np.float32).max])])
        )[0]
        / y.shape[0]
    )
    np.testing.assert_allclose(softmax(intercept), np_int, rtol=1e-6)

    rng = np.random.default_rng(1994)
    w = rng.uniform(low=0, high=1, size=(y.shape[0],))
    intercept = run_clf(X, y, w)
    np.testing.assert_allclose(np.sum(softmax(intercept)), 1.0)
    assert np.all(softmax(intercept) > 0)


# pylint: disable=too-many-locals
def run_adaptive(tree_method: str, weighted: bool, device: Device) -> None:
    """Test for adaptive trees."""
    rng = np.random.RandomState(1994)
    from sklearn.utils import stats

    n_samples = 256
    X, y = make_regression(  # pylint: disable=unbalanced-tuple-unpacking
        n_samples, 16, random_state=rng
    )
    if weighted:
        w = rng.normal(size=n_samples)
        w -= w.min()
        Xy = DMatrix(X, y, weight=w)

        kwargs = {"percentile_rank": 50}
        base_score = stats._weighted_percentile(  # pylint: disable=protected-access
            y, w, **kwargs
        )
    else:
        Xy = DMatrix(X, y)
        base_score = np.median(y)

    # Check the base score is expected.
    booster_0 = train(
        {
            "tree_method": tree_method,
            "base_score": base_score,
            "objective": "reg:absoluteerror",
            "device": device,
        },
        Xy,
        num_boost_round=1,
    )
    booster_1 = train(
        {
            "tree_method": tree_method,
            "objective": "reg:absoluteerror",
            "device": device,
        },
        Xy,
        num_boost_round=1,
    )
    config_0 = json.loads(booster_0.save_config())
    config_1 = json.loads(booster_1.save_config())

    assert get_basescore(config_0) == get_basescore(config_1)

    # check the base score is correctly serialized.
    raw_booster = booster_1.save_raw(raw_format="ubj")
    booster_2 = Booster(model_file=raw_booster)
    config_2 = json.loads(booster_2.save_config())
    assert get_basescore(config_1) == get_basescore(config_2)

    # check we can override the base score.
    booster_0 = train(
        {
            "tree_method": tree_method,
            "base_score": base_score + 1.0,
            "objective": "reg:absoluteerror",
            "device": device,
        },
        Xy,
        num_boost_round=1,
    )
    config_0 = json.loads(booster_0.save_config())
    np.testing.assert_allclose(
        get_basescore(config_0), np.asarray(get_basescore(config_1)) + 1
    )

    # check we can use subsampling.
    evals_result: Dict[str, Dict[str, list]] = {}
    train(
        {
            "tree_method": tree_method,
            "device": device,
            "objective": "reg:absoluteerror",
            "subsample": 0.8,
            "eta": 1.0,
        },
        Xy,
        num_boost_round=10,
        evals=[(Xy, "Train")],
        evals_result=evals_result,
    )
    mae = evals_result["Train"]["mae"]
    assert mae[-1] < 20.0
    assert non_increasing(mae)


def run_exp_family(device: Device) -> None:
    """Exp family has a closed solution."""
    X, y = make_classification(n_samples=128, n_classes=2, weights=[0.8, 0.2])
    Xy = QuantileDMatrix(X, y)
    clf = train(
        {"objective": "binary:logistic", "device": device}, Xy, num_boost_round=1
    )
    reg = train({"objective": "reg:logistic", "device": device}, Xy, num_boost_round=1)
    clf1 = train(
        {"objective": "binary:logitraw", "device": device}, Xy, num_boost_round=1
    )
    # The base score stored in the booster model is un-transformed
    np.testing.assert_allclose([get_basescore(m) for m in (reg, clf, clf1)], y.mean())

    X, y = make_classification(weights=[0.8, 0.2], random_state=2025)
    clf = train(
        {"objective": "binary:logistic", "scale_pos_weight": 4.0, "device": device},
        QuantileDMatrix(X, y),
        num_boost_round=1,
    )
    score = get_basescore(clf)
    np.testing.assert_allclose(score, 0.5, rtol=1e-3)


def run_logistic_degenerate(device: Device) -> None:
    """Test https://github.com/dmlc/xgboost/issues/11499 ."""

    def run(v: float) -> None:
        dtrain = DMatrix(np.asarray([[1.0], [1.0]]), label=[v, v])
        bst = train(
            {"objective": "binary:logistic", "device": device},
            dtrain,
            1,
        )
        intercept = get_basescore(bst)
        assert intercept[0] == v

    run(0.0)
    run(1.0)
