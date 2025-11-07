"""Tests for multi-target training."""

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.datasets import (
    make_classification,
    make_multilabel_classification,
    make_regression,
)

import xgboost.testing as tm

from .._typing import ArrayLike
from ..core import Booster, DMatrix, QuantileDMatrix
from ..objective import TreeObjective
from ..sklearn import XGBClassifier
from ..training import train
from .updater import ResetStrategy
from .utils import Device


def run_multiclass(device: Device, learning_rate: Optional[float]) -> None:
    """Use vector leaf for multi-class models."""
    X, y = make_classification(
        128, n_features=12, n_informative=10, n_classes=4, random_state=2025
    )
    clf = XGBClassifier(
        multi_strategy="multi_output_tree",
        callbacks=[ResetStrategy()],
        n_estimators=10,
        device=device,
        learning_rate=learning_rate,
    )
    clf.fit(X, y, eval_set=[(X, y)])
    assert clf.objective == "multi:softprob"
    assert tm.non_increasing(clf.evals_result()["validation_0"]["mlogloss"])
    if learning_rate is not None and abs(learning_rate - 1.0) < 1e-5:
        assert clf.evals_result()["validation_0"]["mlogloss"][-1] < 0.045

    proba = clf.predict_proba(X)
    assert proba.shape == (y.shape[0], 4)


def run_multilabel(device: Device, learning_rate: Optional[float]) -> None:
    """Use vector leaf for multi-label classification models."""
    X, y = make_multilabel_classification(128, random_state=2025)
    clf = XGBClassifier(
        multi_strategy="multi_output_tree",
        callbacks=[ResetStrategy()],
        n_estimators=10,
        device=device,
        learning_rate=learning_rate,
    )
    clf.fit(X, y, eval_set=[(X, y)])
    assert clf.objective == "binary:logistic"
    assert tm.non_increasing(clf.evals_result()["validation_0"]["logloss"])
    if learning_rate is not None and abs(learning_rate - 1.0) < 1e-5:
        assert clf.evals_result()["validation_0"]["logloss"][-1] < 0.065

    proba = clf.predict_proba(X)
    assert proba.shape == y.shape


def run_reduced_grad(device: Device) -> None:
    """Basic test for using reduced gradient for tree splits."""
    import cupy as cp

    class LsObj(TreeObjective):
        def __call__(
            self, y_pred: ArrayLike, dtrain: DMatrix
        ) -> Tuple[cp.ndarray, cp.ndarray]:
            y_true = dtrain.get_label().reshape(y_pred.shape)
            grad, hess = tm.ls_obj(y_true, y_pred, None)
            return cp.array(grad), cp.array(hess)

        def split_grad(
            self, grad: ArrayLike, hess: ArrayLike
        ) -> Tuple[ArrayLike, ArrayLike]:
            return cp.array(grad), cp.array(hess)

    X, y = make_regression(
        n_samples=1024, n_features=16, random_state=1994, n_targets=5
    )
    Xy = QuantileDMatrix(X, y)

    def run_test(obj: Optional[TreeObjective]) -> Booster:
        evals_result: Dict[str, Dict] = {}
        booster = train(
            {
                "device": device,
                "multi_strategy": "multi_output_tree",
                "learning_rate": 1,
            },
            Xy,
            evals=[(Xy, "Train")],
            obj=obj,
            num_boost_round=8,
            evals_result=evals_result,
        )
        assert tm.non_increasing(evals_result["Train"]["rmse"])
        return booster

    booster_0 = run_test(LsObj())
    booster_1 = run_test(None)
    np.testing.assert_allclose(
        booster_0.inplace_predict(X), booster_1.inplace_predict(X)
    )

    # Use mean gradient, should still converge.
    class LsObj1(LsObj):
        def split_grad(
            self, grad: ArrayLike, hess: ArrayLike
        ) -> Tuple[cp.ndarray, cp.ndarray]:
            sgrad = cp.mean(grad, axis=1)
            shess = cp.mean(hess, axis=1)
            print(shess.shape)
            assert False
            return sgrad, shess

    # booster_2 = run_test(LsObj1())
    # print(booster_2.get_dump())
