"""Tests for multi-target training."""

from typing import Dict, Optional, Tuple

import numpy as np
import pytest
from sklearn.datasets import (
    make_classification,
    make_multilabel_classification,
    make_regression,
)

import xgboost.testing as tm

from .._typing import ArrayLike
from ..compat import import_cupy
from ..core import Booster, DMatrix, ExtMemQuantileDMatrix, QuantileDMatrix, build_info
from ..objective import Objective, TreeObjective
from ..sklearn import XGBClassifier
from ..training import train
from .data import IteratorForTest
from .updater import ResetStrategy
from .utils import Device, assert_allclose


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
    # pylint: disable=unbalanced-tuple-unpacking
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


class LsObj0(TreeObjective):
    """Split grad is the same as value grad."""

    def __call__(
        self, y_pred: ArrayLike, dtrain: DMatrix
    ) -> Tuple[ArrayLike, ArrayLike]:
        cp = import_cupy()

        y_true = dtrain.get_label().reshape(y_pred.shape)
        grad, hess = tm.ls_obj(y_true, y_pred, None)
        return cp.array(grad), cp.array(hess)

    def split_grad(
        self, grad: ArrayLike, hess: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        cp = import_cupy()

        return cp.array(grad), cp.array(hess)


class LsObj1(Objective):
    """No split grad."""

    def __call__(
        self, y_pred: ArrayLike, dtrain: DMatrix
    ) -> Tuple[ArrayLike, ArrayLike]:
        cp = import_cupy()

        y_true = dtrain.get_label().reshape(y_pred.shape)
        grad, hess = tm.ls_obj(y_true, y_pred, None)
        return cp.array(grad), cp.array(hess)


def run_reduced_grad(device: Device) -> None:
    """Basic test for using reduced gradient for tree splits."""
    import cupy as cp

    X, y = make_regression(  # pylint: disable=unbalanced-tuple-unpacking
        n_samples=1024, n_features=16, random_state=1994, n_targets=5
    )
    Xy = QuantileDMatrix(X, y)

    def run_test(
        obj: Optional[Objective], base_score: Optional[list[float]] = None
    ) -> Booster:
        evals_result: Dict[str, Dict] = {}
        booster = train(
            {
                "device": device,
                "multi_strategy": "multi_output_tree",
                "learning_rate": 1,
                "base_score": base_score,
            },
            Xy,
            evals=[(Xy, "Train")],
            obj=obj,
            num_boost_round=8,
            evals_result=evals_result,
        )
        assert tm.non_increasing(evals_result["Train"]["rmse"])
        return booster

    booster_0 = run_test(LsObj0())
    booster_1 = run_test(LsObj1())
    np.testing.assert_allclose(
        booster_0.inplace_predict(X), booster_1.inplace_predict(X)
    )

    booster_2 = run_test(LsObj0(), [0.5] * y.shape[1])
    booster_3 = run_test(None, [0.5] * y.shape[1])
    np.testing.assert_allclose(
        booster_2.inplace_predict(X), booster_3.inplace_predict(X)
    )

    # Use mean gradient, should still converge.
    class LsObj2(LsObj0):
        """Use mean as split grad."""

        def __init__(self, check_used: bool):
            self._chk = check_used

        def split_grad(
            self, grad: ArrayLike, hess: ArrayLike
        ) -> Tuple[cp.ndarray, cp.ndarray]:
            if self._chk:
                assert False
            sgrad = cp.mean(grad, axis=1)
            shess = cp.mean(hess, axis=1)
            return sgrad, shess

    run_test(LsObj2(False))
    with pytest.raises(AssertionError):
        run_test(LsObj2(True))


def run_with_iter(device: Device) -> None:  # pylint: disable=too-many-locals
    """Test vector leaf with external memory."""
    if device == "cuda":
        from cupy import asarray
    else:
        from numpy import asarray

    n_batches = 4
    n_rounds = 8
    n_targets = 3
    intercept = [0.5] * n_targets

    params = {
        "device": device,
        "multi_strategy": "multi_output_tree",
        "learning_rate": 1.0,
        "base_score": intercept,
        "debug_synchronize": True,
    }

    Xs = []
    ys = []
    for i in range(n_batches):
        X_i, y_i = make_regression(  # pylint: disable=unbalanced-tuple-unpacking
            n_samples=4096, n_features=8, random_state=(i + 1), n_targets=n_targets
        )
        Xs.append(asarray(X_i))
        ys.append(asarray(y_i))
    it = IteratorForTest(Xs, ys, None, cache="cache", on_host=True)
    Xy: DMatrix = ExtMemQuantileDMatrix(it, cache_host_ratio=1.0)

    evals_result_0: Dict[str, Dict] = {}
    booster_0 = train(
        params,
        Xy,
        num_boost_round=n_rounds,
        evals=[(Xy, "Train")],
        evals_result=evals_result_0,
    )

    it = IteratorForTest(Xs, ys, None, cache=None)
    Xy = QuantileDMatrix(it)
    evals_result_1: Dict[str, Dict] = {}
    booster_1 = train(
        params,
        Xy,
        num_boost_round=n_rounds,
        evals=[(Xy, "Train")],
        evals_result=evals_result_1,
    )
    np.testing.assert_allclose(
        evals_result_0["Train"]["rmse"], evals_result_1["Train"]["rmse"]
    )
    assert tm.non_increasing(evals_result_0["Train"]["rmse"])
    X, _, _ = it.as_arrays()
    assert_allclose(device, booster_0.inplace_predict(X), booster_1.inplace_predict(X))

    v = build_info()["THRUST_VERSION"]
    if v[0] < 3:
        pytest.xfail("CCCL version too old.")

    it = IteratorForTest(
        Xs,
        ys,
        None,
        cache="cache",
        on_host=True,
        min_cache_page_bytes=X.shape[0] // n_batches * X.shape[1],
    )
    Xy = ExtMemQuantileDMatrix(it, cache_host_ratio=1.0)

    evals_result_2: Dict[str, Dict] = {}
    booster_2 = train(
        params,
        Xy,
        evals=[(Xy, "Train")],
        obj=LsObj0(),
        num_boost_round=n_rounds,
        evals_result=evals_result_2,
    )
    np.testing.assert_allclose(
        evals_result_0["Train"]["rmse"], evals_result_2["Train"]["rmse"]
    )
    assert_allclose(device, booster_0.inplace_predict(X), booster_2.inplace_predict(X))
