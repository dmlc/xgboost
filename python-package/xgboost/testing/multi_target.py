"""Tests for multi-target training."""

# pylint: disable=unbalanced-tuple-unpacking
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pytest
from sklearn.datasets import (
    make_classification,
    make_multilabel_classification,
    make_regression,
)
from sklearn.metrics.pairwise import cosine_similarity

import xgboost.testing as tm

from .._typing import ArrayLike
from ..compat import import_cupy
from ..core import Booster, DMatrix, ExtMemQuantileDMatrix, QuantileDMatrix, build_info
from ..objective import Objective, TreeObjective
from ..sklearn import XGBClassifier
from ..training import train
from .data import IteratorForTest
from .updater import ResetStrategy, train_result
from .utils import Device, assert_allclose, non_increasing


def run_multiclass(device: Device, learning_rate: Optional[float]) -> None:
    """Use vector leaf for multi-class models."""
    X, y = make_classification(
        128, n_features=12, n_informative=10, n_classes=4, random_state=2025
    )
    clf = XGBClassifier(
        debug_synchronize=True,
        multi_strategy="multi_output_tree",
        callbacks=[ResetStrategy()],
        n_estimators=10,
        device=device,
        learning_rate=learning_rate,
    )
    clf.fit(X, y, eval_set=[(X, y)])
    assert clf.objective == "multi:softprob"
    assert non_increasing(clf.evals_result()["validation_0"]["mlogloss"])
    if learning_rate is not None and abs(learning_rate - 1.0) < 1e-5:
        assert clf.evals_result()["validation_0"]["mlogloss"][-1] < 0.045

    proba = clf.predict_proba(X)
    assert proba.shape == (y.shape[0], 4)


def run_multilabel(device: Device, learning_rate: Optional[float]) -> None:
    """Use vector leaf for multi-label classification models."""
    X, y = make_multilabel_classification(128, random_state=2025)
    clf = XGBClassifier(
        debug_synchronize=True,
        multi_strategy="multi_output_tree",
        callbacks=[ResetStrategy()],
        n_estimators=10,
        device=device,
        learning_rate=learning_rate,
    )
    clf.fit(X, y, eval_set=[(X, y)])
    assert clf.objective == "binary:logistic"
    assert non_increasing(clf.evals_result()["validation_0"]["logloss"])
    if learning_rate is not None and abs(learning_rate - 1.0) < 1e-5:
        assert clf.evals_result()["validation_0"]["logloss"][-1] < 0.065

    proba = clf.predict_proba(X)
    assert proba.shape == y.shape


def run_quantile_loss(device: Device, weighted: bool) -> None:
    """Check quantile regression for vector leaf."""
    params = {
        "objective": "reg:quantileerror",
        "device": device,
        "quantile_alpha": [0.45, 0.5, 0.55],
        "multi_strategy": "multi_output_tree",
    }
    n_samples = 2048
    X, y = make_regression(n_samples=n_samples, n_features=16, random_state=2026)

    def no_crossing_first_tree(weight: Optional[np.ndarray]) -> None:
        """The first tree should not generate quantile crossing given sufficient amount
        of samples for quantile interpolation.

        """
        Xy = QuantileDMatrix(X, y, weight=weight)
        booster = train(params, Xy, evals=[(Xy, "Train")], num_boost_round=1)
        y_predt = booster.predict(Xy)
        assert y_predt.shape == (n_samples, 3)
        assert (y_predt[:, 0] <= y_predt[:, 1]).all()
        assert (y_predt[:, 1] <= y_predt[:, 2]).all()

    if not weighted:
        weight = None
    else:
        # Test with weights.
        rng = np.random.default_rng(2026)
        weight = rng.uniform(0.0, 1.0, size=n_samples)

    no_crossing_first_tree(weight)

    Xy = QuantileDMatrix(X, y, weight=weight)
    evals_result = train_result(params, Xy, num_rounds=10)
    assert non_increasing(evals_result["train"]["quantile"])


def run_absolute_error(device: Device) -> None:
    """Test mean absolute error with vector leaf."""
    params = {
        "objective": "reg:absoluteerror",
        "device": device,
        "multi_strategy": "multi_output_tree",
    }
    n_samples = 1024
    X, y = make_regression(
        n_samples=n_samples, n_features=16, n_targets=3, random_state=2026
    )
    Xy = QuantileDMatrix(X, y)
    evals_result: Dict[str, Dict] = {}
    booster = train(
        params,
        Xy,
        evals=[(Xy, "Train")],
        verbose_eval=False,
        evals_result=evals_result,
        num_boost_round=16,
    )
    predt = booster.predict(Xy)
    # make sure different targets are used
    assert np.abs((predt[:, 2] - predt[:, 1]).sum()) > 1000
    assert np.abs((predt[:, 1] - predt[:, 0]).sum()) > 1000
    assert non_increasing(evals_result["Train"]["mae"])
    assert evals_result["Train"]["mae"][-1] < 30.0


def _array_impl(device: Device) -> ModuleType:
    if device == "cuda":
        nda = import_cupy()
    else:
        nda = np
    return nda


class LsObj0(TreeObjective):
    """Split grad is the same as value grad."""

    def __init__(self, device: Device) -> None:
        self.device = device

    def __call__(
        self, iteration: int, y_pred: ArrayLike, dtrain: DMatrix
    ) -> Tuple[ArrayLike, ArrayLike]:
        nda = _array_impl(self.device)

        y_true = dtrain.get_label()
        grad, hess = tm.ls_obj(y_true, y_pred, None)
        return nda.array(grad), nda.array(hess)

    def split_grad(
        self, iteration: int, grad: ArrayLike, hess: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        nda = _array_impl(self.device)
        return nda.array(grad), nda.array(hess)


class LsObj1(Objective):
    """No split grad."""

    def __init__(self, device: Device) -> None:
        self.device = device

    def __call__(
        self, iteration: int, y_pred: ArrayLike, dtrain: DMatrix
    ) -> Tuple[ArrayLike, ArrayLike]:
        nda = _array_impl(self.device)

        y_true = dtrain.get_label()
        grad, hess = tm.ls_obj(y_true, y_pred, None)
        return nda.array(grad), nda.array(hess)


# Use mean gradient, should still converge.
class LsObj2(LsObj0):
    """Use mean as split grad."""

    def __init__(self, device: Device, check_used: bool):
        self._chk = check_used
        super().__init__(device=device)

    def split_grad(
        self, iteration: int, grad: ArrayLike, hess: ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray]:
        nda = _array_impl(self.device)

        if self._chk:
            assert False
        sgrad = nda.mean(grad, axis=1)
        shess = nda.mean(hess, axis=1)
        return sgrad, shess


def run_reduced_grad(device: Device) -> None:
    """Basic test for using reduced gradient for tree splits."""
    X, y = make_regression(
        n_samples=1024, n_features=16, random_state=1994, n_targets=5
    )
    Xy = QuantileDMatrix(X, y)

    def run_test(
        obj: Optional[Objective], base_score: Optional[list[float]] = None
    ) -> Booster:
        evals_result: Dict[str, Dict] = {}
        booster = train(
            {
                "debug_synchronize": True,
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
        assert non_increasing(evals_result["Train"]["rmse"])
        return booster

    booster_0 = run_test(LsObj0(device))
    booster_1 = run_test(LsObj1(device))
    np.testing.assert_allclose(
        booster_0.inplace_predict(X), booster_1.inplace_predict(X)
    )

    booster_2 = run_test(LsObj0(device), [0.5] * y.shape[1])
    booster_3 = run_test(None, [0.5] * y.shape[1])
    np.testing.assert_allclose(
        booster_2.inplace_predict(X), booster_3.inplace_predict(X)
    )

    run_test(LsObj2(device, False))
    with pytest.raises(AssertionError):
        run_test(LsObj2(device, True))


def run_with_iter(device: Device) -> None:  # pylint: disable=too-many-locals
    """Test vector leaf with external memory."""
    nda = _array_impl(device)

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
        X_i, y_i = make_regression(
            n_samples=4096, n_features=8, random_state=(i + 1), n_targets=n_targets
        )
        Xs.append(nda.asarray(X_i))
        ys.append(nda.asarray(y_i))
    it = IteratorForTest(Xs, ys, None, cache="cache", on_host=True)
    Xy: DMatrix = ExtMemQuantileDMatrix(
        it, cache_host_ratio=1.0 if device == "cuda" else None
    )

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
    assert non_increasing(evals_result_0["Train"]["rmse"])
    X, _, _ = it.as_arrays()
    assert_allclose(device, booster_0.inplace_predict(X), booster_1.inplace_predict(X))

    binfo = build_info()
    tv = "THRUST_VERSION"
    if device == "cuda" and tv in binfo and binfo[tv][0] < 3:
        pytest.xfail("CCCL version too old.")

    it = IteratorForTest(
        Xs,
        ys,
        None,
        cache="cache",
        on_host=True,
        min_cache_page_bytes=X.shape[0] // n_batches * X.shape[1],
    )
    Xy = ExtMemQuantileDMatrix(it, cache_host_ratio=1.0 if device == "cuda" else None)

    evals_result_2: Dict[str, Dict] = {}
    booster_2 = train(
        params,
        Xy,
        evals=[(Xy, "Train")],
        obj=LsObj0(device),
        num_boost_round=n_rounds,
        evals_result=evals_result_2,
    )
    np.testing.assert_allclose(
        evals_result_0["Train"]["rmse"], evals_result_2["Train"]["rmse"]
    )
    assert_allclose(device, booster_0.inplace_predict(X), booster_2.inplace_predict(X))


def run_eta(device: Device) -> None:
    """Test for learning rate."""
    X, y = make_regression(512, 16, random_state=2025, n_targets=3)

    def run(obj: Optional[Objective]) -> None:
        params = {
            "device": device,
            "multi_strategy": "multi_output_tree",
            "learning_rate": 1.0,
            "debug_synchronize": True,
            "base_score": 0.0,
        }
        Xy = QuantileDMatrix(X, y)
        booster_0 = train(params, Xy, num_boost_round=1, obj=obj)
        params["learning_rate"] = 0.1
        booster_1 = train(params, Xy, num_boost_round=1, obj=obj)
        params["learning_rate"] = 2.0
        booster_2 = train(params, Xy, num_boost_round=1, obj=obj)

        predt_0 = booster_0.predict(Xy)
        predt_1 = booster_1.predict(Xy)
        predt_2 = booster_2.predict(Xy)

        np.testing.assert_allclose(predt_0, predt_1 * 10, rtol=1e-6)
        np.testing.assert_allclose(predt_0 * 2, predt_2, rtol=1e-6)

    run(None)
    run(LsObj0(device))


def run_deterministic(device: Device) -> None:
    """Check the vector leaf implementation is deterministic."""
    X, y = make_regression(
        n_samples=int(2**16), n_features=64, random_state=1994, n_targets=5
    )

    def run() -> Booster:
        Xy = QuantileDMatrix(X, y)
        params = {
            "device": device,
            "multi_strategy": "multi_output_tree",
            "debug_synchronize": True,
        }
        return train(params, Xy, num_boost_round=16)

    booster_0 = run()
    booster_1 = run()
    raw_0 = booster_0.save_raw()
    raw_1 = booster_1.save_raw()
    assert raw_0 == raw_1


def run_column_sampling(device: Device) -> None:
    """Test column sampling with feature importance for multi-target trees."""
    n_features = 32
    X, y = make_regression(
        n_samples=1024, n_features=n_features, random_state=1994, n_targets=3
    )
    # First half of features have weight, second half has 0 weight (not sampled).
    feature_weights = np.zeros(shape=(n_features, 1), dtype=np.float32)
    feature_weights[: n_features // 2] = 1.0 / (n_features / 2)
    Xy = QuantileDMatrix(X, y, feature_weights=feature_weights)

    params = {
        "device": device,
        "multi_strategy": "multi_output_tree",
        "debug_synchronize": True,
        "colsample_bynode": 0.4,
    }
    booster = train(params, Xy, num_boost_round=16)

    # Test all importance types
    for importance_type in ["weight", "gain", "total_gain", "cover", "total_cover"]:
        scores: dict = booster.get_score(importance_type=importance_type)
        assert len(scores) > 0, f"No scores for {importance_type}"

        # Sampled features (first half) should be in scores
        for f in range(0, n_features // 2):
            assert f"f{f}" in scores, f"f{f} not in {importance_type} scores"

        # Non-sampled features (second half) should NOT be in scores
        for f in range(n_features // 2, n_features):
            assert f"f{f}" not in scores

        for score in scores.values():
            assert isinstance(score, float)
            assert score >= 0

    # sklearn Coef
    X, y = make_multilabel_classification(random_state=1994)
    clf = XGBClassifier(
        multi_strategy="multi_output_tree",
        importance_type="weight",
        device=device,
        colsample_bynode=0.2,
    )
    clf.fit(X, y, feature_weights=np.arange(0, X.shape[1]))
    fi = clf.feature_importances_
    assert fi[0] == 0.0
    assert fi[-1] > fi[1] * 5

    w = np.polynomial.Polynomial.fit(np.arange(0, X.shape[1]), fi, deg=1)
    assert w.coef[1] > 0.03


def run_grow_policy(device: Device, grow_policy: str) -> None:
    """Test grow policy (depthwise and lossguide) for vector leaf."""
    X, y = make_regression(
        n_samples=1024, n_features=16, random_state=1994, n_targets=3
    )
    Xy = QuantileDMatrix(X, y)

    params = {
        "device": device,
        "multi_strategy": "multi_output_tree",
        "debug_synchronize": True,
        "grow_policy": grow_policy,
    }

    evals_result = train_result(params, Xy, num_rounds=10)
    assert non_increasing(evals_result["train"]["rmse"])


def run_mixed_strategy(device: Device) -> None:
    """Test mixed multi_strategy with ResetStrategy callback."""
    X, y = make_classification(
        n_samples=1024, n_informative=8, n_classes=3, random_state=1994
    )
    Xy = DMatrix(data=X, label=y)

    booster = train(
        {
            "num_parallel_tree": 4,
            "num_class": 3,
            "objective": "multi:softprob",
            "multi_strategy": "multi_output_tree",
            "device": device,
            "debug_synchronize": True,
            "base_score": 0,
        },
        num_boost_round=16,
        dtrain=Xy,
        callbacks=[ResetStrategy()],
    )

    # Test model slicing - each boosting round should be iterable
    assert len(list(booster)) == 16

    # Test that sliced predictions sum to full prediction
    predt = booster.predict(Xy, output_margin=True)
    predt_sum = np.zeros(predt.shape)
    for t in booster:
        predt_sum += t.predict(Xy, output_margin=True)
    np.testing.assert_allclose(predt, predt_sum, atol=1e-5)

    # Test feature importance works with mixed trees
    for importance_type in ["weight", "gain", "total_gain", "cover", "total_cover"]:
        scores = booster.get_score(importance_type=importance_type)
        assert len(scores) > 0
        for score in scores.values():
            assert isinstance(score, float)
            assert score >= 0


def run_feature_importance_strategy_compare(device: Device) -> None:
    """Different strategies produce similar feature importance ratios."""
    n_features = 16
    X, y = make_classification(
        n_samples=2048,
        n_features=n_features,
        n_informative=10,
        n_classes=4,
        random_state=1994,
    )
    Xy = DMatrix(data=X, label=y)

    base_params: Dict[str, Any] = {
        "num_class": 4,
        "objective": "multi:softprob",
        "device": device,
        "debug_synchronize": True,
        "max_depth": 5,
    }

    # Train models with different strategies
    boosters = [
        train(
            {**base_params, "multi_strategy": "multi_output_tree"},
            Xy,
            num_boost_round=32,
        ),
        train(
            {**base_params, "multi_strategy": "one_output_per_tree"},
            Xy,
            num_boost_round=32,
        ),
        train(
            {**base_params, "multi_strategy": "multi_output_tree"},
            Xy,
            num_boost_round=32,
            callbacks=[ResetStrategy()],
        ),
    ]

    def get_normalized_importance(booster: Booster, importance_type: str) -> np.ndarray:
        """Get feature importance as normalized array (sums to 1)."""
        scores = booster.get_score(importance_type=importance_type)
        arr = np.array([scores.get(f"f{i}", 0.0) for i in range(n_features)])
        return arr / arr.sum() if arr.sum() > 0 else arr

    for importance_type in ["weight", "gain", "total_gain", "cover", "total_cover"]:
        imps = [get_normalized_importance(b, importance_type) for b in boosters]

        # Check that importances are not exactly the same (different strategies)
        assert not np.allclose(imps[0], imps[1])
        assert not np.allclose(imps[0], imps[2])

        # Check that normalized importances are similar (correlated)
        # All strategies should have reasonably similar importance patterns
        assert cosine_similarity([imps[0]], [imps[1]])[0, 0] > 0.9
        assert cosine_similarity([imps[0]], [imps[2]])[0, 0] > 0.9
        assert cosine_similarity([imps[1]], [imps[2]])[0, 0] > 0.9


# pylint: disable=too-many-arguments, too-many-locals
def _run_regression_objective_test(
    device: Device,
    objective: str,
    metric: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    extra_params: Optional[Dict[str, Any]] = None,
    check_pred_positive: bool = False,
    check_pred_probability: bool = False,
    check_pred_binary: bool = False,
    strictly_non_increasing: bool = True,
) -> None:
    params: Dict[str, Any] = {
        "objective": objective,
        "device": device,
        "multi_strategy": "multi_output_tree",
    }
    if extra_params:
        params.update(extra_params)

    n_samples = X.shape[0]
    n_targets = y.shape[1] if y.ndim > 1 else 1

    Xy = DMatrix(X, y)
    evals_result: Dict[str, Dict] = {}
    booster = train(
        params,
        Xy,
        evals=[(Xy, "Train")],
        verbose_eval=False,
        evals_result=evals_result,
        num_boost_round=16,
    )
    predt = booster.predict(Xy)
    assert predt.shape == (n_samples, n_targets)

    if check_pred_positive:
        assert (predt > 0).all()
    if check_pred_probability:
        assert (predt > 0).all() and (predt < 1).all()
    if check_pred_binary:
        assert set(np.unique(predt)).issubset({0.0, 1.0})

    metric_vals = evals_result["Train"][metric]
    if strictly_non_increasing:
        assert non_increasing(metric_vals)
    else:
        assert metric_vals[-1] < metric_vals[0]


def run_reg_squarederror(device: Device) -> None:
    """Test squared error regression with vector leaf."""
    n_samples, n_targets = 1024, 3
    X, y = make_regression(
        n_samples=n_samples, n_features=16, n_targets=n_targets, random_state=2026
    )
    _run_regression_objective_test(device, "reg:squarederror", "rmse", X, y)


def run_reg_logistic(device: Device) -> None:
    """Test logistic regression for probability with vector leaf."""
    n_samples, n_targets = 1024, 3
    rng = np.random.default_rng(2026)
    X = rng.standard_normal((n_samples, 16))
    y = rng.uniform(0.0, 1.0, (n_samples, n_targets))  # Labels in [0, 1]
    _run_regression_objective_test(
        device, "reg:logistic", "rmse", X, y, check_pred_probability=True
    )


def run_reg_gamma(device: Device) -> None:
    """Test gamma regression with vector leaf."""
    n_samples, n_targets = 1024, 3
    rng = np.random.default_rng(2026)
    X = rng.standard_normal((n_samples, 16))
    y = rng.gamma(2.0, 2.0, (n_samples, n_targets))  # Labels must be positive
    _run_regression_objective_test(
        device, "reg:gamma", "gamma-deviance", X, y, check_pred_positive=True
    )


def run_reg_squaredlogerror(device: Device) -> None:
    """Test squared log error regression with vector leaf."""
    n_samples, n_targets = 1024, 3
    rng = np.random.default_rng(2026)
    X = rng.standard_normal((n_samples, 16))
    y = np.abs(rng.standard_normal((n_samples, n_targets))) + 0.1  # Labels > -1
    _run_regression_objective_test(device, "reg:squaredlogerror", "rmsle", X, y)


def run_reg_pseudohubererror(device: Device) -> None:
    """Test pseudo huber error regression with vector leaf."""
    n_samples, n_targets = 1024, 3
    X, y = make_regression(
        n_samples=n_samples, n_features=16, n_targets=n_targets, random_state=2026
    )
    _run_regression_objective_test(
        device, "reg:pseudohubererror", "mphe", X, y, extra_params={"huber_slope": 1.0}
    )


def run_binary_logitraw(device: Device) -> None:
    """Test binary logitraw with vector leaf (multi-label classification)."""
    n_samples = 1024
    X, y = make_multilabel_classification(n_samples, random_state=2026)
    _run_regression_objective_test(
        device, "binary:logitraw", "logloss", X, y, strictly_non_increasing=False
    )


def run_binary_hinge(device: Device) -> None:
    """Test binary hinge loss with vector leaf (multi-label classification)."""
    n_samples = 1024
    X, y = make_multilabel_classification(n_samples, random_state=2026)
    _run_regression_objective_test(
        device,
        "binary:hinge",
        "error",
        X,
        y,
        check_pred_binary=True,
        strictly_non_increasing=False,
    )


def run_count_poisson(device: Device) -> None:
    """Test Poisson regression with vector leaf."""
    n_samples, n_targets = 1024, 3
    rng = np.random.default_rng(2026)
    X = rng.standard_normal((n_samples, 16))
    y = rng.poisson(5, (n_samples, n_targets)).astype(np.float32)  # Labels >= 0
    _run_regression_objective_test(
        device, "count:poisson", "poisson-nloglik", X, y, check_pred_positive=True
    )


def run_reg_tweedie(device: Device) -> None:
    """Test Tweedie regression with vector leaf."""
    n_samples, n_targets = 1024, 3
    rng = np.random.default_rng(2026)
    X = rng.standard_normal((n_samples, 16))
    y = rng.gamma(2.0, 2.0, (n_samples, n_targets))  # Labels >= 0
    _run_regression_objective_test(
        device, "reg:tweedie", "tweedie-nloglik@1.5", X, y, check_pred_positive=True
    )


def all_reg_objectives() -> List[Callable[[Device], None]]:
    """List of obj tests."""
    objs: List[Callable[[Device], None]] = [
        run_reg_squarederror,
        run_reg_logistic,
        run_reg_gamma,
        run_reg_squaredlogerror,
        run_reg_pseudohubererror,
        run_binary_logitraw,
        run_binary_hinge,
        run_count_poisson,
        run_reg_tweedie,
    ]
    return objs


def _make_subsample_params(device: Device, sampling_method: str) -> dict:
    params = {
        "device": device,
        "tree_method": "hist",
        "multi_strategy": "multi_output_tree",
        "subsample": 0.5,
        "sampling_method": sampling_method,
        "max_depth": 6,
        "debug_synchronize": True,
        "seed": 2026,
    }
    return params


def run_subsample(device: Device, sampling_method: str) -> None:
    """Test row subsampling."""
    n_samples = 2048
    X, y = make_regression(
        n_samples=n_samples, n_features=16, n_targets=3, random_state=2026
    )
    Xy = QuantileDMatrix(X, y)

    params = _make_subsample_params(device, sampling_method)

    evals_result = train_result(params, Xy, num_rounds=16)
    # Training should converge with subsampling
    assert non_increasing(evals_result["train"]["rmse"], tolerance=0.01)

    # Test with quantile regression
    params = _make_subsample_params(device, sampling_method)
    params["objective"] = "reg:quantileerror"
    params["quantile_alpha"] = [0.25, 0.5, 0.75]
    Xy_single = QuantileDMatrix(X, y[:, 0])
    evals_result_q = train_result(params, Xy_single, num_rounds=16)
    assert non_increasing(evals_result_q["train"]["quantile"], tolerance=0.01)


def run_gradient_based_sampling_accuracy(device: Device) -> None:
    """Test that gradient-based sampling provides better accuracy."""
    n_samples = 4096
    X, y = make_regression(
        n_samples=n_samples, n_features=16, n_targets=3, random_state=2026
    )
    Xy = QuantileDMatrix(X, y)

    params_uniform = _make_subsample_params(device, "uniform")

    def run(obj: Callable | None) -> None:
        # Train with uniform sampling
        evals_uniform: Dict[str, Dict] = {}
        train(
            params_uniform,
            Xy,
            num_boost_round=32,
            evals=[(Xy, "train")],
            obj=obj,
            verbose_eval=False,
            evals_result=evals_uniform,
        )

        # Train with gradient-based sampling
        params_grad = _make_subsample_params(device, "gradient_based")
        evals_grad: Dict[str, Dict] = {}
        train(
            params_grad,
            Xy,
            num_boost_round=32,
            evals=[(Xy, "train")],
            obj=obj,
            verbose_eval=False,
            evals_result=evals_grad,
        )

        uniform_final = evals_uniform["train"]["rmse"][-1]
        grad_final = evals_grad["train"]["rmse"][-1]
        assert non_increasing(evals_uniform["train"]["rmse"])
        assert non_increasing(evals_grad["train"]["rmse"])
        if obj is None:
            assert grad_final < uniform_final

    run(None)
    run(LsObj2(device, False))
