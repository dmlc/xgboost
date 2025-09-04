"""Tests for updaters."""

import json
from functools import partial, update_wrapper
from string import ascii_lowercase
from typing import Any, Dict, List, Optional, Union, overload

import numpy as np
import pytest

import xgboost as xgb
import xgboost.testing as tm
from xgboost.core import _parse_version
from xgboost.data import is_pd_cat_dtype

from ..core import DataIter
from .data_iter import CatIter
from .utils import Device


@overload
def get_basescore(model: xgb.XGBModel) -> List[float]: ...


@overload
def get_basescore(model: xgb.Booster) -> List[float]: ...


@overload
def get_basescore(model: Dict[str, Any]) -> List[float]: ...


def get_basescore(
    model: Union[xgb.XGBModel, xgb.Booster, Dict],
) -> List[float]:
    """Get base score from an XGBoost sklearn estimator."""
    if isinstance(model, xgb.XGBModel):
        model = model.get_booster()

    if isinstance(model, dict):
        jintercept = model["learner"]["learner_model_param"]["base_score"]
    else:
        jintercept = json.loads(model.save_config())["learner"]["learner_model_param"][
            "base_score"
        ]
    return json.loads(jintercept)


# pylint: disable=too-many-statements
def check_init_estimation(tree_method: str, device: Device) -> None:
    """Test for init estimation."""
    from sklearn.datasets import (
        make_classification,
        make_multilabel_classification,
        make_regression,
    )

    def run_reg(X: np.ndarray, y: np.ndarray) -> None:  # pylint: disable=invalid-name
        reg = xgb.XGBRegressor(
            tree_method=tree_method, max_depth=1, n_estimators=1, device=device
        )
        reg.fit(X, y, eval_set=[(X, y)])
        base_score_0 = get_basescore(reg)
        score_0 = reg.evals_result()["validation_0"]["rmse"][0]

        n_targets = 1 if y.ndim == 1 else y.shape[1]
        intercept = np.full(shape=(n_targets,), fill_value=0.5, dtype=np.float32)
        reg = xgb.XGBRegressor(
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
        clf = xgb.XGBClassifier(
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
        clf = xgb.XGBClassifier(
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

    X, y = make_classification(
        n_samples=4096, random_state=17, n_classes=5, n_informative=20, n_redundant=0
    )
    intercept = run_clf(X, y)
    np.testing.assert_allclose(np.sum(intercept), 1.0)
    assert np.all(np.array(intercept) > 0)
    np_int = (
        np.histogram(
            y, bins=np.concatenate([np.unique(y), np.array([np.finfo(np.float32).max])])
        )[0]
        / y.shape[0]
    )
    np.testing.assert_allclose(intercept, np_int)

    rng = np.random.default_rng(1994)
    w = rng.uniform(low=0, high=1, size=(y.shape[0],))
    intercept = run_clf(X, y, w)
    np.testing.assert_allclose(np.sum(intercept), 1.0)
    assert np.all(np.array(intercept) > 0)


# pylint: disable=too-many-locals
def check_quantile_loss(tree_method: str, weighted: bool, device: Device) -> None:
    """Test for quantile loss."""
    from sklearn.datasets import make_regression
    from sklearn.metrics import mean_pinball_loss

    from xgboost.sklearn import _metric_decorator

    n_samples = 4096
    n_features = 8
    n_estimators = 8

    rng = np.random.RandomState(1994)
    # pylint: disable=unbalanced-tuple-unpacking
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        random_state=rng,
    )
    if weighted:
        weight = rng.random(size=n_samples)
    else:
        weight = None

    Xy = xgb.QuantileDMatrix(X, y, weight=weight)

    alpha = np.array([0.1, 0.5])
    # non-zero base score can cause floating point difference with GPU predictor.
    # multi-class has small difference than single target in the prediction kernel
    base_score = np.zeros(shape=alpha.shape, dtype=np.float32)
    evals_result: Dict[str, Dict] = {}
    booster_multi = xgb.train(
        {
            "objective": "reg:quantileerror",
            "tree_method": tree_method,
            "device": device,
            "quantile_alpha": alpha,
            "base_score": base_score,
        },
        Xy,
        num_boost_round=n_estimators,
        evals=[(Xy, "Train")],
        evals_result=evals_result,
    )
    predt_multi = booster_multi.predict(Xy, strict_shape=True)

    assert tm.non_increasing(evals_result["Train"]["quantile"])
    assert evals_result["Train"]["quantile"][-1] < 20.0
    # check that there's a way to use custom metric and compare the results.
    metrics = [
        _metric_decorator(
            update_wrapper(
                partial(mean_pinball_loss, sample_weight=weight, alpha=alpha[i]),
                mean_pinball_loss,
            )
        )
        for i in range(alpha.size)
    ]

    predts = np.empty(predt_multi.shape)
    for i in range(alpha.shape[0]):
        a = alpha[i]

        booster_i = xgb.train(
            {
                "objective": "reg:quantileerror",
                "tree_method": tree_method,
                "device": device,
                "quantile_alpha": a,
                "base_score": base_score[i],
            },
            Xy,
            num_boost_round=n_estimators,
            evals=[(Xy, "Train")],
            custom_metric=metrics[i],
            evals_result=evals_result,
        )
        assert tm.non_increasing(evals_result["Train"]["quantile"])
        assert evals_result["Train"]["quantile"][-1] < 30.0
        np.testing.assert_allclose(
            np.array(evals_result["Train"]["quantile"]),
            np.array(evals_result["Train"]["mean_pinball_loss"]),
            atol=1e-6,
            rtol=1e-6,
        )
        predts[:, i] = booster_i.predict(Xy)

    for i in range(alpha.shape[0]):
        np.testing.assert_allclose(predts[:, i], predt_multi[:, i])


def check_quantile_loss_extmem(
    n_samples_per_batch: int,
    n_features: int,
    n_batches: int,
    tree_method: str,
    device: str,
) -> None:
    """Check external memory with the quantile objective."""
    it = tm.IteratorForTest(
        *tm.make_batches(n_samples_per_batch, n_features, n_batches, device != "cpu"),
        cache="cache",
        on_host=False,
    )
    Xy_it = xgb.DMatrix(it)
    params = {
        "tree_method": tree_method,
        "objective": "reg:quantileerror",
        "device": device,
        "quantile_alpha": [0.2, 0.8],
    }
    booster_it = xgb.train(params, Xy_it)
    X, y, w = it.as_arrays()
    Xy = xgb.DMatrix(X, y, weight=w)
    booster = xgb.train(params, Xy)

    predt_it = booster_it.predict(Xy_it)
    predt = booster.predict(Xy)

    np.testing.assert_allclose(predt, predt_it)


def check_extmem_qdm(  # pylint: disable=too-many-arguments
    n_samples_per_batch: int,
    n_features: int,
    *,
    n_batches: int,
    n_bins: int,
    device: str,
    on_host: bool,
    is_cat: bool,
) -> None:
    """Basic test for the `ExtMemQuantileDMatrix`."""

    if is_cat:
        it: DataIter = CatIter(
            n_samples_per_batch=n_samples_per_batch,
            n_features=n_features,
            n_batches=n_batches,
            n_cats=5,
            sparsity=0.0,
            cat_ratio=0.5,
            onehot=False,
            device=device,
            cache="cache",
        )
    else:
        it = tm.IteratorForTest(
            *tm.make_batches(
                n_samples_per_batch, n_features, n_batches, use_cupy=device != "cpu"
            ),
            cache="cache",
            on_host=on_host,
        )

    Xy_it = xgb.ExtMemQuantileDMatrix(it, max_bin=n_bins, enable_categorical=is_cat)
    with pytest.raises(ValueError, match="Only the `hist`"):
        booster_it = xgb.train(
            {"device": device, "tree_method": "approx", "max_bin": n_bins},
            Xy_it,
            num_boost_round=8,
        )

    booster_it = xgb.train(
        {"device": device, "max_bin": n_bins}, Xy_it, num_boost_round=8
    )
    if is_cat:
        it = CatIter(
            n_samples_per_batch=n_samples_per_batch,
            n_features=n_features,
            n_batches=n_batches,
            n_cats=5,
            sparsity=0.0,
            cat_ratio=0.5,
            onehot=False,
            device=device,
            cache=None,
        )
    else:
        it = tm.IteratorForTest(
            *tm.make_batches(
                n_samples_per_batch, n_features, n_batches, use_cupy=device != "cpu"
            ),
            cache=None,
        )
    Xy = xgb.QuantileDMatrix(it, max_bin=n_bins, enable_categorical=is_cat)
    booster = xgb.train({"device": device, "max_bin": n_bins}, Xy, num_boost_round=8)

    cut_it = Xy_it.get_quantile_cut()
    cut = Xy.get_quantile_cut()
    np.testing.assert_allclose(cut_it[0], cut[0])
    np.testing.assert_allclose(cut_it[1], cut[1])

    predt_it = booster_it.predict(Xy_it)
    predt = booster.predict(Xy)
    np.testing.assert_allclose(predt_it, predt)


def check_cut(
    n_entries: int, indptr: np.ndarray, data: np.ndarray, dtypes: Any
) -> None:
    """Check the cut values."""
    assert data.shape[0] == indptr[-1]
    assert data.shape[0] == n_entries

    assert indptr.dtype == np.uint64
    for i in range(1, indptr.size):
        beg = int(indptr[i - 1])
        end = int(indptr[i])
        for j in range(beg + 1, end):
            assert data[j] > data[j - 1]
            if is_pd_cat_dtype(dtypes.iloc[i - 1]):
                assert data[j] == data[j - 1] + 1


def check_get_quantile_cut_device(tree_method: str, use_cupy: bool) -> None:
    """Check with optional cupy."""
    import pandas as pd

    n_samples = 1024
    n_features = 14
    max_bin = 16
    dtypes = pd.Series([np.float32] * n_features)

    # numerical
    X, y, w = tm.make_regression(n_samples, n_features, use_cupy=use_cupy)
    # - qdm
    Xyw: xgb.DMatrix = xgb.QuantileDMatrix(X, y, weight=w, max_bin=max_bin)
    indptr, data = Xyw.get_quantile_cut()
    check_cut((max_bin + 1) * n_features, indptr, data, dtypes)
    # - dm
    Xyw = xgb.DMatrix(X, y, weight=w)
    xgb.train({"tree_method": tree_method, "max_bin": max_bin}, Xyw)
    indptr, data = Xyw.get_quantile_cut()
    check_cut((max_bin + 1) * n_features, indptr, data, dtypes)
    # - ext mem
    n_batches = 3
    n_samples_per_batch = 256
    it = tm.IteratorForTest(
        *tm.make_batches(n_samples_per_batch, n_features, n_batches, use_cupy),
        cache="cache",
        on_host=False,
    )
    Xy: xgb.DMatrix = xgb.DMatrix(it)
    xgb.train({"tree_method": tree_method, "max_bin": max_bin}, Xyw)
    indptr, data = Xyw.get_quantile_cut()
    check_cut((max_bin + 1) * n_features, indptr, data, dtypes)

    # categorical
    n_categories = 32
    X, y = tm.make_categorical(
        n_samples, n_features, n_categories, onehot=False, sparsity=0.8
    )
    if use_cupy:
        import cudf
        import cupy as cp

        X = cudf.from_pandas(X)
        y = cp.array(y)
    # - qdm
    Xy = xgb.QuantileDMatrix(X, y, max_bin=max_bin, enable_categorical=True)
    indptr, data = Xy.get_quantile_cut()
    check_cut(n_categories * n_features, indptr, data, X.dtypes)
    # - dm
    Xy = xgb.DMatrix(X, y, enable_categorical=True)
    xgb.train({"tree_method": tree_method, "max_bin": max_bin}, Xy)
    indptr, data = Xy.get_quantile_cut()
    check_cut(n_categories * n_features, indptr, data, X.dtypes)

    # mixed
    X, y = tm.make_categorical(
        n_samples, n_features, n_categories, onehot=False, sparsity=0.8, cat_ratio=0.5
    )
    n_cat_features = len([0 for dtype in X.dtypes if is_pd_cat_dtype(dtype)])
    n_num_features = n_features - n_cat_features
    n_entries = n_categories * n_cat_features + (max_bin + 1) * n_num_features
    # - qdm
    Xy = xgb.QuantileDMatrix(X, y, max_bin=max_bin, enable_categorical=True)
    indptr, data = Xy.get_quantile_cut()
    check_cut(n_entries, indptr, data, X.dtypes)
    # - dm
    Xy = xgb.DMatrix(X, y, enable_categorical=True)
    xgb.train({"tree_method": tree_method, "max_bin": max_bin}, Xy)
    indptr, data = Xy.get_quantile_cut()
    check_cut(n_entries, indptr, data, X.dtypes)


def check_get_quantile_cut(tree_method: str, device: str) -> None:
    """Check the quantile cut getter."""

    use_cupy = device.startswith("cuda")
    check_get_quantile_cut_device(tree_method, False)
    if use_cupy:
        check_get_quantile_cut_device(tree_method, True)


USE_ONEHOT = np.iinfo(np.int32).max
USE_PART = 1


def _create_dmatrix(  # pylint: disable=too-many-arguments
    n_samples: int,
    n_features: int,
    *,
    n_cats: int,
    device: str,
    sparsity: float,
    tree_method: str,
    onehot: bool,
    extmem: bool,
    enable_categorical: bool,
) -> xgb.DMatrix:
    n_batches = max(min(2, n_samples), 1)
    it = CatIter(
        n_samples // n_batches,
        n_features,
        n_batches=n_batches,
        sparsity=sparsity,
        cat_ratio=1.0,
        n_cats=n_cats,
        onehot=onehot,
        device=device,
        cache="cache" if extmem else None,
    )
    if extmem:
        if tree_method == "hist":
            Xy: xgb.DMatrix = xgb.ExtMemQuantileDMatrix(
                it, enable_categorical=enable_categorical
            )
        elif tree_method == "approx":
            Xy = xgb.DMatrix(it, enable_categorical=enable_categorical)
        else:
            raise ValueError(f"tree_method {tree_method} not supported.")
    else:
        cat, label = it.xy()
        Xy = xgb.DMatrix(cat, label, enable_categorical=enable_categorical)
    return Xy


def check_categorical_ohe(  # pylint: disable=too-many-arguments
    *,
    rows: int,
    cols: int,
    rounds: int,
    cats: int,
    device: str,
    tree_method: str,
    extmem: bool,
) -> None:
    "Test for one-hot encoding with categorical data."

    by_etl_results: Dict[str, Dict[str, List[float]]] = {}
    by_builtin_results: Dict[str, Dict[str, List[float]]] = {}

    parameters: Dict[str, Any] = {
        "tree_method": tree_method,
        # Use one-hot exclusively
        "max_cat_to_onehot": USE_ONEHOT,
        "device": device,
    }

    Xy_onehot = _create_dmatrix(
        rows,
        cols,
        n_cats=cats,
        device=device,
        sparsity=0.0,
        onehot=True,
        tree_method=tree_method,
        extmem=extmem,
        enable_categorical=False,
    )
    xgb.train(
        parameters,
        Xy_onehot,
        num_boost_round=rounds,
        evals=[(Xy_onehot, "Train")],
        evals_result=by_etl_results,
    )

    Xy_cat = _create_dmatrix(
        rows,
        cols,
        n_cats=cats,
        device=device,
        sparsity=0.0,
        tree_method=tree_method,
        onehot=False,
        extmem=extmem,
        enable_categorical=True,
    )
    xgb.train(
        parameters,
        Xy_cat,
        num_boost_round=rounds,
        evals=[(Xy_cat, "Train")],
        evals_result=by_builtin_results,
    )

    # There are guidelines on how to specify tolerance based on considering output
    # as random variables. But in here the tree construction is extremely sensitive
    # to floating point errors. An 1e-5 error in a histogram bin can lead to an
    # entirely different tree. So even though the test is quite lenient, hypothesis
    # can still pick up falsifying examples from time to time.
    np.testing.assert_allclose(
        np.array(by_etl_results["Train"]["rmse"]),
        np.array(by_builtin_results["Train"]["rmse"]),
        rtol=1e-3,
    )
    assert tm.non_increasing(by_builtin_results["Train"]["rmse"])

    by_grouping: Dict[str, Dict[str, List[float]]] = {}
    # switch to partition-based splits
    parameters["max_cat_to_onehot"] = USE_PART
    parameters["reg_lambda"] = 0
    xgb.train(
        parameters,
        Xy_cat,
        num_boost_round=rounds,
        evals=[(Xy_cat, "Train")],
        evals_result=by_grouping,
    )
    rmse_oh = by_builtin_results["Train"]["rmse"]
    rmse_group = by_grouping["Train"]["rmse"]
    # always better or equal to onehot when there's no regularization.
    for a, b in zip(rmse_oh, rmse_group):
        assert a >= b

    parameters["reg_lambda"] = 1.0
    by_grouping = {}
    xgb.train(
        parameters,
        Xy_cat,
        num_boost_round=32,
        evals=[(Xy_cat, "Train")],
        evals_result=by_grouping,
    )
    assert tm.non_increasing(by_grouping["Train"]["rmse"]), by_grouping


def check_categorical_missing(  # pylint: disable=too-many-arguments
    rows: int,
    cols: int,
    cats: int,
    *,
    device: str,
    tree_method: str,
    extmem: bool,
) -> None:
    """Check categorical data with missing values."""
    parameters: Dict[str, Any] = {"tree_method": tree_method, "device": device}
    Xy = _create_dmatrix(
        rows,
        cols,
        n_cats=cats,
        sparsity=0.5,
        device=device,
        tree_method=tree_method,
        onehot=False,
        extmem=extmem,
        enable_categorical=True,
    )
    label = Xy.get_label()

    def run(max_cat_to_onehot: int) -> None:
        # Test with onehot splits
        parameters["max_cat_to_onehot"] = max_cat_to_onehot

        evals_result: Dict[str, Dict] = {}
        booster = xgb.train(
            parameters,
            Xy,
            num_boost_round=16,
            evals=[(Xy, "Train")],
            evals_result=evals_result,
        )
        assert tm.non_increasing(evals_result["Train"]["rmse"])
        y_predt = booster.predict(Xy)

        rmse = tm.root_mean_square(label, y_predt)
        np.testing.assert_allclose(rmse, evals_result["Train"]["rmse"][-1], rtol=2e-5)

    # Test with OHE split
    run(USE_ONEHOT)

    # Test with partition-based split
    run(USE_PART)


def run_max_cat(tree_method: str, device: Device) -> None:
    """Test data with size smaller than number of categories."""
    import pandas as pd

    rng = np.random.default_rng(0)
    n_cat = 100
    n = 5

    X = pd.Series(
        ["".join(rng.choice(list(ascii_lowercase), size=3)) for i in range(n_cat)],
        dtype="category",
    )[:n].to_frame()

    reg = xgb.XGBRegressor(
        enable_categorical=True,
        tree_method=tree_method,
        device=device,
        n_estimators=10,
    )
    y = pd.Series(range(n))
    reg.fit(X=X, y=y, eval_set=[(X, y)])
    assert tm.non_increasing(reg.evals_result()["validation_0"]["rmse"])


def run_invalid_category(tree_method: str, device: Device) -> None:
    """Test with invalid categorical inputs."""
    rng = np.random.default_rng()
    # too large
    X = rng.integers(low=0, high=4, size=1000).reshape(100, 10)
    y = rng.normal(loc=0, scale=1, size=100)
    X[13, 7] = np.iinfo(np.int32).max + 1

    # Check is performed during sketching.
    Xy = xgb.DMatrix(X, y, feature_types=["c"] * 10)
    with pytest.raises(ValueError):
        xgb.train({"tree_method": tree_method, "device": device}, Xy)

    X[13, 7] = 16777216
    Xy = xgb.DMatrix(X, y, feature_types=["c"] * 10)
    with pytest.raises(ValueError):
        xgb.train({"tree_method": tree_method, "device": device}, Xy)

    # mixed positive and negative values
    X = rng.normal(loc=0, scale=1, size=1000).reshape(100, 10)  # type: ignore
    y = rng.normal(loc=0, scale=1, size=100)

    Xy = xgb.DMatrix(X, y, feature_types=["c"] * 10)
    with pytest.raises(ValueError):
        xgb.train({"tree_method": tree_method, "device": device}, Xy)

    if device == "cuda":
        import cupy as cp

        X, y = cp.array(X), cp.array(y)
        with pytest.raises(ValueError):
            Xy = xgb.QuantileDMatrix(X, y, feature_types=["c"] * 10)


def run_adaptive(tree_method: str, weighted: bool, device: Device) -> None:
    """Test for adaptive trees."""
    rng = np.random.RandomState(1994)
    from sklearn import __version__ as sklearn_version
    from sklearn.datasets import make_regression
    from sklearn.utils import stats

    n_samples = 256
    X, y = make_regression(  # pylint: disable=unbalanced-tuple-unpacking
        n_samples, 16, random_state=rng
    )
    if weighted:
        w = rng.normal(size=n_samples)
        w -= w.min()
        Xy = xgb.DMatrix(X, y, weight=w)

        (sk_major, sk_minor, _), _ = _parse_version(sklearn_version)
        if sk_major > 1 or sk_minor >= 7:
            kwargs = {"percentile_rank": 50}
        else:
            kwargs = {"percentile": 50}
        base_score = stats._weighted_percentile(  # pylint: disable=protected-access
            y, w, **kwargs
        )
    else:
        Xy = xgb.DMatrix(X, y)
        base_score = np.median(y)

    booster_0 = xgb.train(
        {
            "tree_method": tree_method,
            "base_score": base_score,
            "objective": "reg:absoluteerror",
            "device": device,
        },
        Xy,
        num_boost_round=1,
    )
    booster_1 = xgb.train(
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

    raw_booster = booster_1.save_raw(raw_format="ubj")
    booster_2 = xgb.Booster(model_file=raw_booster)
    config_2 = json.loads(booster_2.save_config())
    assert get_basescore(config_1) == get_basescore(config_2)

    booster_0 = xgb.train(
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

    evals_result: Dict[str, Dict[str, list]] = {}
    xgb.train(
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
    assert tm.non_increasing(mae)


def train_result(
    param: Dict[str, Any], dmat: xgb.DMatrix, num_rounds: int
) -> Dict[str, Any]:
    """Get training result from parameters and data."""
    result: Dict[str, Any] = {}
    booster = xgb.train(
        param,
        dmat,
        num_rounds,
        evals=[(dmat, "train")],
        verbose_eval=False,
        evals_result=result,
    )
    assert booster.num_features() == dmat.num_col()
    assert booster.num_boosted_rounds() == num_rounds
    assert booster.feature_names == dmat.feature_names
    assert booster.feature_types == dmat.feature_types

    return result


class ResetStrategy(xgb.callback.TrainingCallback):
    """Callback for testing multi-output."""

    def after_iteration(self, model: xgb.Booster, epoch: int, evals_log: dict) -> bool:
        if epoch % 2 == 0:
            model.set_param({"multi_strategy": "multi_output_tree"})
        else:
            model.set_param({"multi_strategy": "one_output_per_tree"})
        return False
