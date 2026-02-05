"""Tests for updaters."""

import json
from functools import partial, update_wrapper
from string import ascii_lowercase
from typing import Any, Dict, List, Union, overload

import numpy as np
import pytest
from sklearn.datasets import make_regression

import xgboost.testing as tm

from ..callback import TrainingCallback
from ..compat import import_cupy
from ..core import (
    Booster,
    DataIter,
    DMatrix,
    ExtMemQuantileDMatrix,
    QuantileDMatrix,
)
from ..data import is_pd_cat_dtype
from ..sklearn import XGBModel, XGBRegressor
from ..training import train
from .data import IteratorForTest, make_batches, make_categorical
from .data_iter import CatIter
from .utils import Device, assert_allclose, non_increasing


@overload
def get_basescore(model: XGBModel) -> List[float]: ...


@overload
def get_basescore(model: Booster) -> List[float]: ...


@overload
def get_basescore(model: Dict[str, Any]) -> List[float]: ...


def get_basescore(
    model: Union[XGBModel, Booster, Dict],
) -> List[float]:
    """Get base score from an XGBoost sklearn estimator."""
    if isinstance(model, XGBModel):
        model = model.get_booster()

    if isinstance(model, dict):
        jintercept = model["learner"]["learner_model_param"]["base_score"]
    else:
        jintercept = json.loads(model.save_config())["learner"]["learner_model_param"][
            "base_score"
        ]
    return json.loads(jintercept)


# pylint: disable=too-many-locals
def check_quantile_loss(tree_method: str, weighted: bool, device: Device) -> None:
    """Test for quantile loss."""
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

    Xy = QuantileDMatrix(X, y, weight=weight)

    alpha = np.array([0.1, 0.5])
    # non-zero base score can cause floating point difference with GPU predictor.
    # multi-class has small difference than single target in the prediction kernel
    base_score = np.zeros(shape=alpha.shape, dtype=np.float32)
    evals_result: Dict[str, Dict] = {}
    booster_multi = train(
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

    assert non_increasing(evals_result["Train"]["quantile"])
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

        booster_i = train(
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
        assert non_increasing(evals_result["Train"]["quantile"])
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


def check_quantile_loss_rf(
    device: Device, tree_method: str, multi_strategy: str
) -> None:
    """Test quantile loss with boosting random forest."""
    # pylint: disable=unbalanced-tuple-unpacking
    X, y = make_regression(n_samples=2048, n_features=16, random_state=2026)
    Xy = DMatrix(X, y)

    def run(params: Dict[str, Any], metric: str) -> None:
        evals_result_0: Dict[str, Dict] = {}
        params["num_parallel_tree"] = 2
        train(
            params,
            Xy,
            num_boost_round=8,
            evals=[(Xy, "Train")],
            evals_result=evals_result_0,
        )

        evals_result_1: Dict[str, Dict] = {}
        params["num_parallel_tree"] = 1
        train(
            params,
            Xy,
            num_boost_round=8,
            evals=[(Xy, "Train")],
            evals_result=evals_result_1,
        )
        # Without subsample, the result should be the same (barring floating point
        # errors).
        np.testing.assert_allclose(
            evals_result_0["Train"][metric], evals_result_1["Train"][metric]
        )
        assert non_increasing(evals_result_0["Train"][metric])

    alpha = np.array([0.1, 0.5, 0.9])
    params = {
        "objective": "reg:quantileerror",
        "tree_method": tree_method,
        "device": device,
        "quantile_alpha": alpha,
        "multi_strategy": multi_strategy,
    }
    run(params, "quantile")

    # Now test with MAE
    params.pop("quantile_alpha")
    params["objective"] = "reg:absoluteerror"
    run(params, "mae")


def check_quantile_loss_extmem(
    n_samples_per_batch: int,
    n_features: int,
    n_batches: int,
    tree_method: str,
    device: str,
) -> None:
    """Check external memory with the quantile objective."""
    it = IteratorForTest(
        *make_batches(n_samples_per_batch, n_features, n_batches, device != "cpu"),
        cache="cache",
        on_host=False,
    )
    Xy_it = DMatrix(it)
    params = {
        "tree_method": tree_method,
        "objective": "reg:quantileerror",
        "device": device,
        "quantile_alpha": [0.2, 0.8],
    }
    booster_it = train(params, Xy_it)
    X, y, w = it.as_arrays()
    Xy = DMatrix(X, y, weight=w)
    booster = train(params, Xy)

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
        it = IteratorForTest(
            *make_batches(
                n_samples_per_batch, n_features, n_batches, use_cupy=device != "cpu"
            ),
            cache="cache",
            on_host=on_host,
        )

    Xy_it = ExtMemQuantileDMatrix(it, max_bin=n_bins, enable_categorical=is_cat)
    with pytest.raises(ValueError, match="Only the `hist`"):
        booster_it = train(
            {"device": device, "tree_method": "approx", "max_bin": n_bins},
            Xy_it,
            num_boost_round=8,
        )

    booster_it = train({"device": device, "max_bin": n_bins}, Xy_it, num_boost_round=8)
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
        it = IteratorForTest(
            *make_batches(
                n_samples_per_batch, n_features, n_batches, use_cupy=device != "cpu"
            ),
            cache=None,
        )
    Xy = QuantileDMatrix(it, max_bin=n_bins, enable_categorical=is_cat)
    booster = train({"device": device, "max_bin": n_bins}, Xy, num_boost_round=8)

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
    Xyw: DMatrix = QuantileDMatrix(X, y, weight=w, max_bin=max_bin)
    indptr, data = Xyw.get_quantile_cut()
    check_cut((max_bin + 1) * n_features, indptr, data, dtypes)
    # - dm
    Xyw = DMatrix(X, y, weight=w)
    train({"tree_method": tree_method, "max_bin": max_bin}, Xyw)
    indptr, data = Xyw.get_quantile_cut()
    check_cut((max_bin + 1) * n_features, indptr, data, dtypes)
    # - ext mem
    n_batches = 3
    n_samples_per_batch = 256
    it = IteratorForTest(
        *make_batches(n_samples_per_batch, n_features, n_batches, use_cupy),
        cache="cache",
        on_host=False,
    )
    Xy: DMatrix = DMatrix(it)
    train({"tree_method": tree_method, "max_bin": max_bin}, Xyw)
    indptr, data = Xyw.get_quantile_cut()
    check_cut((max_bin + 1) * n_features, indptr, data, dtypes)

    # categorical
    n_categories = 32
    X, y = make_categorical(
        n_samples, n_features, n_categories, onehot=False, sparsity=0.8
    )
    if use_cupy:
        import cudf

        cp = import_cupy()

        X = cudf.from_pandas(X)
        y = cp.array(y)
    # - qdm
    Xy = QuantileDMatrix(X, y, max_bin=max_bin, enable_categorical=True)
    indptr, data = Xy.get_quantile_cut()
    check_cut(n_categories * n_features, indptr, data, X.dtypes)
    # - dm
    Xy = DMatrix(X, y, enable_categorical=True)
    train({"tree_method": tree_method, "max_bin": max_bin}, Xy)
    indptr, data = Xy.get_quantile_cut()
    check_cut(n_categories * n_features, indptr, data, X.dtypes)

    # mixed
    X, y = make_categorical(
        n_samples, n_features, n_categories, onehot=False, sparsity=0.8, cat_ratio=0.5
    )
    n_cat_features = len([0 for dtype in X.dtypes if is_pd_cat_dtype(dtype)])
    n_num_features = n_features - n_cat_features
    n_entries = n_categories * n_cat_features + (max_bin + 1) * n_num_features
    # - qdm
    Xy = QuantileDMatrix(X, y, max_bin=max_bin, enable_categorical=True)
    indptr, data = Xy.get_quantile_cut()
    check_cut(n_entries, indptr, data, X.dtypes)
    # - dm
    Xy = DMatrix(X, y, enable_categorical=True)
    train({"tree_method": tree_method, "max_bin": max_bin}, Xy)
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
) -> DMatrix:
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
            Xy: DMatrix = ExtMemQuantileDMatrix(
                it, enable_categorical=enable_categorical
            )
        elif tree_method == "approx":
            Xy = DMatrix(it, enable_categorical=enable_categorical)
        else:
            raise ValueError(f"tree_method {tree_method} not supported.")
    else:
        cat, label = it.xy()
        Xy = DMatrix(cat, label, enable_categorical=enable_categorical)
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
    train(
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
    train(
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
    assert non_increasing(by_builtin_results["Train"]["rmse"])

    by_grouping: Dict[str, Dict[str, List[float]]] = {}
    # switch to partition-based splits
    parameters["max_cat_to_onehot"] = USE_PART
    parameters["reg_lambda"] = 0
    train(
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
    train(
        parameters,
        Xy_cat,
        num_boost_round=32,
        evals=[(Xy_cat, "Train")],
        evals_result=by_grouping,
    )
    assert non_increasing(by_grouping["Train"]["rmse"]), by_grouping


def check_categorical_missing(  # pylint: disable=too-many-arguments
    rows: int,
    cols: int,
    cats: int,
    *,
    device: Device,
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
        booster = train(
            parameters,
            Xy,
            num_boost_round=8,
            evals=[(Xy, "Train")],
            evals_result=evals_result,
        )
        assert non_increasing(evals_result["Train"]["rmse"])
        y_predt = booster.predict(Xy)
        rmse = tm.root_mean_square(label, y_predt)
        assert_allclose(device, rmse, evals_result["Train"]["rmse"][-1], rtol=2e-5)

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

    reg = XGBRegressor(
        enable_categorical=True,
        tree_method=tree_method,
        device=device,
        n_estimators=10,
    )
    y = pd.Series(range(n))
    reg.fit(X=X, y=y, eval_set=[(X, y)])
    assert non_increasing(reg.evals_result()["validation_0"]["rmse"])


def run_invalid_category(tree_method: str, device: Device) -> None:
    """Test with invalid categorical inputs."""
    rng = np.random.default_rng()
    # too large
    X = rng.integers(low=0, high=4, size=1000).reshape(100, 10)
    y = rng.normal(loc=0, scale=1, size=100)
    X[13, 7] = np.iinfo(np.int32).max + 1

    # Check is performed during sketching.
    Xy = DMatrix(X, y, feature_types=["c"] * 10)
    with pytest.raises(ValueError):
        train({"tree_method": tree_method, "device": device}, Xy)

    X[13, 7] = 16777216
    Xy = DMatrix(X, y, feature_types=["c"] * 10)
    with pytest.raises(ValueError):
        train({"tree_method": tree_method, "device": device}, Xy)

    # mixed positive and negative values
    X = rng.normal(loc=0, scale=1, size=1000).reshape(100, 10)  # type: ignore
    y = rng.normal(loc=0, scale=1, size=100)

    Xy = DMatrix(X, y, feature_types=["c"] * 10)
    with pytest.raises(ValueError):
        train({"tree_method": tree_method, "device": device}, Xy)

    if device == "cuda":
        import cupy as cp

        X, y = cp.array(X), cp.array(y)
        with pytest.raises(ValueError):
            QuantileDMatrix(X, y, feature_types=["c"] * 10)


def train_result(
    param: Dict[str, Any], dmat: DMatrix, num_rounds: int
) -> Dict[str, Any]:
    """Get training result from parameters and data."""
    result: Dict[str, Any] = {}
    booster = train(
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


class ResetStrategy(TrainingCallback):
    """Callback for testing multi-output."""

    def after_iteration(self, model: Booster, epoch: int, evals_log: dict) -> bool:
        if epoch % 2 == 0:
            model.set_param({"multi_strategy": "multi_output_tree"})
        else:
            model.set_param({"multi_strategy": "one_output_per_tree"})
        return False
