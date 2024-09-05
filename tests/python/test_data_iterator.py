import os
import tempfile
import weakref
from typing import Any, Callable, Dict, List

import numpy as np
import pytest
from hypothesis import given, settings, strategies
from scipy.sparse import csr_matrix

import xgboost as xgb
from xgboost import testing as tm
from xgboost.data import SingleBatchInternalIter as SingleBatch
from xgboost.testing import IteratorForTest, make_batches, non_increasing
from xgboost.testing.updater import check_extmem_qdm, check_quantile_loss_extmem

pytestmark = tm.timeout(30)


def test_single_batch(tree_method: str = "approx", device: str = "cpu") -> None:
    from sklearn.datasets import load_breast_cancer

    n_rounds = 10
    X, y = load_breast_cancer(return_X_y=True)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    params = {"tree_method": tree_method, "device": device}

    Xy = xgb.DMatrix(SingleBatch(data=X, label=y))
    from_it = xgb.train(params, Xy, num_boost_round=n_rounds)

    Xy = xgb.DMatrix(X, y)
    from_dmat = xgb.train(params, Xy, num_boost_round=n_rounds)
    assert from_it.get_dump() == from_dmat.get_dump()

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X = X.astype(np.float32)
    Xy = xgb.DMatrix(SingleBatch(data=X, label=y))
    from_pd = xgb.train(params, Xy, num_boost_round=n_rounds)
    # remove feature info to generate exact same text representation.
    from_pd.feature_names = None
    from_pd.feature_types = None

    assert from_pd.get_dump() == from_it.get_dump()

    X, y = load_breast_cancer(return_X_y=True)
    X = csr_matrix(X)
    Xy = xgb.DMatrix(SingleBatch(data=X, label=y))
    from_it = xgb.train(params, Xy, num_boost_round=n_rounds)

    X, y = load_breast_cancer(return_X_y=True)
    Xy = xgb.DMatrix(SingleBatch(data=X, label=y), missing=0.0)
    from_np = xgb.train(params, Xy, num_boost_round=n_rounds)
    assert from_np.get_dump() == from_it.get_dump()


def test_with_cat_single() -> None:
    X, y = tm.make_categorical(
        n_samples=128, n_features=3, n_categories=6, onehot=False
    )
    Xy = xgb.DMatrix(SingleBatch(data=X, label=y), enable_categorical=True)
    from_it = xgb.train({}, Xy, num_boost_round=3)

    Xy = xgb.DMatrix(X, y, enable_categorical=True)
    from_Xy = xgb.train({}, Xy, num_boost_round=3)

    jit = from_it.save_raw(raw_format="json")
    jxy = from_Xy.save_raw(raw_format="json")
    assert jit == jxy


def run_data_iterator(
    n_samples_per_batch: int,
    n_features: int,
    n_batches: int,
    tree_method: str,
    subsample: bool,
    device: str,
    use_cupy: bool,
    on_host: bool,
) -> None:
    n_rounds = 2
    # The test is more difficult to pass if the subsample rate is smaller as the root_sum
    # is accumulated in parallel.  Reductions with different number of entries lead to
    # different floating point errors.
    subsample_rate = 0.8 if subsample else 1.0

    it = IteratorForTest(
        *make_batches(n_samples_per_batch, n_features, n_batches, use_cupy),
        cache="cache",
        on_host=on_host,
    )
    if n_batches == 0:
        with pytest.raises(ValueError, match="1 batch"):
            Xy = xgb.DMatrix(it)
        return

    Xy = xgb.DMatrix(it)
    assert Xy.num_row() == n_samples_per_batch * n_batches
    assert Xy.num_col() == n_features

    parameters = {
        "tree_method": tree_method,
        "max_depth": 2,
        "subsample": subsample_rate,
        "device": device,
        "seed": 0,
    }

    if device.find("cuda") != -1:
        parameters["sampling_method"] = "gradient_based"

    results_from_it: Dict[str, Dict[str, List[float]]] = {}
    from_it = xgb.train(
        parameters,
        Xy,
        num_boost_round=n_rounds,
        evals=[(Xy, "Train")],
        evals_result=results_from_it,
        verbose_eval=False,
    )
    if not subsample:
        assert non_increasing(results_from_it["Train"]["rmse"])

    X, y, w = it.as_arrays()
    if use_cupy:
        _y = y.get()
    else:
        _y = y
    np.testing.assert_allclose(Xy.get_label(), _y)

    Xy = xgb.DMatrix(X, y, weight=w)
    assert Xy.num_row() == n_samples_per_batch * n_batches
    assert Xy.num_col() == n_features

    results_from_arrays: Dict[str, Dict[str, List[float]]] = {}
    from_arrays = xgb.train(
        parameters,
        Xy,
        num_boost_round=n_rounds,
        evals=[(Xy, "Train")],
        evals_result=results_from_arrays,
        verbose_eval=False,
    )
    arr_predt = from_arrays.predict(Xy)
    if not subsample:
        assert non_increasing(results_from_arrays["Train"]["rmse"])

    rtol = 1e-2
    # CPU sketching is more memory efficient but less consistent due to small chunks
    it_predt = from_it.predict(Xy)
    arr_predt = from_arrays.predict(Xy)
    np.testing.assert_allclose(it_predt, arr_predt, rtol=rtol)

    np.testing.assert_allclose(
        results_from_it["Train"]["rmse"],
        results_from_arrays["Train"]["rmse"],
        rtol=rtol,
    )


@given(
    strategies.integers(0, 1024),
    strategies.integers(1, 7),
    strategies.integers(0, 13),
    strategies.booleans(),
)
@settings(deadline=None, max_examples=10, print_blob=True)
def test_data_iterator(
    n_samples_per_batch: int,
    n_features: int,
    n_batches: int,
    subsample: bool,
) -> None:
    run_data_iterator(
        n_samples_per_batch,
        n_features,
        n_batches,
        "approx",
        subsample,
        "cpu",
        False,
        False,
    )
    run_data_iterator(
        n_samples_per_batch,
        n_features,
        n_batches,
        "hist",
        subsample,
        "cpu",
        False,
        False,
    )


class IterForCacheTest(xgb.DataIter):
    def __init__(
        self, x: np.ndarray, y: np.ndarray, w: np.ndarray, release_data: bool
    ) -> None:
        self.kwargs = {"data": x, "label": y, "weight": w}
        super().__init__(release_data=release_data)

    def next(self, input_data: Callable) -> int:
        if self.it == 1:
            return 0
        self.it += 1
        input_data(**self.kwargs)
        return 1

    def reset(self) -> None:
        self.it = 0


def test_data_cache() -> None:
    n_batches = 1
    n_features = 2
    n_samples_per_batch = 16
    data = make_batches(n_samples_per_batch, n_features, n_batches, False)
    batches = [v[0] for v in data]

    # Test with a cache.
    it = IterForCacheTest(batches[0], batches[1], batches[2], release_data=False)
    transform = xgb.data._proxy_transform

    called = 0

    def mock(*args: Any, **kwargs: Any) -> Any:
        nonlocal called
        called += 1
        return transform(*args, **kwargs)

    xgb.data._proxy_transform = mock
    xgb.QuantileDMatrix(it)
    assert it._data_ref is weakref.ref(batches[0])
    assert called == 1

    # Test without a cache.
    called = 0
    it = IterForCacheTest(batches[0], batches[1], batches[2], release_data=True)
    xgb.QuantileDMatrix(it)
    assert called == 4

    xgb.data._proxy_transform = transform


def test_cat_check() -> None:
    n_batches = 3
    n_features = 2
    n_samples_per_batch = 16

    batches = []

    for i in range(n_batches):
        X, y = tm.make_categorical(
            n_samples=n_samples_per_batch,
            n_features=n_features,
            n_categories=3,
            onehot=False,
        )
        batches.append((X, y))

    X, y = list(zip(*batches))
    it = tm.IteratorForTest(X, y, None, cache=None, on_host=False)
    Xy: xgb.DMatrix = xgb.QuantileDMatrix(it, enable_categorical=True)

    with pytest.raises(ValueError, match="categorical features"):
        xgb.train({"tree_method": "exact"}, Xy)

    Xy = xgb.DMatrix(X[0], y[0], enable_categorical=True)
    with pytest.raises(ValueError, match="categorical features"):
        xgb.train({"tree_method": "exact"}, Xy)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, "cache")

        it = tm.IteratorForTest(X, y, None, cache=cache_path, on_host=False)
        Xy = xgb.DMatrix(it, enable_categorical=True)
        with pytest.raises(ValueError, match="categorical features"):
            xgb.train({"booster": "gblinear"}, Xy)


@given(
    strategies.integers(1, 64),
    strategies.integers(1, 8),
    strategies.integers(1, 4),
)
@settings(deadline=None, max_examples=10, print_blob=True)
def test_quantile_objective(
    n_samples_per_batch: int, n_features: int, n_batches: int
) -> None:
    check_quantile_loss_extmem(
        n_samples_per_batch,
        n_features,
        n_batches,
        "hist",
        "cpu",
    )
    check_quantile_loss_extmem(
        n_samples_per_batch,
        n_features,
        n_batches,
        "approx",
        "cpu",
    )


@given(
    strategies.integers(1, 4096),
    strategies.integers(1, 8),
    strategies.integers(1, 4),
)
@settings(deadline=None, max_examples=10, print_blob=True)
def test_extmem_qdm(n_samples_per_batch: int, n_features: int, n_batches: int) -> None:
    check_extmem_qdm(n_samples_per_batch, n_features, n_batches, "cpu", False)
