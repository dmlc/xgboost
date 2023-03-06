from typing import Callable, Dict, List

import numpy as np
import pytest
from hypothesis import given, settings, strategies
from scipy.sparse import csr_matrix

import xgboost as xgb
from xgboost import testing as tm
from xgboost.data import SingleBatchInternalIter as SingleBatch
from xgboost.testing import IteratorForTest, make_batches, non_increasing

pytestmark = tm.timeout(30)


def test_single_batch(tree_method: str = "approx") -> None:
    from sklearn.datasets import load_breast_cancer

    n_rounds = 10
    X, y = load_breast_cancer(return_X_y=True)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    Xy = xgb.DMatrix(SingleBatch(data=X, label=y))
    from_it = xgb.train({"tree_method": tree_method}, Xy, num_boost_round=n_rounds)

    Xy = xgb.DMatrix(X, y)
    from_dmat = xgb.train({"tree_method": tree_method}, Xy, num_boost_round=n_rounds)
    assert from_it.get_dump() == from_dmat.get_dump()

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X = X.astype(np.float32)
    Xy = xgb.DMatrix(SingleBatch(data=X, label=y))
    from_pd = xgb.train({"tree_method": tree_method}, Xy, num_boost_round=n_rounds)
    # remove feature info to generate exact same text representation.
    from_pd.feature_names = None
    from_pd.feature_types = None

    assert from_pd.get_dump() == from_it.get_dump()

    X, y = load_breast_cancer(return_X_y=True)
    X = csr_matrix(X)
    Xy = xgb.DMatrix(SingleBatch(data=X, label=y))
    from_it = xgb.train({"tree_method": tree_method}, Xy, num_boost_round=n_rounds)

    X, y = load_breast_cancer(return_X_y=True)
    Xy = xgb.DMatrix(SingleBatch(data=X, label=y), missing=0.0)
    from_np = xgb.train({"tree_method": tree_method}, Xy, num_boost_round=n_rounds)
    assert from_np.get_dump() == from_it.get_dump()


def run_data_iterator(
    n_samples_per_batch: int,
    n_features: int,
    n_batches: int,
    tree_method: str,
    subsample: bool,
    use_cupy: bool,
) -> None:
    n_rounds = 2
    # The test is more difficult to pass if the subsample rate is smaller as the root_sum
    # is accumulated in parallel.  Reductions with different number of entries lead to
    # different floating point errors.
    subsample_rate = 0.8 if subsample else 1.0

    it = IteratorForTest(
        *make_batches(n_samples_per_batch, n_features, n_batches, use_cupy)
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
        "seed": 0,
    }

    if tree_method == "gpu_hist":
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
        n_samples_per_batch, n_features, n_batches, "approx", subsample, False
    )
    run_data_iterator(
        n_samples_per_batch, n_features, n_batches, "hist", subsample, False
    )


class IterForCacheTest(xgb.DataIter):
    def __init__(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        self.kwargs = {"data": x, "label": y, "weight": w}
        super().__init__(release_data=False)

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
    it = IterForCacheTest(*batches)
    xgb.QuantileDMatrix(it)
    assert it._input_id == id(batches[0])
