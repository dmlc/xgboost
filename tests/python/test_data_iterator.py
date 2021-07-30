import xgboost as xgb
from xgboost.data import SingleBatchInternalIter as SingleBatch
import numpy as np
from testing import IteratorForTest
from typing import Tuple, List
import pytest
from hypothesis import given, strategies, settings
from scipy.sparse import csr_matrix


def make_batches(
    n_samples_per_batch: int, n_features: int, n_batches: int, use_cupy: bool = False
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    X = []
    y = []
    if use_cupy:
        import cupy

        rng = cupy.random.RandomState(1994)
    else:
        rng = np.random.RandomState(1994)
    for i in range(n_batches):
        _X = rng.randn(n_samples_per_batch, n_features)
        _y = rng.randn(n_samples_per_batch)
        X.append(_X)
        y.append(_y)
    return X, y


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
    use_cupy: bool,
) -> None:
    n_rounds = 2

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

    results_from_it: xgb.callback.EvaluationMonitor.EvalsLog = {}
    from_it = xgb.train(
        {"tree_method": tree_method, "max_depth": 2},
        Xy,
        num_boost_round=n_rounds,
        evals=[(Xy, "Train")],
        evals_result=results_from_it,
        verbose_eval=False,
    )
    it_predt = from_it.predict(Xy)

    X, y = it.as_arrays()
    Xy = xgb.DMatrix(X, y)
    assert Xy.num_row() == n_samples_per_batch * n_batches
    assert Xy.num_col() == n_features

    results_from_arrays: xgb.callback.EvaluationMonitor.EvalsLog = {}
    from_arrays = xgb.train(
        {"tree_method": tree_method, "max_depth": 2},
        Xy,
        num_boost_round=n_rounds,
        evals=[(Xy, "Train")],
        evals_result=results_from_arrays,
        verbose_eval=False,
    )
    arr_predt = from_arrays.predict(Xy)

    if tree_method != "gpu_hist":
        rtol = 1e-1  # flaky
    else:
        # Model can be sensitive to quantiles, use 1e-2 to relax the test.
        np.testing.assert_allclose(it_predt, arr_predt, rtol=1e-2)
        rtol = 1e-6

    np.testing.assert_allclose(
        results_from_it["Train"]["rmse"],
        results_from_arrays["Train"]["rmse"],
        rtol=rtol,
    )


@given(
    strategies.integers(0, 1024), strategies.integers(1, 7), strategies.integers(0, 13)
)
@settings(deadline=None)
def test_data_iterator(
    n_samples_per_batch: int, n_features: int, n_batches: int
) -> None:
    run_data_iterator(n_samples_per_batch, n_features, n_batches, "approx", False)
    run_data_iterator(n_samples_per_batch, n_features, n_batches, "hist", False)
