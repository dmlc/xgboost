"""Tests related to the `DataIter` interface."""

import numpy as np

import xgboost
from xgboost import testing as tm


def run_mixed_sparsity(device: str) -> None:
    """Check QDM with mixed batches."""
    X_0, y_0, _ = tm.make_regression(128, 16, False)
    if device.startswith("cuda"):
        X_1, y_1 = tm.make_sparse_regression(256, 16, 0.1, True)
    else:
        X_1, y_1 = tm.make_sparse_regression(256, 16, 0.1, False)
    X_2, y_2 = tm.make_sparse_regression(512, 16, 0.9, True)
    X = [X_0, X_1, X_2]
    y = [y_0, y_1, y_2]

    if device.startswith("cuda"):
        import cupy as cp  # pylint: disable=import-error

        X = [cp.array(batch) for batch in X]

    it = tm.IteratorForTest(X, y, None, None)
    Xy_0 = xgboost.QuantileDMatrix(it)

    X_1, y_1 = tm.make_sparse_regression(256, 16, 0.1, True)
    X = [X_0, X_1, X_2]
    y = [y_0, y_1, y_2]
    X_arr = np.concatenate(X, axis=0)
    y_arr = np.concatenate(y, axis=0)
    Xy_1 = xgboost.QuantileDMatrix(X_arr, y_arr)

    assert tm.predictor_equal(Xy_0, Xy_1)
