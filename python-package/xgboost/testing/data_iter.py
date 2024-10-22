"""Tests related to the `DataIter` interface."""

from typing import Callable, Optional

import numpy as np

import xgboost
from xgboost import testing as tm

from ..compat import concat


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

    it = tm.IteratorForTest(X, y, None, cache=None, on_host=False)
    Xy_0 = xgboost.QuantileDMatrix(it)

    X_1, y_1 = tm.make_sparse_regression(256, 16, 0.1, True)
    X = [X_0, X_1, X_2]
    y = [y_0, y_1, y_2]
    X_arr = np.concatenate(X, axis=0)
    y_arr = np.concatenate(y, axis=0)
    Xy_1 = xgboost.QuantileDMatrix(X_arr, y_arr)

    assert tm.predictor_equal(Xy_0, Xy_1)


class CatIter(xgboost.DataIter):  # pylint: disable=too-many-instance-attributes
    """An iterator for testing categorical features."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        n_samples_per_batch: int,
        n_features: int,
        *,
        n_batches: int,
        n_cats: int,
        sparsity: float,
        cat_ratio: float,
        onehot: bool,
        device: str,
        cache: Optional[str],
    ) -> None:
        super().__init__(cache_prefix=cache)
        self.n_samples_per_batch = n_samples_per_batch
        self.n_batches = n_batches
        self.n_cats = n_cats
        self.sparsity = sparsity
        self.onehot = onehot
        self.device = device

        xs, ys = [], []
        for i in range(n_batches):
            cat, y = tm.make_categorical(
                self.n_samples_per_batch,
                n_features,
                n_categories=self.n_cats,
                onehot=self.onehot,
                cat_ratio=cat_ratio,
                sparsity=self.sparsity,
                random_state=self.n_samples_per_batch * n_features * i,
            )
            xs.append(cat)
            ys.append(y)

        self.xs = xs
        self.ys = ys

        self.x = concat(xs)
        self.y = concat(ys)

        self._it = 0

    def xy(self) -> tuple:
        """Return the concatenated data."""
        return self.x, self.y

    def next(self, input_data: Callable) -> bool:
        if self._it == self.n_batches:
            # return False to let XGBoost know this is the end of iteration
            return False
        X, y = self.xs[self._it], self.ys[self._it]
        if self.device == "cuda":
            import cudf  # pylint: disable=import-error
            import cupy  # pylint: disable=import-error

            X = cudf.DataFrame(X)
            y = cupy.array(y)
        input_data(data=X, label=y)
        self._it += 1
        return True

    def reset(self) -> None:
        self._it = 0
