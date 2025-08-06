"""Tests related to the `DataIter` interface."""

from typing import Callable, Optional

import numpy as np

from xgboost import testing as tm

from ..compat import import_cupy
from ..core import DataIter, DMatrix, ExtMemQuantileDMatrix, QuantileDMatrix
from .utils import predictor_equal


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
        cp = import_cupy()

        X = [cp.array(batch) for batch in X]

    it = tm.IteratorForTest(X, y, None, cache=None, on_host=False)
    Xy_0 = QuantileDMatrix(it)

    X_1, y_1 = tm.make_sparse_regression(256, 16, 0.1, True)
    X = [X_0, X_1, X_2]
    y = [y_0, y_1, y_2]
    X_arr = np.concatenate(X, axis=0)
    y_arr = np.concatenate(y, axis=0)
    Xy_1 = QuantileDMatrix(X_arr, y_arr)

    assert predictor_equal(Xy_0, Xy_1)


def check_invalid_cat_batches(device: str) -> None:
    """Check error message for inconsistent feature types."""

    class _InvalidCatIter(DataIter):
        def __init__(self) -> None:
            super().__init__(cache_prefix=None)
            self._it = 0

        def next(self, input_data: Callable) -> bool:
            if self._it == 2:
                return False
            X, y = tm.make_categorical(
                64,
                12,
                4,
                onehot=False,
                sparsity=0.5,
                cat_ratio=1.0 if self._it == 0 else 0.5,
            )
            if device == "cuda":
                import cudf
                import cupy

                X = cudf.DataFrame(X)
                y = cupy.array(y)

            input_data(data=X, label=y)
            self._it += 1
            return True

        def reset(self) -> None:
            self._it = 0

    it = _InvalidCatIter()
    import pytest

    with pytest.raises(ValueError, match="Inconsistent number of categories between"):
        ExtMemQuantileDMatrix(it, enable_categorical=True)

    with pytest.raises(ValueError, match="Inconsistent number of categories between"):
        QuantileDMatrix(it, enable_categorical=True)

    with pytest.raises(ValueError, match="Inconsistent feature types"):
        DMatrix(it, enable_categorical=True)


def check_uneven_sizes(device: str) -> None:
    """Tests for having irregular data shapes."""
    batches = [
        tm.make_regression(n_samples, 16, use_cupy=device == "cuda")
        for n_samples in [512, 256, 1024]
    ]
    unzip = list(zip(*batches))
    it = tm.IteratorForTest(unzip[0], unzip[1], None, cache="cache", on_host=True)

    Xy = DMatrix(it)
    assert Xy.num_col() == 16
    assert Xy.num_row() == sum(x.shape[0] for x in unzip[0])

    Xy = ExtMemQuantileDMatrix(it)
    assert Xy.num_col() == 16
    assert Xy.num_row() == sum(x.shape[0] for x in unzip[0])


class CatIter(DataIter):  # pylint: disable=too-many-instance-attributes
    """An iterator for testing categorical features."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
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
        self.n_batches = n_batches
        self.device = device

        n_samples = n_samples_per_batch * n_batches
        cat, y = tm.make_categorical(
            n_samples,
            n_features,
            n_categories=n_cats,
            onehot=onehot,
            cat_ratio=cat_ratio,
            sparsity=sparsity,
        )
        xs, ys = [], []

        prev = 0
        for _ in range(n_batches):
            n = min(n_samples_per_batch, n_samples - prev)
            X = cat.iloc[prev : prev + n, :]
            xs.append(X)
            ys.append(y[prev : prev + n])
            prev += n_samples_per_batch

        self.xs = xs
        self.ys = ys

        self.x = cat
        self.y = y

        self._it = 0

    def xy(self) -> tuple:
        """Return the concatenated data."""
        return self.x, self.y

    def next(self, input_data: Callable) -> bool:
        if self._it == self.n_batches:
            return False

        X, y = self.xs[self._it], self.ys[self._it]
        if self.device == "cuda":
            import cudf
            import cupy

            X = cudf.DataFrame(X)
            y = cupy.array(y)
        input_data(data=X, label=y)
        self._it += 1
        return True

    def reset(self) -> None:
        self._it = 0
