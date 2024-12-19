"""QuantileDMatrix related tests."""

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

import xgboost as xgb

from .data import make_batches, make_categorical


def check_ref_quantile_cut(device: str) -> None:
    """Check obtaining the same cut values given a reference."""
    X, y, _ = (
        data[0]
        for data in make_batches(
            n_samples_per_batch=8192,
            n_features=16,
            n_batches=1,
            use_cupy=device.startswith("cuda"),
        )
    )

    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    Xy_train = xgb.QuantileDMatrix(X_train, y_train)
    Xy_valid = xgb.QuantileDMatrix(X_valid, y_valid, ref=Xy_train)

    cut_train = Xy_train.get_quantile_cut()
    cut_valid = Xy_valid.get_quantile_cut()

    np.testing.assert_allclose(cut_train[0], cut_valid[0])
    np.testing.assert_allclose(cut_train[1], cut_valid[1])

    Xy_valid = xgb.QuantileDMatrix(X_valid, y_valid)
    cut_valid = Xy_valid.get_quantile_cut()
    assert not np.allclose(cut_train[1], cut_valid[1])


def check_categorical_strings(device: str) -> None:
    """Check string inputs."""
    if device == "cpu":
        pd = pytest.importorskip("pandas")
    else:
        pd = pytest.importorskip("cudf")

    n_categories = 32
    X, y = make_categorical(
        1024,
        8,
        n_categories,
        onehot=False,
        cat_dtype=np.str_,
        cat_ratio=0.5,
        shuffle=True,
    )
    X = pd.DataFrame(X)

    Xy = xgb.QuantileDMatrix(X, y, enable_categorical=True)
    assert Xy.num_col() == 8
    cuts = Xy.get_quantile_cut()
    indptr = cuts[0]
    values = cuts[1]
    for i in range(1, len(indptr)):
        f_idx = i - 1
        if isinstance(X[X.columns[f_idx]].dtype, pd.CategoricalDtype):
            beg, end = indptr[f_idx], indptr[i]
            col = values[beg:end]
            np.testing.assert_allclose(col, np.arange(0, n_categories))
