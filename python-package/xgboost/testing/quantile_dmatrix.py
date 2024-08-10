"""QuantileDMatrix related tests."""

import numpy as np
from sklearn.model_selection import train_test_split

import xgboost as xgb

from .data import make_batches


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
