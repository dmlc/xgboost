import sys
from typing import List

import numpy as np
import pandas as pd
import pytest
import testing as tm

if tm.no_spark()["condition"]:
    pytest.skip(msg=tm.no_spark()["reason"], allow_module_level=True)
if sys.platform.startswith("win") or sys.platform.startswith("darwin"):
    pytest.skip("Skipping PySpark tests on Windows", allow_module_level=True)

from xgboost.spark.data import (
    _read_csr_matrix_from_unwrapped_spark_vec,
    alias,
    create_dmatrix_from_partitions,
    stack_series,
)


def test_stack() -> None:
    a = pd.DataFrame({"a": [[1, 2], [3, 4]]})
    b = stack_series(a["a"])
    assert b.shape == (2, 2)

    a = pd.DataFrame({"a": [[1], [3]]})
    b = stack_series(a["a"])
    assert b.shape == (2, 1)

    a = pd.DataFrame({"a": [np.array([1, 2]), np.array([3, 4])]})
    b = stack_series(a["a"])
    assert b.shape == (2, 2)

    a = pd.DataFrame({"a": [np.array([1]), np.array([3])]})
    b = stack_series(a["a"])
    assert b.shape == (2, 1)


def run_dmatrix_ctor(is_dqm: bool) -> None:
    rng = np.random.default_rng(0)
    dfs: List[pd.DataFrame] = []
    n_features = 16
    n_samples_per_batch = 16
    n_batches = 10
    feature_types = ["float"] * n_features

    for i in range(n_batches):
        X = rng.normal(loc=0, size=256).reshape(n_samples_per_batch, n_features)
        y = rng.normal(loc=0, size=n_samples_per_batch)
        m = rng.normal(loc=0, size=n_samples_per_batch)
        w = rng.normal(loc=0.5, scale=0.5, size=n_samples_per_batch)
        w -= w.min()

        valid = rng.binomial(n=1, p=0.5, size=16).astype(np.bool_)

        df = pd.DataFrame(
            {alias.label: y, alias.margin: m, alias.weight: w, alias.valid: valid}
        )
        if is_dqm:
            for j in range(X.shape[1]):
                df[f"feat-{j}"] = pd.Series(X[:, j])
        else:
            df[alias.data] = pd.Series(list(X))
        dfs.append(df)

    kwargs = {"feature_types": feature_types}
    if is_dqm:
        cols = [f"feat-{i}" for i in range(n_features)]
        train_Xy, valid_Xy = create_dmatrix_from_partitions(
            iter(dfs), cols, 0, kwargs, False
        )
    else:
        train_Xy, valid_Xy = create_dmatrix_from_partitions(
            iter(dfs), None, None, kwargs, False
        )

    assert valid_Xy is not None
    assert valid_Xy.num_row() + train_Xy.num_row() == n_samples_per_batch * n_batches
    assert train_Xy.num_col() == n_features
    assert valid_Xy.num_col() == n_features

    df = pd.concat(dfs, axis=0)
    df_train = df.loc[~df[alias.valid], :]
    df_valid = df.loc[df[alias.valid], :]

    assert df_train.shape[0] == train_Xy.num_row()
    assert df_valid.shape[0] == valid_Xy.num_row()

    # margin
    np.testing.assert_allclose(
        df_train[alias.margin].to_numpy(), train_Xy.get_base_margin()
    )
    np.testing.assert_allclose(
        df_valid[alias.margin].to_numpy(), valid_Xy.get_base_margin()
    )
    # weight
    np.testing.assert_allclose(df_train[alias.weight].to_numpy(), train_Xy.get_weight())
    np.testing.assert_allclose(df_valid[alias.weight].to_numpy(), valid_Xy.get_weight())
    # label
    np.testing.assert_allclose(df_train[alias.label].to_numpy(), train_Xy.get_label())
    np.testing.assert_allclose(df_valid[alias.label].to_numpy(), valid_Xy.get_label())

    np.testing.assert_equal(train_Xy.feature_types, feature_types)
    np.testing.assert_equal(valid_Xy.feature_types, feature_types)


def test_dmatrix_ctor() -> None:
    run_dmatrix_ctor(False)


def test_read_csr_matrix_from_unwrapped_spark_vec() -> None:
    from scipy.sparse import csr_matrix

    pd1 = pd.DataFrame(
        {
            "featureVectorType": [0, 1, 1, 0],
            "featureVectorSize": [3, None, None, 3],
            "featureVectorIndices": [
                np.array([0, 2], dtype=np.int32),
                None,
                None,
                np.array([1, 2], dtype=np.int32),
            ],
            "featureVectorValues": [
                np.array([3.0, 0.0], dtype=np.float64),
                np.array([13.0, 14.0, 0.0], dtype=np.float64),
                np.array([0.0, 24.0, 25.0], dtype=np.float64),
                np.array([0.0, 35.0], dtype=np.float64),
            ],
        }
    )
    sm = _read_csr_matrix_from_unwrapped_spark_vec(pd1)
    assert isinstance(sm, csr_matrix)

    np.testing.assert_array_equal(
        sm.data, [3.0, 0.0, 13.0, 14.0, 0.0, 0.0, 24.0, 25.0, 0.0, 35.0]
    )
    np.testing.assert_array_equal(sm.indptr, [0, 2, 5, 8, 10])
    np.testing.assert_array_equal(sm.indices, [0, 2, 0, 1, 2, 0, 1, 2, 1, 2])
    assert sm.shape == (4, 3)
