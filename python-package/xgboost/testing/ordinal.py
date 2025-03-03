# pylint: disable=invalid-name
"""Tests for the ordinal re-coder."""

import os
import tempfile
from typing import Any, Tuple, Type

import numpy as np

from ..compat import import_cupy
from ..core import DMatrix
from .data import is_pd_cat_dtype, make_categorical


def get_df_impl(device: str) -> Tuple[Type, Type]:
    """Get data frame implementation based on the ]device."""
    if device == "cpu":
        import pandas as pd

        Df = pd.DataFrame
        Ser = pd.Series
    else:
        import cudf

        Df = cudf.DataFrame
        Ser = cudf.Series
    return Df, Ser


def assert_allclose(device: str, a: Any, b: Any) -> None:
    """Dispatch the assert_allclose for devices."""
    if device == "cpu":
        np.testing.assert_allclose(a, b)
    else:
        cp = import_cupy()
        cp.testing.assert_allclose(a, b)


def run_cat_container(device: str) -> None:
    """Basic tests for the container class used by the DMatrix."""
    Df, _ = get_df_impl(device)
    # Basic test with a single feature
    df = Df({"c": ["cdef", "abc"]}, dtype="category")
    categories = df.c.cat.categories

    Xy = DMatrix(df, enable_categorical=True)
    results = Xy.get_categories()
    assert results is not None
    assert len(results["c"]) == len(categories)
    for i in range(len(results["c"])):
        assert str(results["c"][i]) == str(categories[i]), (
            results["c"][i],
            categories[i],
        )

    # Test with missing values.
    df = Df({"c": ["cdef", None, "abc", "abc"]}, dtype="category")
    Xy = DMatrix(df, enable_categorical=True)

    cats = Xy.get_categories()
    assert cats is not None
    ser = cats["c"].to_pandas()
    assert ser.iloc[0] == "abc"
    assert ser.iloc[1] == "cdef"
    assert ser.size == 2

    csr = Xy.get_data()
    assert csr.data.size == 3
    assert_allclose(device, csr.data, np.array([1.0, 0.0, 0.0]))
    assert_allclose(device, csr.indptr, np.array([0, 1, 1, 2, 3]))
    assert_allclose(device, csr.indices, np.array([0, 0, 0]))

    # Test with explicit null-terminated strings.
    df = Df({"c": ["cdef", None, "abc", "abc\0"]}, dtype="category")
    Xy = DMatrix(df, enable_categorical=True)


def run_cat_container_mixed() -> None:
    """Run checks with mixed types."""
    import pandas as pd

    def check(Xy: DMatrix, X: pd.DataFrame) -> None:
        cats = Xy.get_categories()
        assert cats is not None

        for fname in X.columns:
            if is_pd_cat_dtype(X[fname].dtype):
                aw_list = sorted(cats[fname].to_pylist())
                pd_list: list = X[fname].unique().tolist()
                if np.nan in pd_list:
                    pd_list.remove(np.nan)
                pd_list = sorted(pd_list)
                assert aw_list == pd_list
            else:
                assert cats[fname] is None

        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, "DMatrix.binary")
            Xy.save_binary(fname)

            Xy_1 = DMatrix(fname)
            cats_1 = Xy_1.get_categories()
            assert cats_1 is not None

            for k, v_0 in cats.items():
                v_1 = cats_1[k]
                if v_0 is None:
                    assert v_1 is None
                else:
                    assert v_0.to_pylist() == v_1.to_pylist()

    # full str type
    X, y = make_categorical(256, 16, 7, onehot=False, cat_dtype=np.str_)
    Xy = DMatrix(X, y, enable_categorical=True)
    check(Xy, X)

    # str type, mixed with numerical features
    X, y = make_categorical(256, 16, 7, onehot=False, cat_ratio=0.5, cat_dtype=np.str_)
    Xy = DMatrix(X, y, enable_categorical=True)
    check(Xy, X)

    # str type, mixed with numerical features and missing values
    X, y = make_categorical(
        256, 16, 7, onehot=False, cat_ratio=0.5, sparsity=0.5, cat_dtype=np.str_
    )
    Xy = DMatrix(X, y, enable_categorical=True)
    check(Xy, X)

    # int type
    X, y = make_categorical(256, 16, 7, onehot=False, cat_dtype=np.int64)
    Xy = DMatrix(X, y, enable_categorical=True)
    check(Xy, X)

    # int type, mixed with numerical features
    X, y = make_categorical(256, 16, 7, onehot=False, cat_ratio=0.5, cat_dtype=np.int64)
    Xy = DMatrix(X, y, enable_categorical=True)
    check(Xy, X)

    # int type, mixed with numerical features and missing values
    X, y = make_categorical(
        256, 16, 7, onehot=False, cat_ratio=0.5, sparsity=0.5, cat_dtype=np.int64
    )
    Xy = DMatrix(X, y, enable_categorical=True)
    check(Xy, X)
