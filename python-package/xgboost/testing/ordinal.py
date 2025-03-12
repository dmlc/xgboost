# pylint: disable=invalid-name
"""Tests for the ordinal re-coder."""

import os
import tempfile
from typing import Any, Literal, Tuple, Type

import numpy as np

from ..compat import import_cupy
from ..core import DMatrix, ExtMemQuantileDMatrix, QuantileDMatrix
from ..data import _lazy_load_cudf_is_cat
from ..training import train
from .data import IteratorForTest, is_pd_cat_dtype, make_categorical


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


def run_cat_container(device: Literal["cpu", "cuda"]) -> None:
    """Basic tests for the container class used by the DMatrix."""

    def run_dispatch(device: str, DMatrixT: Type) -> None:
        Df, _ = get_df_impl(device)
        # Basic test with a single feature
        df = Df({"c": ["cdef", "abc"]}, dtype="category")
        categories = df.c.cat.categories

        Xy = DMatrixT(df, enable_categorical=True)
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
        Xy = DMatrixT(df, enable_categorical=True)

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
        Xy = DMatrixT(df, enable_categorical=True)

    for dm in (DMatrix, QuantileDMatrix):
        run_dispatch(device, dm)


# pylint: disable=too-many-statements
def run_cat_container_mixed(device: Literal["cpu", "cuda"]) -> None:
    """Run checks with mixed types."""
    import pandas as pd

    try:
        is_cudf_cat = _lazy_load_cudf_is_cat()
    except ImportError:

        def is_cudf_cat(_: Any) -> bool:
            return False

    n_samples = int(2**10)

    def check(Xy: DMatrix, X: pd.DataFrame) -> None:
        cats = Xy.get_categories()
        assert cats is not None

        for fname in X.columns:
            if is_pd_cat_dtype(X[fname].dtype) or is_cudf_cat(X[fname].dtype):
                aw_list = sorted(cats[fname].to_pylist())
                if is_cudf_cat(X[fname].dtype):
                    pd_list: list = X[fname].unique().to_arrow().to_pylist()
                else:
                    pd_list = X[fname].unique().tolist()
                if np.nan in pd_list:  # pandas
                    pd_list.remove(np.nan)
                if None in pd_list:  # cudf
                    pd_list.remove(None)
                pd_list = sorted(pd_list)
                assert aw_list == pd_list
            else:
                assert cats[fname] is None

        if not hasattr(Xy, "ref"):  # not quantile DMatrix.
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

    def run_dispatch(DMatrixT: Type) -> None:
        # full str type
        X, y = make_categorical(
            n_samples, 16, 7, onehot=False, cat_dtype=np.str_, device=device
        )
        Xy = DMatrixT(X, y, enable_categorical=True)
        check(Xy, X)

        # str type, mixed with numerical features
        X, y = make_categorical(
            n_samples,
            16,
            7,
            onehot=False,
            cat_ratio=0.5,
            cat_dtype=np.str_,
            device=device,
        )
        Xy = DMatrixT(X, y, enable_categorical=True)
        check(Xy, X)

        # str type, mixed with numerical features and missing values
        X, y = make_categorical(
            n_samples,
            16,
            7,
            onehot=False,
            cat_ratio=0.5,
            sparsity=0.5,
            cat_dtype=np.str_,
            device=device,
        )
        Xy = DMatrixT(X, y, enable_categorical=True)
        check(Xy, X)

        # int type
        X, y = make_categorical(
            n_samples, 16, 7, onehot=False, cat_dtype=np.int64, device=device
        )
        Xy = DMatrixT(X, y, enable_categorical=True)
        check(Xy, X)

        # int type, mixed with numerical features
        X, y = make_categorical(
            n_samples,
            16,
            7,
            onehot=False,
            cat_ratio=0.5,
            cat_dtype=np.int64,
            device=device,
        )
        Xy = DMatrixT(X, y, enable_categorical=True)
        check(Xy, X)

        # int type, mixed with numerical features and missing values
        X, y = make_categorical(
            n_samples,
            16,
            7,
            onehot=False,
            cat_ratio=0.5,
            sparsity=0.5,
            cat_dtype=np.int64,
            device=device,
        )
        Xy = DMatrixT(X, y, enable_categorical=True)
        check(Xy, X)

    for dm in (DMatrix, QuantileDMatrix):
        run_dispatch(dm)


def run_cat_container_iter(device: Literal["cpu", "cuda"]) -> None:
    """Test the categories container for iterator-based inputs."""
    n_batches = 4
    n_features = 8
    n_samples_per_batch = 64
    n_cats = 5

    X, y = [], []
    for _ in range(n_batches):
        X_i, y_i = make_categorical(
            n_samples_per_batch,
            n_features,
            n_cats,
            onehot=False,
            sparsity=0.5,
            cat_dtype=np.int64,
            device=device,
        )
        X.append(X_i)
        y.append(y_i)

    it = IteratorForTest(X, y, None, cache="cache", on_host=device == "cuda")

    Xy = ExtMemQuantileDMatrix(it, enable_categorical=True)
    cats = Xy.get_categories()
    assert cats is not None and len(cats) == n_features
    for _, v in cats.items():
        assert v.null_count == 0
        assert len(v) == n_cats


def run_cat_predict(device: Literal["cpu", "cuda"]) -> None:
    """Test re-coding during prediction."""
    Df, Ser = get_df_impl(device)

    def run_dispatch(DMatrixT: Type) -> None:
        df = Df({"c": ["cdef", "abc", "def"]}, dtype="category")
        y = np.array([0, 1, 2])

        codes = df.c.cat.codes
        encoded = np.array([codes.iloc[2], codes.iloc[1]])

        Xy = DMatrixT(df, y, enable_categorical=True)
        booster = train({"device": device}, Xy, num_boost_round=4)

        df = Df({"c": ["def", "abc"]}, dtype="category")
        codes = df.c.cat.codes

        predt0 = booster.inplace_predict(df)
        predt1 = booster.inplace_predict(encoded)

        assert_allclose(device, predt0, predt1)

        fmat = DMatrixT(df, enable_categorical=True)
        predt2 = booster.predict(fmat)
        assert_allclose(device, predt0, predt2)

    for dm in (DMatrix, QuantileDMatrix):
        run_dispatch(dm)
