# pylint: disable=invalid-name
"""Tests for the ordinal re-coder."""

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, Tuple, Type, TypeVar

import numpy as np
import pytest

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


def asarray(device: str, data: Any) -> np.ndarray:
    """Wrapper to get an array."""
    if device == "cpu":
        return np.asarray(data)
    import cupy as cp

    return cp.asarray(data)


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
    """Basic tests for re-coding during prediction."""
    Df, _ = get_df_impl(device)

    def run_basic(DMatrixT: Type) -> None:
        df = Df({"c": ["cdef", "abc", "def"]}, dtype="category")
        y = np.array([0, 1, 2])

        codes = df.c.cat.codes
        encoded = np.array([codes.iloc[2], codes.iloc[1]])  # used with the next df

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
        run_basic(dm)

    def run_mixed(DMatrixT: Type) -> None:
        df = Df({"b": [2, 1, 3], "c": ["cdef", "abc", "def"]}, dtype="category")
        y = np.array([0, 1, 2])

        # used with the next df
        b_codes = df.b.cat.codes
        assert_allclose(device, asarray(device, b_codes), np.array([1, 0, 2]))
        # pick codes of 3, 1
        b_encoded = np.array([b_codes.iloc[2], b_codes.iloc[1]])

        c_codes = df.c.cat.codes
        assert_allclose(device, asarray(device, c_codes), np.array([1, 0, 2]))
        # pick codes of "def", "abc"
        c_encoded = np.array([c_codes.iloc[2], c_codes.iloc[1]])
        encoded = np.stack([b_encoded, c_encoded], axis=1)

        Xy = DMatrixT(df, y, enable_categorical=True)
        booster = train({"device": device}, Xy, num_boost_round=4)

        df = Df({"b": [3, 1], "c": ["def", "abc"]}, dtype="category")
        predt0 = booster.inplace_predict(df)
        predt1 = booster.inplace_predict(encoded)
        assert_allclose(device, predt0, predt1)

        fmat = DMatrixT(df, enable_categorical=True)
        predt2 = booster.predict(fmat)
        assert_allclose(device, predt0, predt2)

    for dm in (DMatrix, QuantileDMatrix):
        run_mixed(dm)


def run_cat_invalid(device: Literal["cpu", "cuda"]) -> None:
    """Basic tests for invalid inputs."""
    Df, _ = get_df_impl(device)

    def run_invalid(DMatrixT: Type) -> None:
        df = Df({"b": [2, 1, 3], "c": ["cdef", "abc", "def"]}, dtype="category")
        y = np.array([0, 1, 2])

        Xy = DMatrixT(df, y, enable_categorical=True)
        booster = train({"device": device}, Xy, num_boost_round=4)
        df["b"] = df["b"].astype(np.int64)
        with pytest.raises(ValueError, match="The data type doesn't match"):
            booster.inplace_predict(df)

        Xy = DMatrixT(df, y, enable_categorical=True)
        with pytest.raises(ValueError, match="The data type doesn't match"):
            booster.predict(Xy)

        df = Df(
            {"b": [2, 1, 3, 4], "c": ["cdef", "abc", "def", "bbc"]}, dtype="category"
        )
        with pytest.raises(ValueError, match="Found a category not in the training"):
            booster.inplace_predict(df)

    for dm in (DMatrix, QuantileDMatrix):
        run_invalid(dm)


def run_cat_thread_safety(device: Literal["cpu", "cuda"]) -> None:
    """Basic tests for thread safety."""
    X, y = make_categorical(2048, 16, 112, onehot=False, cat_ratio=0.5, device=device)
    Xy = QuantileDMatrix(X, y, enable_categorical=True)
    booster = train({"device": device}, Xy, num_boost_round=10)

    def run_thread_safety(DMatrixT: Type) -> bool:
        Xy = DMatrixT(X, enable_categorical=True)
        predt0 = booster.predict(Xy)
        predt1 = booster.inplace_predict(X)
        assert_allclose(device, predt0, predt1)
        return True

    futures = []
    for dm in (DMatrix, QuantileDMatrix):
        with ThreadPoolExecutor(max_workers=10) as e:
            for _ in range(10):
                fut = e.submit(run_thread_safety, dm)
                futures.append(fut)

    for f in futures:
        assert f.result()


U = TypeVar("U", DMatrix, QuantileDMatrix)


def _make_dm(DMatrixT: Type[U], ref: DMatrix, *args: Any, **kwargs: Any) -> U:
    if DMatrixT is QuantileDMatrix:
        return DMatrixT(*args, ref=ref, enable_categorical=True, **kwargs)
    return DMatrixT(*args, enable_categorical=True, **kwargs)


def _run_predt(
    device: str,
    DMatrixT: Type,
    pred_contribs: bool,
    pred_interactions: bool,
    pred_leaf: bool,
) -> None:
    Df, _ = get_df_impl(device)

    df = Df({"c": ["cdef", "abc", "def"]}, dtype="category")
    y = np.array([0, 1, 2])

    codes = df.c.cat.codes
    encoded = np.array([codes.iloc[2], codes.iloc[1]])  # used with the next df

    Xy = DMatrixT(df, y, enable_categorical=True)
    booster = train({"device": device}, Xy, num_boost_round=4)

    df = Df({"c": ["def", "abc"]}, dtype="category")
    codes = df.c.cat.codes

    # Contribution
    predt0 = booster.predict(
        _make_dm(DMatrixT, ref=Xy, data=df),
        pred_contribs=pred_contribs,
        pred_interactions=pred_interactions,
        pred_leaf=pred_leaf,
    )
    df = Df({"c": encoded})
    predt1 = booster.predict(
        _make_dm(DMatrixT, ref=Xy, data=encoded.reshape(2, 1), feature_names=["c"]),
        pred_contribs=pred_contribs,
        pred_interactions=pred_interactions,
        pred_leaf=pred_leaf,
    )
    assert_allclose(device, predt0, predt1)


def run_cat_shap(device: Literal["cpu", "cuda"]) -> None:
    """Basic tests for SHAP values."""

    for dm in (DMatrix, QuantileDMatrix):
        _run_predt(
            device, dm, pred_contribs=True, pred_interactions=False, pred_leaf=False
        )

    for dm in (DMatrix, QuantileDMatrix):
        _run_predt(
            device, dm, pred_contribs=False, pred_interactions=True, pred_leaf=False
        )


def run_cat_leaf(device: Literal["cpu", "cuda"]) -> None:
    """Basic tests for leaf prediction."""
    # QuantileDMatrix is not supported by leaf.
    _run_predt(
        device, DMatrix, pred_contribs=False, pred_interactions=False, pred_leaf=True
    )


def run_specified_cat(  # pylint: disable=too-many-locals
    device: Literal["cpu", "cuda"],
) -> None:
    """Run with manually specified category encoding."""
    import pandas as pd

    # Same between old and new, wiht 0 ("a") and 1 ("b") exchanged their position.
    old_cats = ["a", "b", "c", "d"]
    new_cats = ["b", "a", "c", "d"]
    mapping = {0: 1, 1: 0}

    col0 = np.arange(0, 9)
    col1 = pd.Categorical.from_codes(
        # b, b, c, d, a, c, c, d, a
        categories=old_cats,
        codes=[1, 1, 2, 3, 0, 2, 2, 3, 0],
    )
    df = pd.DataFrame({"f0": col0, "f1": col1})
    Df, _ = get_df_impl(device)
    df = Df(df)
    rng = np.random.default_rng(2025)
    y = rng.uniform(size=df.shape[0])

    for dm in (DMatrix, QuantileDMatrix):
        Xy = dm(df, y, enable_categorical=True)
        booster = train({"device": device}, Xy)
        predt0 = booster.predict(Xy)
        predt1 = booster.inplace_predict(df)
        assert_allclose(device, predt0, predt1)

        col1 = pd.Categorical.from_codes(
            # b, b, c, d, a, c, c, d, a
            categories=new_cats,
            codes=[0, 0, 2, 3, 1, 2, 2, 3, 1],
        )
        df1 = Df({"f0": col0, "f1": col1})
        predt2 = booster.inplace_predict(df1)
        assert_allclose(device, predt0, predt2)

    # Test large column numbers. XGBoost makes some specializations for slim datasets,
    # make sure we cover all the cases.
    n_features = 4096
    n_samples = 1024

    col_numeric = rng.uniform(0, 1, size=(n_samples, n_features // 2))
    col_categorical = rng.integers(
        low=0, high=4, size=(n_samples, n_features // 2), dtype=np.int32
    )

    df = {}  # avoid fragmentation warning from pandas
    for c in range(n_features):
        if c % 2 == 0:
            col = col_numeric[:, c // 2]
        else:
            codes = col_categorical[:, c // 2]
            col = pd.Categorical.from_codes(
                categories=old_cats,
                codes=codes,
            )
        df[f"f{c}"] = col

    df = Df(df)
    y = rng.normal(size=n_samples)

    Xy = DMatrix(df, y, enable_categorical=True)
    booster = train({"device": device}, Xy)

    predt0 = booster.predict(Xy)
    predt1 = booster.inplace_predict(df)
    assert_allclose(device, predt0, predt1)

    for c in range(n_features):
        if c % 2 == 0:
            continue

        name = f"f{c}"
        codes_ser = df[name].cat.codes
        if hasattr(codes_ser, "to_pandas"):  # cudf
            codes_ser = codes_ser.to_pandas()
        new_codes = codes_ser.replace(mapping)
        df[name] = pd.Categorical.from_codes(categories=new_cats, codes=new_codes)

    df = Df(df)
    Xy = DMatrix(df, y, enable_categorical=True)
    predt2 = booster.predict(Xy)
    assert_allclose(device, predt0, predt2)

    array = np.empty(shape=(n_samples, n_features))
    array[:, np.arange(0, n_features) % 2 == 0] = col_numeric
    array[:, np.arange(0, n_features) % 2 != 0] = col_categorical

    if device == "cuda":
        import cupy as cp

        array = cp.array(array)

    predt3 = booster.inplace_predict(array)
    assert_allclose(device, predt0, predt3)
