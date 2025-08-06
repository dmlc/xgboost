# pylint: disable=invalid-name
"""Tests for the ordinal re-coder."""

import itertools
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import cache as fcache
from typing import Any, Tuple, Type, TypeVar

import numpy as np
import pytest

from .._typing import EvalsLog
from ..core import DMatrix, ExtMemQuantileDMatrix, QuantileDMatrix
from ..data import _lazy_load_cudf_is_cat
from ..training import train
from .data import (
    IteratorForTest,
    is_pd_cat_dtype,
    make_batches,
    make_categorical,
    memory,
)
from .updater import get_basescore
from .utils import Device, assert_allclose, predictor_equal


@fcache
def get_df_impl(device: Device) -> Tuple[Type, Type]:
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


def asarray(device: Device, data: Any) -> np.ndarray:
    """Wrapper to get an array."""
    if device == "cpu":
        return np.asarray(data)
    import cupy as cp

    return cp.asarray(data)


def comp_booster(device: Device, Xy: DMatrix, booster: str) -> None:
    """Compare the results from DMatrix and Booster."""
    cats_dm = Xy.get_categories(export_to_arrow=True).to_arrow()
    assert cats_dm is not None

    rng = np.random.default_rng(2025)
    Xy.set_label(rng.normal(size=Xy.num_row()))
    bst = train({"booster": booster, "device": device}, Xy, 1)
    cats_bst = bst.get_categories(export_to_arrow=True).to_arrow()
    assert cats_bst is not None
    assert cats_dm == cats_bst


def run_cat_container(device: Device) -> None:
    """Basic tests for the container class used by the DMatrix."""

    def run_dispatch(device: Device, DMatrixT: Type) -> None:
        Df, _ = get_df_impl(device)
        # Basic test with a single feature
        df = Df({"c": ["cdef", "abc"]}, dtype="category")
        categories = df.c.cat.categories

        Xy = DMatrixT(df, enable_categorical=True)
        assert Xy.feature_names == ["c"]
        assert Xy.feature_types == ["c"]
        results = Xy.get_categories(export_to_arrow=True).to_arrow()
        assert results is not None
        results_di = dict(results)
        assert len(results_di["c"]) == len(categories)
        for i in range(len(results_di["c"])):
            assert str(results_di["c"][i]) == str(categories[i]), (
                results_di["c"][i],
                categories[i],
            )

        # Test with missing values.
        df = Df({"c": ["cdef", None, "abc", "abc"]}, dtype="category")
        Xy = DMatrixT(df, enable_categorical=True)

        cats = Xy.get_categories(export_to_arrow=True).to_arrow()
        assert cats is not None
        cats_id = dict(cats)
        ser = cats_id["c"].to_pandas()
        assert ser.iloc[0] == "abc"
        assert ser.iloc[1] == "cdef"
        assert ser.size == 2

        csr = Xy.get_data()
        assert csr.data.size == 3
        assert_allclose(device, csr.data, np.array([1.0, 0.0, 0.0]))
        assert_allclose(device, csr.indptr, np.array([0, 1, 1, 2, 3]))
        assert_allclose(device, csr.indices, np.array([0, 0, 0]))

        comp_booster(device, Xy, "gbtree")
        comp_booster(device, Xy, "dart")

        # Test with explicit null-terminated strings.
        df = Df({"c": ["cdef", None, "abc", "abc\0"]}, dtype="category")
        Xy = DMatrixT(df, enable_categorical=True)

        comp_booster(device, Xy, "gbtree")
        comp_booster(device, Xy, "dart")

        with pytest.raises(ValueError, match="export_to_arrow"):
            Xy.get_categories(export_to_arrow=False).to_arrow()

    for dm in (DMatrix, QuantileDMatrix):
        run_dispatch(device, dm)


# pylint: disable=too-many-statements
def run_cat_container_mixed(device: Device) -> None:
    """Run checks with mixed types."""
    import pandas as pd

    try:
        is_cudf_cat = _lazy_load_cudf_is_cat()
    except ImportError:

        def is_cudf_cat(_: Any) -> bool:
            return False

    n_samples = int(2**10)

    def check(Xy: DMatrix, X: pd.DataFrame) -> None:
        cats = Xy.get_categories(export_to_arrow=True).to_arrow()
        assert cats is not None
        cats_di = dict(cats)

        for fname in X.columns:
            if is_pd_cat_dtype(X[fname].dtype) or is_cudf_cat(X[fname].dtype):
                vf = cats_di[fname]
                assert vf is not None
                aw_list = sorted(vf.to_pylist())
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
                assert cats_di[fname] is None

        if not hasattr(Xy, "ref"):  # not quantile DMatrix.
            assert not isinstance(Xy, QuantileDMatrix)
            with tempfile.TemporaryDirectory() as tmpdir:
                fname = os.path.join(tmpdir, "DMatrix.binary")
                Xy.save_binary(fname)

                Xy_1 = DMatrix(fname)
                cats_1 = Xy_1.get_categories(export_to_arrow=True).to_arrow()
                assert cats_1 is not None
                cats_1_di = dict(cats_1)

                for k, v_0 in cats_di.items():
                    v_1 = cats_1_di[k]
                    if v_0 is None:
                        assert v_1 is None
                    else:
                        assert v_1 is not None
                        assert v_0.to_pylist() == v_1.to_pylist()

        comp_booster(device, Xy, "gbtree")
        comp_booster(device, Xy, "dart")

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

    # No category
    batches = make_batches(
        n_samples_per_batch=128, n_features=4, n_batches=1, use_cupy=device == "cuda"
    )
    X, y, w = map(lambda x: x[0], batches)

    for DMatrixT in (DMatrix, QuantileDMatrix):
        Xy = DMatrixT(X, y, weight=w)
        all_num = Xy.get_categories(export_to_arrow=True).to_arrow()
        assert all_num is not None
        for _, v in all_num:
            assert v is None

        with pytest.raises(ValueError, match="export_to_arrow"):
            Xy.get_categories(export_to_arrow=False).to_arrow()


def run_cat_container_iter(device: Device) -> None:
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
    cats = Xy.get_categories(export_to_arrow=True).to_arrow()
    assert cats is not None and len(cats) == n_features
    cats_di = dict(cats)
    for _, v in cats_di.items():
        assert v is not None
        assert v.null_count == 0
        assert len(v) == n_cats


def _basic_example(device: Device) -> Tuple[Any, Any, np.ndarray, np.ndarray]:
    Df, _ = get_df_impl(device)

    enc = Df({"c": ["cdef", "abc", "def"]}, dtype="category")
    codes = enc.c.cat.codes  # 1, 0, 2
    assert_allclose(device, asarray(device, codes), np.array([1, 0, 2]))
    encoded = np.array([codes.iloc[2], codes.iloc[1]])  # def, abc
    np.testing.assert_allclose(encoded, [2, 0])

    reenc = Df({"c": ["def", "abc"]}, dtype="category")  # same as `encoded`
    codes = reenc.c.cat.codes
    assert_allclose(device, codes, np.array([1, 0]))

    y = np.array([0, 1, 2])

    return enc, reenc, encoded, y


def run_basic_predict(DMatrixT: Type, device: Device, tdevice: Device) -> None:
    """Enable tests with mixed devices."""
    enc, reenc, encoded, y = _basic_example(device)

    Xy = DMatrixT(enc, y, enable_categorical=True)
    booster = train({"device": tdevice}, Xy, num_boost_round=4)

    predt0 = booster.inplace_predict(reenc)
    predt1 = booster.inplace_predict(encoded)
    assert_allclose(device, predt0, predt1)

    fmat = DMatrixT(reenc, enable_categorical=True)
    predt2 = booster.predict(fmat)
    assert_allclose(device, predt0, predt2)


def run_cat_predict(device: Device) -> None:
    """Basic tests for re-coding during prediction."""
    Df, _ = get_df_impl(device)

    for dm in (DMatrix, QuantileDMatrix):
        run_basic_predict(dm, device, device)

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


def run_cat_invalid(device: Device) -> None:
    """Basic tests for invalid inputs."""
    Df, Ser = get_df_impl(device)
    y = np.array([0, 1, 2])

    def run_invalid(DMatrixT: Type) -> None:
        df = Df({"b": [2, 1, 3], "c": ["cdef", "abc", "def"]}, dtype="category")

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

    df = Df({"b": [2, 1, 3], "c": ["cdef", "abc", "def"]}, dtype="category")
    Xy = DMatrix(df, y, enable_categorical=True)
    booster = train({"device": device}, Xy, num_boost_round=4)
    df["c"] = Ser(asarray(device, [0, 1, 1]), dtype="category")

    msg = "index type must match between the training and test set"

    with pytest.raises(ValueError, match=msg):
        booster.inplace_predict(df)

    with pytest.raises(ValueError, match=msg):
        DMatrix(df, enable_categorical=True, feature_types=booster.get_categories())

    with pytest.raises(ValueError, match=msg):
        QuantileDMatrix(
            df, enable_categorical=True, feature_types=booster.get_categories()
        )


def run_cat_thread_safety(device: Device) -> None:
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
    device: Device,
    DMatrixT: Type,
    pred_contribs: bool,
    pred_interactions: bool,
    pred_leaf: bool,
) -> None:
    enc, reenc, encoded, y = _basic_example(device)

    Xy = DMatrixT(enc, y, enable_categorical=True)
    booster = train({"device": device}, Xy, num_boost_round=4)

    predt_0 = booster.predict(
        _make_dm(DMatrixT, ref=Xy, data=reenc),
        pred_contribs=pred_contribs,
        pred_interactions=pred_interactions,
        pred_leaf=pred_leaf,
    )
    predt_1 = booster.predict(
        _make_dm(DMatrixT, ref=Xy, data=encoded.reshape(2, 1), feature_names=["c"]),
        pred_contribs=pred_contribs,
        pred_interactions=pred_interactions,
        pred_leaf=pred_leaf,
    )
    assert_allclose(device, predt_0, predt_1)


def run_cat_shap(device: Device) -> None:
    """Basic tests for SHAP values."""

    for dm in (DMatrix, QuantileDMatrix):
        _run_predt(
            device, dm, pred_contribs=True, pred_interactions=False, pred_leaf=False
        )

    for dm in (DMatrix, QuantileDMatrix):
        _run_predt(
            device, dm, pred_contribs=False, pred_interactions=True, pred_leaf=False
        )


def run_cat_leaf(device: Device) -> None:
    """Basic tests for leaf prediction."""
    # QuantileDMatrix is not supported by leaf.
    _run_predt(
        device, DMatrix, pred_contribs=False, pred_interactions=False, pred_leaf=True
    )


# pylint: disable=too-many-locals
@memory.cache
def make_recoded(device: Device, *, n_features: int = 4096) -> Tuple:
    """Synthesize a test dataset with changed encoding."""
    Df, _ = get_df_impl(device)

    import pandas as pd

    # Test large column numbers. XGBoost makes some specializations for slim datasets,
    # make sure we cover all the cases.
    n_samples = 1024

    # Same between old and new, with 0 ("a") and 1 ("b") exchanged their position.
    old_cats = ["a", "b", "c", "d"]
    new_cats = ["b", "a", "c", "d"]
    mapping = {0: 1, 1: 0}

    rng = np.random.default_rng(2025)

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

    enc = Df(df)
    y = rng.normal(size=n_samples)

    reenc = enc.copy()
    for c in range(n_features):
        if c % 2 == 0:
            continue

        name = f"f{c}"
        codes_ser = reenc[name].cat.codes
        if hasattr(codes_ser, "to_pandas"):  # cudf
            codes_ser = codes_ser.to_pandas()
        new_codes = codes_ser.replace(mapping)
        reenc[name] = pd.Categorical.from_codes(categories=new_cats, codes=new_codes)
    reenc = Df(reenc)
    assert (reenc.iloc[:, 1].cat.codes != enc.iloc[:, 1].cat.codes).any()
    return enc, reenc, y, col_numeric, col_categorical


def run_specified_cat(  # pylint: disable=too-many-locals
    device: Device,
) -> None:
    """Run with manually specified category encoding."""
    import pandas as pd

    # Same between old and new, with 0 ("a") and 1 ("b") exchanged their position.
    old_cats = ["a", "b", "c", "d"]
    new_cats = ["b", "a", "c", "d"]

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

    enc, reenc, y, col_numeric, col_categorical = make_recoded(device)

    Xy = DMatrix(enc, y, enable_categorical=True)
    booster = train({"device": device}, Xy)

    predt0 = booster.predict(Xy)
    predt1 = booster.inplace_predict(enc)
    assert_allclose(device, predt0, predt1)

    Xy = DMatrix(reenc, y, enable_categorical=True)
    predt2 = booster.predict(Xy)
    assert_allclose(device, predt0, predt2)

    array = np.empty(shape=(reenc.shape[0], reenc.shape[1]))

    array[:, enc.dtypes == "category"] = col_categorical
    array[:, enc.dtypes != "category"] = col_numeric

    if device == "cuda":
        import cupy as cp

        array = cp.array(array)

    predt3 = booster.inplace_predict(array)
    assert_allclose(device, predt0, predt3)


def run_validation(device: Device) -> None:
    """Check the validation dataset is using the correct encoding."""
    enc, reenc, y, _, _ = make_recoded(device)

    Xy = DMatrix(enc, y, enable_categorical=True)
    Xy_valid = DMatrix(reenc, y, enable_categorical=True)

    evals_result: EvalsLog = {}
    train(
        {"device": device},
        Xy,
        evals=[(Xy, "Train"), (Xy_valid, "Valid")],
        evals_result=evals_result,
    )

    # Evaluation dataset should have the exact same performance as the training dataset.
    assert_allclose(
        device, evals_result["Train"]["rmse"], evals_result["Valid"]["rmse"]
    )


def run_recode_dmatrix(device: Device) -> None:
    """Test re-coding inpput for DMatrix."""
    import pandas as pd

    Df, _ = get_df_impl(device)

    # String index
    old_cats = ["a", "b", "c", "d"]
    new_cats = ["b", "a", "c", "d"]

    col0 = np.arange(0, 9)
    col1 = pd.Categorical.from_codes(
        # b, b, c, d, a, c, c, d, a
        categories=old_cats,
        codes=[1, 1, 2, 3, 0, 2, 2, 3, 0],
    )
    df = Df({"f0": col0, "f1": col1})

    Xy = DMatrix(df, enable_categorical=True)
    cats_0 = Xy.get_categories(export_to_arrow=True)
    assert Xy.feature_types == ["int", "c"]

    col1 = pd.Categorical.from_codes(
        # b, b, c, d, a, c, c, d, a
        categories=new_cats,
        codes=[0, 0, 2, 3, 1, 2, 2, 3, 1],
    )
    df = Df({"f0": col0, "f1": col1})
    Xy = DMatrix(df, enable_categorical=True, feature_types=cats_0)
    # feature_types is still correct
    assert Xy.feature_names == ["f0", "f1"]
    assert Xy.feature_types == ["int", "c"]
    cats_1 = Xy.get_categories(export_to_arrow=True)
    assert cats_0.to_arrow() == cats_1.to_arrow()

    # Numeric index
    col0 = pd.Categorical.from_codes(
        categories=[5, 6, 7, 8],
        codes=[0, 0, 2, 3, 1, 2, 2, 3, 1],
    )
    Df, _ = get_df_impl(device)
    df = Df({"cat": col0})
    for DMatrixT in (DMatrix, QuantileDMatrix):
        Xy = DMatrixT(df, enable_categorical=True)
        cats_0 = Xy.get_categories(export_to_arrow=True)
        assert cats_0 is not None

        Xy = DMatrixT(df, enable_categorical=True, feature_types=cats_0)
        cats_1 = Xy.get_categories(export_to_arrow=True)
        assert cats_1 is not None

        assert cats_0.to_arrow() == cats_1.to_arrow()

    # Recode
    for DMatrixT in (DMatrix, QuantileDMatrix):
        enc, reenc, y, _, _ = make_recoded(device)
        Xy_0 = DMatrixT(enc, y, enable_categorical=True)
        cats_0 = Xy_0.get_categories(export_to_arrow=True)

        assert cats_0 is not None

        Xy_1 = DMatrixT(reenc, y, feature_types=cats_0, enable_categorical=True)
        cats_1 = Xy_1.get_categories(export_to_arrow=True)
        assert cats_1 is not None

        assert cats_0.to_arrow() == cats_1.to_arrow()
        assert predictor_equal(Xy_0, Xy_1)


def run_training_continuation(device: Device) -> None:
    """Test re-coding for training continuation."""
    enc, reenc, y, _, _ = make_recoded(device)

    def check(Xy_0: DMatrix, Xy_1: DMatrix) -> None:
        params = {"device": device}

        r = 2
        evals_result_0: EvalsLog = {}
        booster_0 = train(
            params,
            Xy_0,
            evals=[(Xy_1, "Valid")],
            num_boost_round=r,
            evals_result=evals_result_0,
        )
        evals_result_1: EvalsLog = {}
        booster_1 = train(
            params,
            Xy_1,
            evals=[(Xy_1, "Valid")],
            xgb_model=booster_0,
            num_boost_round=r,
            evals_result=evals_result_1,
        )
        assert get_basescore(booster_0) == get_basescore(booster_1)

        evals_result_2: EvalsLog = {}
        booster_2 = train(
            params,
            Xy_0,
            evals=[(Xy_1, "Valid")],
            num_boost_round=r * 2,
            evals_result=evals_result_2,
        )
        # Check evaluation results
        eval_concat = evals_result_0["Valid"]["rmse"] + evals_result_1["Valid"]["rmse"]
        eval_full = evals_result_2["Valid"]["rmse"]
        np.testing.assert_allclose(eval_full, eval_concat)

        # Test inference
        for a, b in itertools.product([enc, reenc], [enc, reenc]):
            predt_0 = booster_1.inplace_predict(a)
            predt_1 = booster_2.inplace_predict(b)
            assert_allclose(device, predt_0, predt_1, rtol=1e-5)

        # With DMatrix
        for a, b in itertools.product([Xy_0, Xy_1], [Xy_0, Xy_1]):
            predt_0 = booster_1.predict(a)
            predt_1 = booster_2.predict(b)
            assert_allclose(device, predt_0, predt_1, rtol=1e-5)

    for Train, Valid in itertools.product(
        [DMatrix, QuantileDMatrix], [DMatrix, QuantileDMatrix]
    ):
        Xy_0 = Train(enc, y, enable_categorical=True)
        if Valid is QuantileDMatrix:
            Xy_1 = Valid(
                reenc,
                y,
                enable_categorical=True,
                feature_types=Xy_0.get_categories(),
                ref=Xy_0,
            )
        else:
            Xy_1 = Valid(
                reenc, y, enable_categorical=True, feature_types=Xy_0.get_categories()
            )
        check(Xy_0, Xy_1)


def run_update(device: Device) -> None:
    """Test with individual updaters."""
    enc, reenc, y, _, _ = make_recoded(device)
    Xy = DMatrix(enc, y, enable_categorical=True)
    booster_0 = train({"device": device}, Xy, num_boost_round=4)
    model_0 = booster_0.save_raw()
    cats_0 = booster_0.get_categories()

    Xy_1 = DMatrix(reenc, y, feature_types=cats_0, enable_categorical=True)

    booster_1 = train(
        {
            "device": device,
            "updater": "prune",
            "process_type": "update",
        },
        Xy_1,
        num_boost_round=4,
        xgb_model=booster_0,
    )
    model_1 = booster_1.save_raw()

    assert model_0 == model_1  # also compares the cat container inside


def run_recode_dmatrix_predict(device: Device) -> None:
    """Run prediction with re-coded DMatrix."""
    enc, reenc, y, _, _ = make_recoded(device)

    for DMatrixT in (DMatrix, QuantileDMatrix):
        Xy = DMatrix(enc, y, enable_categorical=True)
        booster = train({"device": device}, Xy, num_boost_round=4)
        cats_0 = booster.get_categories()

        Xy_1 = _make_dm(DMatrixT, Xy, reenc, y, feature_types=cats_0)
        Xy_2 = _make_dm(DMatrixT, Xy, reenc, y)

        predt_0 = booster.predict(Xy)
        predt_1 = booster.predict(Xy_1)
        predt_2 = booster.predict(Xy_2)
        predt_3 = booster.inplace_predict(enc)

        for predt in (predt_1, predt_2, predt_3):
            assert_allclose(device, predt_0, predt)
