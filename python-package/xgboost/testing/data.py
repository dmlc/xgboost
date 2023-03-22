"""Utilities for data generation."""
import os
import zipfile
from typing import Any, Generator, List, Tuple, Union
from urllib import request

import numpy as np
import pytest
from numpy.random import Generator as RNG
from scipy import sparse

import xgboost
from xgboost.data import pandas_pyarrow_mapper

joblib = pytest.importorskip("joblib")
memory = joblib.Memory("./cachedir", verbose=0)


def np_dtypes(
    n_samples: int, n_features: int
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Enumerate all supported dtypes from numpy."""
    import pandas as pd

    rng = np.random.RandomState(1994)
    # Integer and float.
    orig = rng.randint(low=0, high=127, size=n_samples * n_features).reshape(
        n_samples, n_features
    )
    dtypes = [
        np.int32,
        np.int64,
        np.byte,
        np.short,
        np.intc,
        np.int_,
        np.longlong,
        np.uint32,
        np.uint64,
        np.ubyte,
        np.ushort,
        np.uintc,
        np.uint,
        np.ulonglong,
        np.float16,
        np.float32,
        np.float64,
        np.half,
        np.single,
        np.double,
    ]
    for dtype in dtypes:
        X = np.array(orig, dtype=dtype)
        yield orig, X
        yield orig.tolist(), X.tolist()

    for dtype in dtypes:
        X = np.array(orig, dtype=dtype)
        df_orig = pd.DataFrame(orig)
        df = pd.DataFrame(X)
        yield df_orig, df

    # Boolean
    orig = rng.binomial(1, 0.5, size=n_samples * n_features).reshape(
        n_samples, n_features
    )
    for dtype in [np.bool_, bool]:
        X = np.array(orig, dtype=dtype)
        yield orig, X

    for dtype in [np.bool_, bool]:
        X = np.array(orig, dtype=dtype)
        df_orig = pd.DataFrame(orig)
        df = pd.DataFrame(X)
        yield df_orig, df


def pd_dtypes() -> Generator:
    """Enumerate all supported pandas extension types."""
    import pandas as pd

    # Integer
    dtypes = [
        pd.UInt8Dtype(),
        pd.UInt16Dtype(),
        pd.UInt32Dtype(),
        pd.UInt64Dtype(),
        pd.Int8Dtype(),
        pd.Int16Dtype(),
        pd.Int32Dtype(),
        pd.Int64Dtype(),
    ]

    Null: Union[float, None, Any] = np.nan
    orig = pd.DataFrame(
        {"f0": [1, 2, Null, 3], "f1": [4, 3, Null, 1]}, dtype=np.float32
    )
    for Null in (np.nan, None, pd.NA):
        for dtype in dtypes:
            df = pd.DataFrame(
                {"f0": [1, 2, Null, 3], "f1": [4, 3, Null, 1]}, dtype=dtype
            )
            yield orig, df

    # Float
    Null = np.nan
    dtypes = [pd.Float32Dtype(), pd.Float64Dtype()]
    orig = pd.DataFrame(
        {"f0": [1.0, 2.0, Null, 3.0], "f1": [3.0, 2.0, Null, 1.0]}, dtype=np.float32
    )
    for Null in (np.nan, None, pd.NA):
        for dtype in dtypes:
            df = pd.DataFrame(
                {"f0": [1.0, 2.0, Null, 3.0], "f1": [3.0, 2.0, Null, 1.0]}, dtype=dtype
            )
            yield orig, df
            ser_orig = orig["f0"]
            ser = df["f0"]
            assert isinstance(ser, pd.Series)
            assert isinstance(ser_orig, pd.Series)
            yield ser_orig, ser

    # Categorical
    orig = orig.astype("category")
    for Null in (np.nan, None, pd.NA):
        df = pd.DataFrame(
            {"f0": [1.0, 2.0, Null, 3.0], "f1": [3.0, 2.0, Null, 1.0]},
            dtype=pd.CategoricalDtype(),
        )
        yield orig, df

    # Boolean
    for Null in [None, pd.NA]:
        data = {"f0": [True, False, Null, True], "f1": [False, True, Null, True]}
        # pd.NA is not convertible to bool.
        orig = pd.DataFrame(data, dtype=np.bool_ if Null is None else pd.BooleanDtype())
        df = pd.DataFrame(data, dtype=pd.BooleanDtype())
        yield orig, df


def pd_arrow_dtypes() -> Generator:
    """Pandas DataFrame with pyarrow backed type."""
    import pandas as pd
    import pyarrow as pa  # pylint: disable=import-error

    # Integer
    dtypes = pandas_pyarrow_mapper
    Null: Union[float, None, Any] = np.nan
    orig = pd.DataFrame(
        {"f0": [1, 2, Null, 3], "f1": [4, 3, Null, 1]}, dtype=np.float32
    )
    # Create a dictionary-backed dataframe, enable this when the roundtrip is
    # implemented in pandas/pyarrow
    #
    # category = pd.ArrowDtype(pa.dictionary(pa.int32(), pa.int32(), ordered=True))
    # df = pd.DataFrame({"f0": [0, 2, Null, 3], "f1": [4, 3, Null, 1]}, dtype=category)

    # Error:
    # >>> df.astype("category")
    #   Function 'dictionary_encode' has no kernel matching input types
    #   (array[dictionary<values=int32, indices=int32, ordered=0>])

    # Error:
    # pd_cat_df = pd.DataFrame(
    #     {"f0": [0, 2, Null, 3], "f1": [4, 3, Null, 1]},
    #     dtype="category"
    # )
    # pa_catcodes = (
    #     df["f1"].array.__arrow_array__().combine_chunks().to_pandas().cat.codes
    # )
    # pd_catcodes = pd_cat_df["f1"].cat.codes
    # assert pd_catcodes.equals(pa_catcodes)

    for Null in (None, pd.NA):
        for dtype in dtypes:
            if dtype.startswith("float16") or dtype.startswith("bool"):
                continue
            df = pd.DataFrame(
                {"f0": [1, 2, Null, 3], "f1": [4, 3, Null, 1]}, dtype=dtype
            )
            yield orig, df

    orig = pd.DataFrame(
        {"f0": [True, False, pd.NA, True], "f1": [False, True, pd.NA, True]},
        dtype=pd.BooleanDtype(),
    )
    df = pd.DataFrame(
        {"f0": [True, False, pd.NA, True], "f1": [False, True, pd.NA, True]},
        dtype=pd.ArrowDtype(pa.bool_()),
    )
    yield orig, df


def check_inf(rng: RNG) -> None:
    """Validate there's no inf in X."""
    X = rng.random(size=32).reshape(8, 4)
    y = rng.random(size=8)
    X[5, 2] = np.inf

    with pytest.raises(ValueError, match="Input data contains `inf`"):
        xgboost.QuantileDMatrix(X, y)

    with pytest.raises(ValueError, match="Input data contains `inf`"):
        xgboost.DMatrix(X, y)


@memory.cache
def get_california_housing() -> Tuple[np.ndarray, np.ndarray]:
    """Fetch the California housing dataset from sklearn."""
    datasets = pytest.importorskip("sklearn.datasets")
    data = datasets.fetch_california_housing()
    return data.data, data.target


@memory.cache
def get_digits() -> Tuple[np.ndarray, np.ndarray]:
    """Fetch the digits dataset from sklearn."""
    datasets = pytest.importorskip("sklearn.datasets")
    data = datasets.load_digits()
    return data.data, data.target


@memory.cache
def get_cancer() -> Tuple[np.ndarray, np.ndarray]:
    """Fetch the breast cancer dataset from sklearn."""
    datasets = pytest.importorskip("sklearn.datasets")
    return datasets.load_breast_cancer(return_X_y=True)


@memory.cache
def get_sparse() -> Tuple[np.ndarray, np.ndarray]:
    """Generate a sparse dataset."""
    datasets = pytest.importorskip("sklearn.datasets")
    rng = np.random.RandomState(199)
    n = 2000
    sparsity = 0.75
    X, y = datasets.make_regression(n, random_state=rng)
    flag = rng.binomial(1, sparsity, X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if flag[i, j]:
                X[i, j] = np.nan
    return X, y


@memory.cache
def get_ames_housing() -> Tuple[np.ndarray, np.ndarray]:
    """
    Number of samples: 1460
    Number of features: 20
    Number of categorical features: 10
    Number of numerical features: 10
    """
    datasets = pytest.importorskip("sklearn.datasets")
    X, y = datasets.fetch_openml(data_id=42165, as_frame=True, return_X_y=True)

    categorical_columns_subset: List[str] = [
        "BldgType",  # 5 cats, no nan
        "GarageFinish",  # 3 cats, nan
        "LotConfig",  # 5 cats, no nan
        "Functional",  # 7 cats, no nan
        "MasVnrType",  # 4 cats, nan
        "HouseStyle",  # 8 cats, no nan
        "FireplaceQu",  # 5 cats, nan
        "ExterCond",  # 5 cats, no nan
        "ExterQual",  # 4 cats, no nan
        "PoolQC",  # 3 cats, nan
    ]

    numerical_columns_subset: List[str] = [
        "3SsnPorch",
        "Fireplaces",
        "BsmtHalfBath",
        "HalfBath",
        "GarageCars",
        "TotRmsAbvGrd",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "GrLivArea",
        "ScreenPorch",
    ]

    X = X[categorical_columns_subset + numerical_columns_subset]
    X[categorical_columns_subset] = X[categorical_columns_subset].astype("category")
    return X, y


@memory.cache
def get_mq2008(
    dpath: str,
) -> Tuple[
    sparse.csr_matrix,
    np.ndarray,
    np.ndarray,
    sparse.csr_matrix,
    np.ndarray,
    np.ndarray,
    sparse.csr_matrix,
    np.ndarray,
    np.ndarray,
]:
    """Fetch the mq2008 dataset."""
    datasets = pytest.importorskip("sklearn.datasets")
    src = "https://s3-us-west-2.amazonaws.com/xgboost-examples/MQ2008.zip"
    target = os.path.join(dpath, "MQ2008.zip")
    if not os.path.exists(target):
        request.urlretrieve(url=src, filename=target)

    with zipfile.ZipFile(target, "r") as f:
        f.extractall(path=dpath)

    (
        x_train,
        y_train,
        qid_train,
        x_test,
        y_test,
        qid_test,
        x_valid,
        y_valid,
        qid_valid,
    ) = datasets.load_svmlight_files(
        (
            os.path.join(dpath, "MQ2008/Fold1/train.txt"),
            os.path.join(dpath, "MQ2008/Fold1/test.txt"),
            os.path.join(dpath, "MQ2008/Fold1/vali.txt"),
        ),
        query_id=True,
        zero_based=False,
    )

    return (
        x_train,
        y_train,
        qid_train,
        x_test,
        y_test,
        qid_test,
        x_valid,
        y_valid,
        qid_valid,
    )
