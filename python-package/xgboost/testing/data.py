"""Utilities for data generation."""
from typing import Any, Generator, Tuple, Union

import numpy as np


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

    for dtype in dtypes:
        X = np.array(orig, dtype=dtype)
        df_orig = pd.DataFrame(orig)
        df = pd.DataFrame(X)
        yield df_orig, df

    # Boolean
    orig = rng.binomial(1, 0.5, size=n_samples * n_features).reshape(
        n_samples, n_features
    )
    for dtype in [np.bool_, np.bool8, bool]:
        X = np.array(orig, dtype=dtype)
        yield orig, X

    for dtype in [np.bool_, np.bool8, bool]:
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
