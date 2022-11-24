"""Utilities for data generation."""
from typing import Generator, Tuple

import numpy as np


def np_dtypes(
    n_samples: int, n_features: int
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Generate all supported dtypes from numpy."""
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
