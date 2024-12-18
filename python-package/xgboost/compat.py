# pylint: disable=invalid-name,unused-import
"""For compatibility and optional dependencies."""
import functools
import importlib.util
import logging
import sys
import types
from typing import Any, Sequence, cast

import numpy as np

from ._typing import _T

assert sys.version_info[0] == 3, "Python 2 is no longer supported."


def py_str(x: bytes | None) -> str:
    """convert c string back to python string"""
    assert x is not None  # ctypes might return None
    return x.decode("utf-8")  # type: ignore


def lazy_isinstance(instance: Any, module: str, name: str) -> bool:
    """Use string representation to identify a type."""

    # Notice, we use .__class__ as opposed to type() in order
    # to support object proxies such as weakref.proxy
    cls = instance.__class__
    is_same_module = cls.__module__ == module
    has_same_name = cls.__name__ == name
    return is_same_module and has_same_name


# pandas
try:
    from pandas import DataFrame, Series

    PANDAS_INSTALLED = True
except ImportError:
    DataFrame = object
    Series = object
    PANDAS_INSTALLED = False


# sklearn
try:
    from sklearn import __version__ as _sklearn_version
    from sklearn.base import BaseEstimator as XGBModelBase
    from sklearn.base import ClassifierMixin as XGBClassifierBase
    from sklearn.base import RegressorMixin as XGBRegressorBase

    try:
        from sklearn.model_selection import StratifiedKFold as XGBStratifiedKFold
    except ImportError:
        from sklearn.cross_validation import StratifiedKFold as XGBStratifiedKFold

    # sklearn.utils Tags types can be imported unconditionally once
    # xgboost's minimum scikit-learn version is 1.6 or higher
    try:
        from sklearn.utils import Tags as _sklearn_Tags
    except ImportError:
        _sklearn_Tags = object

    SKLEARN_INSTALLED = True

except ImportError:
    SKLEARN_INSTALLED = False

    # used for compatibility without sklearn
    class XGBModelBase:  # type: ignore[no-redef]
        """Dummy class for sklearn.base.BaseEstimator."""

    class XGBClassifierBase:  # type: ignore[no-redef]
        """Dummy class for sklearn.base.ClassifierMixin."""

    class XGBRegressorBase:  # type: ignore[no-redef]
        """Dummy class for sklearn.base.RegressorMixin."""

    XGBStratifiedKFold = None

    _sklearn_Tags = object
    _sklearn_version = object


_logger = logging.getLogger(__name__)


@functools.cache
def is_cudf_available() -> bool:
    """Check cuDF package available or not"""
    if importlib.util.find_spec("cudf") is None:
        return False
    try:
        import cudf

        return True
    except ImportError:
        _logger.exception("Importing cuDF failed, use DMatrix instead of QDM")
        return False


@functools.cache
def is_cupy_available() -> bool:
    """Check cupy package available or not"""
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy

        return True
    except ImportError:
        return False


@functools.cache
def import_cupy() -> types.ModuleType:
    """Import cupy."""
    if not is_cupy_available():
        raise ImportError("`cupy` is required for handling CUDA buffer.")

    import cupy

    return cupy


@functools.cache
def is_pyarrow_available() -> bool:
    """Check pyarrow package available or not"""
    if importlib.util.find_spec("pyarrow") is None:
        return False
    return True


@functools.cache
def import_pyarrow() -> types.ModuleType:
    """Import pyarrow with memory cache."""
    import pyarrow as pa

    return pa


@functools.cache
def import_polars() -> types.ModuleType:
    """Import polars with memory cache."""
    import polars as pl

    return pl


try:
    import scipy.sparse as scipy_sparse
    from scipy.sparse import csr_matrix as scipy_csr
except ImportError:
    scipy_sparse = False
    scipy_csr = object


def concat(value: Sequence[_T]) -> _T:  # pylint: disable=too-many-return-statements
    """Concatenate row-wise."""
    if isinstance(value[0], np.ndarray):
        value_arr = cast(Sequence[np.ndarray], value)
        return np.concatenate(value_arr, axis=0)
    if scipy_sparse and isinstance(value[0], scipy_sparse.csr_matrix):
        return scipy_sparse.vstack(value, format="csr")
    if scipy_sparse and isinstance(value[0], scipy_sparse.csc_matrix):
        return scipy_sparse.vstack(value, format="csc")
    if scipy_sparse and isinstance(value[0], scipy_sparse.spmatrix):
        # other sparse format will be converted to CSR.
        return scipy_sparse.vstack(value, format="csr")
    if PANDAS_INSTALLED and isinstance(value[0], (DataFrame, Series)):
        from pandas import concat as pd_concat

        return pd_concat(value, axis=0)
    if lazy_isinstance(value[0], "cudf.core.dataframe", "DataFrame") or lazy_isinstance(
        value[0], "cudf.core.series", "Series"
    ):
        from cudf import concat as CUDF_concat

        return CUDF_concat(value, axis=0)
    from .data import _is_cupy_alike

    if _is_cupy_alike(value[0]):
        import cupy

        # pylint: disable=c-extension-no-member,no-member
        d = cupy.cuda.runtime.getDevice()
        for v in value:
            arr = cast(cupy.ndarray, v)
            d_v = arr.device.id
            assert d_v == d, "Concatenating arrays on different devices."
        return cupy.concatenate(value, axis=0)
    raise TypeError(f"Unknown type: {type(value[0])}")
