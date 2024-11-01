# pylint: disable= invalid-name,  unused-import
"""For compatibility and optional dependencies."""
import importlib.util
import logging
import sys
import types
from typing import Any, Dict, List, Optional, Sequence, cast

import numpy as np

from ._typing import _T

assert sys.version_info[0] == 3, "Python 2 is no longer supported."


def py_str(x: bytes) -> str:
    """convert c string back to python string"""
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
    from pandas import DataFrame, MultiIndex, Series
    from pandas import concat as pandas_concat

    PANDAS_INSTALLED = True
except ImportError:
    MultiIndex = object
    DataFrame = object
    Series = object
    pandas_concat = None
    PANDAS_INSTALLED = False


# sklearn
try:
    from sklearn.base import BaseEstimator as XGBModelBase
    from sklearn.base import ClassifierMixin as XGBClassifierBase
    from sklearn.base import RegressorMixin as XGBRegressorBase
    from sklearn.preprocessing import LabelEncoder

    try:
        from sklearn.model_selection import KFold as XGBKFold
        from sklearn.model_selection import StratifiedKFold as XGBStratifiedKFold
    except ImportError:
        from sklearn.cross_validation import KFold as XGBKFold
        from sklearn.cross_validation import StratifiedKFold as XGBStratifiedKFold

    SKLEARN_INSTALLED = True

except ImportError:
    SKLEARN_INSTALLED = False

    # used for compatibility without sklearn
    XGBModelBase = object
    XGBClassifierBase = object
    XGBRegressorBase = object
    LabelEncoder = object

    XGBKFold = None
    XGBStratifiedKFold = None


_logger = logging.getLogger(__name__)


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


def is_cupy_available() -> bool:
    """Check cupy package available or not"""
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy

        return True
    except ImportError:
        return False


def import_cupy() -> types.ModuleType:
    """Import cupy."""
    if not is_cupy_available():
        raise ImportError("`cupy` is required for handling CUDA buffer.")

    import cupy  # pylint: disable=import-error

    return cupy


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
        return pandas_concat(value, axis=0)
    if lazy_isinstance(value[0], "cudf.core.dataframe", "DataFrame") or lazy_isinstance(
        value[0], "cudf.core.series", "Series"
    ):
        from cudf import concat as CUDF_concat  # pylint: disable=import-error

        return CUDF_concat(value, axis=0)
    from .data import _is_cupy_alike

    if _is_cupy_alike(value[0]):
        import cupy  # pylint: disable=import-error

        # pylint: disable=c-extension-no-member,no-member
        d = cupy.cuda.runtime.getDevice()
        for v in value:
            arr = cast(cupy.ndarray, v)
            d_v = arr.device.id
            assert d_v == d, "Concatenating arrays on different devices."
        return cupy.concatenate(value, axis=0)
    raise TypeError("Unknown type.")
