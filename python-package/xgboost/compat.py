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


# Modified from tensorflow with added caching.  There's a `LazyLoader` in
# `importlib.utils`, except it's unclear from its document on how to use it.  This one
# seems to be easy to understand and works out of box.


# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
# file except in compliance with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the specific language governing
# permissions and limitations under the License.
class LazyLoader(types.ModuleType):
    """Lazily import a module, mainly to avoid pulling in large dependencies."""

    def __init__(
        self,
        local_name: str,
        parent_module_globals: Dict,
        name: str,
        warning: Optional[str] = None,
    ) -> None:
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals
        self._warning = warning
        self.module: Optional[types.ModuleType] = None

        super().__init__(name)

    def _load(self) -> types.ModuleType:
        """Load the module and insert it into the parent's globals."""
        # Import the target module and insert it into the parent's namespace
        module = importlib.import_module(self.__name__)
        self._parent_module_globals[self._local_name] = module

        # Emit a warning if one was specified
        if self._warning:
            logging.warning(self._warning)
            # Make sure to only warn once.
        self._warning = None

        # Update this object's dict so that if someone keeps a reference to the
        #   LazyLoader, lookups are efficient (__getattr__ is only called on lookups
        #   that fail).
        self.__dict__.update(module.__dict__)

        return module

    def __getattr__(self, item: str) -> Any:
        if not self.module:
            self.module = self._load()
        return getattr(self.module, item)

    def __dir__(self) -> List[str]:
        if not self.module:
            self.module = self._load()
        return dir(self.module)
