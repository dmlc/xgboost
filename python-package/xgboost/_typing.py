# pylint: disable=protected-access
"""Shared typing definition."""
import ctypes
import os
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    overload,
)

# os.PathLike/string/numpy.array/scipy.sparse/pd.DataFrame/dt.Frame/
# cudf.DataFrame/cupy.array/dlpack
import numpy as np

DataType = Any

# xgboost accepts some other possible types in practice due to historical reason, which
# is lesser tested.  For now we encourage users to pass a simple list.
FeatureInfo = Sequence[str]
FeatureNames = FeatureInfo
BoosterParam = Union[List, Dict]  # better be sequence

ArrayLike = Any
PathLike = Union[str, os.PathLike]
CupyT = ArrayLike  # maybe need a stub for cupy arrays
NumpyOrCupy = Any
NumpyDType = Union[str, Type[np.number]]
PandasDType = Any  # real type is pandas.core.dtypes.base.ExtensionDtype

FloatCompatible = Union[float, np.float32, np.float64]

# callables
FPreProcCallable = Callable

# ctypes
# c_bst_ulong corresponds to bst_ulong defined in xgboost/c_api.h
c_bst_ulong = ctypes.c_uint64  # pylint: disable=C0103

CTypeT = TypeVar(
    "CTypeT",
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_uint,
    ctypes.c_size_t,
)

# supported numeric types
CNumeric = Union[
    ctypes.c_float,
    ctypes.c_double,
    ctypes.c_uint,
    ctypes.c_uint64,
    ctypes.c_int32,
    ctypes.c_int64,
]

# c pointer types
if TYPE_CHECKING:
    CStrPtr = ctypes._Pointer[ctypes.c_char]

    CStrPptr = ctypes._Pointer[ctypes.c_char_p]

    CFloatPtr = ctypes._Pointer[ctypes.c_float]

    CNumericPtr = Union[
        ctypes._Pointer[ctypes.c_float],
        ctypes._Pointer[ctypes.c_double],
        ctypes._Pointer[ctypes.c_uint],
        ctypes._Pointer[ctypes.c_uint64],
        ctypes._Pointer[ctypes.c_int32],
        ctypes._Pointer[ctypes.c_int64],
    ]
else:
    CStrPtr = ctypes._Pointer

    CStrPptr = ctypes._Pointer

    CFloatPtr = ctypes._Pointer

    CNumericPtr = Union[
        ctypes._Pointer,
        ctypes._Pointer,
        ctypes._Pointer,
        ctypes._Pointer,
        ctypes._Pointer,
        ctypes._Pointer,
    ]

# template parameter
_T = TypeVar("_T")
_F = TypeVar("_F", bound=Callable[..., Any])


class CatDType:
    """Helper class for passing information about categorical feature. This is useful
    when input data is not a dataframe.

    ..note:: Categorical feature support is experimental.

    Parameters
    ----------

    n_categories :
        Total number of categories for a specific feature.

    """

    def __init__(self, n_categories: int) -> None:
        self.n_categories: int = n_categories

    @staticmethod
    def from_str(type_str: str) -> Union["CatDType", str]:
        """Create from internal string representation."""
        if len(type_str) == 1:
            return type_str
        return CatDType(int(type_str[2:-1]))

    def __str__(self) -> str:
        """Return an internal string representation."""
        return f"c({str(self.n_categories)})"


FeatureTypes = Sequence[Union[str, CatDType]]


@overload
def get_feature_types(ft_str: None) -> None:
    ...


@overload
def get_feature_types(ft_str: Sequence[str]) -> FeatureTypes:
    ...


def get_feature_types(ft_str: Optional[Sequence[str]]) -> Optional[FeatureTypes]:
    """Convert feature types from string to :py:class:`CatDType`."""
    if ft_str is None:
        return None
    res: FeatureTypes = [
        CatDType.from_str(f) if f.startswith("c") else f for f in ft_str
    ]
    return res
