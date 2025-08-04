# pylint: disable=protected-access
"""Shared typing definition."""
import ctypes
import os
from typing import (
    TYPE_CHECKING,
    Any,
    AnyStr,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeAlias,
    TypeVar,
    Union,
)

import numpy as np

DataType = Any

FeatureInfo = Sequence[str]
FeatureNames = FeatureInfo
FeatureTypes = FeatureInfo
BoosterParam = Union[List, Dict[str, Any]]  # better be sequence

ArrayLike = Any
if TYPE_CHECKING:
    import pyarrow as pa

    PathLike = Union[str, os.PathLike[str]]
else:
    PathLike = Union[str, os.PathLike]

ArrowCatCol: TypeAlias = Optional[Union["pa.StringArray", "pa.NumericArray"]]
ArrowCatList: TypeAlias = List[Tuple[str, Optional[ArrowCatCol]]]

CupyT = ArrayLike  # maybe need a stub for cupy arrays
NumpyOrCupy = Any
NumpyDType = Union[str, Type[np.number]]  # pylint: disable=invalid-name
PandasDType = Any  # real type is pandas.core.dtypes.base.ExtensionDtype

FloatCompatible = Union[float, np.float32, np.float64]

# typing.SupportsInt is not suitable here since floating point values are convertible to
# integers as well.
Integer = Union[int, np.integer]
IterationRange = Tuple[Integer, Integer]

# callables
FPreProcCallable = Callable

# ctypes
# c_bst_ulong corresponds to bst_ulong defined in xgboost/c_api.h
c_bst_ulong = ctypes.c_uint64  # pylint: disable=C0103

ModelIn = Union[os.PathLike[AnyStr], bytearray, str]

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

# The second arg is actually Optional[List[cudf.Series]], skipped for easier type check.
# The cudf Series is the obtained cat codes, preserved in the `DataIter` to prevent it
# being freed.
TransformedData = Tuple[Any, Optional[FeatureNames], Optional[FeatureTypes]]

# template parameter
_T = TypeVar("_T")
_F = TypeVar("_F", bound=Callable[..., Any])

_ScoreList = Union[List[float], List[Tuple[float, float]]]
EvalsLog: TypeAlias = Dict[str, Dict[str, _ScoreList]]
