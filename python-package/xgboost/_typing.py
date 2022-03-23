"""Shared typing definition."""
import ctypes
import os
from typing import Optional, List, Any, TypeVar, Union

# os.PathLike/string/numpy.array/scipy.sparse/pd.DataFrame/dt.Frame/
# cudf.DataFrame/cupy.array/dlpack
DataType = Any

# xgboost accepts some other possible types in practice due to historical reason, which is
# lesser tested.  For now we encourage users to pass a simple list of string.
FeatureNames = Optional[List[str]]

ArrayLike = Any
PathLike = Union[str, os.PathLike]
CupyT = ArrayLike  # maybe need a stub for cupy arrays
NumpyOrCupy = Any

# ctypes
# c_bst_ulong corresponds to bst_ulong defined in xgboost/c_api.h
c_bst_ulong = ctypes.c_uint64  # pylint: disable=C0103

CTypeT = Union[
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_uint,
    ctypes.c_size_t,
]

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
# real type should be, as defined in typeshed
# but this has to be put in a .pyi file
# c_str_ptr_t = ctypes.pointer[ctypes.c_char]
CStrPtr = ctypes.pointer
# c_str_pptr_t = ctypes.pointer[ctypes.c_char_p]
CStrPptr = ctypes.pointer
# c_float_ptr_t = ctypes.pointer[ctypes.c_float]
CFloatPtr = ctypes.pointer

# c_numeric_ptr_t = Union[
#  ctypes.pointer[ctypes.c_float], ctypes.pointer[ctypes.c_double],
#  ctypes.pointer[ctypes.c_uint], ctypes.pointer[ctypes.c_uint64],
#  ctypes.pointer[ctypes.c_int32], ctypes.pointer[ctypes.c_int64]
# ]
CNumericPtr = ctypes.pointer

# template parameter
_T = TypeVar("_T")
