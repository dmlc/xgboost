import ctypes
import os
from typing import Optional, List, Any, TypeVar, Union, Dict

import numpy as np

# os.PathLike/string/numpy.array/scipy.sparse/pd.DataFrame/dt.Frame/
# cudf.DataFrame/cupy.array/dlpack
DataType = Any

# xgboost accepts some other possible types in practice due to historical reason, which is
# lesser tested.  For now we encourage users to pass a simple list of string.
FeatNamesT = Optional[List[str]]

ArrayLike = Any
PathLike = Union[str, os.PathLike]
list_or_dict = Union[List, Dict]
cupy_t = ArrayLike  # maybe need a stub for cupy arrays
numpy_or_cupy_t = Union[np.ndarray, cupy_t]

# ctypes
# c_bst_ulong corresponds to bst_ulong defined in xgboost/c_api.h
c_bst_ulong = ctypes.c_uint64

ctype_t = Union[
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
    ctypes.c_float, ctypes.c_uint, ctypes.c_size_t
]

# supported numeric types
c_numeric_t = Union[
    ctypes.c_float, ctypes.c_double, ctypes.c_uint,
    ctypes.c_uint64, ctypes.c_int32, ctypes.c_int64
]

# c pointer types
# real type should be, as defined in typeshed
# but this has to be put in a .pyi file
# c_str_ptr_t = ctypes.pointer[ctypes.c_char]
c_str_ptr_t = ctypes.Array
# c_str_pptr_t = ctypes.pointer[ctypes.c_char_p]
c_str_pptr_t = ctypes.Array
# c_float_ptr_t = ctypes.pointer[ctypes.c_float]
c_float_ptr_t = ctypes.Array

# c_numeric_ptr_t = Union[
#  ctypes.pointer[ctypes.c_float], ctypes.pointer[ctypes.c_double],
#  ctypes.pointer[ctypes.c_uint], ctypes.pointer[ctypes.c_uint64],
#  ctypes.pointer[ctypes.c_int32], ctypes.pointer[ctypes.c_int64]
# ]
c_numeric_ptr_t = ctypes.Array

# template parameter
T = TypeVar("T")
