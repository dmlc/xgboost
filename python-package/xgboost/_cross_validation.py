"""Working-in-progress support for cross-validation."""

import ctypes

from ._c_api import _LIB, _check_call
from .core import ExtMemQuantileDMatrix

_LIB.XGBCvFoldsCreate.restype = ctypes.c_int
_LIB.XGBCvFoldsCreate.argtypes = [ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]

_LIB.XGBCvFoldsFree.restype = ctypes.c_int
_LIB.XGBCvFoldsFree.argtypes = [ctypes.c_void_p]


class CvFolds:
    def __init__(self, k_folds: int) -> None:
        hdl = ctypes.c_void_p()
        _check_call(_LIB.XGBCvFoldsCreate(int(k_folds), ctypes.byref(hdl)))
        self.handle = hdl

    def __del__(self) -> None:
        if hasattr(self, "handle"):
            hdl = self.handle
            del self.handle
            _check_call(_LIB.XGBCvFoldsFree(hdl))


def cross_validate(data: ExtMemQuantileDMatrix) -> CvFolds:
    pass
