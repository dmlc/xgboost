# SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Working-in-progress support for cross-validation."""

import ctypes

from ._c_api import _LIB, _check_call
from .core import ExtMemQuantileDMatrix

_LIB.XGBCvFoldsCreate.restype = ctypes.c_int
_LIB.XGBCvFoldsCreate.argtypes = [ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]

_LIB.XGBCvFoldsFree.restype = ctypes.c_int
_LIB.XGBCvFoldsFree.argtypes = [ctypes.c_void_p]

_LIB.XGBCvFoldInfoBatchesCreate.restype = ctypes.c_int
_LIB.XGBCvFoldInfoBatchesCreate.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_void_p),
]

_LIB.XGBCvFoldInfoBatchesFree.restype = ctypes.c_int
_LIB.XGBCvFoldInfoBatchesFree.argtypes = [ctypes.c_void_p]

_LIB.XGBCvFoldGpairsCreate.restype = ctypes.c_int
_LIB.XGBCvFoldGpairsCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

_LIB.XGBCvFoldGpairsFree.restype = ctypes.c_int
_LIB.XGBCvFoldGpairsFree.argtypes = [ctypes.c_void_p]

_LIB.XGBCvGetGradient.restype = ctypes.c_int
_LIB.XGBCvGetGradient.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


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


class CvFoldInfoBatches:
    def __init__(self, data: ExtMemQuantileDMatrix, k_folds: int) -> None:
        if not isinstance(data, ExtMemQuantileDMatrix):
            raise TypeError(
                "`data` must be an ExtMemQuantileDMatrix for fused cross-validation."
            )

        k_folds = int(k_folds)
        if k_folds <= 0:
            raise ValueError("`k_folds` must be positive.")

        hdl = ctypes.c_void_p()
        _check_call(
            _LIB.XGBCvFoldInfoBatchesCreate(
                data.handle, ctypes.c_size_t(k_folds), ctypes.byref(hdl)
            )
        )
        self.handle = hdl
        self.k_folds = k_folds

    def __del__(self) -> None:
        if hasattr(self, "handle"):
            hdl = self.handle
            del self.handle
            _check_call(_LIB.XGBCvFoldInfoBatchesFree(hdl))


class CvFoldGpairs:
    def __init__(self) -> None:
        hdl = ctypes.c_void_p()
        _check_call(_LIB.XGBCvFoldGpairsCreate(ctypes.byref(hdl)))
        self.handle = hdl

    def __del__(self) -> None:
        if hasattr(self, "handle"):
            hdl = self.handle
            del self.handle
            _check_call(_LIB.XGBCvFoldGpairsFree(hdl))


def cross_validate(data: ExtMemQuantileDMatrix, k_folds: int) -> CvFoldInfoBatches:
    return CvFoldInfoBatches(data, k_folds)


def get_gradient(
    data: ExtMemQuantileDMatrix,
    fold_info: CvFoldInfoBatches,
    iteration: int,
    out: CvFoldGpairs,
) -> CvFoldGpairs:
    _check_call(
        _LIB.XGBCvGetGradient(
            data.handle,
            fold_info.handle,
            out.handle,
            ctypes.c_int(iteration),
        )
    )
    return out
