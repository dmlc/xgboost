# SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Working-in-progress support for cross-validation."""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

import numpy as np

from ._c_api import _LIB, _check_call
from .core import ExtMemQuantileDMatrix

if TYPE_CHECKING:
    import cupy as cp

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

_LIB.XGBCvFoldGpairsGet.restype = ctypes.c_int
_LIB.XGBCvFoldGpairsGet.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_size_t)),
    ctypes.POINTER(ctypes.c_size_t),
]

_LIB.XGBCvFoldGpairsFree.restype = ctypes.c_int
_LIB.XGBCvFoldGpairsFree.argtypes = [ctypes.c_void_p]

_LIB.XGBCvGetGradient.restype = ctypes.c_int
_LIB.XGBCvGetGradient.argtypes = [
    ctypes.c_void_p,
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

    def get(self, k: int) -> tuple[cp.ndarray, cp.ndarray]:
        import cupy as cp

        data = ctypes.POINTER(ctypes.c_float)()
        shape = ctypes.POINTER(ctypes.c_size_t)()
        n_dims = ctypes.c_size_t()
        _check_call(
            _LIB.XGBCvFoldGpairsGet(
                self.handle,
                ctypes.c_size_t(k),
                ctypes.byref(data),
                ctypes.byref(shape),
                ctypes.byref(n_dims),
            )
        )

        array_shape = tuple(int(shape[i]) for i in range(n_dims.value))
        n_elems = int(np.prod(array_shape))
        if n_elems == 0:
            return (
                cp.empty(array_shape, dtype=cp.float32),
                cp.empty(array_shape, dtype=cp.float32),
            )

        data_ptr = ctypes.cast(data, ctypes.c_void_p).value
        assert data_ptr is not None

        float_size = ctypes.sizeof(ctypes.c_float)
        pair_size = 2 * float_size
        strides = []
        stride = 1
        for dim in reversed(array_shape):
            strides.append(stride * pair_size)
            stride *= dim
        strides = list(reversed(strides))

        mem = cp.cuda.UnownedMemory(data_ptr, n_elems * pair_size, self)
        grad, hess = [
            cp.ndarray(
                array_shape,
                dtype=cp.float32,
                memptr=cp.cuda.MemoryPointer(mem, off),
                strides=strides,
            )
            for off in (0, float_size)
        ]
        return grad, hess


def cross_validate(data: ExtMemQuantileDMatrix, k_folds: int) -> CvFoldInfoBatches:
    return CvFoldInfoBatches(data, k_folds)


def get_gradient(
    data: ExtMemQuantileDMatrix,
    cv_folds: CvFolds,
    fold_info: CvFoldInfoBatches,
    iteration: int,
    out: CvFoldGpairs,
) -> CvFoldGpairs:
    _check_call(
        _LIB.XGBCvGetGradient(
            data.handle,
            cv_folds.handle,
            fold_info.handle,
            out.handle,
            ctypes.c_int(iteration),
        )
    )
    return out
