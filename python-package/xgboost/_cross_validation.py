# SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Working-in-progress support for cross-validation."""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from ._c_api import _LIB, _check_call, make_jcargs
from ._typing import ArrayLike
from .core import DataIter, ExtMemQuantileDMatrix, ctypes2buffer

if TYPE_CHECKING:
    import cupy as cp

_LIB.XGBCvFoldModelsCreate.restype = ctypes.c_int
_LIB.XGBCvFoldModelsCreate.argtypes = [
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_void_p),
]

_LIB.XGBCvFoldModelsFree.restype = ctypes.c_int
_LIB.XGBCvFoldModelsFree.argtypes = [ctypes.c_void_p]

_LIB.XGBCvFoldModelsSaveModelToBuffer.restype = ctypes.c_int
_LIB.XGBCvFoldModelsSaveModelToBuffer.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_char)),
]

_LIB.XGBCvFoldInfoBatchesCreate.restype = ctypes.c_int
_LIB.XGBCvFoldInfoBatchesCreate.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_void_p),
]

_LIB.XGBCvFoldInfoBatchesFree.restype = ctypes.c_int
_LIB.XGBCvFoldInfoBatchesFree.argtypes = [ctypes.c_void_p]

_LIB.XGBCvFoldPredictionsCreate.restype = ctypes.c_int
_LIB.XGBCvFoldPredictionsCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

_LIB.XGBCvFoldModelsInitPrediction.restype = ctypes.c_int
_LIB.XGBCvFoldModelsInitPrediction.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]

_LIB.XGBCvFoldPredictionsGet.restype = ctypes.c_int
_LIB.XGBCvFoldPredictionsGet.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_size_t),
]

_LIB.XGBCvFoldPredictionsGetValid.restype = ctypes.c_int
_LIB.XGBCvFoldPredictionsGetValid.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_size_t),
]

_LIB.XGBCvFoldPredictionsGetRefit.restype = ctypes.c_int
_LIB.XGBCvFoldPredictionsGetRefit.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_size_t),
]

_LIB.XGBCvFoldPredictionsFree.restype = ctypes.c_int
_LIB.XGBCvFoldPredictionsFree.argtypes = [ctypes.c_void_p]

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

_LIB.XGBCvFoldGpairsGetRefit.restype = ctypes.c_int
_LIB.XGBCvFoldGpairsGetRefit.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_size_t)),
    ctypes.POINTER(ctypes.c_size_t),
]

_LIB.XGBCvFoldGpairsFree.restype = ctypes.c_int
_LIB.XGBCvFoldGpairsFree.argtypes = [ctypes.c_void_p]

_LIB.XGBCvFoldModelsGetGradient.restype = ctypes.c_int
_LIB.XGBCvFoldModelsGetGradient.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]

_LIB.XGBCvFoldTreeMethodCreate.restype = ctypes.c_int
_LIB.XGBCvFoldTreeMethodCreate.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_void_p),
]

_LIB.XGBCvFoldTreeMethodFree.restype = ctypes.c_int
_LIB.XGBCvFoldTreeMethodFree.argtypes = [ctypes.c_void_p]

_LIB.XGBCvFoldTreeMethodUpdate.restype = ctypes.c_int
_LIB.XGBCvFoldTreeMethodUpdate.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]


class FoldModels:
    """Result of training cross validation.

    Parameters
    ----------
    data :
        The full dataset.
    k_folds :
        Number of cross-validation folds.
    refit :
        Also train a model on the full dataset, inside the same page loop as the fold
        models. Useful when the hyperparameters are fixed and cross-validation only
        selects the number of boosting rounds.

    """

    def __init__(
        self, data: ExtMemQuantileDMatrix, k_folds: int, refit: bool = False
    ) -> None:
        if not isinstance(data, ExtMemQuantileDMatrix):
            raise TypeError(
                "`data` must be an ExtMemQuantileDMatrix for fused cross-validation."
            )

        k_folds = int(k_folds)
        if k_folds <= 0:
            raise ValueError("`k_folds` must be positive.")

        hdl = ctypes.c_void_p()
        _check_call(
            _LIB.XGBCvFoldModelsCreate(
                ctypes.c_size_t(k_folds),
                data.handle,
                ctypes.c_int(int(refit)),
                ctypes.byref(hdl),
            )
        )
        self.handle = hdl
        self.k_folds = k_folds
        self.refit = bool(refit)

    def num_boosted_rounds(self) -> int:
        """Number of boosted rounds shared by all the models, folds and refit alike."""
        rounds = ctypes.c_int()
        _check_call(
            _LIB.XGBCvFoldModelsBoostedRounds(self.handle, ctypes.byref(rounds))
        )
        return rounds.value

    def save_raw(self, raw_format: str = "ubj") -> bytearray:
        """Save every model to an in memory buffer representation.

        The buffer holds one entry per fold under a ``cv_folds`` array, each in the same
        format :py:meth:`Booster.save_raw` produces for a single model. The full-data
        model is not a fold, so it is stored under a ``refit`` key next to the array. The
        key is absent when the run has no refit model.

        Parameters
        ----------
        raw_format :
            Format of output buffer. Can be `json` or `ubj`.

        Returns
        -------
        An in memory buffer representation of the models

        """
        length = ctypes.c_uint64()
        cptr = ctypes.POINTER(ctypes.c_char)()
        config = make_jcargs(format=raw_format)
        _check_call(
            _LIB.XGBCvFoldModelsSaveModelToBuffer(
                self.handle, config, ctypes.byref(length), ctypes.byref(cptr)
            )
        )
        return ctypes2buffer(cptr, length.value)

    def __del__(self) -> None:
        if hasattr(self, "handle"):
            hdl = self.handle
            del self.handle
            _check_call(_LIB.XGBCvFoldModelsFree(hdl))

    def init_prediction(
        self,
        data: ExtMemQuantileDMatrix,
        fold_info: FoldInfoBatches,
        out: FoldPredictions,
    ) -> FoldPredictions:
        """Initialize prediction buffers."""

        _check_call(
            _LIB.XGBCvFoldModelsInitPrediction(
                self.handle,
                data.handle,
                fold_info.handle,
                out.handle,
            )
        )
        return out

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def get_gradient(
        self,
        data: ExtMemQuantileDMatrix,
        iteration: int,
        fold_info: FoldInfoBatches,
        predt: FoldPredictions,
        out: FoldGpairs,
    ) -> FoldGpairs:
        """Calculate the gradient."""

        _check_call(
            _LIB.XGBCvFoldModelsGetGradient(
                self.handle,
                data.handle,
                fold_info.handle,
                predt.handle,
                out.handle,
                ctypes.c_int(iteration),
            )
        )
        return out


class FoldTreeMethod:
    """Optimizer used for fused cross-validation."""

    def __init__(
        self, cv_folds: FoldModels, data: ExtMemQuantileDMatrix, params: dict[str, Any]
    ) -> None:
        hdl = ctypes.c_void_p()
        _check_call(
            _LIB.XGBCvFoldTreeMethodCreate(
                cv_folds.handle,
                data.handle,
                make_jcargs(**(params or {})),
                ctypes.byref(hdl),
            )
        )
        self.handle = hdl

    def __del__(self) -> None:
        if hasattr(self, "handle"):
            hdl = self.handle
            del self.handle
            _check_call(_LIB.XGBCvFoldTreeMethodFree(hdl))

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def update(
        self,
        cv_folds: FoldModels,
        data: ExtMemQuantileDMatrix,
        fold_info: FoldInfoBatches,
        gpairs: FoldGpairs,
        predt: FoldPredictions,
    ) -> None:
        """Grow and commit one fused CV tree for each fold."""

        _check_call(
            _LIB.XGBCvFoldTreeMethodUpdate(
                self.handle,
                cv_folds.handle,
                data.handle,
                fold_info.handle,
                gpairs.handle,
                predt.handle,
            )
        )


class FoldInfoBatches:
    """Meta information used during cross validation."""

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


class FoldPredictions:
    """Prediction buffers for each fold."""

    def __init__(self) -> None:
        hdl = ctypes.c_void_p()
        _check_call(_LIB.XGBCvFoldPredictionsCreate(ctypes.byref(hdl)))
        self.handle = hdl

    def __del__(self) -> None:
        if hasattr(self, "handle"):
            hdl = self.handle
            del self.handle
            _check_call(_LIB.XGBCvFoldPredictionsFree(hdl))

    def _as_array(
        self,
        data: ctypes._Pointer,
        n_rows: ctypes.c_size_t,
        n_columns: ctypes.c_size_t,
        copy: bool,
    ) -> cp.ndarray:
        import cupy as cp

        shape = (int(n_rows.value), int(n_columns.value))
        n_elems = shape[0] * shape[1]
        if n_elems == 0:
            return cp.empty(shape, dtype=cp.float32)

        data_ptr = ctypes.cast(data, ctypes.c_void_p).value
        assert data_ptr is not None
        float_size = ctypes.sizeof(ctypes.c_float)
        mem = cp.cuda.UnownedMemory(data_ptr, n_elems * float_size, self)
        predt = cp.ndarray(  # pylint: disable=unexpected-keyword-arg
            shape,
            dtype=cp.float32,
            memptr=cp.cuda.MemoryPointer(mem, 0),
        )
        return predt.copy() if copy else predt

    def get(self, k: int, copy: bool = True) -> cp.ndarray:
        """Retrieve the training prediction cache of the k^th fold.

        The result is indexed by the global row index, the rows held out by the fold are
        unused padding. Use :py:meth:`get_refit` for the full-data model.

        """
        data = ctypes.POINTER(ctypes.c_float)()
        n_rows = ctypes.c_size_t()
        n_columns = ctypes.c_size_t()
        _check_call(
            _LIB.XGBCvFoldPredictionsGet(
                self.handle,
                ctypes.c_size_t(k),
                ctypes.byref(data),
                ctypes.byref(n_rows),
                ctypes.byref(n_columns),
            )
        )
        return self._as_array(data, n_rows, n_columns, copy)

    def get_valid(self, copy: bool = True) -> cp.ndarray:
        """Retrieve the raw out-of-fold prediction of every row."""
        data = ctypes.POINTER(ctypes.c_float)()
        n_rows = ctypes.c_size_t()
        n_columns = ctypes.c_size_t()
        _check_call(
            _LIB.XGBCvFoldPredictionsGetValid(
                self.handle,
                ctypes.byref(data),
                ctypes.byref(n_rows),
                ctypes.byref(n_columns),
            )
        )
        return self._as_array(data, n_rows, n_columns, copy)

    def get_refit(self, copy: bool = True) -> cp.ndarray:
        """Retrieve the training prediction cache of the full-data model. Requires a run
        created with ``refit=True``.

        """
        data = ctypes.POINTER(ctypes.c_float)()
        n_rows = ctypes.c_size_t()
        n_columns = ctypes.c_size_t()
        _check_call(
            _LIB.XGBCvFoldPredictionsGetRefit(
                self.handle,
                ctypes.byref(data),
                ctypes.byref(n_rows),
                ctypes.byref(n_columns),
            )
        )
        return self._as_array(data, n_rows, n_columns, copy)


class FoldGpairs:
    """Gradient from objective functions."""

    def __init__(self) -> None:
        hdl = ctypes.c_void_p()
        _check_call(_LIB.XGBCvFoldGpairsCreate(ctypes.byref(hdl)))
        self.handle = hdl

    def __del__(self) -> None:
        if hasattr(self, "handle"):
            hdl = self.handle
            del self.handle
            _check_call(_LIB.XGBCvFoldGpairsFree(hdl))

    # pylint: disable=too-many-locals
    def _as_arrays(
        self,
        data: ctypes._Pointer,
        shape: ctypes._Pointer,
        n_dims: ctypes.c_size_t,
        copy: bool,
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """Split an interleaved gradient-hessian buffer into two strided views."""

        import cupy as cp

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
            cp.ndarray(  # pylint: disable=unexpected-keyword-arg
                array_shape,
                dtype=cp.float32,
                memptr=cp.cuda.MemoryPointer(mem, off),
                strides=strides,
            )
            for off in (0, float_size)
        ]
        if copy:
            grad, hess = grad.copy(), hess.copy()
        return grad, hess

    def get(self, k: int, copy: bool = True) -> tuple[cp.ndarray, cp.ndarray]:
        """Retrieve the gradient for the k^th fold.

        The rows held out by the fold are zeroed. Use :py:meth:`get_refit` for the
        full-data model.

        """
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
        return self._as_arrays(data, shape, n_dims, copy)

    def get_refit(self, copy: bool = True) -> tuple[cp.ndarray, cp.ndarray]:
        """Retrieve the gradient of the full-data model. Requires a run created with
        ``refit=True``.

        """
        data = ctypes.POINTER(ctypes.c_float)()
        shape = ctypes.POINTER(ctypes.c_size_t)()
        n_dims = ctypes.c_size_t()
        _check_call(
            _LIB.XGBCvFoldGpairsGetRefit(
                self.handle,
                ctypes.byref(data),
                ctypes.byref(shape),
                ctypes.byref(n_dims),
            )
        )
        return self._as_arrays(data, shape, n_dims, copy)


def _n_rows(data: ArrayLike) -> int:
    shape = getattr(data, "shape", None)
    return len(data) if shape is None else int(shape[0])


def _to_device(data: ArrayLike) -> Any:
    import cupy as cp

    return cp.asarray(data)


DFT_BATCH_SIZE = 2**20


class CvDataIter(DataIter):
    """Split ``X`` and ``y`` into the batches of a fused cross-validation matrix."""

    def __init__(
        self, X: ArrayLike, y: ArrayLike, *, batch_size: int = DFT_BATCH_SIZE
    ) -> None:
        # First, so that the base class is whole even if the arguments are rejected below.
        super().__init__(cache_prefix=None, on_host=True)

        n_samples = _n_rows(X)
        if n_samples == 0:
            raise ValueError("`X` has no row.")
        if _n_rows(y) != n_samples:
            raise ValueError(
                f"`X` has {n_samples} rows, `y` has {_n_rows(y)}, they must match."
            )
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("`batch_size` must be positive.")

        self.X = X
        self.y = y
        # Batch boundaries in the global row index space, the space that the fold indices
        # and the prediction caches are indexed by as well.
        self.batch_ptr = [*range(0, n_samples, batch_size), n_samples]
        self.it = 0

    @property
    def n_batches(self) -> int:
        """Number of batches this iterator produces."""
        return len(self.batch_ptr) - 1

    def next(self, input_data: Callable) -> bool:
        if self.it == self.n_batches:
            return False
        begin, end = self.batch_ptr[self.it], self.batch_ptr[self.it + 1]
        input_data(
            data=_to_device(self.X[begin:end]), label=_to_device(self.y[begin:end])
        )
        self.it += 1
        return True

    def reset(self) -> None:
        self.it = 0


def cross_val_predict(
    X: ArrayLike,
    y: ArrayLike,
    *,
    cv: int = 5,
    params: dict[str, Any] | None = None,
    num_boost_round: int = 10,
) -> cp.ndarray:
    """Generate cross-validated estimates for each input data point.

    Parameters
    ----------
    X :
        Predictor. Must support row slicing, like a numpy or a cupy array.
    y :
        Label, with as many rows as `X`.
    cv :
        Number of folds, at least 2. A splitter object is not accepted. Each fold holds out
        a contiguous window of every external-memory page.
    params :
        Booster parameters accepted by the CV tree method, like `learning_rate`.
    num_boost_round :
        Number of boosting rounds, shared by every fold model.

    Returns
    -------
    The estimate of every row, on the device, in the row order of `X`. The shape is
    ``(n_samples,)`` for a single target and ``(n_samples, n_targets)`` otherwise.

    """
    cv = int(cv)
    if cv < 2:
        raise ValueError(f"`cv` must be at least 2, got {cv}.")
    num_boost_round = int(num_boost_round)
    if num_boost_round < 0:
        raise ValueError(
            f"`num_boost_round` must not be negative, got {num_boost_round}."
        )
    params = params if params is not None else {}
    max_bin = params.get("max_bin", None)

    Xy = ExtMemQuantileDMatrix(CvDataIter(X, y), max_bin=max_bin)
    models = FoldModels(data=Xy, k_folds=cv)
    fold_info = FoldInfoBatches(Xy, k_folds=cv)
    predts = FoldPredictions()
    models.init_prediction(Xy, fold_info, out=predts)
    gpairs = FoldGpairs()
    tree_method = FoldTreeMethod(models, Xy, params=params or {})

    for iteration in range(num_boost_round):
        models.get_gradient(Xy, iteration, fold_info, predts, out=gpairs)
        tree_method.update(models, Xy, fold_info, gpairs, predts)

    # The cache holds raw margins, which the squared error objective leaves in the
    # prediction domain already.
    oof = predts.get_valid()
    return oof.reshape(-1) if oof.shape[1] == 1 else oof
