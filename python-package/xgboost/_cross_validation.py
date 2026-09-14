# SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Working-in-progress support for cross-validation."""

from __future__ import annotations

import ctypes
import json
import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, override

import numpy as np

from ._c_api import _LIB, _check_call, make_jcargs
from ._data_utils import cuda_array_interface
from ._typing import ArrayLike
from .compat import import_cupy, py_str
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
]

_LIB.XGBCvFoldEvaluatorCreate.restype = ctypes.c_int
_LIB.XGBCvFoldEvaluatorCreate.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_void_p),
]

_LIB.XGBCvFoldEvaluatorFree.restype = ctypes.c_int
_LIB.XGBCvFoldEvaluatorFree.argtypes = [ctypes.c_void_p]

_LIB.XGBCvFoldEvaluatorEval.restype = ctypes.c_int
_LIB.XGBCvFoldEvaluatorEval.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
]


_LIB.XGBCvFoldAssignmentCreate.restype = ctypes.c_int
_LIB.XGBCvFoldAssignmentCreate.argtypes = [
    ctypes.c_char_p,
    ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_void_p),
]
_LIB.XGBCvFoldAssignmentFree.restype = ctypes.c_int
_LIB.XGBCvFoldAssignmentFree.argtypes = [ctypes.c_void_p]


def _fold_integer(value: int, name: str, low: int, high: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"`{name}` must be an integer.")
    value = operator.index(value)
    if not low <= value <= high:
        raise ValueError(f"`{name}` must be in [{low}, {high}].")
    return value


class FoldAssignment:
    """Own an immutable GPU copy of one held-out fold ID per row.

    IDs must be integers in ``[0, k_folds)`` with every fold nonempty.

    """

    def __init__(self, fold_ids: ArrayLike, *, k_folds: int) -> None:
        cp = import_cupy()

        ids = cp.asarray(fold_ids)
        if ids.ndim != 1 or ids.size < 2 or ids.dtype.kind not in "iu":
            raise ValueError(
                "Fold IDs must be a one-dimensional integer array with at least two rows."
            )
        k_folds = _fold_integer(
            k_folds, "k_folds", 2, min(ids.size, np.iinfo(np.intc).max - 1)
        )
        ids = cp.ascontiguousarray(ids, dtype=cp.int64)
        hdl = ctypes.c_void_p()
        _check_call(
            _LIB.XGBCvFoldAssignmentCreate(
                cuda_array_interface(ids), k_folds, ctypes.byref(hdl)
            )
        )
        self.handle = hdl
        self._k_folds = k_folds

    @property
    def k_folds(self) -> int:
        """Number of nonempty validation folds."""
        return self._k_folds

    def __del__(self) -> None:
        if hasattr(self, "handle"):
            _check_call(_LIB.XGBCvFoldAssignmentFree(self.handle))
            del self.handle


def make_kfold(
    n_rows: int, k_folds: int, *, shuffle: bool = False, seed: int = 0
) -> cp.ndarray:
    """Return GPU IDs in row-modulo-K order, optionally shuffled globally.

    Pass the result to ``FoldAssignment``. A seed reproduces the permutation for
    fixed inputs and the same CuPy implementation; features retain their row order.
    """
    cp = import_cupy()

    n_rows = _fold_integer(n_rows, "n_rows", 2, np.iinfo(np.uint32).max)
    k_folds = _fold_integer(
        k_folds, "k_folds", 2, min(n_rows, np.iinfo(np.intc).max - 1)
    )
    ids = cp.arange(n_rows, dtype=cp.uint32) % k_folds
    if shuffle:
        cp.random.RandomState(seed).shuffle(ids)
    return ids


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
        out: FoldPredictions,
        *,
        assignment: FoldAssignment,
    ) -> FoldPredictions:
        """Initialize fresh prediction buffers and retain their fold assignment.

        The assignment must match the matrix's row order and the models' fold count.
        """
        if not isinstance(assignment, FoldAssignment):
            raise TypeError("`assignment` must be a FoldAssignment.")

        _check_call(
            _LIB.XGBCvFoldModelsInitPrediction(
                self.handle,
                data.handle,
                assignment.handle,
                out.handle,
            )
        )
        return out

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def get_gradient(
        self,
        data: ExtMemQuantileDMatrix,
        iteration: int,
        predt: FoldPredictions,
        out: FoldGpairs,
    ) -> FoldGpairs:
        """Calculate the gradient."""

        _check_call(
            _LIB.XGBCvFoldModelsGetGradient(
                self.handle,
                data.handle,
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
        gpairs: FoldGpairs,
        predt: FoldPredictions,
    ) -> None:
        """Grow and commit one fused CV tree for each fold."""

        _check_call(
            _LIB.XGBCvFoldTreeMethodUpdate(
                self.handle,
                cv_folds.handle,
                data.handle,
                gpairs.handle,
                predt.handle,
            )
        )


class FoldPredictions:
    """Prediction buffers sharing ownership of their fold assignment.

    Initialize once with :py:meth:`FoldModels.init_prediction` before use.
    """

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
        cp = import_cupy()

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

        cp = import_cupy()

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


@dataclass(frozen=True)
class CvEvalResult:
    """Metric values of one boosting round.

    ``train`` and ``valid`` are ``(n_metrics, k_folds)`` in the order of ``names``, and
    ``train`` is ``None`` under ``eval_train=False``. The CV score of a metric is
    ``valid[m].mean()``. A refit model is not scored.

    """

    names: tuple[str, ...]
    train: np.ndarray | None
    valid: np.ndarray


class FoldEvaluator:
    """Metric evaluation for fused cross-validation.

    No metric parameter is supported yet; ``metrics=None`` uses the objective's default.

    """

    def __init__(
        self,
        cv_folds: FoldModels,
        *,
        metrics: str | Sequence[str] | None = None,
        eval_train: bool = True,
    ) -> None:
        config: dict[str, Any] = {}
        # Absent, not empty, so that C++ tells "use the default" from "evaluate nothing".
        if metrics is not None:
            config["eval_metric"] = (
                [metrics] if isinstance(metrics, str) else list(metrics)
            )
        config["eval_train"] = eval_train

        hdl = ctypes.c_void_p()
        _check_call(
            _LIB.XGBCvFoldEvaluatorCreate(
                cv_folds.handle,
                make_jcargs(**config),
                ctypes.byref(hdl),
            )
        )
        self.handle = hdl

    def __del__(self) -> None:
        if hasattr(self, "handle"):
            hdl = self.handle
            del self.handle
            _check_call(_LIB.XGBCvFoldEvaluatorFree(hdl))

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def evaluate(
        self,
        cv_folds: FoldModels,
        data: ExtMemQuantileDMatrix,
        predt: FoldPredictions,
        iteration: int,
    ) -> CvEvalResult:
        """Evaluate every metric on the round whose trees were last committed."""
        c_meta = ctypes.c_char_p()
        c_values = ctypes.POINTER(ctypes.c_double)()
        _check_call(
            _LIB.XGBCvFoldEvaluatorEval(
                self.handle,
                cv_folds.handle,
                data.handle,
                predt.handle,
                ctypes.c_int(iteration),
                ctypes.byref(c_meta),
                ctypes.byref(c_values),
            )
        )

        # The shape comes back with the names.
        meta = json.loads(py_str(c_meta.value))
        n_sections = meta["shape"][0]
        values = np.ctypeslib.as_array(c_values, shape=tuple(meta["shape"])).copy()

        return CvEvalResult(
            names=tuple(meta["names"]),
            train=values[0] if n_sections == 2 else None,
            valid=values[-1],
        )


def _n_rows(data: ArrayLike) -> int:
    shape = getattr(data, "shape", None)
    return len(data) if shape is None else int(shape[0])


def _to_device(data: ArrayLike) -> Any:
    cp = import_cupy()

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

    @override
    def next(self, input_data: Callable) -> bool:
        if self.it == self.n_batches:
            return False
        begin, end = self.batch_ptr[self.it], self.batch_ptr[self.it + 1]
        input_data(
            data=_to_device(self.X[begin:end]), label=_to_device(self.y[begin:end])
        )
        self.it += 1
        return True

    @override
    def reset(self) -> None:
        self.it = 0


def cross_val_predict(
    X: ArrayLike,
    y: ArrayLike,
    *,
    cv: int | FoldAssignment = 5,
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
        Number of folds, at least 2, or an existing FoldAssignment in the row order
        of X. Integer values use global round-robin membership (row index modulo cv).
    params :
        Booster parameters accepted by the CV tree method, like `learning_rate`.
    num_boost_round :
        Number of boosting rounds, shared by every fold model.

    Returns
    -------
    The estimate of every row, on the device, in the row order of `X`. The shape is
    ``(n_samples,)`` for a single target and ``(n_samples, n_targets)`` otherwise.

    """
    assignment = (
        cv
        if isinstance(cv, FoldAssignment)
        else FoldAssignment(make_kfold(_n_rows(X), cv), k_folds=cv)
    )
    num_boost_round = int(num_boost_round)
    if num_boost_round < 0:
        raise ValueError(
            f"`num_boost_round` must not be negative, got {num_boost_round}."
        )
    params = params if params is not None else {}
    max_bin = params.get("max_bin", None)

    Xy = ExtMemQuantileDMatrix(CvDataIter(X, y), max_bin=max_bin)
    models = FoldModels(data=Xy, k_folds=assignment.k_folds)
    predts = FoldPredictions()
    models.init_prediction(Xy, out=predts, assignment=assignment)
    gpairs = FoldGpairs()
    tree_method = FoldTreeMethod(models, Xy, params=params or {})

    for iteration in range(num_boost_round):
        models.get_gradient(Xy, iteration, predts, out=gpairs)
        tree_method.update(models, Xy, gpairs, predts)

    # The cache holds raw margins, which the squared error objective leaves in the
    # prediction domain already.
    oof = predts.get_valid()
    return oof.reshape(-1) if oof.shape[1] == 1 else oof
