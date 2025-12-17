"""Experimental support for a new objective interface with target dimension
reduction.

.. warning::

  Do not use this module unless you want to participate in development.

.. versionadded:: 3.2.0

"""

import ctypes
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Tuple

import numpy as np

from ._data_utils import (
    _ensure_np_dtype,
    array_interface,
    cuda_array_interface,
    is_flatten,
)
from ._typing import ArrayLike, NumpyOrCupy
from .compat import _is_cupy_alike

if TYPE_CHECKING:
    from .core import DMatrix


__all__ = ["Objective", "TreeObjective"]

# Objective was simply a callable before 3.2
PlainObj = Callable[[np.ndarray, "DMatrix"], Tuple[np.ndarray, np.ndarray]]


# Since 3.2, we are working with batched data in the Python interface. An objective can
# be invoked multiple times for each iteration.
# The `iteration` parameter of the objective prototype is provided as a hint to users
# which iteration the current call belongs to.


class Objective(ABC):
    """Base class for custom objective functions.

    .. warning::

        Do not use this class unless you want to participate in development.

    """

    @abstractmethod
    def __call__(
        self, iteration: int, y_pred: ArrayLike, dtrain: "DMatrix"
    ) -> Tuple[ArrayLike, ArrayLike]: ...


class TreeObjective(Objective):
    """Base class for tree-specific custom objective functions.

    .. warning::

        Do not use this class unless you want to participate in development.

    """

    # pylint: disable=unused-argument
    def split_grad(
        self, iteration: int, grad: ArrayLike, hess: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Provide a different gradient type for finding tree structures."""
        return grad, hess


def _reshape_grad(array: NumpyOrCupy, n_samples: int) -> NumpyOrCupy:
    if array.shape[0] != n_samples and is_flatten(array):
        warnings.warn(
            "Since 2.1.0, the shape of the gradient and hessian is required to"
            " be (n_samples, n_targets) or (n_samples, n_classes).",
            FutureWarning,
        )
        if array.size % n_samples != 0:
            raise ValueError(
                f"Invalid gradient shape: {array.shape}. The number of samples of "
                f"the current batch: {n_samples}."
            )
        array = array.reshape(n_samples, array.size // n_samples)
    return array


def _grad_arrinf(array: NumpyOrCupy) -> bytes:
    """Get array interface for gradient matrices."""
    # Can we check for __array_interface__ instead of a specific type instead?
    msg = (
        "Expecting `np.ndarray` or `cupy.ndarray` for gradient and hessian."
        f" Got: {type(array)}. For CUDA inputs, arrays with "
        "`__cuda_array_interface__` are supported (like torch tensor)."
    )
    if not isinstance(array, np.ndarray) and not _is_cupy_alike(array):
        raise TypeError(msg)

    if isinstance(array, np.ndarray):
        array, _ = _ensure_np_dtype(array, array.dtype)
        interface = array_interface(array)
    elif _is_cupy_alike(array):
        interface = cuda_array_interface(array)
    else:
        raise TypeError(msg)

    return interface


class _GradientContainer:
    """Internal class for storing gradient values produced by custom objectives."""

    def __init__(self, hdl: ctypes.c_void_p) -> None:
        self.handle = hdl

    def push_value_grad(self, grad: ArrayLike, hess: ArrayLike) -> None:
        """Push a batch of tree leaf value gradient into the container."""
        from .core import _LIB, _check_call

        i_grad = _grad_arrinf(grad)
        i_hess = _grad_arrinf(hess)
        _check_call(_LIB.XGGradientContainerPushValueGrad(self.handle, i_grad, i_hess))

    def push_grad(self, grad: NumpyOrCupy, hess: NumpyOrCupy) -> None:
        """Push a batch of (tree split) gradient into the container."""
        from .core import _LIB, _check_call

        i_grad = _grad_arrinf(grad)
        i_hess = _grad_arrinf(hess)
        _check_call(_LIB.XGGradientContainerPushGrad(self.handle, i_grad, i_hess))

    def __del__(self) -> None:
        from .core import _LIB, _check_call

        if hasattr(self, "handle"):
            _check_call(_LIB.XGGradientContainerFree(self.handle))
            del self.handle
