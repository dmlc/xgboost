# pylint: disable=import-error
"""Internal module for dispatching array methods."""
from typing import Sequence, Tuple, Union

import numpy as np

from ._typing import ArrayLike
from .data import _is_cupy_array


def zeros(shape: Tuple[int, ...], is_cupy: bool) -> ArrayLike:
    """Array filled with zeros."""
    if is_cupy:
        import cupy as cp

        return cp.zeros(shape)
    return np.zeros(shape)


def argmax(array: ArrayLike, axis: int) -> ArrayLike:
    """Max index."""
    if _is_cupy_array(array):
        import cupy as cp

        return cp.argmax(array, axis=axis)
    return np.argmax(array, axis=axis)


def repeat(value: Union[int, float], repeats: int, is_cupy: bool) -> ArrayLike:
    """Dispatch numpy/cupy repeat."""
    if is_cupy:
        import cupy as cp

        return cp.repeat(value, repeats=repeats)
    return np.repeat(value, repeats=repeats)


def vstack(tup: Sequence) -> ArrayLike:
    """Dsipatch numpy/cupy vstack."""
    if _is_cupy_array(tup[0]):
        import cupy as cp

        return cp.vstack(tup)

    return np.vstack(tup)


def softmax(values: ArrayLike, axis: int) -> ArrayLike:
    """Softmax function using scipy or cupy."""
    if _is_cupy_array(values):
        import cupy as cp

        x_max = cp.amax(values, axis=axis, keepdims=True)
        exp_x_shifted = cp.exp(values - x_max)
        return exp_x_shifted / cp.sum(exp_x_shifted, axis=axis, keepdims=True)

    from scipy.special import softmax as sss

    return sss(values, axis=axis)
