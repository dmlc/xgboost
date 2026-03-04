# pylint: disable=missing-class-docstring
"""Experimental support for a new objective interface with target dimension
reduction.


This module exposes built-in objectives like ``reg:squarederror`` into the Python
interface, and enables users to specify parameters for some objectives like
``reg:quantileerror``. In addition, one can define a custom ``split_grad`` for training
vector-leaf models.

.. warning::

  Do not use this module unless you want to participate in development.

.. versionadded:: 3.2.0

"""

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Tuple

import numpy as np

from ._data_utils import (
    _ensure_np_dtype,
    _is_flatten,
    array_interface,
    cuda_array_interface,
)
from ._typing import ArrayLike, NumpyOrCupy
from .compat import _is_cupy_alike

if TYPE_CHECKING:
    from .core import DMatrix


class Objective(ABC):
    """Base class for custom objective functions.

    .. warning::

        Do not use this class unless you want to participate in development.

    .. versionadded:: 3.2.0

    """

    @abstractmethod
    def __call__(
        self, iteration: int, y_pred: ArrayLike, dtrain: "DMatrix"
    ) -> Tuple[ArrayLike, ArrayLike]: ...

    # pylint: disable=unused-argument
    def split_grad(
        self, iteration: int, grad: ArrayLike, hess: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike] | None:
        """Provide a different gradient type for finding tree structures."""
        return None


class _BuiltInObjective:
    """Base class for Python wrappers of built-in C++ objective functions."""

    _name: str = ""
    _KNOWN_PARAMS: Dict[str, str] = {}

    def __init__(self, **kwargs: Any) -> None:
        self._params: Dict[str, Any] = {}
        for py_name in self._KNOWN_PARAMS:
            self._params[py_name] = kwargs.pop(py_name, None)
        if kwargs:
            raise TypeError(f"Unknown parameters for {self._name}: {list(kwargs)}")

    @property
    def name(self) -> str:
        """The objective name string."""
        return self._name

    # pylint: disable=missing-function-docstring
    def flat_params(self) -> Dict[str, str]:
        result: Dict[str, str] = {"objective": self._name}
        for py_name, cpp_name in self._KNOWN_PARAMS.items():
            value = self._params[py_name]
            if value is not None:
                result[cpp_name] = _stringify(value)
        return result


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif _is_cupy_alike(value) and hasattr(value, "get"):
        value = value.get().tolist()
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(str(v) for v in value) + "]"
    return str(value)


# Regression objectives


class RegSquaredError(_BuiltInObjective):
    _name = "reg:squarederror"
    _KNOWN_PARAMS = {"scale_pos_weight": "scale_pos_weight"}


class RegSquaredLogError(_BuiltInObjective):
    _name = "reg:squaredlogerror"


class RegAbsoluteError(_BuiltInObjective):
    _name = "reg:absoluteerror"


class RegPseudoHuberError(_BuiltInObjective):
    _name = "reg:pseudohubererror"
    _KNOWN_PARAMS = {"delta": "huber_slope"}


class RegQuantileError(_BuiltInObjective):
    _name = "reg:quantileerror"
    _KNOWN_PARAMS = {"alpha": "quantile_alpha"}


class RegExpectileError(_BuiltInObjective):
    _name = "reg:expectileerror"
    _KNOWN_PARAMS = {"alpha": "expectile_alpha"}


class RegTweedie(_BuiltInObjective):
    _name = "reg:tweedie"
    _KNOWN_PARAMS = {"variance_power": "tweedie_variance_power"}


class CountPoisson(_BuiltInObjective):
    _name = "count:poisson"
    _KNOWN_PARAMS = {"max_delta_step": "max_delta_step"}


# Logistic / classification objectives


class RegLogistic(_BuiltInObjective):
    _name = "reg:logistic"
    _KNOWN_PARAMS = {"scale_pos_weight": "scale_pos_weight"}


class BinaryLogistic(_BuiltInObjective):
    _name = "binary:logistic"
    _KNOWN_PARAMS = {"scale_pos_weight": "scale_pos_weight"}


class RegGamma(_BuiltInObjective):
    _name = "reg:gamma"
    _KNOWN_PARAMS = {"scale_pos_weight": "scale_pos_weight"}


class BinaryLogitRaw(_BuiltInObjective):
    _name = "binary:logitraw"
    _KNOWN_PARAMS = {"scale_pos_weight": "scale_pos_weight"}


class BinaryHinge(_BuiltInObjective):
    _name = "binary:hinge"


# Multiclass objectives


class MultiSoftmax(_BuiltInObjective):
    _name = "multi:softmax"
    _KNOWN_PARAMS = {"num_class": "num_class"}


class MultiSoftprob(_BuiltInObjective):
    _name = "multi:softprob"
    _KNOWN_PARAMS = {"num_class": "num_class"}


# Survival objectives


class SurvivalAFT(_BuiltInObjective):
    _name = "survival:aft"
    _KNOWN_PARAMS = {
        "distribution": "aft_loss_distribution",
        "distribution_scale": "aft_loss_distribution_scale",
    }


class SurvivalCox(_BuiltInObjective):
    _name = "survival:cox"


# Ranking objectives


class RankNDCG(_BuiltInObjective):
    _name = "rank:ndcg"
    _KNOWN_PARAMS = {
        "pair_method": "lambdarank_pair_method",
        "num_pair_per_sample": "lambdarank_num_pair_per_sample",
        "unbiased": "lambdarank_unbiased",
        "exp_gain": "ndcg_exp_gain",
    }


class RankPairwise(_BuiltInObjective):
    _name = "rank:pairwise"
    _KNOWN_PARAMS = {
        "pair_method": "lambdarank_pair_method",
        "num_pair_per_sample": "lambdarank_num_pair_per_sample",
    }


class RankMAP(_BuiltInObjective):
    _name = "rank:map"
    _KNOWN_PARAMS = {
        "pair_method": "lambdarank_pair_method",
        "num_pair_per_sample": "lambdarank_num_pair_per_sample",
    }


def _grad_arrinf(array: NumpyOrCupy, n_samples: int) -> bytes:
    msg = (
        "Expecting `np.ndarray` or `cupy.ndarray` for gradient and hessian."
        f" Got: {type(array)}"
    )
    if not isinstance(array, np.ndarray) and not _is_cupy_alike(array):
        raise TypeError(msg)

    if array.shape[0] != n_samples and _is_flatten(array):
        warnings.warn(
            "Since 2.1.0, the shape of the gradient and hessian is required to"
            " be (n_samples, n_targets) or (n_samples, n_classes).",
            FutureWarning,
        )
        array = array.reshape(n_samples, array.size // n_samples)

    if isinstance(array, np.ndarray):
        array, _ = _ensure_np_dtype(array, array.dtype)
        interface = array_interface(array)
    elif _is_cupy_alike(array):
        interface = cuda_array_interface(array)
    else:
        raise TypeError(msg)

    return interface
