"""Experimental support for a new objective interface with target dimension
reduction.


This module exposes built-in objectives like ``reg:squarederror`` into the Python
interface, and enables users to specify parameters for some objectives like
``reg::quantileerror``. In addition, one can define a custom ``split_grad`` for training
vector-leaf models.

.. warning::

  Do not use this module unless you want to participate in development.

.. versionadded:: 3.2.0

"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

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


class _BuiltInObjective(Objective):
    """Base class for Python wrappers of built-in C++ objective functions."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The objective name string."""

    # pylint: disable=missing-function-docstring
    @abstractmethod
    def flat_params(self) -> Dict[str, str]: ...

    def __call__(
        self, iteration: int, y_pred: ArrayLike, dtrain: "DMatrix"
    ) -> Tuple[ArrayLike, ArrayLike]:
        raise RuntimeError(
            "This method should not be called directly for the built-in objective."
        )


@dataclass(frozen=True)
class _ParamSpec:
    py_name: str
    cpp_name: str
    typ: type


@dataclass(frozen=True)
class _ObjSpec:
    obj_name: str
    params: List[_ParamSpec] = field(default_factory=list)


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


def _make_builtin_objective(spec: _ObjSpec) -> type:
    obj_name = spec.obj_name
    params = spec.params

    doc = f"""Interface for the ``{obj_name}`` objective.

.. versionadded:: 3.3.0

.. warning:: This interface is experimental and may subject to change without notice.

"""
    if params:
        doc += "\nParameters\n----------\n\n"
        for p in params:
            doc += f"{p.py_name} : {p.typ.__name__}\n"

    class _Cls(_BuiltInObjective):
        def __init__(self, **kwargs: Any) -> None:
            self._params: Dict[str, Any] = {}
            for p in params:
                self._params[p.py_name] = kwargs.pop(p.py_name, None)
            if kwargs:
                raise TypeError(f"Unknown parameters for {obj_name}: {list(kwargs)}")

        @property
        def name(self) -> str:
            return obj_name

        def flat_params(self) -> Dict[str, str]:
            result: Dict[str, str] = {"objective": obj_name}
            for p in params:
                value = self._params[p.py_name]
                if value is not None:
                    result[p.cpp_name] = _stringify(value)
            return result

    _Cls.__doc__ = doc

    return _Cls


def _named(name: str, cls: type) -> type:
    cls.__name__ = name
    cls.__qualname__ = name
    return cls


# Regression objectives
RegSquaredError = _named(
    "RegSquaredError",
    _make_builtin_objective(
        _ObjSpec(
            obj_name="reg:squarederror",
            params=[
                _ParamSpec(
                    py_name="scale_pos_weight", cpp_name="scale_pos_weight", typ=float
                )
            ],
        )
    ),
)

RegSquaredLogError = _named(
    "RegSquaredLogError",
    _make_builtin_objective(_ObjSpec(obj_name="reg:squaredlogerror")),
)

RegAbsoluteError = _named(
    "RegAbsoluteError",
    _make_builtin_objective(_ObjSpec(obj_name="reg:absoluteerror")),
)

RegPseudoHuberError = _named(
    "RegPseudoHuberError",
    _make_builtin_objective(
        _ObjSpec(
            obj_name="reg:pseudohubererror",
            params=[_ParamSpec(py_name="delta", cpp_name="huber_slope", typ=float)],
        )
    ),
)

RegQuantileError = _named(
    "RegQuantileError",
    _make_builtin_objective(
        _ObjSpec(
            obj_name="reg:quantileerror",
            params=[_ParamSpec(py_name="alpha", cpp_name="quantile_alpha", typ=list)],
        )
    ),
)

RegExpectileError = _named(
    "RegExpectileError",
    _make_builtin_objective(
        _ObjSpec(
            obj_name="reg:expectileerror",
            params=[_ParamSpec(py_name="alpha", cpp_name="expectile_alpha", typ=list)],
        )
    ),
)

RegTweedie = _named(
    "RegTweedie",
    _make_builtin_objective(
        _ObjSpec(
            obj_name="reg:tweedie",
            params=[
                _ParamSpec(
                    py_name="variance_power",
                    cpp_name="tweedie_variance_power",
                    typ=float,
                )
            ],
        )
    ),
)

CountPoisson = _named(
    "CountPoisson",
    _make_builtin_objective(
        _ObjSpec(
            obj_name="count:poisson",
            params=[
                _ParamSpec(
                    py_name="max_delta_step", cpp_name="max_delta_step", typ=float
                )
            ],
        )
    ),
)

# Logistic / classification objectives
RegLogistic = _named(
    "RegLogistic",
    _make_builtin_objective(
        _ObjSpec(
            obj_name="reg:logistic",
            params=[
                _ParamSpec(
                    py_name="scale_pos_weight", cpp_name="scale_pos_weight", typ=float
                )
            ],
        )
    ),
)

BinaryLogistic = _named(
    "BinaryLogistic",
    _make_builtin_objective(
        _ObjSpec(
            obj_name="binary:logistic",
            params=[
                _ParamSpec(
                    py_name="scale_pos_weight", cpp_name="scale_pos_weight", typ=float
                )
            ],
        )
    ),
)

RegGamma = _named(
    "RegGamma",
    _make_builtin_objective(
        _ObjSpec(
            obj_name="reg:gamma",
            params=[
                _ParamSpec(
                    py_name="scale_pos_weight", cpp_name="scale_pos_weight", typ=float
                )
            ],
        )
    ),
)

BinaryLogitRaw = _named(
    "BinaryLogitRaw",
    _make_builtin_objective(
        _ObjSpec(
            obj_name="binary:logitraw",
            params=[
                _ParamSpec(
                    py_name="scale_pos_weight", cpp_name="scale_pos_weight", typ=float
                )
            ],
        )
    ),
)

BinaryHinge = _named(
    "BinaryHinge",
    _make_builtin_objective(_ObjSpec(obj_name="binary:hinge")),
)

# Multiclass objectives
MultiSoftmax = _named(
    "MultiSoftmax",
    _make_builtin_objective(
        _ObjSpec(
            obj_name="multi:softmax",
            params=[_ParamSpec(py_name="num_class", cpp_name="num_class", typ=int)],
        )
    ),
)

MultiSoftprob = _named(
    "MultiSoftprob",
    _make_builtin_objective(
        _ObjSpec(
            obj_name="multi:softprob",
            params=[_ParamSpec(py_name="num_class", cpp_name="num_class", typ=int)],
        )
    ),
)

# Survival objectives
SurvivalAFT = _named(
    "SurvivalAFT",
    _make_builtin_objective(
        _ObjSpec(
            obj_name="survival:aft",
            params=[
                _ParamSpec(
                    py_name="distribution", cpp_name="aft_loss_distribution", typ=str
                ),
                _ParamSpec(
                    py_name="distribution_scale",
                    cpp_name="aft_loss_distribution_scale",
                    typ=float,
                ),
            ],
        )
    ),
)

SurvivalCox = _named(
    "SurvivalCox",
    _make_builtin_objective(_ObjSpec(obj_name="survival:cox")),
)

# Ranking objectives
RankNDCG = _named(
    "RankNDCG",
    _make_builtin_objective(
        _ObjSpec(
            obj_name="rank:ndcg",
            params=[
                _ParamSpec(
                    py_name="pair_method", cpp_name="lambdarank_pair_method", typ=str
                ),
                _ParamSpec(
                    py_name="num_pair_per_sample",
                    cpp_name="lambdarank_num_pair_per_sample",
                    typ=int,
                ),
                _ParamSpec(
                    py_name="unbiased", cpp_name="lambdarank_unbiased", typ=bool
                ),
                _ParamSpec(py_name="exp_gain", cpp_name="ndcg_exp_gain", typ=bool),
            ],
        )
    ),
)

RankPairwise = _named(
    "RankPairwise",
    _make_builtin_objective(
        _ObjSpec(
            obj_name="rank:pairwise",
            params=[
                _ParamSpec(
                    py_name="pair_method", cpp_name="lambdarank_pair_method", typ=str
                ),
                _ParamSpec(
                    py_name="num_pair_per_sample",
                    cpp_name="lambdarank_num_pair_per_sample",
                    typ=int,
                ),
            ],
        )
    ),
)

RankMAP = _named(
    "RankMAP",
    _make_builtin_objective(
        _ObjSpec(
            obj_name="rank:map",
            params=[
                _ParamSpec(
                    py_name="pair_method", cpp_name="lambdarank_pair_method", typ=str
                ),
                _ParamSpec(
                    py_name="num_pair_per_sample",
                    cpp_name="lambdarank_num_pair_per_sample",
                    typ=int,
                ),
            ],
        )
    ),
)


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
