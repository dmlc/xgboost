"""XGBoost: eXtreme Gradient Boosting library.

Contributors: https://github.com/dmlc/xgboost/blob/master/CONTRIBUTORS.md
"""

from importlib import import_module as _import_module
from typing import TYPE_CHECKING, Any

from . import (
    collective,
    interpret,
    tracker,  # noqa
)
from ._c_api import _py_version
from .config import config_context, get_config, set_config
from .core import (
    Booster,
    DataIter,
    DMatrix,
    ExtMemQuantileDMatrix,
    QuantileDMatrix,
    build_info,
)
from .plotting import plot_importance, plot_tree, to_graphviz
from .tracker import RabitTracker  # noqa
from .training import cv, train

if TYPE_CHECKING:
    from . import sklearn
    from .sklearn import (
        XGBClassifier,
        XGBModel,
        XGBRanker,
        XGBRegressor,
        XGBRFClassifier,
        XGBRFRegressor,
    )


_SKLEARN_ESTIMATORS = (
    "XGBModel",
    "XGBClassifier",
    "XGBRegressor",
    "XGBRanker",
    "XGBRFClassifier",
    "XGBRFRegressor",
)


def __getattr__(name: str) -> Any:
    if name == "sklearn":
        return _import_module(".sklearn", __name__)
    if name in _SKLEARN_ESTIMATORS:
        estimator = getattr(_import_module(".sklearn", __name__), name)
        globals()[name] = estimator
        return estimator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), "sklearn", *_SKLEARN_ESTIMATORS})


__version__ = _py_version()


__all__ = [
    # core
    "DMatrix",
    "QuantileDMatrix",
    "ExtMemQuantileDMatrix",
    "Booster",
    "DataIter",
    "train",
    "cv",
    # utilities
    "RabitTracker",
    "build_info",
    "plot_importance",
    "plot_tree",
    "to_graphviz",
    "set_config",
    "get_config",
    "config_context",
    # sklearn
    *_SKLEARN_ESTIMATORS,
    # collective
    "collective",
    # interpretability
    "interpret",
]
