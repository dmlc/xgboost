"""XGBoost: eXtreme Gradient Boosting library.

Contributors: https://github.com/dmlc/xgboost/blob/master/CONTRIBUTORS.md
"""

from . import tracker  # noqa
from . import collective, dask, rabit
from ._typing import CatDType
from .core import (
    Booster,
    DataIter,
    DeviceQuantileDMatrix,
    DMatrix,
    QuantileDMatrix,
    _py_version,
    build_info,
)
from .tracker import RabitTracker
from .training import cv, train

try:
    from .config import config_context, get_config, set_config
    from .plotting import plot_importance, plot_tree, to_graphviz
    from .sklearn import (
        XGBClassifier,
        XGBModel,
        XGBRanker,
        XGBRegressor,
        XGBRFClassifier,
        XGBRFRegressor,
    )
except ImportError:
    pass


__version__ = _py_version()


__all__ = [
    # core
    "DMatrix",
    "DeviceQuantileDMatrix",
    "QuantileDMatrix",
    "Booster",
    "DataIter",
    "train",
    "cv",
    # utilities
    "CatDType",
    "RabitTracker",
    "build_info",
    "plot_importance",
    "plot_tree",
    "to_graphviz",
    "set_config",
    "get_config",
    "config_context",
    # sklearn
    "XGBModel",
    "XGBClassifier",
    "XGBRegressor",
    "XGBRanker",
    "XGBRFClassifier",
    "XGBRFRegressor",
    # dask
    "dask",
    # collective
    "collective",
    "rabit",
]
