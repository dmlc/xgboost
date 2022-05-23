"""XGBoost: eXtreme Gradient Boosting library.

Contributors: https://github.com/dmlc/xgboost/blob/master/CONTRIBUTORS.md
"""

from .core import (
    DMatrix,
    DeviceQuantileDMatrix,
    Booster,
    DataIter,
    build_info,
    _py_version,
)
from .training import train, cv
from . import rabit  # noqa
from . import tracker  # noqa
from .tracker import RabitTracker  # noqa
from . import dask

try:
    from .sklearn import XGBModel, XGBClassifier, XGBRegressor, XGBRanker
    from .sklearn import XGBRFClassifier, XGBRFRegressor
    from .plotting import plot_importance, plot_tree, to_graphviz
    from .config import set_config, get_config, config_context
except ImportError:
    pass


__version__ = _py_version()


__all__ = [
    # core
    "DMatrix",
    "DeviceQuantileDMatrix",
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
    "XGBModel",
    "XGBClassifier",
    "XGBRegressor",
    "XGBRanker",
    "XGBRFClassifier",
    "XGBRFRegressor",
    # dask
    "dask",
]
