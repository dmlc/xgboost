# coding: utf-8
"""XGBoost: eXtreme Gradient Boosting library.

Contributors: https://github.com/dmlc/xgboost/blob/master/CONTRIBUTORS.md
"""

import os

from .core import DMatrix, DeviceQuantileDMatrix, Booster, DataIter
from .training import train, cv
from . import rabit  # noqa
from . import tracker  # noqa
from .tracker import RabitTracker  # noqa
from . import dask

try:
    from .sklearn import XGBModel, XGBClassifier, XGBRegressor, XGBRanker
    from .sklearn import XGBRFClassifier, XGBRFRegressor
    from .sklearn import XGBMultiOutputRegressor
    from .plotting import plot_importance, plot_tree, to_graphviz
    from .config import set_config, get_config, config_context
except ImportError:
    pass

VERSION_FILE = os.path.join(os.path.dirname(__file__), "VERSION")
with open(VERSION_FILE, encoding="ascii") as f:
    __version__ = f.read().strip()

__all__ = [
    "DMatrix",
    "DeviceQuantileDMatrix",
    "Booster",
    "DataIter",
    "train",
    "cv",
    "RabitTracker",
    "XGBModel",
    "XGBClassifier",
    "XGBRegressor",
    "XGBMultiOutputRegressor",
    "XGBRanker",
    "XGBRFClassifier",
    "XGBRFRegressor",
    "plot_importance",
    "plot_tree",
    "to_graphviz",
    "dask",
    "set_config",
    "get_config",
    "config_context",
]
