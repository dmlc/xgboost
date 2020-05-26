# coding: utf-8
"""XGBoost: eXtreme Gradient Boosting library.

Contributors: https://github.com/dmlc/xgboost/blob/master/CONTRIBUTORS.md
"""

import os
import sys
import warnings

from .core import DMatrix, DeviceQuantileDMatrix, Booster
from .training import train, cv
from . import rabit  # noqa
from . import tracker  # noqa
from .tracker import RabitTracker  # noqa
from . import dask

try:
    from .sklearn import XGBModel, XGBClassifier, XGBRegressor, XGBRanker
    from .sklearn import XGBRFClassifier, XGBRFRegressor
    from .plotting import plot_importance, plot_tree, to_graphviz
except ImportError:
    pass

if sys.version_info[:2] == (3, 5):
    warnings.warn(
        'Python 3.5 support is deprecated; XGBoost will require Python 3.6+ in the near future. ' +
        'Consider upgrading to Python 3.6+.',
        FutureWarning)

VERSION_FILE = os.path.join(os.path.dirname(__file__), 'VERSION')
with open(VERSION_FILE) as f:
    __version__ = f.read().strip()

__all__ = ['DMatrix', 'DeviceQuantileDMatrix', 'Booster',
           'train', 'cv',
           'RabitTracker',
           'XGBModel', 'XGBClassifier', 'XGBRegressor', 'XGBRanker',
           'XGBRFClassifier', 'XGBRFRegressor',
           'plot_importance', 'plot_tree', 'to_graphviz', 'dask']
