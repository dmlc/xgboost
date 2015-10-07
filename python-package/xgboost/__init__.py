# coding: utf-8
"""XGBoost: eXtreme Gradient Boosting library.

Contributors: https://github.com/dmlc/xgboost/blob/master/CONTRIBUTORS.md
"""

from __future__ import absolute_import

import os

from .core import DMatrix, Booster
from .training import train, cv
from .sklearn import XGBModel, XGBClassifier, XGBRegressor
from .plotting import plot_importance, plot_tree, to_graphviz

VERSION_FILE = os.path.join(os.path.dirname(__file__), 'VERSION')
__version__ = open(VERSION_FILE).read().strip()

__all__ = ['DMatrix', 'Booster',
           'train', 'cv',
           'XGBModel', 'XGBClassifier', 'XGBRegressor',
           'plot_importance', 'plot_tree', 'to_graphviz']
