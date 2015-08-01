# coding: utf-8
"""XGBoost: eXtreme Gradient Boosting library.

Contributors: https://github.com/dmlc/xgboost/blob/master/CONTRIBUTORS.md
"""

from __future__ import absolute_import
from .core import DMatrix, Booster
from .training import train, cv
from .sklearn import XGBModel, XGBClassifier, XGBRegressor

__version__ = '0.4'

__all__ = ['DMatrix', 'Booster',
           'train', 'cv',
           'XGBModel', 'XGBClassifier', 'XGBRegressor']
