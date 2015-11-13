# coding: utf-8
# pylint: disable=unused-import, invalid-name
"""For compatibility"""

from __future__ import absolute_import

import sys


PY3 = (sys.version_info[0] == 3)

if PY3:
    # pylint: disable=invalid-name, redefined-builtin
    STRING_TYPES = str,
else:
    # pylint: disable=invalid-name
    STRING_TYPES = basestring,

# pandas
try:
    from pandas import DataFrame
    PANDAS_INSTALLED = True
except ImportError:

    class DataFrame(object):
        """ dummy for pandas.DataFrame """
        pass

    PANDAS_INSTALLED = False

# sklearn
try:
    from sklearn.base import BaseEstimator
    from sklearn.base import RegressorMixin, ClassifierMixin
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_INSTALLED = True

    XGBModelBase = BaseEstimator
    XGBRegressorBase = RegressorMixin
    XGBClassifierBase = ClassifierMixin
except ImportError:
    SKLEARN_INSTALLED = False

    # used for compatiblity without sklearn
    XGBModelBase = object
    XGBClassifierBase = object
    XGBRegressorBase = object
