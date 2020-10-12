# coding: utf-8
# pylint: disable= invalid-name,  unused-import
"""For compatibility and optional dependencies."""
import abc
import os
import sys
from pathlib import PurePath

import numpy as np

assert (sys.version_info[0] == 3), 'Python 2 is no longer supported.'

# pylint: disable=invalid-name, redefined-builtin
STRING_TYPES = (str,)


def py_str(x):
    """convert c string back to python string"""
    return x.decode('utf-8')


def lazy_isinstance(instance, module, name):
    '''Use string representation to identify a type.'''
    module = type(instance).__module__ == module
    name = type(instance).__name__ == name
    return module and name


# pandas
try:
    from pandas import DataFrame, Series
    from pandas import MultiIndex, Int64Index
    from pandas import concat as pandas_concat

    PANDAS_INSTALLED = True
except ImportError:

    MultiIndex = object
    Int64Index = object
    DataFrame = object
    Series = object
    pandas_concat = None
    PANDAS_INSTALLED = False

# cudf
try:
    from cudf import concat as CUDF_concat
except ImportError:
    CUDF_concat = None

# sklearn
try:
    from sklearn.base import BaseEstimator
    from sklearn.base import RegressorMixin, ClassifierMixin
    from sklearn.preprocessing import LabelEncoder

    try:
        from sklearn.model_selection import KFold, StratifiedKFold
    except ImportError:
        from sklearn.cross_validation import KFold, StratifiedKFold

    SKLEARN_INSTALLED = True

    XGBModelBase = BaseEstimator
    XGBRegressorBase = RegressorMixin
    XGBClassifierBase = ClassifierMixin

    XGBKFold = KFold
    XGBStratifiedKFold = StratifiedKFold

    class XGBoostLabelEncoder(LabelEncoder):
        '''Label encoder with JSON serialization methods.'''
        def to_json(self):
            '''Returns a JSON compatible dictionary'''
            meta = dict()
            for k, v in self.__dict__.items():
                if isinstance(v, np.ndarray):
                    meta[k] = v.tolist()
                else:
                    meta[k] = v
            return meta

        def from_json(self, doc):
            # pylint: disable=attribute-defined-outside-init
            '''Load the encoder back from a JSON compatible dict.'''
            meta = dict()
            for k, v in doc.items():
                if k == 'classes_':
                    self.classes_ = np.array(v)
                    continue
                meta[k] = v
            self.__dict__.update(meta)
except ImportError:
    SKLEARN_INSTALLED = False

    # used for compatibility without sklearn
    XGBModelBase = object
    XGBClassifierBase = object
    XGBRegressorBase = object

    XGBKFold = None
    XGBStratifiedKFold = None
    XGBoostLabelEncoder = None


# dask
try:
    import dask
    DASK_INSTALLED = True
except ImportError:
    dask = None
    DASK_INSTALLED = False


try:
    import sparse
    import scipy.sparse as scipy_sparse
    SCIPY_INSTALLED = True
except ImportError:
    sparse = False
    scipy_sparse = False
    SCIPY_INSTALLED = False
