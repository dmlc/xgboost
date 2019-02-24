# coding: utf-8
# pylint: disable= invalid-name,  unused-import
"""For compatibility"""

from __future__ import absolute_import

import sys


PY3 = (sys.version_info[0] == 3)

if PY3:
    # pylint: disable=invalid-name, redefined-builtin
    STRING_TYPES = str,

    def py_str(x):
        """convert c string back to python string"""
        return x.decode('utf-8')
else:
    # pylint: disable=invalid-name
    STRING_TYPES = basestring,

    def py_str(x):
        """convert c string back to python string"""
        return x

try:
    import cPickle as pickle   # noqa
except ImportError:
    import pickle              # noqa


# pandas
try:
    from pandas import DataFrame
    from pandas import MultiIndex
    PANDAS_INSTALLED = True
except ImportError:

    class MultiIndex(object):
        """ dummy for pandas.MultiIndex """
        pass

    class DataFrame(object):
        """ dummy for pandas.DataFrame """
        pass

    PANDAS_INSTALLED = False

# cudf
try:
    from cudf.dataframe import DataFrame as CUDF
    from cudf.dataframe.column import Column as CUDF_COL
    from libgdf_cffi import ffi as CUDF_FFI
    CUDF_INSTALLED = True
except ImportError:

    class CUDF(object):
        """ dummy object for cudf.dataframe.DataFrame """
        pass

    class CUDF_COL(object):
        """ dummy object for cudf.dataframe.column.Column """
        pass

    class CUDF_FFI(object):
        """ dummy object for libgdf_cffi.ffi ... FFI bindings to cudf """
        def new(self, *args, **kwargs):
            pass

        def cast(self, *args, **kwargs):
            pass

    CUDF_INSTALLED = False

# dt
try:
    import datatable
    if hasattr(datatable, "Frame"):
        DataTable = datatable.Frame
    else:
        DataTable = datatable.DataTable
    DT_INSTALLED = True
except ImportError:

    class DataTable(object):
        """ dummy for datatable.DataTable """
        pass

    DT_INSTALLED = False

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
    XGBLabelEncoder = LabelEncoder
except ImportError:
    SKLEARN_INSTALLED = False

    # used for compatibility without sklearn
    XGBModelBase = object
    XGBClassifierBase = object
    XGBRegressorBase = object

    XGBKFold = None
    XGBStratifiedKFold = None
    XGBLabelEncoder = None
