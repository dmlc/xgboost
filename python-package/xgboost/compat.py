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


###############################################################################
# START NUMPY PATHLIB ATTRIBUTION
###############################################################################
# os.PathLike compatibility used in  Numpy:
# https://github.com/numpy/numpy/tree/v1.17.0
# Attribution:
# https://github.com/numpy/numpy/blob/v1.17.0/numpy/compat/py3k.py#L188-L247
# Backport os.fs_path, os.PathLike, and PurePath.__fspath__
if sys.version_info[:2] >= (3, 6):
    os_fspath = os.fspath
    os_PathLike = os.PathLike
else:
    def _PurePath__fspath__(self):
        return str(self)

    class os_PathLike(abc.ABC):
        """Abstract base class for implementing the file system path protocol."""

        @abc.abstractmethod
        def __fspath__(self):
            """Return the file system path representation of the object."""
            raise NotImplementedError

        @classmethod
        def __subclasshook__(cls, subclass):
            if issubclass(subclass, PurePath):
                return True
            return hasattr(subclass, '__fspath__')

    def os_fspath(path):
        """Return the path representation of a path-like object.
        If str or bytes is passed in, it is returned unchanged. Otherwise the
        os.PathLike interface is used to get the path representation. If the
        path representation is not str or bytes, TypeError is raised. If the
        provided path is not str, bytes, or os.PathLike, TypeError is raised.
        """
        if isinstance(path, (str, bytes)):
            return path

        # Work from the object's type to match method resolution of other magic
        # methods.
        path_type = type(path)
        try:
            path_repr = path_type.__fspath__(path)
        except AttributeError:
            if hasattr(path_type, '__fspath__'):
                raise
            if issubclass(path_type, PurePath):
                return _PurePath__fspath__(path)
            raise TypeError("expected str, bytes or os.PathLike object, "
                            "not " + path_type.__name__)
        if isinstance(path_repr, (str, bytes)):
            return path_repr
        raise TypeError("expected {}.__fspath__() to return str or bytes, "
                        "not {}".format(path_type.__name__,
                                        type(path_repr).__name__))
###############################################################################
# END NUMPY PATHLIB ATTRIBUTION
###############################################################################


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
    from cudf import DataFrame as CUDF_DataFrame
    from cudf import Series as CUDF_Series
    from cudf import MultiIndex as CUDF_MultiIndex
    from cudf import concat as CUDF_concat
    CUDF_INSTALLED = True
except ImportError:
    CUDF_DataFrame = object
    CUDF_Series = object
    CUDF_MultiIndex = object
    CUDF_INSTALLED = False
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
    from dask import delayed
    from dask import dataframe as dd
    from dask import array as da
    from dask.distributed import Client, get_client
    from dask.distributed import comm as distributed_comm
    from dask.distributed import wait as distributed_wait
    from distributed import get_worker as distributed_get_worker

    DASK_INSTALLED = True
except ImportError:
    dd = None
    da = None
    Client = None
    delayed = None
    get_client = None
    distributed_comm = None
    distributed_wait = None
    distributed_get_worker = None
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
