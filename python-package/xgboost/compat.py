# coding: utf-8
# pylint: disable= invalid-name,  unused-import
"""For compatibility and optional dependencies."""
from typing import Any
import sys
import types
import importlib.util
import logging
import numpy as np

assert (sys.version_info[0] == 3), 'Python 2 is no longer supported.'

# pylint: disable=invalid-name, redefined-builtin
STRING_TYPES = (str,)


def py_str(x):
    """convert c string back to python string"""
    return x.decode('utf-8')


def lazy_isinstance(instance, module, name):
    """Use string representation to identify a type."""

    # Notice, we use .__class__ as opposed to type() in order
    # to support object proxies such as weakref.proxy
    cls = instance.__class__
    module = cls.__module__ == module
    name = cls.__name__ == name
    return module and name


# pandas
try:
    from pandas import DataFrame, Series
    from pandas import MultiIndex
    from pandas import concat as pandas_concat

    PANDAS_INSTALLED = True
except ImportError:

    MultiIndex = object
    DataFrame: Any = object
    Series = object
    pandas_concat = None
    PANDAS_INSTALLED = False

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
            meta = {}
            for k, v in self.__dict__.items():
                if isinstance(v, np.ndarray):
                    meta[k] = v.tolist()
                else:
                    meta[k] = v
            return meta

        def from_json(self, doc):
            # pylint: disable=attribute-defined-outside-init
            '''Load the encoder back from a JSON compatible dict.'''
            meta = {}
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
    import pkg_resources
    pkg_resources.get_distribution('dask')
    DASK_INSTALLED = True
except pkg_resources.DistributionNotFound:
    dask = None
    DASK_INSTALLED = False


try:
    import scipy.sparse as scipy_sparse
    from scipy.sparse import csr_matrix as scipy_csr
    SCIPY_INSTALLED = True
except ImportError:
    scipy_sparse = False
    scipy_csr: Any = object
    SCIPY_INSTALLED = False


# Modified from tensorflow with added caching.  There's a `LazyLoader` in
# `importlib.utils`, except it's unclear from its document on how to use it.  This one
# seems to be easy to understand and works out of box.

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
# file except in compliance with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the specific language governing
# permissions and limitations under the License.
class LazyLoader(types.ModuleType):
    """Lazily import a module, mainly to avoid pulling in large dependencies.
    """

    def __init__(self, local_name, parent_module_globals, name, warning=None):
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals
        self._warning = warning
        self.module = None

        super().__init__(name)

    def _load(self):
        """Load the module and insert it into the parent's globals."""
        # Import the target module and insert it into the parent's namespace
        module = importlib.import_module(self.__name__)
        self._parent_module_globals[self._local_name] = module

        # Emit a warning if one was specified
        if self._warning:
            logging.warning(self._warning)
            # Make sure to only warn once.
        self._warning = None

        # Update this object's dict so that if someone keeps a reference to the
        #   LazyLoader, lookups are efficient (__getattr__ is only called on lookups
        #   that fail).
        self.__dict__.update(module.__dict__)

        return module

    def __getattr__(self, item):
        if not self.module:
            self.module = self._load()
        return getattr(self.module, item)

    def __dir__(self):
        if not self.module:
            self.module = self._load()
        return dir(self.module)
