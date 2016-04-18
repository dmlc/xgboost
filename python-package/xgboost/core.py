# coding: utf-8
# pylint: disable=too-many-arguments, too-many-branches
"""Core XGBoost Library."""
from __future__ import absolute_import

import os
import ctypes
import collections

import numpy as np
import scipy.sparse

from .libpath import find_lib_path

from .compat import STRING_TYPES, PY3, DataFrame, py_str

class XGBoostError(Exception):
    """Error throwed by xgboost trainer."""
    pass


def from_pystr_to_cstr(data):
    """Convert a list of Python str to C pointer

    Parameters
    ----------
    data : list
        list of str
    """

    if isinstance(data, list):
        pointers = (ctypes.c_char_p * len(data))()
        if PY3:
            data = [bytes(d, 'utf-8') for d in data]
        else:
            data = [d.encode('utf-8') if isinstance(d, unicode) else d
                    for d in data]
        pointers[:] = data
        return pointers
    else:
        # copy from above when we actually use it
        raise NotImplementedError


def from_cstr_to_pystr(data, length):
    """Revert C pointer to Python str

    Parameters
    ----------
    data : ctypes pointer
        pointer to data
    length : ctypes pointer
        pointer to length of data
    """
    if PY3:
        res = []
        for i in range(length.value):
            try:
                res.append(str(data[i].decode('ascii')))
            except UnicodeDecodeError:
                res.append(str(data[i].decode('utf-8')))
    else:
        res = []
        for i in range(length.value):
            try:
                res.append(str(data[i].decode('ascii')))
            except UnicodeDecodeError:
                res.append(unicode(data[i].decode('utf-8')))
    return res


def _load_lib():
    """Load xgboost Library."""
    lib_path = find_lib_path()
    if len(lib_path) == 0:
        return None
    lib = ctypes.cdll.LoadLibrary(lib_path[0])
    lib.XGBGetLastError.restype = ctypes.c_char_p
    return lib


# load the XGBoost library globally
_LIB = _load_lib()

def _check_call(ret):
    """Check the return value of C API call

    This function will raise exception when error occurs.
    Wrap every API call with this function

    Parameters
    ----------
    ret : int
        return value from API calls
    """
    if ret != 0:
        raise XGBoostError(_LIB.XGBGetLastError())


def ctypes2numpy(cptr, length, dtype):
    """Convert a ctypes pointer array to a numpy array.
    """
    if not isinstance(cptr, ctypes.POINTER(ctypes.c_float)):
        raise RuntimeError('expected float pointer')
    res = np.zeros(length, dtype=dtype)
    if not ctypes.memmove(res.ctypes.data, cptr, length * res.strides[0]):
        raise RuntimeError('memmove failed')
    return res


def ctypes2buffer(cptr, length):
    """Convert ctypes pointer to buffer type."""
    if not isinstance(cptr, ctypes.POINTER(ctypes.c_char)):
        raise RuntimeError('expected char pointer')
    res = bytearray(length)
    rptr = (ctypes.c_char * length).from_buffer(res)
    if not ctypes.memmove(rptr, cptr, length):
        raise RuntimeError('memmove failed')
    return res


def c_str(string):
    """Convert a python string to cstring."""
    return ctypes.c_char_p(string.encode('utf-8'))


def c_array(ctype, values):
    """Convert a python string to c array."""
    return (ctype * len(values))(*values)



PANDAS_DTYPE_MAPPER = {'int8': 'int', 'int16': 'int', 'int32': 'int', 'int64': 'int',
                       'uint8': 'int', 'uint16': 'int', 'uint32': 'int', 'uint64': 'int',
                       'float16': 'float', 'float32': 'float', 'float64': 'float',
                       'bool': 'i'}


def _maybe_pandas_data(data, feature_names, feature_types):
    """ Extract internal data from pd.DataFrame for DMatrix data """

    if not isinstance(data, DataFrame):
        return data, feature_names, feature_types

    data_dtypes = data.dtypes
    if not all(dtype.name in PANDAS_DTYPE_MAPPER for dtype in data_dtypes):
        bad_fields = [data.columns[i] for i, dtype in enumerate(data_dtypes) if dtype.name not in PANDAS_DTYPE_MAPPER ]  
        raise ValueError('DataFrame.dtypes for data must be int, float or bool.\nDid not expect the data types in fie    lds '+', '.join(bad_fields))

    if feature_names is None:
        feature_names = data.columns.format()

    if feature_types is None:
        feature_types = [PANDAS_DTYPE_MAPPER[dtype.name] for dtype in data_dtypes]

    data = data.values.astype('float')

    return data, feature_names, feature_types


def _maybe_pandas_label(label):
    """ Extract internal data from pd.DataFrame for DMatrix label """

    if isinstance(label, DataFrame):
        if len(label.columns) > 1:
            raise ValueError('DataFrame for label cannot have multiple columns')

        label_dtypes = label.dtypes
        if not all(dtype.name in PANDAS_DTYPE_MAPPER for dtype in label_dtypes):
            raise ValueError('DataFrame.dtypes for label must be int, float or bool')
        else:
            label = label.values.astype('float')
    # pd.Series can be passed to xgb as it is

    return label

class DMatrix(object):
    """Data Matrix used in XGBoost.

    DMatrix is a internal data structure that used by XGBoost
    which is optimized for both memory efficiency and training speed.
    You can construct DMatrix from numpy.arrays
    """

    _feature_names = None  # for previous version's pickle
    _feature_types = None

    def __init__(self, data, label=None, missing=None,
                 weight=None, silent=False,
                 feature_names=None, feature_types=None):
        """
        Data matrix used in XGBoost.

        Parameters
        ----------
        data : string/numpy array/scipy.sparse/pd.DataFrame
            Data source of DMatrix.
            When data is string type, it represents the path libsvm format txt file,
            or binary file that xgboost can read from.
        label : list or numpy 1-D array, optional
            Label of the training data.
        missing : float, optional
            Value in the data which needs to be present as a missing value. If
            None, defaults to np.nan.
        weight : list or numpy 1-D array , optional
            Weight for each instance.
        silent : boolean, optional
            Whether print messages during construction
        feature_names : list, optional
            Set names for features.
        feature_types : list, optional
            Set types for features.
        """
        # force into void_p, mac need to pass things in as void_p
        if data is None:
            self.handle = None
            return

        data, feature_names, feature_types = _maybe_pandas_data(data,
                                                                feature_names,
                                                                feature_types)
        label = _maybe_pandas_label(label)

        if isinstance(data, STRING_TYPES):
            self.handle = ctypes.c_void_p()
            _check_call(_LIB.XGDMatrixCreateFromFile(c_str(data),
                                                     int(silent),
                                                     ctypes.byref(self.handle)))
        elif isinstance(data, scipy.sparse.csr_matrix):
            self._init_from_csr(data)
        elif isinstance(data, scipy.sparse.csc_matrix):
            self._init_from_csc(data)
        elif isinstance(data, np.ndarray):
            self._init_from_npy2d(data, missing)
        else:
            try:
                csr = scipy.sparse.csr_matrix(data)
                self._init_from_csr(csr)
            except:
                raise TypeError('can not initialize DMatrix from {}'.format(type(data).__name__))
        if label is not None:
            self.set_label(label)
        if weight is not None:
            self.set_weight(weight)

        self.feature_names = feature_names
        self.feature_types = feature_types

    def _init_from_csr(self, csr):
        """
        Initialize data from a CSR matrix.
        """
        if len(csr.indices) != len(csr.data):
            raise ValueError('length mismatch: {} vs {}'.format(len(csr.indices), len(csr.data)))
        self.handle = ctypes.c_void_p()
        _check_call(_LIB.XGDMatrixCreateFromCSR(c_array(ctypes.c_ulong, csr.indptr),
                                                c_array(ctypes.c_uint, csr.indices),
                                                c_array(ctypes.c_float, csr.data),
                                                len(csr.indptr), len(csr.data),
                                                ctypes.byref(self.handle)))

    def _init_from_csc(self, csc):
        """
        Initialize data from a CSC matrix.
        """
        if len(csc.indices) != len(csc.data):
            raise ValueError('length mismatch: {} vs {}'.format(len(csc.indices), len(csc.data)))
        self.handle = ctypes.c_void_p()
        _check_call(_LIB.XGDMatrixCreateFromCSC(c_array(ctypes.c_ulong, csc.indptr),
                                                c_array(ctypes.c_uint, csc.indices),
                                                c_array(ctypes.c_float, csc.data),
                                                len(csc.indptr), len(csc.data),
                                                ctypes.byref(self.handle)))

    def _init_from_npy2d(self, mat, missing):
        """
        Initialize data from a 2-D numpy matrix.
        """
        if len(mat.shape) != 2:
            raise ValueError('Input numpy.ndarray must be 2 dimensional')
        data = np.array(mat.reshape(mat.size), dtype=np.float32)
        self.handle = ctypes.c_void_p()
        missing = missing if missing is not None else np.nan
        _check_call(_LIB.XGDMatrixCreateFromMat(data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                                mat.shape[0], mat.shape[1],
                                                ctypes.c_float(missing),
                                                ctypes.byref(self.handle)))

    def __del__(self):
        _check_call(_LIB.XGDMatrixFree(self.handle))

    def get_float_info(self, field):
        """Get float property from the DMatrix.

        Parameters
        ----------
        field: str
            The field name of the information

        Returns
        -------
        info : array
            a numpy array of float information of the data
        """
        length = ctypes.c_ulong()
        ret = ctypes.POINTER(ctypes.c_float)()
        _check_call(_LIB.XGDMatrixGetFloatInfo(self.handle,
                                               c_str(field),
                                               ctypes.byref(length),
                                               ctypes.byref(ret)))
        return ctypes2numpy(ret, length.value, np.float32)

    def get_uint_info(self, field):
        """Get unsigned integer property from the DMatrix.

        Parameters
        ----------
        field: str
            The field name of the information

        Returns
        -------
        info : array
            a numpy array of float information of the data
        """
        length = ctypes.c_ulong()
        ret = ctypes.POINTER(ctypes.c_uint)()
        _check_call(_LIB.XGDMatrixGetUIntInfo(self.handle,
                                              c_str(field),
                                              ctypes.byref(length),
                                              ctypes.byref(ret)))
        return ctypes2numpy(ret, length.value, np.uint32)

    def set_float_info(self, field, data):
        """Set float type property into the DMatrix.

        Parameters
        ----------
        field: str
            The field name of the information

        data: numpy array
            The array ofdata to be set
        """
        _check_call(_LIB.XGDMatrixSetFloatInfo(self.handle,
                                               c_str(field),
                                               c_array(ctypes.c_float, data),
                                               len(data)))

    def set_uint_info(self, field, data):
        """Set uint type property into the DMatrix.

        Parameters
        ----------
        field: str
            The field name of the information

        data: numpy array
            The array ofdata to be set
        """
        _check_call(_LIB.XGDMatrixSetUIntInfo(self.handle,
                                              c_str(field),
                                              c_array(ctypes.c_uint, data),
                                              len(data)))

    def save_binary(self, fname, silent=True):
        """Save DMatrix to an XGBoost buffer.

        Parameters
        ----------
        fname : string
            Name of the output buffer file.
        silent : bool (optional; default: True)
            If set, the output is suppressed.
        """
        _check_call(_LIB.XGDMatrixSaveBinary(self.handle,
                                             c_str(fname),
                                             int(silent)))

    def set_label(self, label):
        """Set label of dmatrix

        Parameters
        ----------
        label: array like
            The label information to be set into DMatrix
        """
        self.set_float_info('label', label)

    def set_weight(self, weight):
        """ Set weight of each instance.

        Parameters
        ----------
        weight : array like
            Weight for each data point
        """
        self.set_float_info('weight', weight)

    def set_base_margin(self, margin):
        """ Set base margin of booster to start from.

        This can be used to specify a prediction value of
        existing model to be base_margin
        However, remember margin is needed, instead of transformed prediction
        e.g. for logistic regression: need to put in value before logistic transformation
        see also example/demo.py

        Parameters
        ----------
        margin: array like
            Prediction margin of each datapoint
        """
        self.set_float_info('base_margin', margin)

    def set_group(self, group):
        """Set group size of DMatrix (used for ranking).

        Parameters
        ----------
        group : array like
            Group size of each group
        """
        _check_call(_LIB.XGDMatrixSetGroup(self.handle,
                                           c_array(ctypes.c_uint, group),
                                           len(group)))

    def get_label(self):
        """Get the label of the DMatrix.

        Returns
        -------
        label : array
        """
        return self.get_float_info('label')

    def get_weight(self):
        """Get the weight of the DMatrix.

        Returns
        -------
        weight : array
        """
        return self.get_float_info('weight')

    def get_base_margin(self):
        """Get the base margin of the DMatrix.

        Returns
        -------
        base_margin : float
        """
        return self.get_float_info('base_margin')

    def num_row(self):
        """Get the number of rows in the DMatrix.

        Returns
        -------
        number of rows : int
        """
        ret = ctypes.c_ulong()
        _check_call(_LIB.XGDMatrixNumRow(self.handle,
                                         ctypes.byref(ret)))
        return ret.value

    def num_col(self):
        """Get the number of columns (features) in the DMatrix.

        Returns
        -------
        number of columns : int
        """
        ret = ctypes.c_uint()
        _check_call(_LIB.XGDMatrixNumCol(self.handle,
                                         ctypes.byref(ret)))
        return ret.value

    def slice(self, rindex):
        """Slice the DMatrix and return a new DMatrix that only contains `rindex`.

        Parameters
        ----------
        rindex : list
            List of indices to be selected.

        Returns
        -------
        res : DMatrix
            A new DMatrix containing only selected indices.
        """
        res = DMatrix(None, feature_names=self.feature_names)
        res.handle = ctypes.c_void_p()
        _check_call(_LIB.XGDMatrixSliceDMatrix(self.handle,
                                               c_array(ctypes.c_int, rindex),
                                               len(rindex),
                                               ctypes.byref(res.handle)))
        return res

    @property
    def feature_names(self):
        """Get feature names (column labels).

        Returns
        -------
        feature_names : list or None
        """
        return self._feature_names

    @property
    def feature_types(self):
        """Get feature types (column types).

        Returns
        -------
        feature_types : list or None
        """
        return self._feature_types

    @feature_names.setter
    def feature_names(self, feature_names):
        """Set feature names (column labels).

        Parameters
        ----------
        feature_names : list or None
            Labels for features. None will reset existing feature names
        """
        if feature_names is not None:
            # validate feature name
            if not isinstance(feature_names, list):
                feature_names = list(feature_names)
            if len(feature_names) != len(set(feature_names)):
                raise ValueError('feature_names must be unique')
            if len(feature_names) != self.num_col():
                msg = 'feature_names must have the same length as data'
                raise ValueError(msg)
            # prohibit to use symbols may affect to parse. e.g. []<
            if not all(isinstance(f, STRING_TYPES) and
                       not any(x in f for x in set(('[', ']', '<')))
                       for f in feature_names):
                raise ValueError('feature_names may not contain [, ] or <')
        else:
            # reset feature_types also
            self.feature_types = None
        self._feature_names = feature_names

    @feature_types.setter
    def feature_types(self, feature_types):
        """Set feature types (column types).

        This is for displaying the results and unrelated
        to the learning process.

        Parameters
        ----------
        feature_types : list or None
            Labels for features. None will reset existing feature names
        """
        if feature_types is not None:

            if self.feature_names is None:
                msg = 'Unable to set feature types before setting names'
                raise ValueError(msg)

            if isinstance(feature_types, STRING_TYPES):
                # single string will be applied to all columns
                feature_types = [feature_types] * self.num_col()

            if not isinstance(feature_types, list):
                feature_types = list(feature_types)
            if len(feature_types) != self.num_col():
                msg = 'feature_types must have the same length as data'
                raise ValueError(msg)

            valid = ('int', 'float', 'i', 'q')
            if not all(isinstance(f, STRING_TYPES) and f in valid
                       for f in feature_types):
                raise ValueError('All feature_names must be {int, float, i, q}')
        self._feature_types = feature_types


class Booster(object):
    """"A Booster of of XGBoost.

    Booster is the model of xgboost, that contains low level routines for
    training, prediction and evaluation.
    """

    feature_names = None

    def __init__(self, params=None, cache=(), model_file=None):
        # pylint: disable=invalid-name
        """Initialize the Booster.

        Parameters
        ----------
        params : dict
            Parameters for boosters.
        cache : list
            List of cache items.
        model_file : string
            Path to the model file.
        """
        for d in cache:
            if not isinstance(d, DMatrix):
                raise TypeError('invalid cache item: {}'.format(type(d).__name__))
            self._validate_features(d)

        dmats = c_array(ctypes.c_void_p, [d.handle for d in cache])
        self.handle = ctypes.c_void_p()
        _check_call(_LIB.XGBoosterCreate(dmats, len(cache), ctypes.byref(self.handle)))
        self.set_param({'seed': 0})
        self.set_param(params or {})
        if model_file is not None:
            self.load_model(model_file)

    def __del__(self):
        _LIB.XGBoosterFree(self.handle)

    def __getstate__(self):
        # can't pickle ctypes pointers
        # put model content in bytearray
        this = self.__dict__.copy()
        handle = this['handle']
        if handle is not None:
            raw = self.save_raw()
            this["handle"] = raw
        return this

    def __setstate__(self, state):
        # reconstruct handle from raw data
        handle = state['handle']
        if handle is not None:
            buf = handle
            dmats = c_array(ctypes.c_void_p, [])
            handle = ctypes.c_void_p()
            _check_call(_LIB.XGBoosterCreate(dmats, 0, ctypes.byref(handle)))
            length = ctypes.c_ulong(len(buf))
            ptr = (ctypes.c_char * len(buf)).from_buffer(buf)
            _check_call(_LIB.XGBoosterLoadModelFromBuffer(handle, ptr, length))
            state['handle'] = handle
        self.__dict__.update(state)
        self.set_param({'seed': 0})

    def __copy__(self):
        return self.__deepcopy__(None)

    def __deepcopy__(self, memo):
        return Booster(model_file=self.save_raw())

    def copy(self):
        """Copy the booster object.

        Returns
        -------
        booster: `Booster`
            a copied booster model
        """
        return self.__copy__()

    def load_rabit_checkpoint(self):
        """Initialize the model by load from rabit checkpoint.

        Returns
        -------
        version: integer
            The version number of the model.
        """
        version = ctypes.c_int()
        _check_call(_LIB.XGBoosterLoadRabitCheckpoint(
            self.handle, ctypes.byref(version)))
        return version.value

    def save_rabit_checkpoint(self):
        """Save the current booster to rabit checkpoint."""
        _check_call(_LIB.XGBoosterSaveRabitCheckpoint(self.handle))

    def attr(self, key):
        """Get attribute string from the Booster.

        Parameters
        ----------
        key : str
            The key to get attribute from.

        Returns
        -------
        value : str
            The attribute value of the key, returns None if attribute do not exist.
        """
        ret = ctypes.c_char_p()
        success = ctypes.c_int()
        _check_call(_LIB.XGBoosterGetAttr(
            self.handle, c_str(key), ctypes.byref(ret), ctypes.byref(success)))
        if success.value != 0:
            return py_str(ret.value)
        else:
            return None

    def set_attr(self, **kwargs):
        """Set the attribute of the Booster.

        Parameters
        ----------
        **kwargs
            The attributes to set
        """
        for key, value in kwargs.items():
            if not isinstance(value, STRING_TYPES):
                raise ValueError("Set Attr only accepts string values")
            _check_call(_LIB.XGBoosterSetAttr(
                self.handle, c_str(key), c_str(str(value))))

    def set_param(self, params, value=None):
        """Set parameters into the Booster.

        Parameters
        ----------
        params: dict/list/str
           list of key,value paris, dict of key to value or simply str key
        value: optional
           value of the specified parameter, when params is str key
        """
        if isinstance(params, collections.Mapping):
            params = params.items()
        elif isinstance(params, STRING_TYPES) and value is not None:
            params = [(params, value)]
        for key, val in params:
            _check_call(_LIB.XGBoosterSetParam(self.handle, c_str(key), c_str(str(val))))

    def update(self, dtrain, iteration, fobj=None):
        """
        Update for one iteration, with objective function calculated internally.

        Parameters
        ----------
        dtrain : DMatrix
            Training data.
        iteration : int
            Current iteration number.
        fobj : function
            Customized objective function.
        """
        if not isinstance(dtrain, DMatrix):
            raise TypeError('invalid training matrix: {}'.format(type(dtrain).__name__))
        self._validate_features(dtrain)

        if fobj is None:
            _check_call(_LIB.XGBoosterUpdateOneIter(self.handle, iteration, dtrain.handle))
        else:
            pred = self.predict(dtrain)
            grad, hess = fobj(pred, dtrain)
            self.boost(dtrain, grad, hess)

    def boost(self, dtrain, grad, hess):
        """
        Boost the booster for one iteration, with customized gradient statistics.

        Parameters
        ----------
        dtrain : DMatrix
            The training DMatrix.
        grad : list
            The first order of gradient.
        hess : list
            The second order of gradient.
        """
        if len(grad) != len(hess):
            raise ValueError('grad / hess length mismatch: {} / {}'.format(len(grad), len(hess)))
        if not isinstance(dtrain, DMatrix):
            raise TypeError('invalid training matrix: {}'.format(type(dtrain).__name__))
        self._validate_features(dtrain)

        _check_call(_LIB.XGBoosterBoostOneIter(self.handle, dtrain.handle,
                                               c_array(ctypes.c_float, grad),
                                               c_array(ctypes.c_float, hess),
                                               len(grad)))

    def eval_set(self, evals, iteration=0, feval=None):
        # pylint: disable=invalid-name
        """Evaluate  a set of data.

        Parameters
        ----------
        evals : list of tuples (DMatrix, string)
            List of items to be evaluated.
        iteration : int
            Current iteration.
        feval : function
            Custom evaluation function.

        Returns
        -------
        result: str
            Evaluation result string.
        """
        if feval is None:
            for d in evals:
                if not isinstance(d[0], DMatrix):
                    raise TypeError('expected DMatrix, got {}'.format(type(d[0]).__name__))
                if not isinstance(d[1], STRING_TYPES):
                    raise TypeError('expected string, got {}'.format(type(d[1]).__name__))
                self._validate_features(d[0])

            dmats = c_array(ctypes.c_void_p, [d[0].handle for d in evals])
            evnames = c_array(ctypes.c_char_p, [c_str(d[1]) for d in evals])
            msg = ctypes.c_char_p()
            _check_call(_LIB.XGBoosterEvalOneIter(self.handle, iteration,
                                                  dmats, evnames, len(evals),
                                                  ctypes.byref(msg)))
            return msg.value
        else:
            res = '[%d]' % iteration
            for dmat, evname in evals:
                feval_ret = feval(self.predict(dmat), dmat)
                if isinstance(feval_ret, list):
                    for name, val in feval_ret:
                        res += '\t%s-%s:%f' % (evname, name, val)
                else:
                    name, val = feval_ret
                    res += '\t%s-%s:%f' % (evname, name, val)
            return res

    def eval(self, data, name='eval', iteration=0):
        """Evaluate the model on mat.

        Parameters
        ----------
        data : DMatrix
            The dmatrix storing the input.

        name : str, optional
            The name of the dataset.

        iteration : int, optional
            The current iteration number.

        Returns
        -------
        result: str
            Evaluation result string.
        """
        self._validate_features(data)
        return self.eval_set([(data, name)], iteration)

    def predict(self, data, output_margin=False, ntree_limit=0, pred_leaf=False):
        """
        Predict with data.

        NOTE: This function is not thread safe.
              For each booster object, predict can only be called from one thread.
              If you want to run prediction using multiple thread, call bst.copy() to make copies
              of model object and then call predict

        Parameters
        ----------
        data : DMatrix
            The dmatrix storing the input.

        output_margin : bool
            Whether to output the raw untransformed margin value.

        ntree_limit : int
            Limit number of trees in the prediction; defaults to 0 (use all trees).

        pred_leaf : bool
            When this option is on, the output will be a matrix of (nsample, ntrees)
            with each record indicating the predicted leaf index of each sample in each tree.
            Note that the leaf index of a tree is unique per tree, so you may find leaf 1
            in both tree 1 and tree 0.

        Returns
        -------
        prediction : numpy array
        """
        option_mask = 0x00
        if output_margin:
            option_mask |= 0x01
        if pred_leaf:
            option_mask |= 0x02

        self._validate_features(data)

        length = ctypes.c_ulong()
        preds = ctypes.POINTER(ctypes.c_float)()
        _check_call(_LIB.XGBoosterPredict(self.handle, data.handle,
                                          option_mask, ntree_limit,
                                          ctypes.byref(length),
                                          ctypes.byref(preds)))
        preds = ctypes2numpy(preds, length.value, np.float32)
        if pred_leaf:
            preds = preds.astype(np.int32)
        nrow = data.num_row()
        if preds.size != nrow and preds.size % nrow == 0:
            preds = preds.reshape(nrow, preds.size / nrow)
        return preds

    def save_model(self, fname):
        """
        Save the model to a file.

        Parameters
        ----------
        fname : string
            Output file name
        """
        if isinstance(fname, STRING_TYPES):  # assume file name
            _check_call(_LIB.XGBoosterSaveModel(self.handle, c_str(fname)))
        else:
            raise TypeError("fname must be a string")

    def save_raw(self):
        """
        Save the model to a in memory buffer represetation

        Returns
        -------
        a in memory buffer represetation of the model
        """
        length = ctypes.c_ulong()
        cptr = ctypes.POINTER(ctypes.c_char)()
        _check_call(_LIB.XGBoosterGetModelRaw(self.handle,
                                              ctypes.byref(length),
                                              ctypes.byref(cptr)))
        return ctypes2buffer(cptr, length.value)

    def load_model(self, fname):
        """
        Load the model from a file.

        Parameters
        ----------
        fname : string or a memory buffer
            Input file name or memory buffer(see also save_raw)
        """
        if isinstance(fname, STRING_TYPES):
            # assume file name, cannot use os.path.exist to check, file can be from URL.
            _check_call(_LIB.XGBoosterLoadModel(self.handle, c_str(fname)))
        else:
            buf = fname
            length = ctypes.c_ulong(len(buf))
            ptr = (ctypes.c_char * len(buf)).from_buffer(buf)
            _check_call(_LIB.XGBoosterLoadModelFromBuffer(self.handle, ptr, length))

    def dump_model(self, fout, fmap='', with_stats=False):
        # pylint: disable=consider-using-enumerate
        """
        Dump model into a text file.

        Parameters
        ----------
        foout : string
            Output file name.
        fmap : string, optional
            Name of the file containing feature map names.
        with_stats : bool (optional)
            Controls whether the split statistics are output.
        """
        if isinstance(fout, STRING_TYPES):
            fout = open(fout, 'w')
            need_close = True
        else:
            need_close = False
        ret = self.get_dump(fmap, with_stats)
        for i in range(len(ret)):
            fout.write('booster[{}]:\n'.format(i))
            fout.write(ret[i])
        if need_close:
            fout.close()

    def get_dump(self, fmap='', with_stats=False):
        """
        Returns the dump the model as a list of strings.
        """

        length = ctypes.c_ulong()
        sarr = ctypes.POINTER(ctypes.c_char_p)()
        if self.feature_names is not None and fmap == '':
            flen = int(len(self.feature_names))

            fname = from_pystr_to_cstr(self.feature_names)

            if self.feature_types is None:
                # use quantitative as default
                # {'q': quantitative, 'i': indicator}
                ftype = from_pystr_to_cstr(['q'] * flen)
            else:
                ftype = from_pystr_to_cstr(self.feature_types)
            _check_call(_LIB.XGBoosterDumpModelWithFeatures(self.handle,
                                                            flen,
                                                            fname,
                                                            ftype,
                                                            int(with_stats),
                                                            ctypes.byref(length),
                                                            ctypes.byref(sarr)))
        else:
            if fmap != '' and not os.path.exists(fmap):
                raise ValueError("No such file: {0}".format(fmap))
            _check_call(_LIB.XGBoosterDumpModel(self.handle,
                                                c_str(fmap),
                                                int(with_stats),
                                                ctypes.byref(length),
                                                ctypes.byref(sarr)))
        res = from_cstr_to_pystr(sarr, length)
        return res

    def get_fscore(self, fmap=''):
        """Get feature importance of each feature.

        Parameters
        ----------
        fmap: str (optional)
           The name of feature map file
        """
        trees = self.get_dump(fmap)
        fmap = {}
        for tree in trees:
            for line in tree.split('\n'):
                arr = line.split('[')
                if len(arr) == 1:
                    continue
                fid = arr[1].split(']')[0]
                fid = fid.split('<')[0]
                if fid not in fmap:
                    fmap[fid] = 1
                else:
                    fmap[fid] += 1
        return fmap

    def _validate_features(self, data):
        """
        Validate Booster and data's feature_names are identical.
        Set feature_names and feature_types from DMatrix
        """
        if self.feature_names is None:
            self.feature_names = data.feature_names
            self.feature_types = data.feature_types
        else:
            # Booster can't accept data with different feature names
            if self.feature_names != data.feature_names:
                dat_missing = set(self.feature_names) - set(data.feature_names)
                my_missing = set(data.feature_names) - set(self.feature_names)
                msg = 'feature_names mismatch: {0} {1}'
                if dat_missing: msg +='\nexpected ' + ', '.join(str(s) for s in dat_missing) +' in input data'
                if my_missing: msg +='\ntraining data did not have the following fields: ' + ', '.join(str(s) for s in my_missing)
                raise ValueError(msg.format(self.feature_names,
                                            data.feature_names))
