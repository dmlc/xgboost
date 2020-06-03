# coding: utf-8
# pylint: disable=too-many-arguments, too-many-branches, invalid-name
# pylint: disable=too-many-branches, too-many-lines, too-many-locals
# pylint: disable=too-many-public-methods
"""Core XGBoost Library."""
import collections
# pylint: disable=no-name-in-module,import-error
from collections.abc import Mapping  # Python 3
# pylint: enable=no-name-in-module,import-error
import ctypes
import os
import re
import sys
import warnings
import json

import numpy as np
import scipy.sparse

from .compat import (
    STRING_TYPES, DataFrame, MultiIndex, Int64Index, py_str,
    PANDAS_INSTALLED, CUDF_INSTALLED,
    CUDF_DataFrame, CUDF_Series, CUDF_MultiIndex,
    os_fspath, os_PathLike, lazy_isinstance)
from .libpath import find_lib_path

# c_bst_ulong corresponds to bst_ulong defined in xgboost/c_api.h
c_bst_ulong = ctypes.c_uint64


class XGBoostError(Exception):
    """Error thrown by xgboost trainer."""


class EarlyStopException(Exception):
    """Exception to signal early stopping.

    Parameters
    ----------
    best_iteration : int
        The best iteration stopped.
    """

    def __init__(self, best_iteration):
        super(EarlyStopException, self).__init__()
        self.best_iteration = best_iteration


# Callback environment used by callbacks
CallbackEnv = collections.namedtuple(
    "XGBoostCallbackEnv",
    ["model",
     "cvfolds",
     "iteration",
     "begin_iteration",
     "end_iteration",
     "rank",
     "evaluation_result_list"])


def from_pystr_to_cstr(data):
    """Convert a list of Python str to C pointer

    Parameters
    ----------
    data : list
        list of str
    """

    if not isinstance(data, list):
        raise NotImplementedError
    pointers = (ctypes.c_char_p * len(data))()
    data = [bytes(d, 'utf-8') for d in data]
    pointers[:] = data
    return pointers


def from_cstr_to_pystr(data, length):
    """Revert C pointer to Python str

    Parameters
    ----------
    data : ctypes pointer
        pointer to data
    length : ctypes pointer
        pointer to length of data
    """
    res = []
    for i in range(length.value):
        try:
            res.append(str(data[i].decode('ascii')))
        except UnicodeDecodeError:
            res.append(str(data[i].decode('utf-8')))
    return res


def _expect(expectations, got):
    """Translate input error into string.

    Parameters
    ----------
    expectations: sequence
        a list of expected value.
    got:
        actual input

    Returns
    -------
    msg: str
    """
    msg = 'Expecting '
    for t in range(len(expectations) - 1):
        msg += str(expectations[t])
        msg += ' or '
    msg += str(expectations[-1])
    msg += '.  Got ' + str(got)
    return msg


def _log_callback(msg):
    """Redirect logs from native library into Python console"""
    print("{0:s}".format(py_str(msg)))


def _get_log_callback_func():
    """Wrap log_callback() method in ctypes callback type"""
    # pylint: disable=invalid-name
    CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_char_p)
    return CALLBACK(_log_callback)


def _load_lib():
    """Load xgboost Library."""
    lib_paths = find_lib_path()
    if not lib_paths:
        return None
    try:
        pathBackup = os.environ['PATH'].split(os.pathsep)
    except KeyError:
        pathBackup = []
    lib_success = False
    os_error_list = []
    for lib_path in lib_paths:
        try:
            # needed when the lib is linked with non-system-available
            # dependencies
            os.environ['PATH'] = os.pathsep.join(
                pathBackup + [os.path.dirname(lib_path)])
            lib = ctypes.cdll.LoadLibrary(lib_path)
            lib_success = True
        except OSError as e:
            os_error_list.append(str(e))
            continue
        finally:
            os.environ['PATH'] = os.pathsep.join(pathBackup)
    if not lib_success:
        libname = os.path.basename(lib_paths[0])
        raise XGBoostError(
            'XGBoost Library ({}) could not be loaded.\n'.format(libname) +
            'Likely causes:\n' +
            '  * OpenMP runtime is not installed ' +
            '(vcomp140.dll or libgomp-1.dll for Windows, libomp.dylib for Mac OSX, ' +
            'libgomp.so for Linux and other UNIX-like OSes). Mac OSX users: Run ' +
            '`brew install libomp` to install OpenMP runtime.\n' +
            '  * You are running 32-bit Python on a 64-bit OS\n' +
            'Error message(s): {}\n'.format(os_error_list))
    lib.XGBGetLastError.restype = ctypes.c_char_p
    lib.callback = _get_log_callback_func()
    if lib.XGBRegisterLogCallback(lib.callback) != 0:
        raise XGBoostError(lib.XGBGetLastError())
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
        raise XGBoostError(py_str(_LIB.XGBGetLastError()))


def ctypes2numpy(cptr, length, dtype):
    """Convert a ctypes pointer array to a numpy array."""
    NUMPY_TO_CTYPES_MAPPING = {
        np.float32: ctypes.c_float,
        np.uint32: ctypes.c_uint,
    }
    if dtype not in NUMPY_TO_CTYPES_MAPPING:
        raise RuntimeError('Supported types: {}'.format(
            NUMPY_TO_CTYPES_MAPPING.keys()))
    ctype = NUMPY_TO_CTYPES_MAPPING[dtype]
    if not isinstance(cptr, ctypes.POINTER(ctype)):
        raise RuntimeError('expected {} pointer'.format(ctype))
    res = np.zeros(length, dtype=dtype)
    if not ctypes.memmove(res.ctypes.data, cptr, length * res.strides[0]):
        raise RuntimeError('memmove failed')
    return res


def ctypes2cupy(cptr, length, dtype):
    """Convert a ctypes pointer array to a cupy array."""
    # pylint: disable=import-error
    import cupy
    from cupy.cuda.memory import MemoryPointer
    from cupy.cuda.memory import UnownedMemory
    CUPY_TO_CTYPES_MAPPING = {
        cupy.float32: ctypes.c_float,
        cupy.uint32: ctypes.c_uint
    }
    if dtype not in CUPY_TO_CTYPES_MAPPING.keys():
        raise RuntimeError('Supported types: {}'.format(
            CUPY_TO_CTYPES_MAPPING.keys()
        ))
    addr = ctypes.cast(cptr, ctypes.c_void_p).value
    # pylint: disable=c-extension-no-member,no-member
    device = cupy.cuda.runtime.pointerGetAttributes(addr).device
    # The owner field is just used to keep the memory alive with ref count.  As
    # unowned's life time is scoped within this function we don't need that.
    unownd = UnownedMemory(
        addr, length.value * ctypes.sizeof(CUPY_TO_CTYPES_MAPPING[dtype]),
        owner=None)
    memptr = MemoryPointer(unownd, 0)
    # pylint: disable=unexpected-keyword-arg
    mem = cupy.ndarray((length.value, ), dtype=dtype, memptr=memptr)
    assert mem.device.id == device
    arr = cupy.array(mem, copy=True)
    return arr


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
    if (isinstance(values, np.ndarray)
            and values.dtype.itemsize == ctypes.sizeof(ctype)):
        return (ctype * len(values)).from_buffer_copy(values)
    return (ctype * len(values))(*values)


PANDAS_DTYPE_MAPPER = {'int8': 'int', 'int16': 'int', 'int32': 'int', 'int64':
                       'int', 'uint8': 'int', 'uint16': 'int', 'uint32': 'int',
                       'uint64': 'int', 'float16': 'float', 'float32': 'float',
                       'float64': 'float', 'bool': 'i'}


# Either object has cuda array interface or contains columns with interfaces
def _has_cuda_array_interface(data):
    return hasattr(data, '__cuda_array_interface__') or (
        CUDF_INSTALLED and isinstance(data, CUDF_DataFrame))


def _maybe_pandas_data(data, feature_names, feature_types,
                       meta=None, meta_type=None):
    """Extract internal data from pd.DataFrame for DMatrix data"""
    if not (PANDAS_INSTALLED and isinstance(data, DataFrame)):
        return data, feature_names, feature_types
    from pandas.api.types import is_sparse

    data_dtypes = data.dtypes
    if not all(dtype.name in PANDAS_DTYPE_MAPPER or is_sparse(dtype)
               for dtype in data_dtypes):
        bad_fields = [
            str(data.columns[i]) for i, dtype in enumerate(data_dtypes)
            if dtype.name not in PANDAS_DTYPE_MAPPER
        ]

        msg = """DataFrame.dtypes for data must be int, float or bool.
                Did not expect the data types in fields """
        raise ValueError(msg + ', '.join(bad_fields))

    if feature_names is None and meta is None:
        if isinstance(data.columns, MultiIndex):
            feature_names = [
                ' '.join([str(x) for x in i]) for i in data.columns
            ]
        elif isinstance(data.columns, Int64Index):
            feature_names = list(map(str, data.columns))
        else:
            feature_names = data.columns.format()

    if feature_types is None and meta is None:
        feature_types = []
        for dtype in data_dtypes:
            if is_sparse(dtype):
                feature_types.append(PANDAS_DTYPE_MAPPER[dtype.subtype.name])
            else:
                feature_types.append(PANDAS_DTYPE_MAPPER[dtype.name])

    if meta and len(data.columns) > 1:
        raise ValueError(
            'DataFrame for {meta} cannot have multiple columns'.format(
                meta=meta))

    dtype = meta_type if meta_type else 'float'
    data = data.values.astype(dtype)

    return data, feature_names, feature_types


def _cudf_array_interfaces(df):
    '''Extract CuDF __cuda_array_interface__'''
    interfaces = []
    for col in df:
        interface = df[col].__cuda_array_interface__
        if 'mask' in interface:
            interface['mask'] = interface['mask'].__cuda_array_interface__
        interfaces.append(interface)
    interfaces_str = bytes(json.dumps(interfaces, indent=2), 'utf-8')
    return interfaces_str


def _maybe_cudf_dataframe(data, feature_names, feature_types):
    """Extract internal data from cudf.DataFrame for DMatrix data."""
    if not (CUDF_INSTALLED and isinstance(data,
                                          (CUDF_DataFrame, CUDF_Series))):
        return data, feature_names, feature_types
    if feature_names is None:
        if isinstance(data, CUDF_Series):
            feature_names = [data.name]
        elif isinstance(data.columns, CUDF_MultiIndex):
            feature_names = [
                ' '.join([str(x) for x in i])
                for i in data.columns
            ]
        else:
            feature_names = data.columns.format()
    if feature_types is None:
        if isinstance(data, CUDF_Series):
            dtypes = [data.dtype]
        else:
            dtypes = data.dtypes
        feature_types = [PANDAS_DTYPE_MAPPER[d.name] for d in dtypes]
    return data, feature_names, feature_types


DT_TYPE_MAPPER = {'bool': 'bool', 'int': 'int', 'real': 'float'}

DT_TYPE_MAPPER2 = {'bool': 'i', 'int': 'int', 'real': 'float'}


def _maybe_dt_data(data, feature_names, feature_types,
                   meta=None, meta_type=None):
    """Validate feature names and types if data table"""
    if (not lazy_isinstance(data, 'datatable', 'Frame') and
            not lazy_isinstance(data, 'datatable', 'DataTable')):
        return data, feature_names, feature_types

    if meta and data.shape[1] > 1:
        raise ValueError(
            'DataTable for label or weight cannot have multiple columns')
    if meta:
        # below requires new dt version
        # extract first column
        data = data.to_numpy()[:, 0].astype(meta_type)
        return data, None, None

    data_types_names = tuple(lt.name for lt in data.ltypes)
    bad_fields = [data.names[i]
                  for i, type_name in enumerate(data_types_names)
                  if type_name not in DT_TYPE_MAPPER]
    if bad_fields:
        msg = """DataFrame.types for data must be int, float or bool.
                Did not expect the data types in fields """
        raise ValueError(msg + ', '.join(bad_fields))

    if feature_names is None and meta is None:
        feature_names = data.names

        # always return stypes for dt ingestion
        if feature_types is not None:
            raise ValueError(
                'DataTable has own feature types, cannot pass them in.')
        feature_types = np.vectorize(DT_TYPE_MAPPER2.get)(data_types_names)

    return data, feature_names, feature_types

def _is_dlpack(x):
    return 'PyCapsule' in str(type(x)) and "dltensor" in str(x)

# Just convert dlpack into cupy (zero copy)
def _maybe_dlpack_data(data, feature_names, feature_types):
    if not _is_dlpack(data):
        return data, feature_names, feature_types
    from cupy import fromDlpack # pylint: disable=E0401
    data = fromDlpack(data)
    return data, feature_names, feature_types


def _convert_dataframes(data, feature_names, feature_types,
                        meta=None, meta_type=None):
    data, feature_names, feature_types = _maybe_pandas_data(data,
                                                            feature_names,
                                                            feature_types,
                                                            meta,
                                                            meta_type)

    data, feature_names, feature_types = _maybe_dt_data(data,
                                                        feature_names,
                                                        feature_types,
                                                        meta,
                                                        meta_type)

    data, feature_names, feature_types = _maybe_cudf_dataframe(
        data, feature_names, feature_types)

    data, feature_names, feature_types = _maybe_dlpack_data(
        data, feature_names, feature_types)

    return data, feature_names, feature_types


def _maybe_np_slice(data, dtype=np.float32):
    '''Handle numpy slice.  This can be removed if we use __array_interface__.
    '''
    try:
        if not data.flags.c_contiguous:
            warnings.warn(
                "Use subset (sliced data) of np.ndarray is not recommended " +
                "because it will generate extra copies and increase " +
                "memory consumption")
            data = np.array(data, copy=True, dtype=dtype)
        else:
            data = np.array(data, copy=False, dtype=dtype)
    except AttributeError:
        data = np.array(data, copy=False, dtype=dtype)
    return data


class DMatrix(object):
    """Data Matrix used in XGBoost.

    DMatrix is a internal data structure that used by XGBoost
    which is optimized for both memory efficiency and training speed.
    You can construct DMatrix from numpy.arrays
    """

    _feature_names = None  # for previous version's pickle
    _feature_types = None

    def __init__(self, data, label=None, weight=None, base_margin=None,
                 missing=None,
                 silent=False,
                 feature_names=None,
                 feature_types=None,
                 nthread=None):
        """Parameters
        ----------
        data : os.PathLike/string/numpy.array/scipy.sparse/pd.DataFrame/
               dt.Frame/cudf.DataFrame/cupy.array/dlpack
            Data source of DMatrix.
            When data is string or os.PathLike type, it represents the path
            libsvm format txt file, csv file (by specifying uri parameter
            'path_to_csv?format=csv'), or binary file that xgboost can read
            from.
        label : list, numpy 1-D array or cudf.DataFrame, optional
            Label of the training data.
        missing : float, optional
            Value in the input data which needs to be present as a missing
            value. If None, defaults to np.nan.
        weight : list, numpy 1-D array or cudf.DataFrame , optional
            Weight for each instance.

            .. note:: For ranking task, weights are per-group.

                In ranking task, one weight is assigned to each group (not each
                data point). This is because we only care about the relative
                ordering of data points within each group, so it doesn't make
                sense to assign weights to individual data points.

        silent : boolean, optional
            Whether print messages during construction
        feature_names : list, optional
            Set names for features.
        feature_types : list, optional
            Set types for features.
        nthread : integer, optional
            Number of threads to use for loading data when parallelization is
            applicable. If -1, uses maximum threads available on the system.

        """
        # force into void_p, mac need to pass things in as void_p
        if data is None:
            self.handle = None

            if feature_names is not None:
                self._feature_names = feature_names
            if feature_types is not None:
                self._feature_types = feature_types
            return

        if isinstance(data, list):
            raise TypeError('Input data can not be a list.')

        data, feature_names, feature_types = _convert_dataframes(
            data, feature_names, feature_types
        )
        missing = missing if missing is not None else np.nan
        nthread = nthread if nthread is not None else 1

        if isinstance(data, (STRING_TYPES, os_PathLike)):
            handle = ctypes.c_void_p()
            _check_call(_LIB.XGDMatrixCreateFromFile(c_str(os_fspath(data)),
                                                     ctypes.c_int(silent),
                                                     ctypes.byref(handle)))
            self.handle = handle
        elif isinstance(data, scipy.sparse.csr_matrix):
            self._init_from_csr(data)
        elif isinstance(data, scipy.sparse.csc_matrix):
            self._init_from_csc(data)
        elif isinstance(data, np.ndarray):
            self._init_from_npy2d(data, missing, nthread)
        elif lazy_isinstance(data, 'datatable', 'Frame'):
            self._init_from_dt(data, nthread)
        elif hasattr(data, "__cuda_array_interface__"):
            self._init_from_array_interface(data, missing, nthread)
        elif CUDF_INSTALLED and isinstance(data, CUDF_DataFrame):
            self._init_from_array_interface_columns(data, missing, nthread)
        else:
            try:
                csr = scipy.sparse.csr_matrix(data)
                self._init_from_csr(csr)
            except Exception:
                raise TypeError('can not initialize DMatrix from'
                                ' {}'.format(type(data).__name__))

        if label is not None:
            self.set_label(label)
        if weight is not None:
            self.set_weight(weight)
        if base_margin is not None:
            self.set_base_margin(base_margin)

        self.feature_names = feature_names
        self.feature_types = feature_types

    def _init_from_csr(self, csr):
        """Initialize data from a CSR matrix."""
        if len(csr.indices) != len(csr.data):
            raise ValueError('length mismatch: {} vs {}'.format(
                len(csr.indices), len(csr.data)))
        handle = ctypes.c_void_p()
        _check_call(_LIB.XGDMatrixCreateFromCSREx(
            c_array(ctypes.c_size_t, csr.indptr),
            c_array(ctypes.c_uint, csr.indices),
            c_array(ctypes.c_float, csr.data),
            ctypes.c_size_t(len(csr.indptr)),
            ctypes.c_size_t(len(csr.data)),
            ctypes.c_size_t(csr.shape[1]),
            ctypes.byref(handle)))
        self.handle = handle

    def _init_from_csc(self, csc):
        """Initialize data from a CSC matrix."""
        if len(csc.indices) != len(csc.data):
            raise ValueError('length mismatch: {} vs {}'.format(
                len(csc.indices), len(csc.data)))
        handle = ctypes.c_void_p()
        _check_call(_LIB.XGDMatrixCreateFromCSCEx(
            c_array(ctypes.c_size_t, csc.indptr),
            c_array(ctypes.c_uint, csc.indices),
            c_array(ctypes.c_float, csc.data),
            ctypes.c_size_t(len(csc.indptr)),
            ctypes.c_size_t(len(csc.data)),
            ctypes.c_size_t(csc.shape[0]),
            ctypes.byref(handle)))
        self.handle = handle

    def _init_from_npy2d(self, mat, missing, nthread):
        """Initialize data from a 2-D numpy matrix.

        If ``mat`` does not have ``order='C'`` (aka row-major) or is
        not contiguous, a temporary copy will be made.

        If ``mat`` does not have ``dtype=numpy.float32``, a temporary copy will
        be made.

        So there could be as many as two temporary data copies; be mindful of
        input layout and type if memory use is a concern.

        """
        if len(mat.shape) != 2:
            raise ValueError('Expecting 2 dimensional numpy.ndarray, got: ',
                             mat.shape)
        # flatten the array by rows and ensure it is float32.  we try to avoid
        # data copies if possible (reshape returns a view when possible and we
        # explicitly tell np.array to try and avoid copying)
        data = np.array(mat.reshape(mat.size), copy=False, dtype=np.float32)
        handle = ctypes.c_void_p()
        _check_call(_LIB.XGDMatrixCreateFromMat_omp(
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            c_bst_ulong(mat.shape[0]),
            c_bst_ulong(mat.shape[1]),
            ctypes.c_float(missing),
            ctypes.byref(handle),
            ctypes.c_int(nthread)))
        self.handle = handle

    def _init_from_dt(self, data, nthread):
        """Initialize data from a datatable Frame."""
        ptrs = (ctypes.c_void_p * data.ncols)()
        if hasattr(data, "internal") and hasattr(data.internal, "column"):
            # datatable>0.8.0
            for icol in range(data.ncols):
                col = data.internal.column(icol)
                ptr = col.data_pointer
                ptrs[icol] = ctypes.c_void_p(ptr)
        else:
            # datatable<=0.8.0
            from datatable.internal import \
                frame_column_data_r  # pylint: disable=no-name-in-module,import-error
            for icol in range(data.ncols):
                ptrs[icol] = frame_column_data_r(data, icol)

        # always return stypes for dt ingestion
        feature_type_strings = (ctypes.c_char_p * data.ncols)()
        for icol in range(data.ncols):
            feature_type_strings[icol] = ctypes.c_char_p(
                data.stypes[icol].name.encode('utf-8'))

        handle = ctypes.c_void_p()
        _check_call(_LIB.XGDMatrixCreateFromDT(
            ptrs, feature_type_strings,
            c_bst_ulong(data.shape[0]),
            c_bst_ulong(data.shape[1]),
            ctypes.byref(handle),
            ctypes.c_int(nthread)))
        self.handle = handle

    def _init_from_array_interface_columns(self, df, missing, nthread):
        """Initialize DMatrix from columnar memory format."""
        interfaces_str = _cudf_array_interfaces(df)
        handle = ctypes.c_void_p()
        _check_call(
            _LIB.XGDMatrixCreateFromArrayInterfaceColumns(
                interfaces_str,
                ctypes.c_float(missing),
                ctypes.c_int(nthread),
                ctypes.byref(handle)))
        self.handle = handle

    def _init_from_array_interface(self, data, missing, nthread):
        """Initialize DMatrix from cupy ndarray."""
        interface = data.__cuda_array_interface__
        if 'mask' in interface:
            interface['mask'] = interface['mask'].__cuda_array_interface__
        interface_str = bytes(json.dumps(interface, indent=2), 'utf-8')

        handle = ctypes.c_void_p()
        _check_call(
            _LIB.XGDMatrixCreateFromArrayInterface(
                interface_str,
                ctypes.c_float(missing),
                ctypes.c_int(nthread),
                ctypes.byref(handle)))
        self.handle = handle

    def __del__(self):
        if hasattr(self, "handle") and self.handle:
            _check_call(_LIB.XGDMatrixFree(self.handle))
            self.handle = None

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
        length = c_bst_ulong()
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
            a numpy array of unsigned integer information of the data
        """
        length = c_bst_ulong()
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
            The array of data to be set
        """
        data, _, _ = _convert_dataframes(data, None, None, field, 'float')
        if isinstance(data, np.ndarray):
            self.set_float_info_npy2d(field, data)
            return
        c_data = c_array(ctypes.c_float, data)
        _check_call(_LIB.XGDMatrixSetFloatInfo(self.handle,
                                               c_str(field),
                                               c_data,
                                               c_bst_ulong(len(data))))

    def set_float_info_npy2d(self, field, data):
        """Set float type property into the DMatrix
           for numpy 2d array input

        Parameters
        ----------
        field: str
            The field name of the information

        data: numpy array
            The array of data to be set
        """
        data = _maybe_np_slice(data, np.float32)
        c_data = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        _check_call(_LIB.XGDMatrixSetFloatInfo(self.handle,
                                               c_str(field),
                                               c_data,
                                               c_bst_ulong(len(data))))

    def set_uint_info(self, field, data):
        """Set uint type property into the DMatrix.

        Parameters
        ----------
        field: str
            The field name of the information

        data: numpy array
            The array of data to be set
        """
        data = _maybe_np_slice(data, np.uint32)
        data, _, _ = _convert_dataframes(data, None, None, field, 'uint32')
        data = np.array(data, copy=False, dtype=ctypes.c_uint)
        _check_call(_LIB.XGDMatrixSetUIntInfo(self.handle,
                                              c_str(field),
                                              c_array(ctypes.c_uint, data),
                                              c_bst_ulong(len(data))))

    def set_interface_info(self, field, data):
        """Set info type property into DMatrix."""
        # If we are passed a dataframe, extract the series
        if CUDF_INSTALLED and isinstance(data, CUDF_DataFrame):
            if len(data.columns) != 1:
                raise ValueError(
                    'Expecting meta-info to contain a single column')
            data = data[data.columns[0]]

        interface = bytes(json.dumps([data.__cuda_array_interface__],
                                     indent=2), 'utf-8')
        _check_call(_LIB.XGDMatrixSetInfoFromInterface(self.handle,
                                                       c_str(field),
                                                       interface))

    def save_binary(self, fname, silent=True):
        """Save DMatrix to an XGBoost buffer.  Saved binary can be later loaded
        by providing the path to :py:func:`xgboost.DMatrix` as input.

        Parameters
        ----------
        fname : string or os.PathLike
            Name of the output buffer file.
        silent : bool (optional; default: True)
            If set, the output is suppressed.
        """
        _check_call(_LIB.XGDMatrixSaveBinary(self.handle,
                                             c_str(os_fspath(fname)),
                                             ctypes.c_int(silent)))

    def set_label(self, label):
        """Set label of dmatrix

        Parameters
        ----------
        label: array like
            The label information to be set into DMatrix
        """
        if _has_cuda_array_interface(label):
            self.set_interface_info('label', label)
        else:
            self.set_float_info('label', label)

    def set_weight(self, weight):
        """Set weight of each instance.

        Parameters
        ----------
        weight : array like
            Weight for each data point

            .. note:: For ranking task, weights are per-group.

                In ranking task, one weight is assigned to each group (not each
                data point). This is because we only care about the relative
                ordering of data points within each group, so it doesn't make
                sense to assign weights to individual data points.

        """
        if _has_cuda_array_interface(weight):
            self.set_interface_info('weight', weight)
        else:
            self.set_float_info('weight', weight)

    def set_base_margin(self, margin):
        """Set base margin of booster to start from.

        This can be used to specify a prediction value of existing model to be
        base_margin However, remember margin is needed, instead of transformed
        prediction e.g. for logistic regression: need to put in value before
        logistic transformation see also example/demo.py

        Parameters
        ----------
        margin: array like
            Prediction margin of each datapoint

        """
        if _has_cuda_array_interface(margin):
            self.set_interface_info('base_margin', margin)
        else:
            self.set_float_info('base_margin', margin)

    def set_group(self, group):
        """Set group size of DMatrix (used for ranking).

        Parameters
        ----------
        group : array like
            Group size of each group
        """
        if _has_cuda_array_interface(group):
            self.set_interface_info('group', group)
        else:
            self.set_uint_info('group', group)

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
        ret = c_bst_ulong()
        _check_call(_LIB.XGDMatrixNumRow(self.handle,
                                         ctypes.byref(ret)))
        return ret.value

    def num_col(self):
        """Get the number of columns (features) in the DMatrix.

        Returns
        -------
        number of columns : int
        """
        ret = c_bst_ulong()
        _check_call(_LIB.XGDMatrixNumCol(self.handle,
                                         ctypes.byref(ret)))
        return ret.value

    def slice(self, rindex, allow_groups=False):
        """Slice the DMatrix and return a new DMatrix that only contains `rindex`.

        Parameters
        ----------
        rindex : list
            List of indices to be selected.
        allow_groups : boolean
            Allow slicing of a matrix with a groups attribute

        Returns
        -------
        res : DMatrix
            A new DMatrix containing only selected indices.
        """
        res = DMatrix(None, feature_names=self.feature_names,
                      feature_types=self.feature_types)
        res.handle = ctypes.c_void_p()
        _check_call(_LIB.XGDMatrixSliceDMatrixEx(self.handle,
                                                 c_array(ctypes.c_int, rindex),
                                                 c_bst_ulong(len(rindex)),
                                                 ctypes.byref(res.handle),
                                                 ctypes.c_int(1 if allow_groups else 0)))
        return res

    @property
    def feature_names(self):
        """Get feature names (column labels).

        Returns
        -------
        feature_names : list or None
        """
        if self._feature_names is None:
            self._feature_names = ['f{0}'.format(i)
                                   for i in range(self.num_col())]
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
            try:
                if not isinstance(feature_names, str):
                    feature_names = list(feature_names)
                else:
                    feature_names = [feature_names]
            except TypeError:
                feature_names = [feature_names]

            if len(feature_names) != len(set(feature_names)):
                raise ValueError('feature_names must be unique')
            if len(feature_names) != self.num_col():
                msg = 'feature_names must have the same length as data'
                raise ValueError(msg)
            # prohibit to use symbols may affect to parse. e.g. []<
            if not all(isinstance(f, STRING_TYPES) and
                       not any(x in f for x in set(('[', ']', '<')))
                       for f in feature_names):
                raise ValueError('feature_names must be string, and may not contain [, ] or <')
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
            if self._feature_names is None:
                msg = 'Unable to set feature types before setting names'
                raise ValueError(msg)

            if isinstance(feature_types, STRING_TYPES):
                # single string will be applied to all columns
                feature_types = [feature_types] * self.num_col()

            try:
                if not isinstance(feature_types, str):
                    feature_types = list(feature_types)
                else:
                    feature_types = [feature_types]
            except TypeError:
                feature_types = [feature_types]

            if len(feature_types) != self.num_col():
                msg = 'feature_types must have the same length as data'
                raise ValueError(msg)

            valid = ('int', 'float', 'i', 'q')
            if not all(isinstance(f, STRING_TYPES) and f in valid
                       for f in feature_types):
                raise ValueError('All feature_names must be {int, float, i, q}')
        self._feature_types = feature_types


class DeviceQuantileDMatrix(DMatrix):
    """Device memory Data Matrix used in XGBoost for training with tree_method='gpu_hist'. Do not
    use this for test/validation tasks as some information may be lost in quantisation. This
    DMatrix is primarily designed to save memory in training from device memory inputs by
    avoiding intermediate storage. Implementation does not currently consider weights in
    quantisation process(unlike DMatrix). Set max_bin to control the number of bins during
    quantisation.

    You can construct DeviceQuantileDMatrix from cupy/cudf/dlpack.

    .. versionadded:: 1.1.0

    """

    def __init__(self, data, label=None, weight=None, base_margin=None,
                 missing=None,
                 silent=False,
                 feature_names=None,
                 feature_types=None,
                 nthread=None, max_bin=256):
        self.max_bin = max_bin
        if not (hasattr(data, "__cuda_array_interface__") or (
                CUDF_INSTALLED and isinstance(data, CUDF_DataFrame)) or _is_dlpack(data)):
            raise ValueError('Only cupy/cudf/dlpack currently supported for DeviceQuantileDMatrix')

        super().__init__(data, label=label, weight=weight, base_margin=base_margin,
                         missing=missing,
                         silent=silent,
                         feature_names=feature_names,
                         feature_types=feature_types,
                         nthread=nthread)

    def _init_from_array_interface_columns(self, df, missing, nthread):
        """Initialize DMatrix from columnar memory format."""
        interfaces_str = _cudf_array_interfaces(df)
        handle = ctypes.c_void_p()
        missing = missing if missing is not None else np.nan
        nthread = nthread if nthread is not None else 1
        _check_call(
            _LIB.XGDeviceQuantileDMatrixCreateFromArrayInterfaceColumns(
                interfaces_str,
                ctypes.c_float(missing), ctypes.c_int(nthread),
                ctypes.c_int(self.max_bin), ctypes.byref(handle)))
        self.handle = handle

    def _init_from_array_interface(self, data, missing, nthread):
        """Initialize DMatrix from cupy ndarray."""
        interface = data.__cuda_array_interface__
        if 'mask' in interface:
            interface['mask'] = interface['mask'].__cuda_array_interface__
        interface_str = bytes(json.dumps(interface, indent=2), 'utf-8')

        handle = ctypes.c_void_p()
        missing = missing if missing is not None else np.nan
        nthread = nthread if nthread is not None else 1
        _check_call(
            _LIB.XGDeviceQuantileDMatrixCreateFromArrayInterface(
                interface_str,
                ctypes.c_float(missing), ctypes.c_int(nthread),
                ctypes.c_int(self.max_bin), ctypes.byref(handle)))
        self.handle = handle


class Booster(object):
    # pylint: disable=too-many-public-methods
    """A Booster of XGBoost.

    Booster is the model of xgboost, that contains low level routines for
    training, prediction and evaluation.
    """

    feature_names = None

    def __init__(self, params=None, cache=(), model_file=None):
        # pylint: disable=invalid-name
        """
        Parameters
        ----------
        params : dict
            Parameters for boosters.
        cache : list
            List of cache items.
        model_file : string or os.PathLike
            Path to the model file.
        """
        for d in cache:
            if not isinstance(d, DMatrix):
                raise TypeError('invalid cache item: {}'.format(type(d).__name__), cache)
            self._validate_features(d)

        dmats = c_array(ctypes.c_void_p, [d.handle for d in cache])
        self.handle = ctypes.c_void_p()
        _check_call(_LIB.XGBoosterCreate(dmats, c_bst_ulong(len(cache)),
                                         ctypes.byref(self.handle)))
        params = params or {}
        if isinstance(params, list):
            params.append(('validate_parameters', True))
        else:
            params['validate_parameters'] = True

        self.set_param(params or {})
        if (params is not None) and ('booster' in params):
            self.booster = params['booster']
        else:
            self.booster = 'gbtree'
        if isinstance(model_file, Booster):
            assert self.handle is not None
            # We use the pickle interface for getting memory snapshot from
            # another model, and load the snapshot with this booster.
            state = model_file.__getstate__()
            handle = state['handle']
            del state['handle']
            ptr = (ctypes.c_char * len(handle)).from_buffer(handle)
            length = c_bst_ulong(len(handle))
            _check_call(
                _LIB.XGBoosterUnserializeFromBuffer(self.handle, ptr, length))
            self.__dict__.update(state)
        elif isinstance(model_file, (STRING_TYPES, os_PathLike, bytearray)):
            self.load_model(model_file)
        elif model_file is None:
            pass
        else:
            raise TypeError('Unknown type:', model_file)

    def __del__(self):
        if hasattr(self, 'handle') and self.handle is not None:
            _check_call(_LIB.XGBoosterFree(self.handle))
            self.handle = None

    def __getstate__(self):
        # can't pickle ctypes pointers, put model content in bytearray
        this = self.__dict__.copy()
        handle = this['handle']
        if handle is not None:
            length = c_bst_ulong()
            cptr = ctypes.POINTER(ctypes.c_char)()
            _check_call(_LIB.XGBoosterSerializeToBuffer(self.handle,
                                                        ctypes.byref(length),
                                                        ctypes.byref(cptr)))
            buf = ctypes2buffer(cptr, length.value)
            this["handle"] = buf
        return this

    def __setstate__(self, state):
        # reconstruct handle from raw data
        handle = state['handle']
        if handle is not None:
            buf = handle
            dmats = c_array(ctypes.c_void_p, [])
            handle = ctypes.c_void_p()
            _check_call(_LIB.XGBoosterCreate(
                dmats, c_bst_ulong(0), ctypes.byref(handle)))
            length = c_bst_ulong(len(buf))
            ptr = (ctypes.c_char * len(buf)).from_buffer(buf)
            _check_call(
                _LIB.XGBoosterUnserializeFromBuffer(handle, ptr, length))
            state['handle'] = handle
        self.__dict__.update(state)

    def save_config(self):
        '''Output internal parameter configuration of Booster as a JSON
        string.

        .. versionadded:: 1.0.0
        '''
        json_string = ctypes.c_char_p()
        length = c_bst_ulong()
        _check_call(_LIB.XGBoosterSaveJsonConfig(
            self.handle,
            ctypes.byref(length),
            ctypes.byref(json_string)))
        json_string = json_string.value.decode()
        return json_string

    def load_config(self, config):
        '''Load configuration returned by `save_config`.

        .. versionadded:: 1.0.0
        '''
        assert isinstance(config, str)
        _check_call(_LIB.XGBoosterLoadJsonConfig(
            self.handle,
            c_str(config)))

    def __copy__(self):
        return self.__deepcopy__(None)

    def __deepcopy__(self, _):
        '''Return a copy of booster.'''
        return Booster(model_file=self)

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
        return None

    def attributes(self):
        """Get attributes stored in the Booster as a dictionary.

        Returns
        -------
        result : dictionary of  attribute_name: attribute_value pairs of strings.
            Returns an empty dict if there's no attributes.
        """
        length = c_bst_ulong()
        sarr = ctypes.POINTER(ctypes.c_char_p)()
        _check_call(_LIB.XGBoosterGetAttrNames(self.handle,
                                               ctypes.byref(length),
                                               ctypes.byref(sarr)))
        attr_names = from_cstr_to_pystr(sarr, length)
        return {n: self.attr(n) for n in attr_names}

    def set_attr(self, **kwargs):
        """Set the attribute of the Booster.

        Parameters
        ----------
        **kwargs
            The attributes to set. Setting a value to None deletes an attribute.
        """
        for key, value in kwargs.items():
            if value is not None:
                if not isinstance(value, STRING_TYPES):
                    raise ValueError("Set Attr only accepts string values")
                value = c_str(str(value))
            _check_call(_LIB.XGBoosterSetAttr(
                self.handle, c_str(key), value))

    def set_param(self, params, value=None):
        """Set parameters into the Booster.

        Parameters
        ----------
        params: dict/list/str
           list of key,value pairs, dict of key to value or simply str key
        value: optional
           value of the specified parameter, when params is str key
        """
        if isinstance(params, Mapping):
            params = params.items()
        elif isinstance(params, STRING_TYPES) and value is not None:
            params = [(params, value)]
        for key, val in params:
            if val is not None:
                _check_call(_LIB.XGBoosterSetParam(self.handle, c_str(key),
                                                   c_str(str(val))))

    def update(self, dtrain, iteration, fobj=None):
        """Update for one iteration, with objective function calculated
        internally.  This function should not be called directly by users.

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
            raise TypeError('invalid training matrix: {}'.format(
                type(dtrain).__name__))
        self._validate_features(dtrain)

        if fobj is None:
            _check_call(_LIB.XGBoosterUpdateOneIter(self.handle,
                                                    ctypes.c_int(iteration),
                                                    dtrain.handle))
        else:
            pred = self.predict(dtrain, output_margin=True, training=True)
            grad, hess = fobj(pred, dtrain)
            self.boost(dtrain, grad, hess)

    def boost(self, dtrain, grad, hess):
        """Boost the booster for one iteration, with customized gradient
        statistics.  Like :func:`xgboost.core.Booster.update`, this
        function should not be called directly by users.

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
                                               c_bst_ulong(len(grad))))

    def eval_set(self, evals, iteration=0, feval=None):
        # pylint: disable=invalid-name
        """Evaluate a set of data.

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
        for d in evals:
            if not isinstance(d[0], DMatrix):
                raise TypeError('expected DMatrix, got {}'.format(
                    type(d[0]).__name__))
            if not isinstance(d[1], STRING_TYPES):
                raise TypeError('expected string, got {}'.format(
                    type(d[1]).__name__))
            self._validate_features(d[0])

        dmats = c_array(ctypes.c_void_p, [d[0].handle for d in evals])
        evnames = c_array(ctypes.c_char_p, [c_str(d[1]) for d in evals])
        msg = ctypes.c_char_p()
        _check_call(_LIB.XGBoosterEvalOneIter(self.handle,
                                              ctypes.c_int(iteration),
                                              dmats, evnames,
                                              c_bst_ulong(len(evals)),
                                              ctypes.byref(msg)))
        res = msg.value.decode()
        if feval is not None:
            for dmat, evname in evals:
                feval_ret = feval(self.predict(dmat, training=False), dmat)
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

    # pylint: disable=too-many-function-args
    def predict(self,
                data,
                output_margin=False,
                ntree_limit=0,
                pred_leaf=False,
                pred_contribs=False,
                approx_contribs=False,
                pred_interactions=False,
                validate_features=True,
                training=False):
        """Predict with data.

        .. note:: This function is not thread safe except for ``gbtree``
                  booster.

          For ``gbtree`` booster, the thread safety is guaranteed by locks.
          For lock free prediction use ``inplace_predict`` instead.  Also, the
          safety does not hold when used in conjunction with other methods.

          When using booster other than ``gbtree``, predict can only be called
          from one thread.  If you want to run prediction using multiple
          thread, call ``bst.copy()`` to make copies of model object and then
          call ``predict()``.

        Parameters
        ----------
        data : DMatrix
            The dmatrix storing the input.

        output_margin : bool
            Whether to output the raw untransformed margin value.

        ntree_limit : int
            Limit number of trees in the prediction; defaults to 0 (use all
            trees).

        pred_leaf : bool
            When this option is on, the output will be a matrix of (nsample,
            ntrees) with each record indicating the predicted leaf index of
            each sample in each tree.  Note that the leaf index of a tree is
            unique per tree, so you may find leaf 1 in both tree 1 and tree 0.

        pred_contribs : bool
            When this is True the output will be a matrix of size (nsample,
            nfeats + 1) with each record indicating the feature contributions
            (SHAP values) for that prediction. The sum of all feature
            contributions is equal to the raw untransformed margin value of the
            prediction. Note the final column is the bias term.

        approx_contribs : bool
            Approximate the contributions of each feature

        pred_interactions : bool
            When this is True the output will be a matrix of size (nsample,
            nfeats + 1, nfeats + 1) indicating the SHAP interaction values for
            each pair of features. The sum of each row (or column) of the
            interaction values equals the corresponding SHAP value (from
            pred_contribs), and the sum of the entire matrix equals the raw
            untransformed margin value of the prediction. Note the last row and
            column correspond to the bias term.

        validate_features : bool
            When this is True, validate that the Booster's and data's
            feature_names are identical.  Otherwise, it is assumed that the
            feature_names are the same.

        training : bool
            Whether the prediction value is used for training.  This can effect
            `dart` booster, which performs dropouts during training iterations.

            .. versionadded:: 1.0.0

        .. note:: Using ``predict()`` with DART booster

          If the booster object is DART type, ``predict()`` will not perform
          dropouts, i.e. all the trees will be evaluated.  If you want to
          obtain result with dropouts, provide `training=True`.

        Returns
        -------
        prediction : numpy array

        """
        option_mask = 0x00
        if output_margin:
            option_mask |= 0x01
        if pred_leaf:
            option_mask |= 0x02
        if pred_contribs:
            option_mask |= 0x04
        if approx_contribs:
            option_mask |= 0x08
        if pred_interactions:
            option_mask |= 0x10

        if not isinstance(data, DMatrix):
            raise TypeError('Expecting data to be a DMatrix object, got: ',
                            type(data))

        if validate_features:
            self._validate_features(data)

        length = c_bst_ulong()
        preds = ctypes.POINTER(ctypes.c_float)()
        _check_call(_LIB.XGBoosterPredict(self.handle, data.handle,
                                          ctypes.c_int(option_mask),
                                          ctypes.c_uint(ntree_limit),
                                          ctypes.c_int(training),
                                          ctypes.byref(length),
                                          ctypes.byref(preds)))
        preds = ctypes2numpy(preds, length.value, np.float32)
        if pred_leaf:
            preds = preds.astype(np.int32)
        nrow = data.num_row()
        if preds.size != nrow and preds.size % nrow == 0:
            chunk_size = int(preds.size / nrow)

            if pred_interactions:
                ngroup = int(chunk_size / ((data.num_col() + 1) *
                                           (data.num_col() + 1)))
                if ngroup == 1:
                    preds = preds.reshape(nrow,  # pylint: disable=too-many-function-args
                                          data.num_col() + 1,
                                          data.num_col() + 1)
                else:
                    preds = preds.reshape(nrow, ngroup,
                                          data.num_col() + 1,
                                          data.num_col() + 1)
            elif pred_contribs:
                ngroup = int(chunk_size / (data.num_col() + 1))
                if ngroup == 1:
                    preds = preds.reshape(nrow, data.num_col() + 1)
                else:
                    preds = preds.reshape(nrow, ngroup, data.num_col() + 1)
            else:
                preds = preds.reshape(nrow, chunk_size)
        return preds

    def inplace_predict(self, data, iteration_range=(0, 0),
                        predict_type='value', missing=np.nan):
        '''Run prediction in-place, Unlike ``predict`` method, inplace prediction does
        not cache the prediction result.

        Calling only ``inplace_predict`` in multiple threads is safe and lock
        free.  But the safety does not hold when used in conjunction with other
        methods. E.g. you can't train the booster in one thread and perform
        prediction in the other.

        .. code-block:: python

            booster.set_param({'predictor': 'gpu_predictor'})
            booster.inplace_predict(cupy_array)

            booster.set_param({'predictor': 'cpu_predictor})
            booster.inplace_predict(numpy_array)

        .. versionadded:: 1.1.0

        Parameters
        ----------
        data : numpy.ndarray/scipy.sparse.csr_matrix/cupy.ndarray/
               cudf.DataFrame/pd.DataFrame
            The input data, must not be a view for numpy array.  Set
            ``predictor`` to ``gpu_predictor`` for running prediction on CuPy
            array or CuDF DataFrame.
        iteration_range : tuple
            Specifies which layer of trees are used in prediction.  For
            example, if a random forest is trained with 100 rounds.  Specifying
            `iteration_range=(10, 20)`, then only the forests built during [10,
            20) (open set) rounds are used in this prediction.
        predict_type : str
            * `value` Output model prediction values.
            * `margin` Output the raw untransformed margin value.
        missing : float
            Value in the input data which needs to be present as a missing
            value.

        Returns
        -------
        prediction : numpy.ndarray/cupy.ndarray
            The prediction result.  When input data is on GPU, prediction
            result is stored in a cupy array.

        '''

        def reshape_output(predt, rows):
            '''Reshape for multi-output prediction.'''
            if predt.size != rows and predt.size % rows == 0:
                cols = int(predt.size / rows)
                predt = predt.reshape(rows, cols)
                return predt
            return predt

        length = c_bst_ulong()
        preds = ctypes.POINTER(ctypes.c_float)()
        iteration_range = (ctypes.c_uint(iteration_range[0]),
                           ctypes.c_uint(iteration_range[1]))

        # once caching is supported, we can pass id(data) as cache id.
        if isinstance(data, DataFrame):
            data = data.values
        if isinstance(data, np.ndarray):
            assert data.flags.c_contiguous
            arr = np.array(data.reshape(data.size), copy=False,
                           dtype=np.float32)
            _check_call(_LIB.XGBoosterPredictFromDense(
                self.handle,
                arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                c_bst_ulong(data.shape[0]),
                c_bst_ulong(data.shape[1]),
                ctypes.c_float(missing),
                iteration_range[0],
                iteration_range[1],
                c_str(predict_type),
                c_bst_ulong(0),
                ctypes.byref(length),
                ctypes.byref(preds)
            ))
            preds = ctypes2numpy(preds, length.value, np.float32)
            rows = data.shape[0]
            return reshape_output(preds, rows)
        if isinstance(data, scipy.sparse.csr_matrix):
            csr = data
            _check_call(_LIB.XGBoosterPredictFromCSR(
                self.handle,
                c_array(ctypes.c_size_t, csr.indptr),
                c_array(ctypes.c_uint, csr.indices),
                c_array(ctypes.c_float, csr.data),
                ctypes.c_size_t(len(csr.indptr)),
                ctypes.c_size_t(len(csr.data)),
                ctypes.c_size_t(csr.shape[1]),
                ctypes.c_float(missing),
                iteration_range[0],
                iteration_range[1],
                c_str(predict_type),
                c_bst_ulong(0),
                ctypes.byref(length),
                ctypes.byref(preds)))
            preds = ctypes2numpy(preds, length.value, np.float32)
            rows = data.shape[0]
            return reshape_output(preds, rows)
        if lazy_isinstance(data, 'cupy.core.core', 'ndarray'):
            assert data.flags.c_contiguous
            interface = data.__cuda_array_interface__
            if 'mask' in interface:
                interface['mask'] = interface['mask'].__cuda_array_interface__
            interface_str = bytes(json.dumps(interface, indent=2), 'utf-8')
            _check_call(_LIB.XGBoosterPredictFromArrayInterface(
                self.handle,
                interface_str,
                ctypes.c_float(missing),
                iteration_range[0],
                iteration_range[1],
                c_str(predict_type),
                c_bst_ulong(0),
                ctypes.byref(length),
                ctypes.byref(preds)))
            mem = ctypes2cupy(preds, length, np.float32)
            rows = data.shape[0]
            return reshape_output(mem, rows)
        if lazy_isinstance(data, 'cudf.core.dataframe', 'DataFrame'):
            interfaces_str = _cudf_array_interfaces(data)
            _check_call(_LIB.XGBoosterPredictFromArrayInterfaceColumns(
                self.handle,
                interfaces_str,
                ctypes.c_float(missing),
                iteration_range[0],
                iteration_range[1],
                c_str(predict_type),
                c_bst_ulong(0),
                ctypes.byref(length),
                ctypes.byref(preds)))
            mem = ctypes2cupy(preds, length, np.float32)
            rows = data.shape[0]
            predt = reshape_output(mem, rows)
            return predt

        raise TypeError('Data type:' + str(type(data)) +
                        ' not supported by inplace prediction.')

    def save_model(self, fname):
        """Save the model to a file.

        The model is saved in an XGBoost internal format which is universal
        among the various XGBoost interfaces. Auxiliary attributes of the
        Python Booster object (such as feature_names) will not be saved.  To
        preserve all attributes, pickle the Booster object.

        Parameters
        ----------
        fname : string or os.PathLike
            Output file name

        """
        if isinstance(fname, (STRING_TYPES, os_PathLike)):  # assume file name
            _check_call(_LIB.XGBoosterSaveModel(
                self.handle, c_str(os_fspath(fname))))
        else:
            raise TypeError("fname must be a string or os_PathLike")

    def save_raw(self):
        """Save the model to a in memory buffer representation

        Returns
        -------
        a in memory buffer representation of the model
        """
        length = c_bst_ulong()
        cptr = ctypes.POINTER(ctypes.c_char)()
        _check_call(_LIB.XGBoosterGetModelRaw(self.handle,
                                              ctypes.byref(length),
                                              ctypes.byref(cptr)))
        return ctypes2buffer(cptr, length.value)

    def load_model(self, fname):
        """Load the model from a file or bytearray. Path to file can be local
        or as an URI.

        The model is loaded from an XGBoost format which is universal among the
        various XGBoost interfaces. Auxiliary attributes of the Python Booster
        object (such as feature_names) will not be loaded.  To preserve all
        attributes, pickle the Booster object.

        Parameters
        ----------
        fname : string, os.PathLike, or a memory buffer
            Input file name or memory buffer(see also save_raw)

        """
        if isinstance(fname, (STRING_TYPES, os_PathLike)):
            # assume file name, cannot use os.path.exist to check, file can be
            # from URL.
            _check_call(_LIB.XGBoosterLoadModel(
                self.handle, c_str(os_fspath(fname))))
        elif isinstance(fname, bytearray):
            buf = fname
            length = c_bst_ulong(len(buf))
            ptr = (ctypes.c_char * len(buf)).from_buffer(buf)
            _check_call(_LIB.XGBoosterLoadModelFromBuffer(self.handle, ptr,
                                                          length))
        else:
            raise TypeError('Unknown file type: ', fname)

    def dump_model(self, fout, fmap='', with_stats=False, dump_format="text"):
        """Dump model into a text or JSON file.

        Parameters
        ----------
        fout : string or os.PathLike
            Output file name.
        fmap : string or os.PathLike, optional
            Name of the file containing feature map names.
        with_stats : bool, optional
            Controls whether the split statistics are output.
        dump_format : string, optional
            Format of model dump file. Can be 'text' or 'json'.
        """
        if isinstance(fout, (STRING_TYPES, os_PathLike)):
            fout = open(os_fspath(fout), 'w')
            need_close = True
        else:
            need_close = False
        ret = self.get_dump(fmap, with_stats, dump_format)
        if dump_format == 'json':
            fout.write('[\n')
            for i, _ in enumerate(ret):
                fout.write(ret[i])
                if i < len(ret) - 1:
                    fout.write(",\n")
            fout.write('\n]')
        else:
            for i, _ in enumerate(ret):
                fout.write('booster[{}]:\n'.format(i))
                fout.write(ret[i])
        if need_close:
            fout.close()

    def get_dump(self, fmap='', with_stats=False, dump_format="text"):
        """Returns the model dump as a list of strings.

        Parameters
        ----------
        fmap : string or os.PathLike, optional
            Name of the file containing feature map names.
        with_stats : bool, optional
            Controls whether the split statistics are output.
        dump_format : string, optional
            Format of model dump. Can be 'text', 'json' or 'dot'.
        """
        fmap = os_fspath(fmap)
        length = c_bst_ulong()
        sarr = ctypes.POINTER(ctypes.c_char_p)()
        if self.feature_names is not None and fmap == '':
            flen = len(self.feature_names)

            fname = from_pystr_to_cstr(self.feature_names)

            if self.feature_types is None:
                # use quantitative as default
                # {'q': quantitative, 'i': indicator}
                ftype = from_pystr_to_cstr(['q'] * flen)
            else:
                ftype = from_pystr_to_cstr(self.feature_types)
            _check_call(_LIB.XGBoosterDumpModelExWithFeatures(
                self.handle,
                ctypes.c_int(flen),
                fname,
                ftype,
                ctypes.c_int(with_stats),
                c_str(dump_format),
                ctypes.byref(length),
                ctypes.byref(sarr)))
        else:
            if fmap != '' and not os.path.exists(fmap):
                raise ValueError("No such file: {0}".format(fmap))
            _check_call(_LIB.XGBoosterDumpModelEx(self.handle,
                                                  c_str(fmap),
                                                  ctypes.c_int(with_stats),
                                                  c_str(dump_format),
                                                  ctypes.byref(length),
                                                  ctypes.byref(sarr)))
        res = from_cstr_to_pystr(sarr, length)
        return res

    def get_fscore(self, fmap=''):
        """Get feature importance of each feature.

        .. note:: Feature importance is defined only for tree boosters

            Feature importance is only defined when the decision tree model is chosen as base
            learner (`booster=gbtree`). It is not defined for other base learner types, such
            as linear learners (`booster=gblinear`).

        .. note:: Zero-importance features will not be included

           Keep in mind that this function does not include zero-importance feature, i.e.
           those features that have not been used in any split conditions.

        Parameters
        ----------
        fmap: str or os.PathLike (optional)
           The name of feature map file
        """

        return self.get_score(fmap, importance_type='weight')

    def get_score(self, fmap='', importance_type='weight'):
        """Get feature importance of each feature.
        Importance type can be defined as:

        * 'weight': the number of times a feature is used to split the data across all trees.
        * 'gain': the average gain across all splits the feature is used in.
        * 'cover': the average coverage across all splits the feature is used in.
        * 'total_gain': the total gain across all splits the feature is used in.
        * 'total_cover': the total coverage across all splits the feature is used in.

        .. note:: Feature importance is defined only for tree boosters

            Feature importance is only defined when the decision tree model is chosen as base
            learner (`booster=gbtree`). It is not defined for other base learner types, such
            as linear learners (`booster=gblinear`).

        Parameters
        ----------
        fmap: str or os.PathLike (optional)
           The name of feature map file.
        importance_type: str, default 'weight'
            One of the importance types defined above.
        """
        fmap = os_fspath(fmap)
        if getattr(self, 'booster', None) is not None and self.booster not in {'gbtree', 'dart'}:
            raise ValueError('Feature importance is not defined for Booster type {}'
                             .format(self.booster))

        allowed_importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
        if importance_type not in allowed_importance_types:
            msg = ("importance_type mismatch, got '{}', expected one of " +
                   repr(allowed_importance_types))
            raise ValueError(msg.format(importance_type))

        # if it's weight, then omap stores the number of missing values
        if importance_type == 'weight':
            # do a simpler tree dump to save time
            trees = self.get_dump(fmap, with_stats=False)
            fmap = {}
            for tree in trees:
                for line in tree.split('\n'):
                    # look for the opening square bracket
                    arr = line.split('[')
                    # if no opening bracket (leaf node), ignore this line
                    if len(arr) == 1:
                        continue

                    # extract feature name from string between []
                    fid = arr[1].split(']')[0].split('<')[0]

                    if fid not in fmap:
                        # if the feature hasn't been seen yet
                        fmap[fid] = 1
                    else:
                        fmap[fid] += 1

            return fmap

        average_over_splits = True
        if importance_type == 'total_gain':
            importance_type = 'gain'
            average_over_splits = False
        elif importance_type == 'total_cover':
            importance_type = 'cover'
            average_over_splits = False

        trees = self.get_dump(fmap, with_stats=True)

        importance_type += '='
        fmap = {}
        gmap = {}
        for tree in trees:
            for line in tree.split('\n'):
                # look for the opening square bracket
                arr = line.split('[')
                # if no opening bracket (leaf node), ignore this line
                if len(arr) == 1:
                    continue

                # look for the closing bracket, extract only info within that bracket
                fid = arr[1].split(']')

                # extract gain or cover from string after closing bracket
                g = float(fid[1].split(importance_type)[1].split(',')[0])

                # extract feature name from string before closing bracket
                fid = fid[0].split('<')[0]

                if fid not in fmap:
                    # if the feature hasn't been seen yet
                    fmap[fid] = 1
                    gmap[fid] = g
                else:
                    fmap[fid] += 1
                    gmap[fid] += g

        # calculate average value (gain/cover) for each feature
        if average_over_splits:
            for fid in gmap:
                gmap[fid] = gmap[fid] / fmap[fid]

        return gmap

    def trees_to_dataframe(self, fmap=''):
        """Parse a boosted tree model text dump into a pandas DataFrame structure.

        This feature is only defined when the decision tree model is chosen as base
        learner (`booster in {gbtree, dart}`). It is not defined for other base learner
        types, such as linear learners (`booster=gblinear`).

        Parameters
        ----------
        fmap: str or os.PathLike (optional)
           The name of feature map file.
        """
        # pylint: disable=too-many-locals
        fmap = os_fspath(fmap)
        if not PANDAS_INSTALLED:
            raise Exception(('pandas must be available to use this method.'
                             'Install pandas before calling again.'))

        if getattr(self, 'booster', None) is not None and self.booster not in {'gbtree', 'dart'}:
            raise ValueError('This method is not defined for Booster type {}'
                             .format(self.booster))

        tree_ids = []
        node_ids = []
        fids = []
        splits = []
        y_directs = []
        n_directs = []
        missings = []
        gains = []
        covers = []

        trees = self.get_dump(fmap, with_stats=True)
        for i, tree in enumerate(trees):
            for line in tree.split('\n'):
                arr = line.split('[')
                # Leaf node
                if len(arr) == 1:
                    # Last element of line.split is an empy string
                    if arr == ['']:
                        continue
                    # parse string
                    parse = arr[0].split(':')
                    stats = re.split('=|,', parse[1])

                    # append to lists
                    tree_ids.append(i)
                    node_ids.append(int(re.findall(r'\b\d+\b', parse[0])[0]))
                    fids.append('Leaf')
                    splits.append(float('NAN'))
                    y_directs.append(float('NAN'))
                    n_directs.append(float('NAN'))
                    missings.append(float('NAN'))
                    gains.append(float(stats[1]))
                    covers.append(float(stats[3]))
                # Not a Leaf Node
                else:
                    # parse string
                    fid = arr[1].split(']')
                    parse = fid[0].split('<')
                    stats = re.split('=|,', fid[1])

                    # append to lists
                    tree_ids.append(i)
                    node_ids.append(int(re.findall(r'\b\d+\b', arr[0])[0]))
                    fids.append(parse[0])
                    splits.append(float(parse[1]))
                    str_i = str(i)
                    y_directs.append(str_i + '-' + stats[1])
                    n_directs.append(str_i + '-' + stats[3])
                    missings.append(str_i + '-' + stats[5])
                    gains.append(float(stats[7]))
                    covers.append(float(stats[9]))

        ids = [str(t_id) + '-' + str(n_id) for t_id, n_id in zip(tree_ids, node_ids)]
        df = DataFrame({'Tree': tree_ids, 'Node': node_ids, 'ID': ids,
                        'Feature': fids, 'Split': splits, 'Yes': y_directs,
                        'No': n_directs, 'Missing': missings, 'Gain': gains,
                        'Cover': covers})

        if callable(getattr(df, 'sort_values', None)):
            # pylint: disable=no-member
            return df.sort_values(['Tree', 'Node']).reset_index(drop=True)
        # pylint: disable=no-member
        return df.sort(['Tree', 'Node']).reset_index(drop=True)

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

                if dat_missing:
                    msg += ('\nexpected ' + ', '.join(
                        str(s) for s in dat_missing) + ' in input data')

                if my_missing:
                    msg += ('\ntraining data did not have the following fields: ' +
                            ', '.join(str(s) for s in my_missing))

                raise ValueError(msg.format(self.feature_names,
                                            data.feature_names))

    def get_split_value_histogram(self, feature, fmap='', bins=None,
                                  as_pandas=True):
        """Get split value histogram of a feature

        Parameters
        ----------
        feature: str
            The name of the feature.
        fmap: str or os.PathLike (optional)
            The name of feature map file.
        bin: int, default None
            The maximum number of bins.
            Number of bins equals number of unique split values n_unique,
            if bins == None or bins > n_unique.
        as_pandas: bool, default True
            Return pd.DataFrame when pandas is installed.
            If False or pandas is not installed, return numpy ndarray.

        Returns
        -------
        a histogram of used splitting values for the specified feature
        either as numpy array or pandas DataFrame.
        """
        xgdump = self.get_dump(fmap=fmap)
        values = []
        regexp = re.compile(r"\[{0}<([\d.Ee+-]+)\]".format(feature))
        for i, _ in enumerate(xgdump):
            m = re.findall(regexp, xgdump[i])
            values.extend([float(x) for x in m])

        n_unique = len(np.unique(values))
        bins = max(min(n_unique, bins) if bins is not None else n_unique, 1)

        nph = np.histogram(values, bins=bins)
        nph = np.column_stack((nph[1][1:], nph[0]))
        nph = nph[nph[:, 1] > 0]

        if as_pandas and PANDAS_INSTALLED:
            return DataFrame(nph, columns=['SplitValue', 'Count'])
        if as_pandas and not PANDAS_INSTALLED:
            sys.stderr.write(
                "Returning histogram as ndarray (as_pandas == True, but pandas is not installed).")
        return nph
