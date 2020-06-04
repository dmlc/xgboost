# pylint: disable=too-many-arguments, no-self-use
'''Data dispatching for DMatrix.'''
import ctypes
import abc
import json
import warnings

import numpy as np

from .core import c_array, _LIB, _check_call, c_str, _cudf_array_interfaces
from .compat import lazy_isinstance, STRING_TYPES, os_fspath, os_PathLike

c_bst_ulong = ctypes.c_uint64   # pylint: disable=invalid-name


class DataHandler(abc.ABC):
    '''Base class for various data handler.'''
    def __init__(self, missing, nthread, silent, meta=None, meta_type=None):
        self.missing = missing
        self.nthread = nthread
        self.silent = silent

        self.meta = meta
        self.meta_type = meta_type

    def _warn_unused_missing(self, data):
        if not (np.isnan(np.nan) or None):
            warnings.warn(
                '`missing` is not used for current input data type:' +
                str(type(data)))

    def check_complex(self, data):
        '''Test whether data is complex using `dtype` attribute.'''
        complex_dtypes = (np.complex128, np.complex64,
                          np.cfloat, np.cdouble, np.clongdouble)
        if hasattr(data, 'dtype') and data.dtype in complex_dtypes:
            raise ValueError('Complex data not supported')

    def transform(self, data):
        '''Optional method for transforming data before being accepted by
        other XGBoost API.'''
        return data, None, None

    @abc.abstractmethod
    def handle_input(self, data, feature_names, feature_types):
        '''Abstract method for handling different data input.'''


class DMatrixDataManager:
    '''The registry class for various data handler.'''
    def __init__(self):
        self.__data_handlers = {}
        self.__data_handlers_dly = []

    def register_handler(self, module, name, handler):
        '''Register a data handler handling specfic type of data.'''
        self.__data_handlers['.'.join([module, name])] = handler

    def register_handler_opaque(self, func, handler):
        '''Register a data handler that handles data with opaque type.

        Parameters
        ----------
        func : callable
            A function with a single parameter `data`.  It should return True
            if the handler can handle this data, otherwise returns False.
        handler : xgboost.data.DataHandler
            The handler class that is a subclass of `DataHandler`.
        '''
        self.__data_handlers_dly.append((func, handler))

    def get_handler(self, data):
        '''Get a handler of `data`, returns None if handler not found.'''
        module, name = type(data).__module__, type(data).__name__
        if '.'.join([module, name]) in self.__data_handlers.keys():
            handler = self.__data_handlers['.'.join([module, name])]
            return handler
        for f, handler in self.__data_handlers_dly:
            if f(data):
                return handler
        return None


__dmatrix_registry = DMatrixDataManager()  # pylint: disable=invalid-name


def get_dmatrix_data_handler(data, missing, nthread, silent,
                             meta=None, meta_type=None):
    '''Get a handler of `data` for DMatrix.

    .. versionadded:: 1.2.0

    Parameters
    ----------
    data : any
        The input data.
    missing : float
        Same as `missing` for DMatrix.
    nthread : int
        Same as `nthread` for DMatrix.
    silent : boolean
        Same as `silent` for DMatrix.
    meta : str
        Field name of meta data, like `label`.  Used only for getting handler
        for meta info.
    meta_type : str/np.dtype
        Type of meta data.

    Returns
    -------
    handler : DataHandler
    '''
    handler = __dmatrix_registry.get_handler(data)
    if handler is None:
        return None
    return handler(missing, nthread, silent, meta, meta_type)


class FileHandler(DataHandler):
    '''Handler of path like input.'''
    def handle_input(self, data, feature_names, feature_types):
        self._warn_unused_missing(data)
        handle = ctypes.c_void_p()
        _check_call(_LIB.XGDMatrixCreateFromFile(c_str(os_fspath(data)),
                                                 ctypes.c_int(self.silent),
                                                 ctypes.byref(handle)))
        return handle, feature_names, feature_types


__dmatrix_registry.register_handler_opaque(
    lambda data: isinstance(data, (STRING_TYPES, os_PathLike)),
    FileHandler)


class CSRHandler(DataHandler):
    '''Handler of `scipy.sparse.csr.csr_matrix`.'''
    def handle_input(self, data, feature_names, feature_types):
        '''Initialize data from a CSR matrix.'''
        if len(data.indices) != len(data.data):
            raise ValueError('length mismatch: {} vs {}'.format(
                len(data.indices), len(data.data)))
        self._warn_unused_missing(data)
        handle = ctypes.c_void_p()
        _check_call(_LIB.XGDMatrixCreateFromCSREx(
            c_array(ctypes.c_size_t, data.indptr),
            c_array(ctypes.c_uint, data.indices),
            c_array(ctypes.c_float, data.data),
            ctypes.c_size_t(len(data.indptr)),
            ctypes.c_size_t(len(data.data)),
            ctypes.c_size_t(data.shape[1]),
            ctypes.byref(handle)))
        return handle, feature_names, feature_types


__dmatrix_registry.register_handler(
    'scipy.sparse.csr', 'csr_matrix', CSRHandler)


class CSCHandler(DataHandler):
    '''Handler of `scipy.sparse.csc.csc_matrix`.'''
    def handle_input(self, data, feature_names, feature_types):
        if len(data.indices) != len(data.data):
            raise ValueError('length mismatch: {} vs {}'.format(
                len(data.indices), len(data.data)))
        self._warn_unused_missing(data)
        handle = ctypes.c_void_p()
        _check_call(_LIB.XGDMatrixCreateFromCSCEx(
            c_array(ctypes.c_size_t, data.indptr),
            c_array(ctypes.c_uint, data.indices),
            c_array(ctypes.c_float, data.data),
            ctypes.c_size_t(len(data.indptr)),
            ctypes.c_size_t(len(data.data)),
            ctypes.c_size_t(data.shape[0]),
            ctypes.byref(handle)))
        return handle, feature_names, feature_types


__dmatrix_registry.register_handler(
    'scipy.sparse.csc', 'csc_matrix', CSCHandler)


class NumpyHandler(DataHandler):
    '''Handler of `numpy.ndarray`.'''
    def _maybe_np_slice(self, data, dtype):
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

    def transform(self, data):
        return self._maybe_np_slice(data, self.meta_type), None, None

    def handle_input(self, data, feature_names, feature_types):
        """Initialize data from a 2-D numpy matrix.

        If ``mat`` does not have ``order='C'`` (aka row-major) or is
        not contiguous, a temporary copy will be made.

        If ``mat`` does not have ``dtype=numpy.float32``, a temporary copy will
        be made.

        So there could be as many as two temporary data copies; be mindful of
        input layout and type if memory use is a concern.

        """
        if not isinstance(data, np.ndarray) and hasattr(data, '__array__'):
            data = np.array(data, copy=False)
        if len(data.shape) != 2:
            raise ValueError('Expecting 2 dimensional numpy.ndarray, got: ',
                             data.shape)
        # flatten the array by rows and ensure it is float32.  we try to avoid
        # data copies if possible (reshape returns a view when possible and we
        # explicitly tell np.array to try and avoid copying)
        flatten = np.array(data.reshape(data.size), copy=False,
                           dtype=np.float32)
        flatten = self._maybe_np_slice(flatten, np.float32)
        self.check_complex(data)
        handle = ctypes.c_void_p()
        _check_call(_LIB.XGDMatrixCreateFromMat_omp(
            flatten.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            c_bst_ulong(data.shape[0]),
            c_bst_ulong(data.shape[1]),
            ctypes.c_float(self.missing),
            ctypes.byref(handle),
            ctypes.c_int(self.nthread)))
        return handle, feature_names, feature_types


__dmatrix_registry.register_handler('numpy', 'ndarray', NumpyHandler)
__dmatrix_registry.register_handler('numpy', 'matrix', NumpyHandler)
__dmatrix_registry.register_handler_opaque(
    lambda x: hasattr(x, '__array__'), NumpyHandler)


class ListHandler(NumpyHandler):
    '''Handler of builtin list and tuple'''
    def handle_input(self, data, feature_names, feature_types):
        assert self.meta is None, 'List input data is not supported for X'
        data = np.array(data)
        return super().handle_input(data, feature_names, feature_types)


__dmatrix_registry.register_handler('builtins', 'list', NumpyHandler)
__dmatrix_registry.register_handler('builtins', 'tuple', NumpyHandler)


class PandasHandler(NumpyHandler):
    '''Handler of data structures defined by `pandas`.'''
    pandas_dtype_mapper = {
        'int8': 'int',
        'int16': 'int',
        'int32': 'int',
        'int64': 'int',
        'uint8': 'int',
        'uint16': 'int',
        'uint32': 'int',
        'uint64': 'int',
        'float16': 'float',
        'float32': 'float',
        'float64': 'float',
        'bool': 'i'
    }

    def _maybe_pandas_data(self, data, feature_names, feature_types,
                           meta=None, meta_type=None):
        """Extract internal data from pd.DataFrame for DMatrix data"""
        if lazy_isinstance(data, 'pandas.core.series', 'Series'):
            dtype = meta_type if meta_type else 'float'
            return data.values.astype(dtype), feature_names, feature_types

        from pandas.api.types import is_sparse
        from pandas import MultiIndex, Int64Index

        data_dtypes = data.dtypes
        if not all(dtype.name in self.pandas_dtype_mapper or is_sparse(dtype)
                   for dtype in data_dtypes):
            bad_fields = [
                str(data.columns[i]) for i, dtype in enumerate(data_dtypes)
                if dtype.name not in self.pandas_dtype_mapper
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
                    feature_types.append(self.pandas_dtype_mapper[
                        dtype.subtype.name])
                else:
                    feature_types.append(self.pandas_dtype_mapper[dtype.name])

        if meta and len(data.columns) > 1:
            raise ValueError(
                'DataFrame for {meta} cannot have multiple columns'.format(
                    meta=meta))

        dtype = meta_type if meta_type else 'float'
        data = data.values.astype(dtype)

        return data, feature_names, feature_types

    def transform(self, data):
        return self._maybe_pandas_data(data, None, None, self.meta,
                                       self.meta_type)

    def handle_input(self, data, feature_names, feature_types):
        data, feature_names, feature_types = self._maybe_pandas_data(
            data, feature_names, feature_types, self.meta, self.meta_type)
        return super().handle_input(data, feature_names, feature_types)


__dmatrix_registry.register_handler(
    'pandas.core.frame', 'DataFrame', PandasHandler)
__dmatrix_registry.register_handler(
    'pandas.core.series', 'Series', PandasHandler)


class DTHandler(DataHandler):
    '''Handler of datatable.'''
    dt_type_mapper = {'bool': 'bool', 'int': 'int', 'real': 'float'}
    dt_type_mapper2 = {'bool': 'i', 'int': 'int', 'real': 'float'}

    def _maybe_dt_data(self, data, feature_names, feature_types,
                       meta=None, meta_type=None):
        """Validate feature names and types if data table"""
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
                      if type_name not in self.dt_type_mapper]
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
            feature_types = np.vectorize(self.dt_type_mapper2.get)(
                data_types_names)

        return data, feature_names, feature_types

    def transform(self, data):
        return self._maybe_dt_data(data, None, None, self.meta, self.meta_type)

    def handle_input(self, data, feature_names, feature_types):
        data, feature_names, feature_types = self._maybe_dt_data(
            data, feature_names, feature_types, self.meta, self.meta_type)

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

        self._warn_unused_missing(data)
        handle = ctypes.c_void_p()
        _check_call(_LIB.XGDMatrixCreateFromDT(
            ptrs, feature_type_strings,
            c_bst_ulong(data.shape[0]),
            c_bst_ulong(data.shape[1]),
            ctypes.byref(handle),
            ctypes.c_int(self.nthread)))
        return handle, feature_names, feature_types


__dmatrix_registry.register_handler('datatable', 'Frame', DTHandler)
__dmatrix_registry.register_handler('datatable', 'DataTable', DTHandler)


class CudaArrayInterfaceHandler(DataHandler):
    '''Handler of data with `__cuda_array_interface__` (cupy.ndarray).'''
    def handle_input(self, data, feature_names, feature_types):
        """Initialize DMatrix from cupy ndarray."""
        interface = data.__cuda_array_interface__
        if 'mask' in interface:
            interface['mask'] = interface['mask'].__cuda_array_interface__
        interface_str = bytes(json.dumps(interface, indent=2), 'utf-8')

        handle = ctypes.c_void_p()
        _check_call(
            _LIB.XGDMatrixCreateFromArrayInterface(
                interface_str,
                ctypes.c_float(self.missing),
                ctypes.c_int(self.nthread),
                ctypes.byref(handle)))
        return handle, feature_names, feature_types


__dmatrix_registry.register_handler('cupy.core.core', 'ndarray',
                                    CudaArrayInterfaceHandler)


class CudaColumnarHandler(DataHandler):
    '''Handler of CUDA based columnar data. (cudf.DataFrame)'''
    def _maybe_cudf_dataframe(self, data, feature_names, feature_types):
        """Extract internal data from cudf.DataFrame for DMatrix data."""
        if feature_names is None:
            if lazy_isinstance(data, 'cudf.core.series', 'Series'):
                feature_names = [data.name]
            elif lazy_isinstance(
                    data.columns, 'cudf.core.multiindex', 'MultiIndex'):
                feature_names = [
                    ' '.join([str(x) for x in i])
                    for i in data.columns
                ]
            else:
                feature_names = data.columns.format()
        if feature_types is None:
            if lazy_isinstance(data, 'cudf.core.series', 'Series'):
                dtypes = [data.dtype]
            else:
                dtypes = data.dtypes
            feature_types = [PandasHandler.pandas_dtype_mapper[d.name]
                             for d in dtypes]
        return data, feature_names, feature_types

    def transform(self, data):
        return self._maybe_cudf_dataframe(data, None, None)

    def handle_input(self, data, feature_names, feature_types):
        """Initialize DMatrix from columnar memory format."""
        data, feature_names, feature_types = self._maybe_cudf_dataframe(
            data, feature_names, feature_types)
        interfaces_str = _cudf_array_interfaces(data)
        handle = ctypes.c_void_p()
        _check_call(
            _LIB.XGDMatrixCreateFromArrayInterfaceColumns(
                interfaces_str,
                ctypes.c_float(self.missing),
                ctypes.c_int(self.nthread),
                ctypes.byref(handle)))
        return handle, feature_names, feature_types


__dmatrix_registry.register_handler('cudf.core.dataframe', 'DataFrame',
                                    CudaColumnarHandler)
__dmatrix_registry.register_handler('cudf.core.series', 'Series',
                                    CudaColumnarHandler)


class DLPackHandler(CudaArrayInterfaceHandler):
    '''Handler of `dlpack`.'''
    def _maybe_dlpack_data(self, data, feature_names, feature_types):
        from cupy import fromDlpack  # pylint: disable=E0401
        data = fromDlpack(data)
        return data, feature_names, feature_types

    def transform(self, data):
        return self._maybe_dlpack_data(data, None, None)

    def handle_input(self, data, feature_names, feature_types):
        data, feature_names, feature_types = self._maybe_dlpack_data(
            data, feature_names, feature_types)
        return super().handle_input(
            data, feature_names, feature_types)


__dmatrix_registry.register_handler_opaque(
    lambda x: 'PyCapsule' in str(type(x)) and "dltensor" in str(x),
    DLPackHandler)


class DeviceQuantileDMatrixDataHandler(DataHandler):  # pylint: disable=abstract-method
    '''Base class of data handler for `DeviceQuantileDMatrix`.'''
    def __init__(self, max_bin, missing, nthread, silent,
                 meta=None, meta_type=None):
        self.max_bin = max_bin
        super().__init__(missing, nthread, silent, meta, meta_type)


__device_quantile_dmatrix_registry = DMatrixDataManager()  # pylint: disable=invalid-name


def get_device_quantile_dmatrix_data_handler(
        data, max_bin, missing, nthread, silent):
    '''Get data handler for `DeviceQuantileDMatrix`. Similar to
    `get_dmatrix_data_handler`.

    .. versionadded:: 1.2.0

    '''
    handler = __device_quantile_dmatrix_registry.get_handler(
        data)
    assert handler, 'Current data type ' + str(type(data)) +\
        ' is not supported for DeviceQuantileDMatrix'
    return handler(max_bin, missing, nthread, silent)


class DeviceQuantileCudaArrayInterfaceHandler(
        DeviceQuantileDMatrixDataHandler):
    '''Handler of data with `__cuda_array_interface__`, for
    `DeviceQuantileDMatrix`.

    '''
    def handle_input(self, data, feature_names, feature_types):
        """Initialize DMatrix from cupy ndarray."""
        if not hasattr(data, '__cuda_array_interface__') and hasattr(
                data, '__array__'):
            import cupy         # pylint: disable=import-error
            data = cupy.array(data, copy=False)

        interface = data.__cuda_array_interface__
        if 'mask' in interface:
            interface['mask'] = interface['mask'].__cuda_array_interface__
        interface_str = bytes(json.dumps(interface, indent=2), 'utf-8')

        handle = ctypes.c_void_p()
        _check_call(
            _LIB.XGDeviceQuantileDMatrixCreateFromArrayInterface(
                interface_str,
                ctypes.c_float(self.missing), ctypes.c_int(self.nthread),
                ctypes.c_int(self.max_bin), ctypes.byref(handle)))
        return handle, feature_names, feature_types


__device_quantile_dmatrix_registry.register_handler(
    'cupy.core.core', 'ndarray', DeviceQuantileCudaArrayInterfaceHandler)


class DeviceQuantileCudaColumnarHandler(DeviceQuantileDMatrixDataHandler,
                                        CudaColumnarHandler):
    '''Handler of CUDA based columnar data, for `DeviceQuantileDMatrix`.'''
    def __init__(self, max_bin, missing, nthread, silent,
                 meta=None, meta_type=None):
        super().__init__(
            max_bin=max_bin, missing=missing, nthread=nthread, silent=silent,
            meta=meta, meta_type=meta_type
        )

    def handle_input(self, data, feature_names, feature_types):
        """Initialize Quantile Device DMatrix from columnar memory format."""
        data, feature_names, feature_types = self._maybe_cudf_dataframe(
            data, feature_names, feature_types)
        interfaces_str = _cudf_array_interfaces(data)
        handle = ctypes.c_void_p()
        _check_call(
            _LIB.XGDeviceQuantileDMatrixCreateFromArrayInterfaceColumns(
                interfaces_str,
                ctypes.c_float(self.missing), ctypes.c_int(self.nthread),
                ctypes.c_int(self.max_bin), ctypes.byref(handle)))
        return handle, feature_names, feature_types


__device_quantile_dmatrix_registry.register_handler(
    'cudf.core.dataframe', 'DataFrame', DeviceQuantileCudaColumnarHandler)
__device_quantile_dmatrix_registry.register_handler(
    'cudf.core.series', 'Series', DeviceQuantileCudaColumnarHandler)


class DeviceQuantileDLPackHandler(DeviceQuantileCudaArrayInterfaceHandler,
                                  DLPackHandler):
    '''Handler of `dlpack`, for `DeviceQuantileDMatrix`.'''
    def __init__(self, max_bin, missing, nthread, silent,
                 meta=None, meta_type=None):
        super().__init__(
            max_bin=max_bin, missing=missing, nthread=nthread, silent=silent,
            meta=meta, meta_type=meta_type
        )

    def handle_input(self, data, feature_names, feature_types):
        data, feature_names, feature_types = self._maybe_dlpack_data(
            data, feature_names, feature_types)
        return super().handle_input(
            data, feature_names, feature_types)


__device_quantile_dmatrix_registry.register_handler_opaque(
    lambda x: 'PyCapsule' in str(type(x)) and "dltensor" in str(x),
    DeviceQuantileDLPackHandler)
