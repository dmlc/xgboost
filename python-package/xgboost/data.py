import scipy
import ctypes
import abc
import json
import warnings
import numpy as np
from .core import c_array, _LIB, _check_call, c_str, _cudf_array_interfaces
from .compat import lazy_isinstance, STRING_TYPES, os_fspath, os_PathLike

c_bst_ulong = ctypes.c_uint64


# Notes:
# Existing handler functions:
# - csr
# - csc
# - numpy array
#   + 1d
#   + 2d
#   + slice
# - pandas
#   + SparseArray
#   + MultiIndex
#   + DataFrame
#   + Series
# - cudf
#   + MultiIndex
#   + Dataframe
#   + Series
# - dt
# - dlpack
# - cupy

# To be added:
# - Arrow

class DataHandler(abc.ABC):
    def __init__(self, missing, nthread, silent, meta=None, meta_type=None):
        self.missing = missing
        self.nthread = nthread
        self.silent = silent

        self.meta = meta
        self.meta_type = meta_type

    def _warn_unused_missing(self):
        if self.missing != np.nan:
            warnings.warn('`missing` is not used for current input data type.')

    def transform(self, data):
        '''Optional method for transforming data before being accepted by
        other XGBoost API.'''
        return data, None, None

    def handle_input(self, data, feature_names, feature_types):
        '''Abstract method for handling different data input.'''


class DMatrixDataManager:
    def __init__(self):
        self.__data_handlers = {}
        self.__data_handlers_dly = []

    def register_handler(self, module, name, handler):
        self.__data_handlers['.'.join([module, name])] = handler

    def register_handler_dly(self, func, handler):
        self.__data_handlers_dly.append((func, handler))

    def get_handler(self, data):
        module, name = type(data).__module__, type(data).__name__
        if '.'.join([module, name]) in self.__data_handlers.keys():
            handler = self.__data_handlers['.'.join([module, name])]
            return handler
        else:
            for f, handler in self.__data_handlers_dly:
                if f(data):
                    return handler
        return None


__dmatrix_registry = DMatrixDataManager()


class FileHandler(DataHandler):
    def handle_input(self, data, feature_names, feature_types):
        self._warn_unused_missing()
        handle = ctypes.c_void_p()
        _check_call(_LIB.XGDMatrixCreateFromFile(c_str(os_fspath(data)),
                                                 ctypes.c_int(self.silent),
                                                 ctypes.byref(handle)))
        return handle, feature_names, feature_types


__dmatrix_registry.register_handler_dly(
    lambda data: isinstance(data, (STRING_TYPES, os_PathLike)),
    FileHandler)


class CSRHandler(DataHandler):
    def handle_input(self, data, feature_names, feature_types):
        '''Initialize data from a CSR matrix.'''
        if len(data.indices) != len(data.data):
            raise ValueError('length mismatch: {} vs {}'.format(
                len(data.indices), len(data.data)))
        self._warn_unused_missing()
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
    def handle_input(self, csc, feature_names, feature_types):
        if len(csc.indices) != len(csc.data):
            raise ValueError('length mismatch: {} vs {}'.format(
                len(csc.indices), len(csc.data)))
        self._warn_unused_missing()
        handle = ctypes.c_void_p()
        _check_call(_LIB.XGDMatrixCreateFromCSCEx(
            c_array(ctypes.c_size_t, csc.indptr),
            c_array(ctypes.c_uint, csc.indices),
            c_array(ctypes.c_float, csc.data),
            ctypes.c_size_t(len(csc.indptr)),
            ctypes.c_size_t(len(csc.data)),
            ctypes.c_size_t(csc.shape[0]),
            ctypes.byref(handle)))
        return handle, feature_names, feature_types


__dmatrix_registry.register_handler(
    'scipy.sparse.csc', 'csc_matrix', CSCHandler)


class NumpyHandler(DataHandler):
    def _maybe_np_slice(self, data, dtype):
        '''Handle numpy slice.  This can be removed if we use __array_interface__.
        '''
        import numpy as np
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

    def handle_input(self, mat, feature_names, feature_types):
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
        import numpy as np
        # flatten the array by rows and ensure it is float32.  we try to avoid
        # data copies if possible (reshape returns a view when possible and we
        # explicitly tell np.array to try and avoid copying)
        data = np.array(mat.reshape(mat.size), copy=False, dtype=np.float32)
        data = self._maybe_np_slice(data, np.float32)
        handle = ctypes.c_void_p()
        _check_call(_LIB.XGDMatrixCreateFromMat_omp(
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            c_bst_ulong(mat.shape[0]),
            c_bst_ulong(mat.shape[1]),
            ctypes.c_float(self.missing),
            ctypes.byref(handle),
            ctypes.c_int(self.nthread)))
        return handle, feature_names, feature_types


__dmatrix_registry.register_handler('numpy', 'ndarray', NumpyHandler)


class PandasHandler(NumpyHandler):
    PANDAS_DTYPE_MAPPER = {
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
        from pandas.api.types import is_sparse
        from pandas import MultiIndex, Int64Index

        data_dtypes = data.dtypes
        if not all(dtype.name in self.PANDAS_DTYPE_MAPPER or is_sparse(dtype)
                   for dtype in data_dtypes):
            bad_fields = [
                str(data.columns[i]) for i, dtype in enumerate(data_dtypes)
                if dtype.name not in self.PANDAS_DTYPE_MAPPER
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
                    feature_types.append(self.PANDAS_DTYPE_MAPPER[
                        dtype.subtype.name])
                else:
                    feature_types.append(self.PANDAS_DTYPE_MAPPER[dtype.name])

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


class DTHandler(DataHandler):
    DT_TYPE_MAPPER = {'bool': 'bool', 'int': 'int', 'real': 'float'}
    DT_TYPE_MAPPER2 = {'bool': 'i', 'int': 'int', 'real': 'float'}

    def _maybe_dt_data(self, data, feature_names, feature_types,
                       meta=None, meta_type=None):
        """Validate feature names and types if data table"""
        import numpy as np
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
                      if type_name not in self.DT_TYPE_MAPPER]
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
            feature_types = np.vectorize(self.DT_TYPE_MAPPER2.get)(
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

        self._warn_unused_missing()
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


class ArrayInterfaceHandler(DataHandler):
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
                                    ArrayInterfaceHandler)


class CudaColumnarHandler(DataHandler):
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
            feature_types = [PandasHandler.PANDAS_DTYPE_MAPPER[d.name]
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


class DLPackHandler(ArrayInterfaceHandler):
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


__dmatrix_registry.register_handler_dly(
    lambda x: 'PyCapsule' in str(type(x)) and "dltensor" in str(x),
    DLPackHandler)


def get_dmatrix_data_handler(data, missing, nthread, silent,
                             meta=None, meta_type=None):
    handler = __dmatrix_registry.get_handler(data)
    if handler is None:
        warnings.warn(
            f'Unknown data type {type(data)}, coverting it to csr_matrix')
        try:
            data = scipy.sparse.csr_matrix(data)
            handler = __dmatrix_registry.get_handler(data)
        except Exception:
            raise TypeError('Can not initialize DMatrix from'
                            ' {}'.format(type(data).__name__))
    return handler(missing, nthread, silent, meta, meta_type)


class DeviceQuantileDMatrixDataHandler(DataHandler):
    def __init__(self, max_bin, missing, nthread, silent,
                 meta=None, meta_type=None):
        self.max_bin = max_bin
        self.missing = missing
        self.nthread = nthread
        self.silent = silent

        self.meta = meta
        self.meta_type = meta_type


__device_quantile_dmatrix_registry = DMatrixDataManager()


class DeviceQuantileArrayInterfaceHandler(DeviceQuantileDMatrixDataHandler):
    def handle_input(self, data, feature_names, feature_types):
        """Initialize DMatrix from cupy ndarray."""
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
    'cupy.core.core', 'ndarray', DeviceQuantileArrayInterfaceHandler)


class DeviceQuantileCudaColumnarHandler(DeviceQuantileDMatrixDataHandler,
                                        CudaColumnarHandler):
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


class DeviceQuantileDLPackHandler(DeviceQuantileArrayInterfaceHandler,
                                  DLPackHandler):
    def __init__(self, max_bin, missing, nthread, silent,
                 meta=None, meta_type=None):
        super(DeviceQuantileArrayInterfaceHandler, self).__init__(
            max_bin=max_bin, missing=missing, nthread=nthread, silent=silent,
            meta=meta, meta_type=meta_type
        )

    def handle_input(self, data, feature_names, feature_types):
        data, feature_names, feature_types = self._maybe_dlpack_data(
            data, feature_names, feature_types)
        return super().handle_input(
            data, feature_names, feature_types)


__device_quantile_dmatrix_registry.register_handler_dly(
    lambda x: 'PyCapsule' in str(type(x)) and "dltensor" in str(x),
    DeviceQuantileDLPackHandler)


def get_device_quantile_dmatrix_data_handler(
        data, max_bin, missing, nthread, silent):
    handler = __device_quantile_dmatrix_registry.get_handler(
        data)
    assert handler, f'Current data type {type(data)} is not supported' + \
        ' for DeviceQuantileDMatrix'
    return handler(max_bin, missing, nthread, silent)
