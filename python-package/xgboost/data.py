# pylint: disable=too-many-arguments, too-many-branches
# pylint: disable=too-many-return-statements, import-error
'''Data dispatching for DMatrix.'''
import ctypes
import json
import warnings
import os

import numpy as np

from .core import c_array, _LIB, _check_call, c_str
from .core import DataIter, DeviceQuantileDMatrix, DMatrix
from .compat import lazy_isinstance

c_bst_ulong = ctypes.c_uint64   # pylint: disable=invalid-name


def _warn_unused_missing(data, missing):
    if (missing is not None) and (not np.isnan(missing)):
        warnings.warn(
            '`missing` is not used for current input data type:' +
            str(type(data)), UserWarning)


def _check_complex(data):
    '''Test whether data is complex using `dtype` attribute.'''
    complex_dtypes = (np.complex128, np.complex64,
                      np.cfloat, np.cdouble, np.clongdouble)
    if hasattr(data, 'dtype') and data.dtype in complex_dtypes:
        raise ValueError('Complex data not supported')


def _is_scipy_csr(data):
    try:
        import scipy
    except ImportError:
        scipy = None
        return False
    return isinstance(data, scipy.sparse.csr_matrix)


def _from_scipy_csr(data, missing, feature_names, feature_types):
    '''Initialize data from a CSR matrix.'''
    if len(data.indices) != len(data.data):
        raise ValueError('length mismatch: {} vs {}'.format(
            len(data.indices), len(data.data)))
    _warn_unused_missing(data, missing)
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


def _is_scipy_csc(data):
    try:
        import scipy
    except ImportError:
        scipy = None
        return False
    return isinstance(data, scipy.sparse.csc_matrix)


def _from_scipy_csc(data, missing, feature_names, feature_types):
    if len(data.indices) != len(data.data):
        raise ValueError('length mismatch: {} vs {}'.format(
            len(data.indices), len(data.data)))
    _warn_unused_missing(data, missing)
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


def _is_scipy_coo(data):
    try:
        import scipy
    except ImportError:
        scipy = None
        return False
    return isinstance(data, scipy.sparse.coo_matrix)


def _is_numpy_array(data):
    return isinstance(data, (np.ndarray, np.matrix))


def _maybe_np_slice(data, dtype):
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


def _transform_np_array(data: np.ndarray):
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
    flatten = _maybe_np_slice(flatten, np.float32)
    _check_complex(data)
    return flatten


def _from_numpy_array(data, missing, nthread, feature_names, feature_types):
    """Initialize data from a 2-D numpy matrix.

    If ``mat`` does not have ``order='C'`` (aka row-major) or is
    not contiguous, a temporary copy will be made.

    If ``mat`` does not have ``dtype=numpy.float32``, a temporary copy will
    be made.

    So there could be as many as two temporary data copies; be mindful of
    input layout and type if memory use is a concern.

    """
    flatten = _transform_np_array(data)
    handle = ctypes.c_void_p()
    _check_call(_LIB.XGDMatrixCreateFromMat_omp(
        flatten.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        c_bst_ulong(data.shape[0]),
        c_bst_ulong(data.shape[1]),
        ctypes.c_float(missing),
        ctypes.byref(handle),
        ctypes.c_int(nthread)))
    return handle, feature_names, feature_types


def _is_pandas_df(data):
    try:
        import pandas as pd
    except ImportError:
        return False
    return isinstance(data, pd.DataFrame)


def _is_modin_df(data):
    try:
        import modin.pandas as pd
    except ImportError:
        return False
    return isinstance(data, pd.DataFrame)


_pandas_dtype_mapper = {
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


def _transform_pandas_df(data, enable_categorical,
                         feature_names=None, feature_types=None,
                         meta=None, meta_type=None):
    from pandas import MultiIndex, Int64Index
    from pandas.api.types import is_sparse, is_categorical_dtype

    data_dtypes = data.dtypes
    if not all(dtype.name in _pandas_dtype_mapper or is_sparse(dtype) or
               (is_categorical_dtype(dtype) and enable_categorical)
               for dtype in data_dtypes):
        bad_fields = [
            str(data.columns[i]) for i, dtype in enumerate(data_dtypes)
            if dtype.name not in _pandas_dtype_mapper
        ]

        msg = """DataFrame.dtypes for data must be int, float, bool or categorical.  When
                categorical type is supplied, DMatrix parameter
                `enable_categorical` must be set to `True`."""
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
                feature_types.append(_pandas_dtype_mapper[
                    dtype.subtype.name])
            elif is_categorical_dtype(dtype) and enable_categorical:
                feature_types.append('categorical')
            else:
                feature_types.append(_pandas_dtype_mapper[dtype.name])

    if meta and len(data.columns) > 1:
        raise ValueError(
            'DataFrame for {meta} cannot have multiple columns'.format(
                meta=meta))

    dtype = meta_type if meta_type else np.float32
    data = np.ascontiguousarray(data.values, dtype=dtype)
    return data, feature_names, feature_types


def _from_pandas_df(data, enable_categorical, missing, nthread,
                    feature_names, feature_types):
    data, feature_names, feature_types = _transform_pandas_df(
        data, enable_categorical, feature_names, feature_types)
    return _from_numpy_array(data, missing, nthread, feature_names,
                             feature_types)


def _is_pandas_series(data):
    try:
        import pandas as pd
    except ImportError:
        return False
    return isinstance(data, pd.Series)


def _is_modin_series(data):
    try:
        import modin.pandas as pd
    except ImportError:
        return False
    return isinstance(data, pd.Series)


def _from_pandas_series(data, missing, nthread, feature_types, feature_names):
    return _from_numpy_array(data.values.astype('float'), missing, nthread,
                             feature_names, feature_types)


def _is_dt_df(data):
    return lazy_isinstance(data, 'datatable', 'Frame') or \
        lazy_isinstance(data, 'datatable', 'DataTable')


_dt_type_mapper = {'bool': 'bool', 'int': 'int', 'real': 'float'}
_dt_type_mapper2 = {'bool': 'i', 'int': 'int', 'real': 'float'}


def _transform_dt_df(data, feature_names, feature_types, meta=None,
                     meta_type=None):
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
                  if type_name not in _dt_type_mapper]
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
        feature_types = np.vectorize(_dt_type_mapper2.get)(
            data_types_names).tolist()

    return data, feature_names, feature_types


def _from_dt_df(data, missing, nthread, feature_names, feature_types):
    data, feature_names, feature_types = _transform_dt_df(
        data, feature_names, feature_types, None, None)

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
            frame_column_data_r  # pylint: disable=no-name-in-module
        for icol in range(data.ncols):
            ptrs[icol] = frame_column_data_r(data, icol)

    # always return stypes for dt ingestion
    feature_type_strings = (ctypes.c_char_p * data.ncols)()
    for icol in range(data.ncols):
        feature_type_strings[icol] = ctypes.c_char_p(
            data.stypes[icol].name.encode('utf-8'))

    _warn_unused_missing(data, missing)
    handle = ctypes.c_void_p()
    _check_call(_LIB.XGDMatrixCreateFromDT(
        ptrs, feature_type_strings,
        c_bst_ulong(data.shape[0]),
        c_bst_ulong(data.shape[1]),
        ctypes.byref(handle),
        ctypes.c_int(nthread)))
    return handle, feature_names, feature_types


def _is_cudf_df(data):
    try:
        import cudf
    except ImportError:
        return False
    return hasattr(cudf, 'DataFrame') and isinstance(data, cudf.DataFrame)


def _cudf_array_interfaces(data):
    '''Extract CuDF __cuda_array_interface__'''
    interfaces = []
    if _is_cudf_ser(data):
        interfaces.append(data.__cuda_array_interface__)
    else:
        for col in data:
            interface = data[col].__cuda_array_interface__
            if 'mask' in interface:
                interface['mask'] = interface['mask'].__cuda_array_interface__
            interfaces.append(interface)
    interfaces_str = bytes(json.dumps(interfaces, indent=2), 'utf-8')
    return interfaces_str


def _transform_cudf_df(data, feature_names, feature_types):
    if feature_names is None:
        if _is_cudf_ser(data):
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
        if _is_cudf_ser(data):
            dtypes = [data.dtype]
        else:
            dtypes = data.dtypes
        feature_types = [_pandas_dtype_mapper[d.name]
                         for d in dtypes]
    return data, feature_names, feature_types


def _from_cudf_df(data, missing, nthread, feature_names, feature_types):
    data, feature_names, feature_types = _transform_cudf_df(
        data, feature_names, feature_types)
    interfaces_str = _cudf_array_interfaces(data)
    handle = ctypes.c_void_p()
    _check_call(
        _LIB.XGDMatrixCreateFromArrayInterfaceColumns(
            interfaces_str,
            ctypes.c_float(missing),
            ctypes.c_int(nthread),
            ctypes.byref(handle)))
    return handle, feature_names, feature_types


def _is_cudf_ser(data):
    try:
        import cudf
    except ImportError:
        return False
    return isinstance(data, cudf.Series)


def _is_cupy_array(data):
    try:
        import cupy
    except ImportError:
        return False
    return isinstance(data, cupy.ndarray)


def _transform_cupy_array(data):
    if not hasattr(data, '__cuda_array_interface__') and hasattr(
            data, '__array__'):
        import cupy             # pylint: disable=import-error
        data = cupy.array(data, copy=False)
    data = data.astype(dtype=data.dtype, order='C', copy=False)
    return data


def _from_cupy_array(data, missing, nthread, feature_names, feature_types):
    """Initialize DMatrix from cupy ndarray."""
    data = _transform_cupy_array(data)
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
    return handle, feature_names, feature_types


def _is_cupy_csr(data):
    try:
        import cupyx
    except ImportError:
        return False
    return isinstance(data, cupyx.scipy.sparse.csr_matrix)


def _is_cupy_csc(data):
    try:
        import cupyx
    except ImportError:
        return False
    return isinstance(data, cupyx.scipy.sparse.csc_matrix)


def _is_dlpack(data):
    return 'PyCapsule' in str(type(data)) and "dltensor" in str(data)


def _transform_dlpack(data):
    from cupy import fromDlpack  # pylint: disable=E0401
    assert 'used_dltensor' not in str(data)
    data = fromDlpack(data)
    return data


def _from_dlpack(data, missing, nthread, feature_names, feature_types):
    data = _transform_dlpack(data)
    return _from_cupy_array(data, missing, nthread, feature_names,
                            feature_types)


def _is_uri(data):
    return isinstance(data, (str, os.PathLike))


def _from_uri(data, missing, feature_names, feature_types):
    _warn_unused_missing(data, missing)
    handle = ctypes.c_void_p()
    _check_call(_LIB.XGDMatrixCreateFromFile(c_str(os.fspath(data)),
                                             ctypes.c_int(1),
                                             ctypes.byref(handle)))
    return handle, feature_names, feature_types


def _is_list(data):
    return isinstance(data, list)


def _from_list(data, missing, feature_names, feature_types):
    raise TypeError('List input data is not supported for data')


def _is_tuple(data):
    return isinstance(data, tuple)


def _from_tuple(data, missing, feature_names, feature_types):
    return _from_list(data, missing, feature_names, feature_types)


def _is_iter(data):
    return isinstance(data, DataIter)


def _has_array_protocol(data):
    return hasattr(data, '__array__')


def dispatch_data_backend(data, missing, threads,
                          feature_names, feature_types,
                          enable_categorical=False):
    '''Dispatch data for DMatrix.'''
    if _is_scipy_csr(data):
        return _from_scipy_csr(data, missing, feature_names, feature_types)
    if _is_scipy_csc(data):
        return _from_scipy_csc(data, missing, feature_names, feature_types)
    if _is_scipy_coo(data):
        return _from_scipy_csr(data.tocsr(), missing, feature_names, feature_types)
    if _is_numpy_array(data):
        return _from_numpy_array(data, missing, threads, feature_names,
                                 feature_types)
    if _is_uri(data):
        return _from_uri(data, missing, feature_names, feature_types)
    if _is_list(data):
        return _from_list(data, missing, feature_names, feature_types)
    if _is_tuple(data):
        return _from_tuple(data, missing, feature_names, feature_types)
    if _is_pandas_df(data):
        return _from_pandas_df(data, enable_categorical, missing, threads,
                               feature_names, feature_types)
    if _is_pandas_series(data):
        return _from_pandas_series(data, missing, threads, feature_names,
                                   feature_types)
    if _is_cudf_df(data):
        return _from_cudf_df(data, missing, threads, feature_names,
                             feature_types)
    if _is_cudf_ser(data):
        return _from_cudf_df(data, missing, threads, feature_names,
                             feature_types)
    if _is_cupy_array(data):
        return _from_cupy_array(data, missing, threads, feature_names,
                                feature_types)
    if _is_cupy_csr(data):
        raise TypeError('cupyx CSR is not supported yet.')
    if _is_cupy_csc(data):
        raise TypeError('cupyx CSC is not supported yet.')
    if _is_dlpack(data):
        return _from_dlpack(data, missing, threads, feature_names,
                            feature_types)
    if _is_dt_df(data):
        _warn_unused_missing(data, missing)
        return _from_dt_df(data, missing, threads, feature_names,
                           feature_types)
    if _is_modin_df(data):
        return _from_pandas_df(data, enable_categorical, missing, threads,
                               feature_names, feature_types)
    if _is_modin_series(data):
        return _from_pandas_series(data, missing, threads, feature_names,
                                   feature_types)
    if _has_array_protocol(data):
        pass
    raise TypeError('Not supported type for data.' + str(type(data)))


def _to_data_type(dtype: str, name: str):
    dtype_map = {'float32': 1, 'float64': 2, 'uint32': 3, 'uint64': 4}
    if dtype not in dtype_map.keys():
        raise TypeError(
            f'Expecting float32, float64, uint32, uint64, got {dtype} ' +
            f'for {name}.')
    return dtype_map[dtype]


def _validate_meta_shape(data):
    if hasattr(data, 'shape'):
        assert len(data.shape) == 1 or (
            len(data.shape) == 2 and
            (data.shape[1] == 0 or data.shape[1] == 1))


def _meta_from_numpy(data, field, dtype, handle):
    data = _maybe_np_slice(data, dtype)
    interface = data.__array_interface__
    assert interface.get('mask', None) is None, 'Masked array is not supported'
    size = data.shape[0]

    c_type = _to_data_type(str(data.dtype), field)
    ptr = interface['data'][0]
    ptr = ctypes.c_void_p(ptr)
    _check_call(_LIB.XGDMatrixSetDenseInfo(
        handle,
        c_str(field),
        ptr,
        c_bst_ulong(size),
        c_type
    ))


def _meta_from_list(data, field, dtype, handle):
    data = np.array(data)
    _meta_from_numpy(data, field, dtype, handle)


def _meta_from_tuple(data, field, dtype, handle):
    return _meta_from_list(data, field, dtype, handle)


def _meta_from_cudf_df(data, field, handle):
    if len(data.columns) != 1:
        raise ValueError(
            'Expecting meta-info to contain a single column')
    data = data[data.columns[0]]

    interface = bytes(json.dumps([data.__cuda_array_interface__],
                                 indent=2), 'utf-8')
    _check_call(_LIB.XGDMatrixSetInfoFromInterface(handle,
                                                   c_str(field),
                                                   interface))


def _meta_from_cudf_series(data, field, handle):
    interface = bytes(json.dumps([data.__cuda_array_interface__],
                                 indent=2), 'utf-8')
    _check_call(_LIB.XGDMatrixSetInfoFromInterface(handle,
                                                   c_str(field),
                                                   interface))


def _meta_from_cupy_array(data, field, handle):
    data = _transform_cupy_array(data)
    interface = bytes(json.dumps([data.__cuda_array_interface__],
                                 indent=2), 'utf-8')
    _check_call(_LIB.XGDMatrixSetInfoFromInterface(handle,
                                                   c_str(field),
                                                   interface))


def _meta_from_dt(data, field, dtype, handle):
    data, _, _ = _transform_dt_df(data, None, None)
    _meta_from_numpy(data, field, dtype, handle)


def dispatch_meta_backend(matrix: DMatrix, data, name: str, dtype: str = None):
    '''Dispatch for meta info.'''
    handle = matrix.handle
    _validate_meta_shape(data)
    if data is None:
        return
    if _is_list(data):
        _meta_from_list(data, name, dtype, handle)
        return
    if _is_tuple(data):
        _meta_from_tuple(data, name, dtype, handle)
        return
    if _is_numpy_array(data):
        _meta_from_numpy(data, name, dtype, handle)
        return
    if _is_pandas_df(data):
        data, _, _ = _transform_pandas_df(data, False, meta=name,
                                          meta_type=dtype)
        _meta_from_numpy(data, name, dtype, handle)
        return
    if _is_pandas_series(data):
        data = data.values.astype('float')
        assert len(data.shape) == 1 or data.shape[1] == 0 or data.shape[1] == 1
        _meta_from_numpy(data, name, dtype, handle)
        return
    if _is_dlpack(data):
        data = _transform_dlpack(data)
        _meta_from_cupy_array(data, name, handle)
        return
    if _is_cupy_array(data):
        _meta_from_cupy_array(data, name, handle)
        return
    if _is_cudf_ser(data):
        _meta_from_cudf_series(data, name, handle)
        return
    if _is_cudf_df(data):
        _meta_from_cudf_df(data, name, handle)
        return
    if _is_dt_df(data):
        _meta_from_dt(data, name, dtype, handle)
        return
    if _is_modin_df(data):
        data, _, _ = _transform_pandas_df(
            data, False, meta=name, meta_type=dtype)
        _meta_from_numpy(data, name, dtype, handle)
        return
    if _is_modin_series(data):
        data = data.values.astype('float')
        assert len(data.shape) == 1 or data.shape[1] == 0 or data.shape[1] == 1
        _meta_from_numpy(data, name, dtype, handle)
        return
    if _has_array_protocol(data):
        pass
    raise TypeError('Unsupported type for ' + name, str(type(data)))


class SingleBatchInternalIter(DataIter):  # pylint: disable=R0902
    '''An iterator for single batch data to help creating device DMatrix.
    Transforming input directly to histogram with normal single batch data API
    can not access weight for sketching.  So this iterator acts as a staging
    area for meta info.

    '''
    def __init__(self, data, label, weight, base_margin, group,
                 label_lower_bound, label_upper_bound,
                 feature_names, feature_types):
        self.data = data
        self.label = label
        self.weight = weight
        self.base_margin = base_margin
        self.group = group
        self.label_lower_bound = label_lower_bound
        self.label_upper_bound = label_upper_bound
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.it = 0             # pylint: disable=invalid-name
        super().__init__()

    def next(self, input_data):
        if self.it == 1:
            return 0
        self.it += 1
        input_data(data=self.data, label=self.label,
                   weight=self.weight, base_margin=self.base_margin,
                   group=self.group,
                   label_lower_bound=self.label_lower_bound,
                   label_upper_bound=self.label_upper_bound,
                   feature_names=self.feature_names,
                   feature_types=self.feature_types)
        return 1

    def reset(self):
        self.it = 0


def init_device_quantile_dmatrix(
        data, missing, max_bin, threads, feature_names, feature_types, **meta):
    '''Constructor for DeviceQuantileDMatrix.'''
    if not any([_is_cudf_df(data), _is_cudf_ser(data), _is_cupy_array(data),
                _is_dlpack(data), _is_iter(data)]):
        raise TypeError(str(type(data)) +
                        ' is not supported for DeviceQuantileDMatrix')
    if _is_dlpack(data):
        # We specialize for dlpack because cupy will take the memory from it so
        # it can't be transformed twice.
        data = _transform_dlpack(data)
    if _is_iter(data):
        it = data
    else:
        it = SingleBatchInternalIter(
            data, **meta, feature_names=feature_names,
            feature_types=feature_types)

    reset_factory = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    reset_callback = reset_factory(it.reset_wrapper)
    next_factory = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.c_void_p,
    )
    next_callback = next_factory(it.next_wrapper)
    handle = ctypes.c_void_p()
    ret = _LIB.XGDeviceQuantileDMatrixCreateFromCallback(
        None,
        it.proxy.handle,
        reset_callback,
        next_callback,
        ctypes.c_float(missing),
        ctypes.c_int(threads),
        ctypes.c_int(max_bin),
        ctypes.byref(handle)
    )
    if it.exception:
        raise it.exception
    # delay check_call to throw intermediate exception first
    _check_call(ret)
    matrix = DeviceQuantileDMatrix(handle)
    feature_names = matrix.feature_names
    feature_types = matrix.feature_types
    matrix.handle = None
    return handle, feature_names, feature_types


def _device_quantile_transform(data, feature_names, feature_types):
    if _is_cudf_df(data):
        return _transform_cudf_df(data, feature_names, feature_types)
    if _is_cudf_ser(data):
        return _transform_cudf_df(data, feature_names, feature_types)
    if _is_cupy_array(data):
        data = _transform_cupy_array(data)
        return data, feature_names, feature_types
    if _is_dlpack(data):
        return _transform_dlpack(data), feature_names, feature_types
    raise TypeError('Value type is not supported for data iterator:' +
                    str(type(data)))


def dispatch_device_quantile_dmatrix_set_data(proxy, data):
    '''Dispatch for DeviceQuantileDMatrix.'''
    if _is_cudf_df(data):
        proxy._set_data_from_cuda_columnar(data)  # pylint: disable=W0212
        return
    if _is_cudf_ser(data):
        proxy._set_data_from_cuda_columnar(data)  # pylint: disable=W0212
        return
    if _is_cupy_array(data):
        proxy._set_data_from_cuda_interface(data)  # pylint: disable=W0212
        return
    if _is_dlpack(data):
        data = _transform_dlpack(data)
        proxy._set_data_from_cuda_interface(data)  # pylint: disable=W0212
        return
    raise TypeError('Value type is not supported for data iterator:' +
                    str(type(data)))
