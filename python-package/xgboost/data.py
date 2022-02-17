# pylint: disable=too-many-arguments, too-many-branches, too-many-lines
# pylint: disable=too-many-return-statements, import-error
'''Data dispatching for DMatrix.'''
import ctypes
import json
import warnings
import os
from typing import Any, Tuple, Callable, Union, Optional, List, cast, Sequence

import numpy as np

from .core import c_array, _LIB, _check_call, c_str
from .core import _cuda_array_interface
from .core import DataIter, _ProxyDMatrix, DMatrix, FeatureNames
from .compat import lazy_isinstance

from .typing import ArrayLike, FloatT, CSRLike, NPArrayLike, DFLike, NativeInput
from .typing import CuArrayLike, CuDFLike, FeatureTypes, DTypeLike

c_bst_ulong = ctypes.c_uint64   # pylint: disable=invalid-name

_CtorReturnT = Tuple[ctypes.c_void_p, Optional[List[str]], FeatureTypes]

CAT_T = "c"

# meta info that can be a matrix instead of vector.
# For now it's base_margin for multi-class, but it can be extended to label once we have
# multi-output.
_matrix_meta = {"base_margin", "label"}


def _warn_unused_missing(data: NativeInput, missing: FloatT) -> None:
    if (missing is not None) and (not np.isnan(missing)):
        warnings.warn(
            '`missing` is not used for current input data type:' +
            str(type(data)), UserWarning)


def _check_complex(data: ArrayLike) -> None:
    '''Test whether data is complex using `dtype` attribute.'''
    complex_dtypes = (np.complex128, np.complex64,
                      np.cfloat, np.cdouble, np.clongdouble)
    if hasattr(data, 'dtype') and data.dtype in complex_dtypes:
        raise ValueError('Complex data not supported')


def _check_data_shape(data: NativeInput) -> None:
    if hasattr(data, "shape") and len(cast(ArrayLike, data).shape) != 2:
        raise ValueError("Please reshape the input data into 2-dimensional matrix.")


def _is_scipy_csr(data: NativeInput) -> bool:
    try:
        import scipy
    except ImportError:
        scipy = None
        return False
    return isinstance(data, scipy.sparse.csr_matrix)


def _array_interface(data: NPArrayLike) -> bytes:
    assert (
        data.dtype.hasobject is False
    ), "Input data contains `object` dtype.  Expecting numeric data."
    interface = data.__array_interface__
    if "mask" in interface:
        interface["mask"] = cast(NPArrayLike, interface["mask"]).__array_interface__
    interface_str = bytes(json.dumps(interface), "utf-8")
    return interface_str


def _from_scipy_csr(
    data: CSRLike,
    missing: FloatT,
    nthread: int,
    feature_names: FeatureNames,
    feature_types: FeatureTypes,
) -> _CtorReturnT:
    """Initialize data from a CSR matrix."""
    if len(data.indices) != len(data.data):
        raise ValueError(
            f"length mismatch: {len(data.indices)} vs {len(data.data)}"
        )
    handle = ctypes.c_void_p()
    args = {
        "missing": float(missing),
        "nthread": int(nthread),
    }
    config = bytes(json.dumps(args), "utf-8")
    _check_call(
        _LIB.XGDMatrixCreateFromCSR(
            _array_interface(data.indptr),
            _array_interface(data.indices),
            _array_interface(data.data),
            ctypes.c_size_t(data.shape[1]),
            config,
            ctypes.byref(handle),
        )
    )
    return handle, feature_names, feature_types


def _is_scipy_csc(data: NativeInput) -> bool:
    try:
        import scipy
    except ImportError:
        scipy = None
        return False
    return isinstance(data, scipy.sparse.csc_matrix)


def _from_scipy_csc(
    data: CSRLike,
    missing: FloatT,
    feature_names: FeatureNames,
    feature_types: FeatureTypes,
) -> _CtorReturnT:
    if len(data.indices) != len(data.data):
        raise ValueError(f"length mismatch: {len(data.indices)} vs {len(data.data)}")
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


def _is_scipy_coo(data: NativeInput) -> bool:
    try:
        import scipy
    except ImportError:
        scipy = None
        return False
    return isinstance(data, scipy.sparse.coo_matrix)


def _is_numpy_array(data: NativeInput) -> bool:
    return isinstance(data, (np.ndarray, np.matrix))


def _ensure_np_dtype(
    data: np.ndarray, dtype: DTypeLike
) -> Tuple[np.ndarray, DTypeLike]:
    if data.dtype.hasobject or data.dtype in [np.float16, np.bool_]:
        data = data.astype(np.float32, copy=False)
        dtype = np.float32
    return data, dtype


def _maybe_np_slice(
    data: Union[List[int], np.ndarray], dtype: DTypeLike
) -> np.ndarray:
    '''Handle numpy slice.  This can be removed if we use __array_interface__.
    '''
    try:
        if not cast(np.ndarray, data).flags.c_contiguous:
            data = np.array(data, copy=True, dtype=dtype)
        else:
            data = np.array(data, copy=False, dtype=dtype)
    except AttributeError:
        data = np.array(data, copy=False, dtype=dtype)
    data, dtype = _ensure_np_dtype(data, dtype)
    return data


def _from_numpy_array(
    data: np.ndarray,
    missing: FloatT,
    nthread: int,
    feature_names: FeatureNames,
    feature_types: FeatureTypes,
) -> _CtorReturnT:
    """Initialize data from a 2-D numpy matrix.

    """
    if len(data.shape) != 2:
        raise ValueError(
            "Expecting 2 dimensional numpy.ndarray, got: ", data.shape
        )
    data, _ = _ensure_np_dtype(data, data.dtype)
    handle = ctypes.c_void_p()
    args = {
        "missing": float(missing),
        "nthread": int(nthread),
    }
    config = bytes(json.dumps(args), "utf-8")
    _check_call(
        _LIB.XGDMatrixCreateFromDense(
            _array_interface(data),
            config,
            ctypes.byref(handle),
        )
    )
    return handle, feature_names, feature_types


def _is_pandas_df(data: NativeInput) -> bool:
    try:
        import pandas as pd
    except ImportError:
        return False
    return isinstance(data, pd.DataFrame)


def _is_modin_df(data: NativeInput) -> bool:
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
    'bool': 'i',
}


def _invalid_dataframe_dtype(data: Any) -> None:
    # pandas series has `dtypes` but it's just a single object
    # cudf series doesn't have `dtypes`.
    if hasattr(data, "dtypes") and hasattr(data.dtypes, "__iter__"):
        bad_fields = [
            str(data.columns[i])
            for i, dtype in enumerate(data.dtypes)
            if dtype.name not in _pandas_dtype_mapper
        ]
        err = " Invalid columns:" + ", ".join(bad_fields)
    else:
        err = ""

    msg = """DataFrame.dtypes for data must be int, float, bool or category.  When
categorical type is supplied, DMatrix parameter `enable_categorical` must
be set to `True`.""" + err
    raise ValueError(msg)


def _transform_pandas_df(
    df: DFLike,
    enable_categorical: bool,
    feature_names: Optional[List[str]] = None,
    feature_types: FeatureTypes = None,
    meta: str = None,
    meta_type: DTypeLike = None,
) -> Tuple[np.ndarray, Optional[List[str]], FeatureTypes]:
    import pandas as pd
    from pandas.api.types import is_sparse, is_categorical_dtype
    data = cast(pd.DataFrame, df)

    if not all(
        dtype.name in _pandas_dtype_mapper
        or is_sparse(dtype)
        or (is_categorical_dtype(dtype) and enable_categorical)
        for dtype in data.dtypes
    ):
        _invalid_dataframe_dtype(data)

    # handle feature names
    if feature_names is None and meta is None:
        if isinstance(data.columns, pd.MultiIndex):
            feature_names = [" ".join([str(x) for x in i]) for i in data.columns]
        elif isinstance(data.columns, (pd.Index, pd.RangeIndex)):
            feature_names = list(map(str, data.columns))
        else:
            feature_names = data.columns.format()

    # handle feature types
    if feature_types is None and meta is None:
        feature_types_lst = []
        for i, dtype in enumerate(data.dtypes):
            if is_sparse(dtype):
                feature_types_lst.append(_pandas_dtype_mapper[dtype.subtype.name])
            elif is_categorical_dtype(dtype) and enable_categorical:
                feature_types_lst.append(CAT_T)
            else:
                feature_types_lst.append(_pandas_dtype_mapper[dtype.name])
        feature_types = feature_types_lst

    # handle category codes.
    transformed = pd.DataFrame()
    # Avoid transformation due to: PerformanceWarning: DataFrame is highly fragmented
    if enable_categorical and any(is_categorical_dtype(dtype) for dtype in data.dtypes):
        for i, dtype in enumerate(data.dtypes):
            if is_categorical_dtype(dtype):
                # pandas uses -1 as default missing value for categorical data
                transformed[data.columns[i]] = (
                    data[data.columns[i]]
                    .cat.codes.astype(np.float32)
                    .replace(-1.0, np.NaN)
                )
            else:
                transformed[data.columns[i]] = data[data.columns[i]]
    else:
        transformed = data

    if meta and len(data.columns) > 1 and meta not in _matrix_meta:
        raise ValueError(f"DataFrame for {meta} cannot have multiple columns")

    dtype = meta_type if meta_type else np.float32
    arr = transformed.values
    if meta_type:
        arr = arr.astype(meta_type)
    return arr, feature_names, feature_types


def _from_pandas_df(
    data: DFLike,
    enable_categorical: bool,
    missing: FloatT,
    nthread: int,
    feature_names: FeatureNames,
    feature_types: FeatureTypes,
) -> _CtorReturnT:
    arr, feature_names, feature_types = _transform_pandas_df(
        data, enable_categorical, feature_names, feature_types
    )
    return _from_numpy_array(
        arr, missing, nthread, feature_names, feature_types
    )


def _is_pandas_series(data: NativeInput) -> bool:
    try:
        import pandas as pd
    except ImportError:
        return False
    return isinstance(data, pd.Series)


def _meta_from_pandas_series(
    data: DFLike, name: str, dtype: DTypeLike, handle: ctypes.c_void_p
) -> None:
    """Help transform pandas series for meta data like labels"""
    import pandas as pd
    df = cast(pd.Series, data)
    arr = df.values.astype('float')
    from pandas.api.types import is_sparse
    if is_sparse(arr):
        arr = arr.to_dense()
    assert len(arr.shape) == 1 or arr.shape[1] == 0 or arr.shape[1] == 1
    _meta_from_numpy(arr, name, dtype, handle)


def _is_modin_series(data: NativeInput) -> bool:
    try:
        import modin.pandas as pd
    except ImportError:
        return False
    return isinstance(data, pd.Series)


def _from_pandas_series(
    data: DFLike,
    missing: FloatT,
    nthread: int,
    enable_categorical: bool,
    feature_names: FeatureNames,
    feature_types: FeatureTypes,
) -> _CtorReturnT:
    import pandas as pd
    from pandas.api.types import is_categorical_dtype
    series = cast(pd.Series, data)

    if (series.dtype.name not in _pandas_dtype_mapper) and not (
        is_categorical_dtype(series.dtype) and enable_categorical
    ):
        _invalid_dataframe_dtype(series)
    if enable_categorical and is_categorical_dtype(series.dtype):
        series = series.cat.codes

    return _from_numpy_array(
        series.values.reshape(series.shape[0], 1).astype("float"),
        missing,
        nthread,
        feature_names,
        feature_types,
    )


def _is_dt_df(data: NativeInput) -> bool:
    return lazy_isinstance(data, 'datatable', 'Frame') or \
        lazy_isinstance(data, 'datatable', 'DataTable')


_dt_type_mapper = {'bool': 'bool', 'int': 'int', 'real': 'float'}
_dt_type_mapper2 = {'bool': 'i', 'int': 'int', 'real': 'float'}


def _transform_dt_df(
    data: Any,
    feature_names: Optional[List[str]],
    feature_types: FeatureTypes,
    meta: str = None,
    meta_type: DTypeLike = None,
) -> Tuple[np.ndarray, Optional[List[str]], FeatureTypes]:
    """Validate feature names and types if data table"""
    if meta and data.shape[1] > 1:
        raise ValueError('DataTable for meta info cannot have multiple columns')
    if meta:
        meta_type = "float" if meta_type is None else meta_type
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


def _from_dt_df(
    data: Any,
    missing: FloatT,
    nthread: int,
    feature_names: Optional[List[str]],
    feature_types: FeatureTypes,
    enable_categorical: bool,
) -> _CtorReturnT:
    if enable_categorical:
        raise ValueError("categorical data in datatable is not supported yet.")
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


def _is_cudf_df(data: NativeInput) -> bool:
    try:
        import cudf
    except ImportError:
        return False
    return hasattr(cudf, 'DataFrame') and isinstance(data, cudf.DataFrame)


def _cudf_array_interfaces(data: CuDFLike, cat_codes: Optional[list]) -> bytes:
    """Extract CuDF __cuda_array_interface__.  This is special as it returns a new list of
    data and a list of array interfaces.  The data is list of categorical codes that
    caller can safely ignore, but have to keep their reference alive until usage of array
    interface is finished.

    """
    try:
        from cudf.api.types import is_categorical_dtype
    except ImportError:
        from cudf.utils.dtypes import is_categorical_dtype

    interfaces = []
    if _is_cudf_ser(data):
        if is_categorical_dtype(data.dtype):
            assert cat_codes is not None
            interface = cat_codes[0].__cuda_array_interface__
        else:
            interface = data.__cuda_array_interface__
        if "mask" in interface:
            interface["mask"] = interface["mask"].__cuda_array_interface__
        interfaces.append(interface)
    else:
        for i, col in enumerate(data):
            if is_categorical_dtype(data[col].dtype):
                assert cat_codes is not None
                codes = cat_codes[i]
                interface = codes.__cuda_array_interface__
            else:
                interface = data[col].__cuda_array_interface__
            if "mask" in interface:
                interface["mask"] = interface["mask"].__cuda_array_interface__
            interfaces.append(interface)
    interfaces_str = bytes(json.dumps(interfaces, indent=2), "utf-8")
    return interfaces_str


def _transform_cudf_df(
    data: CuDFLike,
    feature_names: FeatureNames,
    feature_types: FeatureTypes,
    enable_categorical: bool,
) -> Tuple[CuDFLike, Optional[list], Optional[List[str]], FeatureTypes]:
    try:
        from cudf.api.types import is_categorical_dtype
    except ImportError:
        from cudf.utils.dtypes import is_categorical_dtype

    import cudf
    df = cast(cudf.DataFrame, data)

    if _is_cudf_ser(df):
        dtypes: Sequence = [df.dtype]
    else:
        dtypes = df.dtypes

    if not all(
        dtype.name in _pandas_dtype_mapper
        or (is_categorical_dtype(dtype) and enable_categorical)
        for dtype in dtypes
    ):
        _invalid_dataframe_dtype(df)

    # handle feature names
    if feature_names is None:
        if _is_cudf_ser(df):
            feature_names = [df.name]
        elif lazy_isinstance(df.columns, "cudf.core.multiindex", "MultiIndex"):
            feature_names = [" ".join([str(x) for x in i]) for i in df.columns]
        elif (
            lazy_isinstance(df.columns, "cudf.core.index", "RangeIndex")
            or lazy_isinstance(df.columns, "cudf.core.index", "Int64Index")
            # Unique to cuDF, no equivalence in pandas 1.3.3
            or lazy_isinstance(df.columns, "cudf.core.index", "Int32Index")
        ):
            feature_names = list(map(str, df.columns))
        else:
            feature_names = df.columns.format()

    # handle feature types
    if feature_types is None:
        feature_types_lst = []
        for dtype in dtypes:
            if is_categorical_dtype(dtype) and enable_categorical:
                feature_types_lst.append(CAT_T)
            else:
                feature_types_lst.append(_pandas_dtype_mapper[dtype.name])
        feature_types = feature_types_lst

    # handle categorical data
    cat_codes = []
    if _is_cudf_ser(df):
        # unlike pandas, cuDF uses NA for missing data.
        if is_categorical_dtype(df.dtype) and enable_categorical:
            codes = df.cat.codes
            cat_codes.append(codes)
    else:
        for col in df:
            if is_categorical_dtype(df[col].dtype) and enable_categorical:
                codes = df[col].cat.codes
                cat_codes.append(codes)

    return df, cat_codes, feature_names, feature_types


def _from_cudf_df(
    data: CuDFLike,
    missing: FloatT,
    nthread: int,
    feature_names: FeatureNames,
    feature_types: FeatureTypes,
    enable_categorical: bool,
) -> _CtorReturnT:
    df, cat_codes, feature_names, feature_types = _transform_cudf_df(
        data, feature_names, feature_types, enable_categorical
    )
    interfaces_str = _cudf_array_interfaces(df, cat_codes)
    handle = ctypes.c_void_p()
    config = bytes(json.dumps({"missing": missing, "nthread": nthread}), "utf-8")
    _check_call(
        _LIB.XGDMatrixCreateFromCudaColumnar(
            interfaces_str,
            config,
            ctypes.byref(handle),
        )
    )
    return handle, feature_names, feature_types


def _is_cudf_ser(data: NativeInput) -> bool:
    try:
        import cudf
    except ImportError:
        return False
    return isinstance(data, cudf.Series)


def _is_cupy_array(data: NativeInput) -> bool:
    try:
        import cupy
    except ImportError:
        return False
    return isinstance(data, cupy.ndarray)


def _transform_cupy_array(arr: CuArrayLike) -> CuArrayLike:
    import cupy  # pylint: disable=import-error
    data = cast(cupy.ndarray, arr)
    if not hasattr(data, '__cuda_array_interface__') and hasattr(
            data, '__array__'):
        data = cupy.array(data, copy=False)
    if data.dtype.hasobject or data.dtype in [cupy.float16, cupy.bool_]:
        data = data.astype(cupy.float32, copy=False)
    return data


def _from_cupy_array(
    data: CuArrayLike,
    missing: FloatT,
    nthread: int,
    feature_names: Optional[List[str]],
    feature_types: FeatureTypes,
) -> _CtorReturnT:
    """Initialize DMatrix from cupy ndarray."""
    data = _transform_cupy_array(data)
    interface_str = _cuda_array_interface(data)
    handle = ctypes.c_void_p()
    config = bytes(json.dumps({"missing": missing, "nthread": nthread}), "utf-8")
    _check_call(
        _LIB.XGDMatrixCreateFromCudaArrayInterface(
            interface_str,
            config,
            ctypes.byref(handle)))
    return handle, feature_names, feature_types


def _is_cupy_csr(data: NativeInput) -> bool:
    try:
        import cupyx
    except ImportError:
        return False
    return isinstance(data, cupyx.scipy.sparse.csr_matrix)


def _is_cupy_csc(data: NativeInput) -> bool:
    try:
        import cupyx
    except ImportError:
        return False
    return isinstance(data, cupyx.scipy.sparse.csc_matrix)


def _is_dlpack(data: NativeInput) -> bool:
    return 'PyCapsule' in str(type(data)) and "dltensor" in str(data)


def _transform_dlpack(data: Any) -> CuArrayLike:
    from cupy import fromDlpack  # pylint: disable=E0401
    assert 'used_dltensor' not in str(data)
    data = fromDlpack(data)
    return data


def _from_dlpack(
    data: Any,
    missing: FloatT,
    nthread: int,
    feature_names: Optional[List[str]],
    feature_types: FeatureTypes,
) -> _CtorReturnT:
    data = _transform_dlpack(data)
    return _from_cupy_array(data, missing, nthread, feature_names,
                            feature_types)


def _is_uri(data: NativeInput) -> bool:
    return isinstance(data, (str, os.PathLike))


def _from_uri(
    data: Union[os.PathLike, str],
    missing: FloatT,
    feature_names: FeatureNames,
    feature_types: FeatureTypes,
) -> _CtorReturnT:
    _warn_unused_missing(data, missing)
    handle = ctypes.c_void_p()
    data = os.fspath(os.path.expanduser(data))
    _check_call(_LIB.XGDMatrixCreateFromFile(c_str(data),
                                             ctypes.c_int(1),
                                             ctypes.byref(handle)))
    return handle, feature_names, feature_types


def _is_list(data: NativeInput) -> bool:
    return isinstance(data, list)


def _from_list(
    data: list,
    missing: FloatT,
    n_threads: int,
    feature_names: FeatureNames,
    feature_types: FeatureTypes,
) -> _CtorReturnT:
    array = np.array(data)
    _check_data_shape(array)
    return _from_numpy_array(array, missing, n_threads, feature_names, feature_types)


def _is_tuple(data: Any) -> bool:
    return isinstance(data, tuple)


def _from_tuple(
    data: tuple,
    missing: FloatT,
    n_threads: int,
    feature_names: FeatureNames,
    feature_types: FeatureTypes,
) -> _CtorReturnT:
    return _from_list(cast(list, data), missing, n_threads, feature_names, feature_types)


def _is_iter(data: Any) -> bool:
    return isinstance(data, DataIter)


def _has_array_protocol(data: Any) -> bool:
    return hasattr(data, '__array__')


def _convert_unknown_data(data: Any) -> Optional[Any]:
    warnings.warn(
        f'Unknown data type: {type(data)}, trying to convert it to csr_matrix',
        UserWarning
    )
    try:
        import scipy
    except ImportError:
        return None

    try:
        data = scipy.sparse.csr_matrix(data)
    except Exception:           # pylint: disable=broad-except
        return None

    return data


def dispatch_data_backend(
    data: NativeInput,
    missing: FloatT,
    threads: int,
    feature_names: FeatureNames,
    feature_types: FeatureTypes,
    enable_categorical: bool = False,
) -> _CtorReturnT:
    '''Dispatch data for DMatrix.'''
    if not _is_cudf_ser(data) and not _is_pandas_series(data):
        _check_data_shape(data)
    if _is_scipy_csr(data):
        return _from_scipy_csr(
            cast(CSRLike, data), missing, threads, feature_names, feature_types
        )
    if _is_scipy_csc(data):
        return _from_scipy_csc(cast(CSRLike, data), missing, feature_names, feature_types)
    if _is_scipy_coo(data):
        from scipy.sparse import coo_matrix
        return _from_scipy_csr(
            cast(coo_matrix, data).tocsr(), missing, threads, feature_names, feature_types
        )
    if _is_numpy_array(data):
        return _from_numpy_array(
            cast(np.ndarray, data), missing, threads, feature_names, feature_types
        )
    if _is_uri(data):
        return _from_uri(
            cast(Union[os.PathLike, str], data), missing, feature_names, feature_types
        )
    if _is_list(data):
        return _from_list(
            cast(list, data), missing, threads, feature_names, feature_types
        )
    if _is_tuple(data):
        return _from_tuple(
            cast(tuple, data), missing, threads, feature_names, feature_types
        )
    if _is_pandas_df(data):
        return _from_pandas_df(cast(DFLike, data), enable_categorical, missing, threads,
                               feature_names, feature_types)
    if _is_pandas_series(data):
        return _from_pandas_series(
            cast(DFLike, data), missing, threads, enable_categorical, feature_names, feature_types
        )
    if _is_cudf_df(data) or _is_cudf_ser(data):
        df = cast(CuDFLike, data)
        return _from_cudf_df(
            df, missing, threads, feature_names, feature_types, enable_categorical
        )
    if _is_cupy_array(data):
        return _from_cupy_array(
            cast(CuArrayLike, data), missing, threads, feature_names, feature_types
        )
    if _is_cupy_csr(data):
        raise TypeError('cupyx CSR is not supported yet.')
    if _is_cupy_csc(data):
        raise TypeError('cupyx CSC is not supported yet.')
    if _is_dlpack(data):
        return _from_dlpack(data, missing, threads, feature_names,
                            feature_types)
    if _is_dt_df(data):
        _warn_unused_missing(data, missing)
        return _from_dt_df(
            data, missing, threads, feature_names, feature_types, enable_categorical
        )
    if _is_modin_df(data):
        return _from_pandas_df(
            cast(DFLike, data), enable_categorical, missing, threads,
            feature_names, feature_types
        )
    if _is_modin_series(data):
        return _from_pandas_series(
            cast(DFLike, data), missing, threads, enable_categorical, feature_names, feature_types
        )
    if _has_array_protocol(data):
        array = np.asarray(data)
        return _from_numpy_array(array, missing, threads, feature_names, feature_types)

    converted = _convert_unknown_data(data)
    if converted is not None:
        return _from_scipy_csr(converted, missing, threads, feature_names, feature_types)

    raise TypeError('Not supported type for data.' + str(type(data)))


def _to_data_type(dtype: str, name: str) -> int:
    dtype_map = {'float32': 1, 'float64': 2, 'uint32': 3, 'uint64': 4}
    if dtype not in dtype_map:
        raise TypeError(
            f'Expecting float32, float64, uint32, uint64, got {dtype} ' +
            f'for {name}.')
    return dtype_map[dtype]


def _validate_meta_shape(data: ArrayLike, name: str) -> None:
    if hasattr(data, "shape"):
        msg = f"Invalid shape: {data.shape} for {name}"
        if name in _matrix_meta:
            if len(data.shape) > 2:
                raise ValueError(msg)
            return

        if len(data.shape) > 2 or (
            len(data.shape) == 2 and (data.shape[1] != 0 and data.shape[1] != 1)
        ):
            raise ValueError(f"Invalid shape: {data.shape} for {name}")


def _meta_from_numpy(
    data: np.ndarray,
    field: str,
    dtype: DTypeLike,
    handle: ctypes.c_void_p,
) -> None:
    data, dtype = _ensure_np_dtype(data, dtype)
    interface = data.__array_interface__
    if interface.get("mask", None) is not None:
        raise ValueError("Masked array is not supported.")
    interface_str = _array_interface(data)
    _check_call(_LIB.XGDMatrixSetInfoFromInterface(handle, c_str(field), interface_str))


def _meta_from_list(
    data: list, field: str, dtype: DTypeLike, handle: ctypes.c_void_p
) -> None:
    arr = np.asarray(data)
    _meta_from_numpy(arr, field, dtype, handle)


def _meta_from_tuple(
    data: tuple, field: str, dtype: DTypeLike, handle: ctypes.c_void_p
) -> None:
    return _meta_from_list(cast(list, data), field, dtype, handle)


def _meta_from_cudf_df(data: CuDFLike, field: str, handle: ctypes.c_void_p) -> None:
    import cudf
    df = cast(cudf.DataFrame, data)
    if field not in _matrix_meta:
        _meta_from_cudf_series(df.iloc[:, 0], field, handle)
    else:
        arr = data.values
        interface = _cuda_array_interface(arr)
        _check_call(_LIB.XGDMatrixSetInfoFromInterface(handle, c_str(field), interface))


def _meta_from_cudf_series(data: CuDFLike, field: str, handle: ctypes.c_void_p) -> None:
    interface = bytes(json.dumps([data.__cuda_array_interface__],
                                 indent=2), 'utf-8')
    _check_call(_LIB.XGDMatrixSetInfoFromInterface(handle,
                                                   c_str(field),
                                                   interface))


def _meta_from_cupy_array(data: CuArrayLike, field: str, handle: ctypes.c_void_p) -> None:
    data = _transform_cupy_array(data)
    interface = bytes(json.dumps([data.__cuda_array_interface__],
                                 indent=2), 'utf-8')
    _check_call(_LIB.XGDMatrixSetInfoFromInterface(handle,
                                                   c_str(field),
                                                   interface))


def _meta_from_dt(
    data: Any, field: str, dtype: DTypeLike, handle: ctypes.c_void_p
) -> None:
    data, _, _ = _transform_dt_df(data, None, None, field, dtype)
    _meta_from_numpy(data, field, dtype, handle)


def dispatch_meta_backend(
    matrix: DMatrix, data: ArrayLike, name: str, dtype: DTypeLike = None
) -> None:
    '''Dispatch for meta info.'''
    handle = matrix.handle
    assert handle is not None
    _validate_meta_shape(data, name)
    if data is None:
        return
    if _is_list(data):
        _meta_from_list(cast(list, data), name, dtype, handle)
        return
    if _is_tuple(data):
        _meta_from_tuple(cast(tuple, data), name, dtype, handle)
        return
    if _is_numpy_array(data):
        _meta_from_numpy(cast(np.ndarray, data), name, dtype, handle)
        return
    if _is_pandas_df(data):
        arr, _, _ = _transform_pandas_df(
            cast(DFLike, data), False, meta=name, meta_type=dtype
        )
        _meta_from_numpy(arr, name, dtype, handle)
        return
    if _is_pandas_series(data):
        _meta_from_pandas_series(cast(DFLike, data), name, dtype, handle)
        return
    if _is_dlpack(data):
        data = _transform_dlpack(data)
        _meta_from_cupy_array(data, name, handle)
        return
    if _is_cupy_array(data):
        _meta_from_cupy_array(cast(CuArrayLike, data), name, handle)
        return
    if _is_cudf_ser(data):
        _meta_from_cudf_series(cast(CuDFLike, data), name, handle)
        return
    if _is_cudf_df(data):
        _meta_from_cudf_df(cast(CuDFLike, data), name, handle)
        return
    if _is_dt_df(data):
        _meta_from_dt(data, name, dtype, handle)
        return
    if _is_modin_df(data):
        data, _, _ = _transform_pandas_df(
            cast(DFLike, data), False, meta=name, meta_type=dtype
        )
        _meta_from_numpy(data, name, dtype, handle)
        return
    if _is_modin_series(data):
        arr = cast(np.ndarray, cast(DFLike, data).values).astype('float')
        assert len(arr.shape) == 1 or arr.shape[1] == 0 or arr.shape[1] == 1
        _meta_from_numpy(arr, name, dtype, handle)
        return
    if _has_array_protocol(data):
        array = np.asarray(data)
        _meta_from_numpy(array, name, dtype, handle)
        return
    raise TypeError('Unsupported type for ' + name, str(type(data)))


class SingleBatchInternalIter(DataIter):  # pylint: disable=R0902
    '''An iterator for single batch data to help creating device DMatrix.
    Transforming input directly to histogram with normal single batch data API
    can not access weight for sketching.  So this iterator acts as a staging
    area for meta info.

    '''
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self.it = 0             # pylint: disable=invalid-name
        super().__init__()

    def next(self, input_data: Callable) -> int:
        if self.it == 1:
            return 0
        self.it += 1
        input_data(**self.kwargs)
        return 1

    def reset(self) -> None:
        self.it = 0


def _proxy_transform(
    data: ArrayLike,
    feature_names: FeatureNames,
    feature_types: FeatureTypes,
    enable_categorical: bool,
) -> Tuple[ArrayLike, Optional[list], Optional[List[str]], FeatureTypes]:
    if _is_cudf_df(data) or _is_cudf_ser(data):
        return _transform_cudf_df(
            cast(CuDFLike, data), feature_names, feature_types, enable_categorical
        )
    if _is_cupy_array(data):
        data = _transform_cupy_array(cast(CuArrayLike, data))
        return data, None, feature_names, feature_types
    if _is_dlpack(data):
        return _transform_dlpack(data), None, feature_names, feature_types
    if _is_numpy_array(data):
        return data, None, feature_names, feature_types
    if _is_scipy_csr(data):
        return data, None, feature_names, feature_types
    if _is_pandas_df(data):
        arr, feature_names, feature_types = _transform_pandas_df(
            cast(DFLike, data), enable_categorical, feature_names, feature_types
        )
        return arr, None, feature_names, feature_types
    raise TypeError("Value type is not supported for data iterator:" + str(type(data)))


def dispatch_proxy_set_data(
    proxy: _ProxyDMatrix,
    data: Any,
    cat_codes: Optional[List[CuDFLike]],
    allow_host: bool,
) -> None:
    """Dispatch for DeviceQuantileDMatrix."""
    if not _is_cudf_ser(data) and not _is_pandas_series(data):
        _check_data_shape(data)

    if _is_cudf_df(data):
        # pylint: disable=W0212
        proxy._set_data_from_cuda_columnar(data, cat_codes)
        return
    if _is_cudf_ser(data):
        # pylint: disable=W0212
        proxy._set_data_from_cuda_columnar(data, cat_codes)
        return
    if _is_cupy_array(data):
        # pylint: disable=W0212
        proxy._set_data_from_cuda_interface(cast(CuArrayLike, data))
        return
    if _is_dlpack(data):
        data = _transform_dlpack(data)
        proxy._set_data_from_cuda_interface(data)  # pylint: disable=W0212
        return

    err = TypeError("Value type is not supported for data iterator:" + str(type(data)))

    if not allow_host:
        raise err

    if _is_numpy_array(data):
        proxy._set_data_from_array(cast(NPArrayLike,  data))  # pylint: disable=W0212
        return
    if _is_scipy_csr(data):
        proxy._set_data_from_csr(cast(CSRLike, data))  # pylint: disable=W0212
        return
    raise err
