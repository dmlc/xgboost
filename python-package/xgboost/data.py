# pylint: disable=too-many-arguments, too-many-branches, too-many-lines
# pylint: disable=too-many-return-statements, import-error
'''Data dispatching for DMatrix.'''
import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, Union, cast

import numpy as np

from ._typing import (
    CupyT,
    DataType,
    FeatureNames,
    FeatureTypes,
    FloatCompatible,
    NumpyDType,
    PandasDType,
    c_bst_ulong,
)
from .compat import DataFrame, lazy_isinstance
from .core import (
    _LIB,
    DataIter,
    DMatrix,
    _check_call,
    _cuda_array_interface,
    _ProxyDMatrix,
    c_array,
    c_str,
    from_pystr_to_cstr,
    make_jcargs,
)

DispatchedDataBackendReturnType = Tuple[
    ctypes.c_void_p, Optional[FeatureNames], Optional[FeatureTypes]]

CAT_T = "c"

# meta info that can be a matrix instead of vector.
_matrix_meta = {"base_margin", "label"}


def _warn_unused_missing(data: DataType, missing: Optional[FloatCompatible]) -> None:
    if (missing is not None) and (not np.isnan(missing)):
        warnings.warn(
            '`missing` is not used for current input data type:' +
            str(type(data)), UserWarning)


def _check_complex(data: DataType) -> None:
    '''Test whether data is complex using `dtype` attribute.'''
    complex_dtypes = (np.complex128, np.complex64,
                      np.cfloat, np.cdouble, np.clongdouble)
    if hasattr(data, 'dtype') and data.dtype in complex_dtypes:
        raise ValueError('Complex data not supported')


def _check_data_shape(data: DataType) -> None:
    if hasattr(data, "shape") and len(data.shape) != 2:
        raise ValueError("Please reshape the input data into 2-dimensional matrix.")


def _is_scipy_csr(data: DataType) -> bool:
    try:
        import scipy.sparse
    except ImportError:
        return False
    return isinstance(data, scipy.sparse.csr_matrix)


def _array_interface(data: np.ndarray) -> bytes:
    assert (
        data.dtype.hasobject is False
    ), "Input data contains `object` dtype.  Expecting numeric data."
    interface = data.__array_interface__
    if "mask" in interface:
        interface["mask"] = interface["mask"].__array_interface__
    interface_str = bytes(json.dumps(interface), "utf-8")
    return interface_str


def _transform_scipy_csr(data: DataType) -> DataType:
    from scipy.sparse import csr_matrix

    indptr, _ = _ensure_np_dtype(data.indptr, data.indptr.dtype)
    indices, _ = _ensure_np_dtype(data.indices, data.indices.dtype)
    values, _ = _ensure_np_dtype(data.data, data.data.dtype)
    if (
        indptr is not data.indptr
        or indices is not data.indices
        or values is not data.data
    ):
        data = csr_matrix((values, indices, indptr), shape=data.shape)
    return data


def _from_scipy_csr(
    data: DataType,
    missing: FloatCompatible,
    nthread: int,
    feature_names: Optional[FeatureNames],
    feature_types: Optional[FeatureTypes],
) -> DispatchedDataBackendReturnType:
    """Initialize data from a CSR matrix."""
    if len(data.indices) != len(data.data):
        raise ValueError(
            f"length mismatch: {len(data.indices)} vs {len(data.data)}"
        )
    handle = ctypes.c_void_p()
    data = _transform_scipy_csr(data)
    _check_call(
        _LIB.XGDMatrixCreateFromCSR(
            _array_interface(data.indptr),
            _array_interface(data.indices),
            _array_interface(data.data),
            c_bst_ulong(data.shape[1]),
            make_jcargs(missing=float(missing), nthread=int(nthread)),
            ctypes.byref(handle),
        )
    )
    return handle, feature_names, feature_types


def _is_scipy_csc(data: DataType) -> bool:
    try:
        import scipy.sparse
    except ImportError:
        return False
    return isinstance(data, scipy.sparse.csc_matrix)


def _from_scipy_csc(
    data: DataType,
    missing: Optional[FloatCompatible],
    feature_names: Optional[FeatureNames],
    feature_types: Optional[FeatureTypes],
) -> DispatchedDataBackendReturnType:
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


def _is_scipy_coo(data: DataType) -> bool:
    try:
        import scipy.sparse
    except ImportError:
        return False
    return isinstance(data, scipy.sparse.coo_matrix)


def _is_numpy_array(data: DataType) -> bool:
    return isinstance(data, (np.ndarray, np.matrix))


def _ensure_np_dtype(
    data: DataType, dtype: Optional[NumpyDType]
) -> Tuple[np.ndarray, Optional[NumpyDType]]:
    if data.dtype.hasobject or data.dtype in [np.float16, np.bool_]:
        dtype = np.float32
        data = data.astype(dtype, copy=False)
    if not data.flags.aligned:
        data = np.require(data, requirements="A")
    return data, dtype


def _maybe_np_slice(data: DataType, dtype: Optional[NumpyDType]) -> np.ndarray:
    '''Handle numpy slice.  This can be removed if we use __array_interface__.
    '''
    try:
        if not data.flags.c_contiguous:
            data = np.array(data, copy=True, dtype=dtype)
        else:
            data = np.array(data, copy=False, dtype=dtype)
    except AttributeError:
        data = np.array(data, copy=False, dtype=dtype)
    data, dtype = _ensure_np_dtype(data, dtype)
    return data


def _from_numpy_array(
    data: DataType,
    missing: FloatCompatible,
    nthread: int,
    feature_names: Optional[FeatureNames],
    feature_types: Optional[FeatureTypes],
) -> DispatchedDataBackendReturnType:
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


def _is_pandas_df(data: DataType) -> bool:
    try:
        import pandas as pd
    except ImportError:
        return False
    return isinstance(data, pd.DataFrame)


def _is_modin_df(data: DataType) -> bool:
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
    # nullable types
    "Int16": "int",
    "Int32": "int",
    "Int64": "int",
    "Float32": "float",
    "Float64": "float",
    "boolean": "i",
}


_ENABLE_CAT_ERR = (
    "When categorical type is supplied, The experimental DMatrix parameter"
    "`enable_categorical` must be set to `True`."
)


def _invalid_dataframe_dtype(data: DataType) -> None:
    # pandas series has `dtypes` but it's just a single object
    # cudf series doesn't have `dtypes`.
    if hasattr(data, "dtypes") and hasattr(data.dtypes, "__iter__"):
        bad_fields = [
            f"{data.columns[i]}: {dtype}"
            for i, dtype in enumerate(data.dtypes)
            if dtype.name not in _pandas_dtype_mapper
        ]
        err = " Invalid columns:" + ", ".join(bad_fields)
    else:
        err = ""

    type_err = "DataFrame.dtypes for data must be int, float, bool or category."
    msg = f"""{type_err} {_ENABLE_CAT_ERR} {err}"""
    raise ValueError(msg)


def _pandas_feature_info(
    data: DataFrame,
    meta: Optional[str],
    feature_names: Optional[FeatureNames],
    feature_types: Optional[FeatureTypes],
    enable_categorical: bool,
) -> Tuple[Optional[FeatureNames], Optional[FeatureTypes]]:
    import pandas as pd
    from pandas.api.types import (
        is_sparse,
        is_categorical_dtype,
    )

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
        feature_types = []
        for dtype in data.dtypes:
            if is_sparse(dtype):
                feature_types.append(_pandas_dtype_mapper[dtype.subtype.name])
            elif is_categorical_dtype(dtype) and enable_categorical:
                feature_types.append(CAT_T)
            else:
                feature_types.append(_pandas_dtype_mapper[dtype.name])
    return feature_names, feature_types


def is_nullable_dtype(dtype: PandasDType) -> bool:
    """Wether dtype is a pandas nullable type."""
    from pandas.api.types import (
        is_integer_dtype,
        is_bool_dtype,
        is_float_dtype,
        is_categorical_dtype,
    )

    # dtype: pd.core.arrays.numeric.NumericDtype
    nullable_alias = {"Int16", "Int32", "Int64", "Float32", "Float64", "category"}
    is_int = is_integer_dtype(dtype) and dtype.name in nullable_alias
    # np.bool has alias `bool`, while pd.BooleanDtype has `bzoolean`.
    is_bool = is_bool_dtype(dtype) and dtype.name == "boolean"
    is_float = is_float_dtype(dtype) and dtype.name in nullable_alias
    return is_int or is_bool or is_float or is_categorical_dtype(dtype)


def _pandas_cat_null(data: DataFrame) -> DataFrame:
    from pandas.api.types import is_categorical_dtype
    # handle category codes and nullable.
    cat_columns = [
        col
        for col, dtype in zip(data.columns, data.dtypes)
        if is_categorical_dtype(dtype)
    ]
    nul_columns = [
        col for col, dtype in zip(data.columns, data.dtypes) if is_nullable_dtype(dtype)
    ]
    if cat_columns or nul_columns:
        # Avoid transformation due to: PerformanceWarning: DataFrame is highly
        # fragmented
        transformed = data.copy()
    else:
        transformed = data

    if cat_columns:
        # DF doesn't have the cat attribute, so we use apply here
        transformed[cat_columns] = (
            transformed[cat_columns]
            .apply(lambda x: x.cat.codes)
            .astype(np.float32)
            .replace(-1.0, np.NaN)
        )
    if nul_columns:
        transformed[nul_columns] = transformed[nul_columns].astype(np.float32)

    return transformed


def _transform_pandas_df(
    data: DataFrame,
    enable_categorical: bool,
    feature_names: Optional[FeatureNames] = None,
    feature_types: Optional[FeatureTypes] = None,
    meta: Optional[str] = None,
    meta_type: Optional[NumpyDType] = None,
) -> Tuple[np.ndarray, Optional[FeatureNames], Optional[FeatureTypes]]:
    from pandas.api.types import (
        is_sparse,
        is_categorical_dtype,
    )

    if not all(
        dtype.name in _pandas_dtype_mapper
        or is_sparse(dtype)
        or (is_nullable_dtype(dtype) and not is_categorical_dtype(dtype))
        or (is_categorical_dtype(dtype) and enable_categorical)
        for dtype in data.dtypes
    ):
        _invalid_dataframe_dtype(data)

    feature_names, feature_types = _pandas_feature_info(
        data, meta, feature_names, feature_types, enable_categorical
    )

    transformed = _pandas_cat_null(data)

    if meta and len(data.columns) > 1 and meta not in _matrix_meta:
        raise ValueError(f"DataFrame for {meta} cannot have multiple columns")

    dtype = meta_type if meta_type else np.float32
    arr: np.ndarray = transformed.values
    if meta_type:
        arr = arr.astype(dtype)
    return arr, feature_names, feature_types


def _from_pandas_df(
    data: DataFrame,
    enable_categorical: bool,
    missing: FloatCompatible,
    nthread: int,
    feature_names: Optional[FeatureNames],
    feature_types: Optional[FeatureTypes],
) -> DispatchedDataBackendReturnType:
    data, feature_names, feature_types = _transform_pandas_df(
        data, enable_categorical, feature_names, feature_types
    )
    return _from_numpy_array(data, missing, nthread, feature_names, feature_types)


def _is_pandas_series(data: DataType) -> bool:
    try:
        import pandas as pd
    except ImportError:
        return False
    return isinstance(data, pd.Series)


def _meta_from_pandas_series(
    data: DataType,
    name: str,
    dtype: Optional[NumpyDType],
    handle: ctypes.c_void_p
) -> None:
    """Help transform pandas series for meta data like labels"""
    data = data.values.astype('float')
    from pandas.api.types import is_sparse
    if is_sparse(data):
        data = data.to_dense()  # type: ignore
    assert len(data.shape) == 1 or data.shape[1] == 0 or data.shape[1] == 1
    _meta_from_numpy(data, name, dtype, handle)


def _is_modin_series(data: DataType) -> bool:
    try:
        import modin.pandas as pd
    except ImportError:
        return False
    return isinstance(data, pd.Series)


def _from_pandas_series(
    data: DataType,
    missing: FloatCompatible,
    nthread: int,
    enable_categorical: bool,
    feature_names: Optional[FeatureNames],
    feature_types: Optional[FeatureTypes],
) -> DispatchedDataBackendReturnType:
    from pandas.api.types import is_categorical_dtype

    if (data.dtype.name not in _pandas_dtype_mapper) and not (
        is_categorical_dtype(data.dtype) and enable_categorical
    ):
        _invalid_dataframe_dtype(data)
    if enable_categorical and is_categorical_dtype(data.dtype):
        data = data.cat.codes
    return _from_numpy_array(
        data.values.reshape(data.shape[0], 1).astype("float"),
        missing,
        nthread,
        feature_names,
        feature_types,
    )


def _is_dt_df(data: DataType) -> bool:
    return lazy_isinstance(data, 'datatable', 'Frame') or \
        lazy_isinstance(data, 'datatable', 'DataTable')


_dt_type_mapper = {'bool': 'bool', 'int': 'int', 'real': 'float'}
_dt_type_mapper2 = {'bool': 'i', 'int': 'int', 'real': 'float'}


def _transform_dt_df(
    data: DataType,
    feature_names: Optional[FeatureNames],
    feature_types: Optional[FeatureTypes],
    meta: Optional[str] = None,
    meta_type: Optional[NumpyDType] = None,
) -> Tuple[np.ndarray, Optional[FeatureNames], Optional[FeatureTypes]]:
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
    data: DataType,
    missing: Optional[FloatCompatible],
    nthread: int,
    feature_names: Optional[FeatureNames],
    feature_types: Optional[FeatureTypes],
    enable_categorical: bool,
) -> DispatchedDataBackendReturnType:
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


def _is_arrow(data: DataType) -> bool:
    try:
        import pyarrow as pa
        from pyarrow import dataset as arrow_dataset
        return isinstance(data, (pa.Table, arrow_dataset.Dataset))
    except ImportError:
        return False


def record_batch_data_iter(data_iter: Iterator) -> Callable:
    """Data iterator used to ingest Arrow columnar record batches. We are not using
    class DataIter because it is only intended for building Device DMatrix and external
    memory DMatrix.

    """
    from pyarrow.cffi import ffi

    c_schemas: List[ffi.CData] = []
    c_arrays: List[ffi.CData] = []

    def _next(data_handle: int) -> int:
        from pyarrow.cffi import ffi

        try:
            batch = next(data_iter)
            c_schemas.append(ffi.new("struct ArrowSchema*"))
            c_arrays.append(ffi.new("struct ArrowArray*"))
            ptr_schema = int(ffi.cast("uintptr_t", c_schemas[-1]))
            ptr_array = int(ffi.cast("uintptr_t", c_arrays[-1]))
            # pylint: disable=protected-access
            batch._export_to_c(ptr_array, ptr_schema)
            _check_call(
                _LIB.XGImportArrowRecordBatch(
                    ctypes.c_void_p(data_handle),
                    ctypes.c_void_p(ptr_array),
                    ctypes.c_void_p(ptr_schema),
                )
            )
            return 1
        except StopIteration:
            return 0

    return _next


def _from_arrow(
    data: DataType,
    missing: FloatCompatible,
    nthread: int,
    feature_names: Optional[FeatureNames],
    feature_types: Optional[FeatureTypes],
    enable_categorical: bool,
) -> DispatchedDataBackendReturnType:
    import pyarrow as pa

    if not all(
        pa.types.is_integer(t) or pa.types.is_floating(t) for t in data.schema.types
    ):
        raise ValueError(
            "Features in dataset can only be integers or floating point number"
        )
    if enable_categorical:
        raise ValueError("categorical data in arrow is not supported yet.")

    batches = data.to_batches()
    rb_iter = iter(batches)
    it = record_batch_data_iter(rb_iter)
    next_callback = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)(it)
    handle = ctypes.c_void_p()
    config = from_pystr_to_cstr(
        json.dumps({"missing": missing, "nthread": nthread, "nbatch": len(batches)})
    )
    _check_call(
        _LIB.XGDMatrixCreateFromArrowCallback(
            next_callback,
            config,
            ctypes.byref(handle),
        )
    )
    return handle, feature_names, feature_types


def _is_cudf_df(data: DataType) -> bool:
    return lazy_isinstance(data, "cudf.core.dataframe", "DataFrame")


def _cudf_array_interfaces(data: DataType, cat_codes: list) -> bytes:
    """Extract CuDF __cuda_array_interface__.  This is special as it returns a new list
    of data and a list of array interfaces.  The data is list of categorical codes that
    caller can safely ignore, but have to keep their reference alive until usage of
    array interface is finished.

    """
    try:
        from cudf.api.types import is_categorical_dtype
    except ImportError:
        from cudf.utils.dtypes import is_categorical_dtype

    interfaces = []

    def append(interface: dict) -> None:
        if "mask" in interface:
            interface["mask"] = interface["mask"].__cuda_array_interface__
        interfaces.append(interface)

    if _is_cudf_ser(data):
        if is_categorical_dtype(data.dtype):
            interface = cat_codes[0].__cuda_array_interface__
        else:
            interface = data.__cuda_array_interface__
        append(interface)
    else:
        for i, col in enumerate(data):
            if is_categorical_dtype(data[col].dtype):
                codes = cat_codes[i]
                interface = codes.__cuda_array_interface__
            else:
                interface = data[col].__cuda_array_interface__
            append(interface)
    interfaces_str = from_pystr_to_cstr(json.dumps(interfaces))
    return interfaces_str


def _transform_cudf_df(
    data: DataType,
    feature_names: Optional[FeatureNames],
    feature_types: Optional[FeatureTypes],
    enable_categorical: bool,
) -> Tuple[ctypes.c_void_p, list, Optional[FeatureNames], Optional[FeatureTypes]]:
    try:
        from cudf.api.types import is_categorical_dtype
    except ImportError:
        from cudf.utils.dtypes import is_categorical_dtype

    if _is_cudf_ser(data):
        dtypes = [data.dtype]
    else:
        dtypes = data.dtypes

    if not all(
        dtype.name in _pandas_dtype_mapper
        or (is_categorical_dtype(dtype) and enable_categorical)
        for dtype in dtypes
    ):
        _invalid_dataframe_dtype(data)

    # handle feature names
    if feature_names is None:
        if _is_cudf_ser(data):
            feature_names = [data.name]
        elif lazy_isinstance(data.columns, "cudf.core.multiindex", "MultiIndex"):
            feature_names = [" ".join([str(x) for x in i]) for i in data.columns]
        elif (
            lazy_isinstance(data.columns, "cudf.core.index", "RangeIndex")
            or lazy_isinstance(data.columns, "cudf.core.index", "Int64Index")
            # Unique to cuDF, no equivalence in pandas 1.3.3
            or lazy_isinstance(data.columns, "cudf.core.index", "Int32Index")
        ):
            feature_names = list(map(str, data.columns))
        else:
            feature_names = data.columns.format()

    # handle feature types
    if feature_types is None:
        feature_types = []
        for dtype in dtypes:
            if is_categorical_dtype(dtype) and enable_categorical:
                feature_types.append(CAT_T)
            else:
                feature_types.append(_pandas_dtype_mapper[dtype.name])

    # handle categorical data
    cat_codes = []
    if _is_cudf_ser(data):
        # unlike pandas, cuDF uses NA for missing data.
        if is_categorical_dtype(data.dtype) and enable_categorical:
            codes = data.cat.codes
            cat_codes.append(codes)
    else:
        for col in data:
            dtype = data[col].dtype
            if is_categorical_dtype(dtype) and enable_categorical:
                codes = data[col].cat.codes
                cat_codes.append(codes)
            elif is_categorical_dtype(dtype):
                raise ValueError(_ENABLE_CAT_ERR)
            else:
                cat_codes.append([])

    return data, cat_codes, feature_names, feature_types


def _from_cudf_df(
    data: DataType,
    missing: FloatCompatible,
    nthread: int,
    feature_names: Optional[FeatureNames],
    feature_types: Optional[FeatureTypes],
    enable_categorical: bool,
) -> DispatchedDataBackendReturnType:
    data, cat_codes, feature_names, feature_types = _transform_cudf_df(
        data, feature_names, feature_types, enable_categorical
    )
    interfaces_str = _cudf_array_interfaces(data, cat_codes)
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


def _is_cudf_ser(data: DataType) -> bool:
    return lazy_isinstance(data, "cudf.core.series", "Series")


def _is_cupy_array(data: DataType) -> bool:
    return any(
        lazy_isinstance(data, n, "ndarray")
        for n in ("cupy.core.core", "cupy", "cupy._core.core")
    )


def _transform_cupy_array(data: DataType) -> CupyT:
    import cupy  # pylint: disable=import-error
    if not hasattr(data, '__cuda_array_interface__') and hasattr(
            data, '__array__'):
        data = cupy.array(data, copy=False)
    if data.dtype.hasobject or data.dtype in [cupy.float16, cupy.bool_]:
        data = data.astype(cupy.float32, copy=False)
    return data


def _from_cupy_array(
    data: DataType,
    missing: FloatCompatible,
    nthread: int,
    feature_names: Optional[FeatureNames],
    feature_types: Optional[FeatureTypes],
) -> DispatchedDataBackendReturnType:
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


def _is_cupy_csr(data: DataType) -> bool:
    try:
        import cupyx
    except ImportError:
        return False
    return isinstance(data, cupyx.scipy.sparse.csr_matrix)


def _is_cupy_csc(data: DataType) -> bool:
    try:
        import cupyx
    except ImportError:
        return False
    return isinstance(data, cupyx.scipy.sparse.csc_matrix)


def _is_dlpack(data: DataType) -> bool:
    return 'PyCapsule' in str(type(data)) and "dltensor" in str(data)


def _transform_dlpack(data: DataType) -> bool:
    from cupy import fromDlpack  # pylint: disable=E0401
    assert 'used_dltensor' not in str(data)
    data = fromDlpack(data)
    return data


def _from_dlpack(
    data: DataType,
    missing: FloatCompatible,
    nthread: int,
    feature_names: Optional[FeatureNames],
    feature_types: Optional[FeatureTypes],
) -> DispatchedDataBackendReturnType:
    data = _transform_dlpack(data)
    return _from_cupy_array(data, missing, nthread, feature_names,
                            feature_types)


def _is_uri(data: DataType) -> bool:
    return isinstance(data, (str, os.PathLike))


def _from_uri(
    data: DataType,
    missing: Optional[FloatCompatible],
    feature_names: Optional[FeatureNames],
    feature_types: Optional[FeatureTypes],
) -> DispatchedDataBackendReturnType:
    _warn_unused_missing(data, missing)
    handle = ctypes.c_void_p()
    data = os.fspath(os.path.expanduser(data))
    _check_call(_LIB.XGDMatrixCreateFromFile(c_str(data),
                                             ctypes.c_int(1),
                                             ctypes.byref(handle)))
    return handle, feature_names, feature_types


def _is_list(data: DataType) -> bool:
    return isinstance(data, list)


def _from_list(
    data: Sequence,
    missing: FloatCompatible,
    n_threads: int,
    feature_names: Optional[FeatureNames],
    feature_types: Optional[FeatureTypes],
) -> DispatchedDataBackendReturnType:
    array = np.array(data)
    _check_data_shape(data)
    return _from_numpy_array(array, missing, n_threads, feature_names, feature_types)


def _is_tuple(data: DataType) -> bool:
    return isinstance(data, tuple)


def _from_tuple(
    data: Sequence,
    missing: FloatCompatible,
    n_threads: int,
    feature_names: Optional[FeatureNames],
    feature_types: Optional[FeatureTypes],
) -> DispatchedDataBackendReturnType:
    return _from_list(data, missing, n_threads, feature_names, feature_types)


def _is_iter(data: DataType) -> bool:
    return isinstance(data, DataIter)


def _has_array_protocol(data: DataType) -> bool:
    return hasattr(data, '__array__')


def _convert_unknown_data(data: DataType) -> DataType:
    warnings.warn(
        f'Unknown data type: {type(data)}, trying to convert it to csr_matrix',
        UserWarning
    )
    try:
        import scipy.sparse
    except ImportError:
        return None

    try:
        data = scipy.sparse.csr_matrix(data)
    except Exception:           # pylint: disable=broad-except
        return None

    return data


def dispatch_data_backend(
    data: DataType,
    missing: FloatCompatible,  # Or Optional[Float]
    threads: int,
    feature_names: Optional[FeatureNames],
    feature_types: Optional[FeatureTypes],
    enable_categorical: bool = False,
) -> DispatchedDataBackendReturnType:
    '''Dispatch data for DMatrix.'''
    if not _is_cudf_ser(data) and not _is_pandas_series(data):
        _check_data_shape(data)
    if _is_scipy_csr(data):
        return _from_scipy_csr(data, missing, threads, feature_names, feature_types)
    if _is_scipy_csc(data):
        return _from_scipy_csc(data, missing, feature_names, feature_types)
    if _is_scipy_coo(data):
        return _from_scipy_csr(
            data.tocsr(), missing, threads, feature_names, feature_types
        )
    if _is_numpy_array(data):
        return _from_numpy_array(data, missing, threads, feature_names, feature_types)
    if _is_uri(data):
        return _from_uri(data, missing, feature_names, feature_types)
    if _is_list(data):
        return _from_list(data, missing, threads, feature_names, feature_types)
    if _is_tuple(data):
        return _from_tuple(data, missing, threads, feature_names, feature_types)
    if _is_pandas_df(data):
        return _from_pandas_df(data, enable_categorical, missing, threads,
                               feature_names, feature_types)
    if _is_pandas_series(data):
        return _from_pandas_series(
            data, missing, threads, enable_categorical, feature_names, feature_types
        )
    if _is_cudf_df(data) or _is_cudf_ser(data):
        return _from_cudf_df(
            data, missing, threads, feature_names, feature_types, enable_categorical
        )
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
        return _from_dt_df(
            data, missing, threads, feature_names, feature_types, enable_categorical
        )
    if _is_modin_df(data):
        return _from_pandas_df(data, enable_categorical, missing, threads,
                               feature_names, feature_types)
    if _is_modin_series(data):
        return _from_pandas_series(
            data, missing, threads, enable_categorical, feature_names, feature_types
        )
    if _is_arrow(data):
        return _from_arrow(
            data, missing, threads, feature_names, feature_types, enable_categorical)
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


def _validate_meta_shape(data: DataType, name: str) -> None:
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
    dtype: Optional[NumpyDType],
    handle: ctypes.c_void_p,
) -> None:
    data, dtype = _ensure_np_dtype(data, dtype)
    interface = data.__array_interface__
    if interface.get("mask", None) is not None:
        raise ValueError("Masked array is not supported.")
    interface_str = _array_interface(data)
    _check_call(_LIB.XGDMatrixSetInfoFromInterface(handle, c_str(field), interface_str))


def _meta_from_list(
    data: Sequence,
    field: str,
    dtype: Optional[NumpyDType],
    handle: ctypes.c_void_p
) -> None:
    data_np = np.array(data)
    _meta_from_numpy(data_np, field, dtype, handle)


def _meta_from_tuple(
    data: Sequence,
    field: str,
    dtype: Optional[NumpyDType],
    handle: ctypes.c_void_p
) -> None:
    return _meta_from_list(data, field, dtype, handle)


def _meta_from_cudf_df(data: DataType, field: str, handle: ctypes.c_void_p) -> None:
    if field not in _matrix_meta:
        _meta_from_cudf_series(data.iloc[:, 0], field, handle)
    else:
        data = data.values
        interface = _cuda_array_interface(data)
        _check_call(_LIB.XGDMatrixSetInfoFromInterface(handle, c_str(field), interface))


def _meta_from_cudf_series(data: DataType, field: str, handle: ctypes.c_void_p) -> None:
    interface = bytes(json.dumps([data.__cuda_array_interface__],
                                 indent=2), 'utf-8')
    _check_call(_LIB.XGDMatrixSetInfoFromInterface(handle,
                                                   c_str(field),
                                                   interface))


def _meta_from_cupy_array(data: DataType, field: str, handle: ctypes.c_void_p) -> None:
    data = _transform_cupy_array(data)
    interface = bytes(json.dumps([data.__cuda_array_interface__],
                                 indent=2), 'utf-8')
    _check_call(_LIB.XGDMatrixSetInfoFromInterface(handle,
                                                   c_str(field),
                                                   interface))


def _meta_from_dt(
    data: DataType,
    field: str,
    dtype: Optional[NumpyDType],
    handle: ctypes.c_void_p
) -> None:
    data, _, _ = _transform_dt_df(data, None, None, field, dtype)
    _meta_from_numpy(data, field, dtype, handle)


def dispatch_meta_backend(
    matrix: DMatrix,
    data: DataType,
    name: str,
    dtype: Optional[NumpyDType] = None
) -> None:
    '''Dispatch for meta info.'''
    handle = matrix.handle
    assert handle is not None
    _validate_meta_shape(data, name)
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
        data, _, _ = _transform_pandas_df(data, False, meta=name, meta_type=dtype)
        _meta_from_numpy(data, name, dtype, handle)
        return
    if _is_pandas_series(data):
        _meta_from_pandas_series(data, name, dtype, handle)
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
        data, _, _ = _transform_pandas_df(data, False, meta=name, meta_type=dtype)
        _meta_from_numpy(data, name, dtype, handle)
        return
    if _is_modin_series(data):
        data = data.values.astype('float')
        assert len(data.shape) == 1 or data.shape[1] == 0 or data.shape[1] == 1
        _meta_from_numpy(data, name, dtype, handle)
        return
    if _has_array_protocol(data):
        # pyarrow goes here.
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
    def __init__(self, **kwargs: Any) -> None:
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
    data: DataType,
    feature_names: Optional[FeatureNames],
    feature_types: Optional[FeatureTypes],
    enable_categorical: bool,
) -> Tuple[
    Union[bool, ctypes.c_void_p, np.ndarray],
        Optional[list], Optional[FeatureNames], Optional[FeatureTypes]]:
    if _is_cudf_df(data) or _is_cudf_ser(data):
        return _transform_cudf_df(
            data, feature_names, feature_types, enable_categorical
        )
    if _is_cupy_array(data):
        data = _transform_cupy_array(data)
        return data, None, feature_names, feature_types
    if _is_dlpack(data):
        return _transform_dlpack(data), None, feature_names, feature_types
    if _is_numpy_array(data):
        data, _ = _ensure_np_dtype(data, data.dtype)
        return data, None, feature_names, feature_types
    if _is_scipy_csr(data):
        data = _transform_scipy_csr(data)
        return data, None, feature_names, feature_types
    if _is_pandas_df(data):
        arr, feature_names, feature_types = _transform_pandas_df(
            data, enable_categorical, feature_names, feature_types
        )
        arr, _ = _ensure_np_dtype(arr, arr.dtype)
        return arr, None, feature_names, feature_types
    raise TypeError("Value type is not supported for data iterator:" + str(type(data)))


def dispatch_proxy_set_data(
    proxy: _ProxyDMatrix,
    data: DataType,
    cat_codes: Optional[list],
    allow_host: bool,
) -> None:
    """Dispatch for DeviceQuantileDMatrix."""
    if not _is_cudf_ser(data) and not _is_pandas_series(data):
        _check_data_shape(data)

    if _is_cudf_df(data):
        # pylint: disable=W0212
        proxy._set_data_from_cuda_columnar(data, cast(List, cat_codes))
        return
    if _is_cudf_ser(data):
        # pylint: disable=W0212
        proxy._set_data_from_cuda_columnar(data, cast(List, cat_codes))
        return
    if _is_cupy_array(data):
        proxy._set_data_from_cuda_interface(data)  # pylint: disable=W0212
        return
    if _is_dlpack(data):
        data = _transform_dlpack(data)
        proxy._set_data_from_cuda_interface(data)  # pylint: disable=W0212
        return

    err = TypeError("Value type is not supported for data iterator:" + str(type(data)))

    if not allow_host:
        raise err

    if _is_numpy_array(data):
        proxy._set_data_from_array(data)  # pylint: disable=W0212
        return
    if _is_scipy_csr(data):
        proxy._set_data_from_csr(data)  # pylint: disable=W0212
        return
    raise err
