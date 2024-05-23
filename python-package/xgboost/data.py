# pylint: disable=too-many-arguments, too-many-branches, too-many-lines
# pylint: disable=too-many-return-statements, import-error
"""Data dispatching for DMatrix."""
import ctypes
import json
import os
import warnings
from typing import Any, Callable, List, Optional, Sequence, Tuple, cast

import numpy as np

from ._typing import (
    CupyT,
    DataType,
    FeatureNames,
    FeatureTypes,
    FloatCompatible,
    NumpyDType,
    PandasDType,
    TransformedData,
    c_bst_ulong,
)
from .compat import DataFrame, lazy_isinstance
from .core import (
    _LIB,
    DataIter,
    DataSplitMode,
    DMatrix,
    _array_hasobject,
    _check_call,
    _cuda_array_interface,
    _ProxyDMatrix,
    c_str,
    from_pystr_to_cstr,
    make_jcargs,
)

DispatchedDataBackendReturnType = Tuple[
    ctypes.c_void_p, Optional[FeatureNames], Optional[FeatureTypes]
]

CAT_T = "c"

# meta info that can be a matrix instead of vector.
_matrix_meta = {"base_margin", "label"}


def _warn_unused_missing(data: DataType, missing: Optional[FloatCompatible]) -> None:
    if (missing is not None) and (not np.isnan(missing)):
        warnings.warn(
            "`missing` is not used for current input data type:" + str(type(data)),
            UserWarning,
        )


def _check_data_shape(data: DataType) -> None:
    if hasattr(data, "shape") and len(data.shape) != 2:
        raise ValueError("Please reshape the input data into 2-dimensional matrix.")


def is_scipy_csr(data: DataType) -> bool:
    """Predicate for scipy CSR input."""
    is_array = False
    is_matrix = False
    try:
        from scipy.sparse import csr_array

        is_array = isinstance(data, csr_array)
    except ImportError:
        pass
    try:
        from scipy.sparse import csr_matrix

        is_matrix = isinstance(data, csr_matrix)
    except ImportError:
        pass
    return is_array or is_matrix


def _array_interface_dict(data: np.ndarray) -> dict:
    if _array_hasobject(data):
        raise ValueError("Input data contains `object` dtype.  Expecting numeric data.")
    interface = data.__array_interface__
    if "mask" in interface:
        interface["mask"] = interface["mask"].__array_interface__
    return interface


def _array_interface(data: np.ndarray) -> bytes:
    interface = _array_interface_dict(data)
    interface_str = bytes(json.dumps(interface), "utf-8")
    return interface_str


def transform_scipy_sparse(data: DataType, is_csr: bool) -> DataType:
    """Ensure correct data alignment and data type for scipy sparse inputs. Input should
    be either csr or csc matrix.

    """
    from scipy.sparse import csc_matrix, csr_matrix

    if len(data.indices) != len(data.data):
        raise ValueError(f"length mismatch: {len(data.indices)} vs {len(data.data)}")

    indptr, _ = _ensure_np_dtype(data.indptr, data.indptr.dtype)
    indices, _ = _ensure_np_dtype(data.indices, data.indices.dtype)
    values, _ = _ensure_np_dtype(data.data, data.data.dtype)
    if (
        indptr is not data.indptr
        or indices is not data.indices
        or values is not data.data
    ):
        if is_csr:
            data = csr_matrix((values, indices, indptr), shape=data.shape)
        else:
            data = csc_matrix((values, indices, indptr), shape=data.shape)
    return data


def _from_scipy_csr(
    data: DataType,
    missing: FloatCompatible,
    nthread: int,
    feature_names: Optional[FeatureNames],
    feature_types: Optional[FeatureTypes],
    data_split_mode: DataSplitMode = DataSplitMode.ROW,
) -> DispatchedDataBackendReturnType:
    """Initialize data from a CSR matrix."""

    handle = ctypes.c_void_p()
    data = transform_scipy_sparse(data, True)
    _check_call(
        _LIB.XGDMatrixCreateFromCSR(
            _array_interface(data.indptr),
            _array_interface(data.indices),
            _array_interface(data.data),
            c_bst_ulong(data.shape[1]),
            make_jcargs(
                missing=float(missing),
                nthread=int(nthread),
                data_split_mode=int(data_split_mode),
            ),
            ctypes.byref(handle),
        )
    )
    return handle, feature_names, feature_types


def is_scipy_csc(data: DataType) -> bool:
    """Predicate for scipy CSC input."""
    is_array = False
    is_matrix = False
    try:
        from scipy.sparse import csc_array

        is_array = isinstance(data, csc_array)
    except ImportError:
        pass
    try:
        from scipy.sparse import csc_matrix

        is_matrix = isinstance(data, csc_matrix)
    except ImportError:
        pass
    return is_array or is_matrix


def _from_scipy_csc(
    data: DataType,
    missing: FloatCompatible,
    nthread: int,
    feature_names: Optional[FeatureNames],
    feature_types: Optional[FeatureTypes],
    data_split_mode: DataSplitMode = DataSplitMode.ROW,
) -> DispatchedDataBackendReturnType:
    """Initialize data from a CSC matrix."""
    handle = ctypes.c_void_p()
    transform_scipy_sparse(data, False)
    _check_call(
        _LIB.XGDMatrixCreateFromCSC(
            _array_interface(data.indptr),
            _array_interface(data.indices),
            _array_interface(data.data),
            c_bst_ulong(data.shape[0]),
            make_jcargs(
                missing=float(missing),
                nthread=int(nthread),
                data_split_mode=int(data_split_mode),
            ),
            ctypes.byref(handle),
        )
    )
    return handle, feature_names, feature_types


def is_scipy_coo(data: DataType) -> bool:
    """Predicate for scipy COO input."""
    is_array = False
    is_matrix = False
    try:
        from scipy.sparse import coo_array

        is_array = isinstance(data, coo_array)
    except ImportError:
        pass
    try:
        from scipy.sparse import coo_matrix

        is_matrix = isinstance(data, coo_matrix)
    except ImportError:
        pass
    return is_array or is_matrix


def _is_np_array_like(data: DataType) -> bool:
    return hasattr(data, "__array_interface__")


def _ensure_np_dtype(
    data: DataType, dtype: Optional[NumpyDType]
) -> Tuple[np.ndarray, Optional[NumpyDType]]:
    if _array_hasobject(data) or data.dtype in [np.float16, np.bool_]:
        dtype = np.float32
        data = data.astype(dtype, copy=False)
    if not data.flags.aligned:
        data = np.require(data, requirements="A")
    return data, dtype


def _maybe_np_slice(data: DataType, dtype: Optional[NumpyDType]) -> np.ndarray:
    """Handle numpy slice.  This can be removed if we use __array_interface__."""
    try:
        if not data.flags.c_contiguous:
            data = np.array(data, copy=True, dtype=dtype)
        else:
            data = np.asarray(data, dtype=dtype)
    except AttributeError:
        data = np.asarray(data, dtype=dtype)
    data, dtype = _ensure_np_dtype(data, dtype)
    return data


def _from_numpy_array(
    data: DataType,
    missing: FloatCompatible,
    nthread: int,
    feature_names: Optional[FeatureNames],
    feature_types: Optional[FeatureTypes],
    data_split_mode: DataSplitMode = DataSplitMode.ROW,
) -> DispatchedDataBackendReturnType:
    """Initialize data from a 2-D numpy matrix."""
    _check_data_shape(data)
    data, _ = _ensure_np_dtype(data, data.dtype)
    handle = ctypes.c_void_p()
    _check_call(
        _LIB.XGDMatrixCreateFromDense(
            _array_interface(data),
            make_jcargs(
                missing=float(missing),
                nthread=int(nthread),
                data_split_mode=int(data_split_mode),
            ),
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
    "int8": "int",
    "int16": "int",
    "int32": "int",
    "int64": "int",
    "uint8": "int",
    "uint16": "int",
    "uint32": "int",
    "uint64": "int",
    "float16": "float",
    "float32": "float",
    "float64": "float",
    "bool": "i",
}

# nullable types
pandas_nullable_mapper = {
    "Int8": "int",
    "Int16": "int",
    "Int32": "int",
    "Int64": "int",
    "UInt8": "int",
    "UInt16": "int",
    "UInt32": "int",
    "UInt64": "int",
    "Float32": "float",
    "Float64": "float",
    "boolean": "i",
}

pandas_pyarrow_mapper = {
    "int8[pyarrow]": "int",
    "int16[pyarrow]": "int",
    "int32[pyarrow]": "int",
    "int64[pyarrow]": "int",
    "uint8[pyarrow]": "int",
    "uint16[pyarrow]": "int",
    "uint32[pyarrow]": "int",
    "uint64[pyarrow]": "int",
    "float[pyarrow]": "float",
    "float32[pyarrow]": "float",
    "double[pyarrow]": "float",
    "float64[pyarrow]": "float",
    "bool[pyarrow]": "i",
}

_pandas_dtype_mapper.update(pandas_nullable_mapper)
_pandas_dtype_mapper.update(pandas_pyarrow_mapper)


_ENABLE_CAT_ERR = (
    "When categorical type is supplied, the experimental DMatrix parameter"
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


def pandas_feature_info(
    data: DataFrame,
    meta: Optional[str],
    feature_names: Optional[FeatureNames],
    feature_types: Optional[FeatureTypes],
    enable_categorical: bool,
) -> Tuple[Optional[FeatureNames], Optional[FeatureTypes]]:
    """Handle feature info for pandas dataframe."""
    import pandas as pd

    # handle feature names
    if feature_names is None and meta is None:
        if isinstance(data.columns, pd.MultiIndex):
            feature_names = [" ".join([str(x) for x in i]) for i in data.columns]
        else:
            feature_names = list(data.columns.map(str))

    # handle feature types
    if feature_types is None and meta is None:
        feature_types = []
        for dtype in data.dtypes:
            if is_pd_sparse_dtype(dtype):
                feature_types.append(_pandas_dtype_mapper[dtype.subtype.name])
            elif (
                is_pd_cat_dtype(dtype) or is_pa_ext_categorical_dtype(dtype)
            ) and enable_categorical:
                feature_types.append(CAT_T)
            else:
                feature_types.append(_pandas_dtype_mapper[dtype.name])
    return feature_names, feature_types


def is_nullable_dtype(dtype: PandasDType) -> bool:
    """Whether dtype is a pandas nullable type."""
    from pandas.api.types import is_bool_dtype, is_float_dtype, is_integer_dtype

    is_int = is_integer_dtype(dtype) and dtype.name in pandas_nullable_mapper
    # np.bool has alias `bool`, while pd.BooleanDtype has `boolean`.
    is_bool = is_bool_dtype(dtype) and dtype.name == "boolean"
    is_float = is_float_dtype(dtype) and dtype.name in pandas_nullable_mapper
    return is_int or is_bool or is_float or is_pd_cat_dtype(dtype)


def is_pa_ext_dtype(dtype: Any) -> bool:
    """Return whether dtype is a pyarrow extension type for pandas"""
    return hasattr(dtype, "pyarrow_dtype")


def is_pa_ext_categorical_dtype(dtype: Any) -> bool:
    """Check whether dtype is a dictionary type."""
    return lazy_isinstance(
        getattr(dtype, "pyarrow_dtype", None), "pyarrow.lib", "DictionaryType"
    )


def is_pd_cat_dtype(dtype: PandasDType) -> bool:
    """Wrapper for testing pandas category type."""
    import pandas as pd

    if hasattr(pd.util, "version") and hasattr(pd.util.version, "Version"):
        Version = pd.util.version.Version
        if Version(pd.__version__) >= Version("2.1.0"):
            from pandas import CategoricalDtype

            return isinstance(dtype, CategoricalDtype)

    from pandas.api.types import is_categorical_dtype

    return is_categorical_dtype(dtype)


def is_pd_sparse_dtype(dtype: PandasDType) -> bool:
    """Wrapper for testing pandas sparse type."""
    import pandas as pd

    if hasattr(pd.util, "version") and hasattr(pd.util.version, "Version"):
        Version = pd.util.version.Version
        if Version(pd.__version__) >= Version("2.1.0"):
            from pandas import SparseDtype

            return isinstance(dtype, SparseDtype)

    from pandas.api.types import is_sparse

    return is_sparse(dtype)


def pandas_pa_type(ser: Any) -> np.ndarray:
    """Handle pandas pyarrow extention."""
    import pandas as pd
    import pyarrow as pa

    # No copy, callstack:
    # pandas.core.internals.managers.SingleBlockManager.array_values()
    # pandas.core.internals.blocks.EABackedBlock.values
    d_array: pd.arrays.ArrowExtensionArray = ser.array
    # no copy in __arrow_array__
    # ArrowExtensionArray._data is a chunked array
    aa: pa.ChunkedArray = d_array.__arrow_array__()
    # combine_chunks takes the most significant amount of time
    chunk: pa.Array = aa.combine_chunks()
    # When there's null value, we have to use copy
    zero_copy = chunk.null_count == 0
    # Alternately, we can use chunk.buffers(), which returns a list of buffers and
    # we need to concatenate them ourselves.
    # FIXME(jiamingy): Is there a better way to access the arrow buffer along with
    # its mask?
    # Buffers from chunk.buffers() have the address attribute, but don't expose the
    # mask.
    arr: np.ndarray = chunk.to_numpy(zero_copy_only=zero_copy, writable=False)
    arr, _ = _ensure_np_dtype(arr, arr.dtype)
    return arr


def pandas_transform_data(data: DataFrame) -> List[np.ndarray]:
    """Handle categorical dtype and extension types from pandas."""
    import pandas as pd
    from pandas import Float32Dtype, Float64Dtype

    result: List[np.ndarray] = []

    def cat_codes(ser: pd.Series) -> np.ndarray:
        if is_pd_cat_dtype(ser.dtype):
            return _ensure_np_dtype(
                ser.cat.codes.astype(np.float32)
                .replace(-1.0, np.nan)
                .to_numpy(na_value=np.nan),
                np.float32,
            )[0]
        # Not yet supported, the index is not ordered for some reason. Alternately:
        # `combine_chunks().to_pandas().cat.codes`. The result is the same.
        assert is_pa_ext_categorical_dtype(ser.dtype)
        return (
            ser.array.__arrow_array__()
            .combine_chunks()
            .dictionary_encode()
            .indices.astype(np.float32)
            .replace(-1.0, np.nan)
        )

    def nu_type(ser: pd.Series) -> np.ndarray:
        # Avoid conversion when possible
        if isinstance(dtype, Float32Dtype):
            res_dtype: NumpyDType = np.float32
        elif isinstance(dtype, Float64Dtype):
            res_dtype = np.float64
        else:
            res_dtype = np.float32
        return _ensure_np_dtype(
            ser.to_numpy(dtype=res_dtype, na_value=np.nan), res_dtype
        )[0]

    def oth_type(ser: pd.Series) -> np.ndarray:
        # The dtypes module is added in 1.25.
        npdtypes = np.lib.NumpyVersion(np.__version__) > np.lib.NumpyVersion("1.25.0")
        npdtypes = npdtypes and isinstance(
            ser.dtype,
            (
                # pylint: disable=no-member
                np.dtypes.Float32DType,  # type: ignore
                # pylint: disable=no-member
                np.dtypes.Float64DType,  # type: ignore
            ),
        )

        if npdtypes or dtype in {np.float32, np.float64}:
            array = ser.to_numpy()
        else:
            # Specifying the dtype can significantly slow down the conversion (about
            # 15% slow down for dense inplace-predict)
            array = ser.to_numpy(dtype=np.float32, na_value=np.nan)
        return _ensure_np_dtype(array, array.dtype)[0]

    for col, dtype in zip(data.columns, data.dtypes):
        if is_pa_ext_categorical_dtype(dtype):
            raise ValueError(
                "pyarrow dictionary type is not supported. Use pandas category instead."
            )
        if is_pd_cat_dtype(dtype):
            result.append(cat_codes(data[col]))
        elif is_pa_ext_dtype(dtype):
            result.append(pandas_pa_type(data[col]))
        elif is_nullable_dtype(dtype):
            result.append(nu_type(data[col]))
        elif is_pd_sparse_dtype(dtype):
            arr = cast(pd.arrays.SparseArray, data[col].values)
            arr = arr.to_dense()
            if _is_np_array_like(arr):
                arr, _ = _ensure_np_dtype(arr, arr.dtype)
            result.append(arr)
        else:
            result.append(oth_type(data[col]))

    # FIXME(jiamingy): Investigate the possibility of using dataframe protocol or arrow
    # IPC format for pandas so that we can apply the data transformation inside XGBoost
    # for better memory efficiency.
    return result


def pandas_check_dtypes(data: DataFrame, enable_categorical: bool) -> None:
    """Validate the input types, returns True if the dataframe is backed by arrow."""
    sparse_extension = False

    for dtype in data.dtypes:
        if not (
            (dtype.name in _pandas_dtype_mapper)
            or is_pd_sparse_dtype(dtype)
            or (is_pd_cat_dtype(dtype) and enable_categorical)
            or is_pa_ext_dtype(dtype)
        ):
            _invalid_dataframe_dtype(data)

        if is_pd_sparse_dtype(dtype):
            sparse_extension = True

    if sparse_extension:
        warnings.warn("Sparse arrays from pandas are converted into dense.")


class PandasTransformed:
    """A storage class for transformed pandas DataFrame."""

    def __init__(self, columns: List[np.ndarray]) -> None:
        self.columns = columns

    def array_interface(self) -> bytes:
        """Return a byte string for JSON encoded array interface."""
        aitfs = list(map(_array_interface_dict, self.columns))
        sarrays = bytes(json.dumps(aitfs), "utf-8")
        return sarrays

    @property
    def shape(self) -> Tuple[int, int]:
        """Return shape of the transformed DataFrame."""
        return self.columns[0].shape[0], len(self.columns)


def _transform_pandas_df(
    data: DataFrame,
    enable_categorical: bool,
    feature_names: Optional[FeatureNames] = None,
    feature_types: Optional[FeatureTypes] = None,
    meta: Optional[str] = None,
) -> Tuple[PandasTransformed, Optional[FeatureNames], Optional[FeatureTypes]]:
    pandas_check_dtypes(data, enable_categorical)
    if meta and len(data.columns) > 1 and meta not in _matrix_meta:
        raise ValueError(f"DataFrame for {meta} cannot have multiple columns")

    feature_names, feature_types = pandas_feature_info(
        data, meta, feature_names, feature_types, enable_categorical
    )

    arrays = pandas_transform_data(data)
    return PandasTransformed(arrays), feature_names, feature_types


def _meta_from_pandas_df(
    data: DataType,
    name: str,
    dtype: Optional[NumpyDType],
    handle: ctypes.c_void_p,
) -> None:
    data, _, _ = _transform_pandas_df(data, False, meta=name)
    if len(data.columns) == 1:
        array = data.columns[0]
    else:
        array = np.stack(data.columns).T

    array, dtype = _ensure_np_dtype(array, dtype)
    _meta_from_numpy(array, name, dtype, handle)


def _from_pandas_df(
    data: DataFrame,
    enable_categorical: bool,
    missing: FloatCompatible,
    nthread: int,
    feature_names: Optional[FeatureNames],
    feature_types: Optional[FeatureTypes],
    data_split_mode: DataSplitMode = DataSplitMode.ROW,
) -> DispatchedDataBackendReturnType:
    df, feature_names, feature_types = _transform_pandas_df(
        data, enable_categorical, feature_names, feature_types
    )

    handle = ctypes.c_void_p()
    _check_call(
        _LIB.XGDMatrixCreateFromColumnar(
            df.array_interface(),
            make_jcargs(
                nthread=nthread, missing=missing, data_split_mode=data_split_mode
            ),
            ctypes.byref(handle),
        )
    )
    return handle, feature_names, feature_types


def _is_pandas_series(data: DataType) -> bool:
    try:
        import pandas as pd
    except ImportError:
        return False
    return isinstance(data, pd.Series)


def _meta_from_pandas_series(
    data: DataType, name: str, dtype: Optional[NumpyDType], handle: ctypes.c_void_p
) -> None:
    """Help transform pandas series for meta data like labels"""
    if is_pd_sparse_dtype(data.dtype):
        data = data.values.to_dense().astype(np.float32)
    elif is_pa_ext_dtype(data.dtype):
        data = pandas_pa_type(data)
    else:
        data = data.to_numpy(np.float32, na_value=np.nan)

    if is_pd_sparse_dtype(getattr(data, "dtype", data)):
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
    if (data.dtype.name not in _pandas_dtype_mapper) and not (
        is_pd_cat_dtype(data.dtype) and enable_categorical
    ):
        _invalid_dataframe_dtype(data)
    if enable_categorical and is_pd_cat_dtype(data.dtype):
        data = data.cat.codes
    return _from_numpy_array(
        data.values.reshape(data.shape[0], 1).astype("float"),
        missing,
        nthread,
        feature_names,
        feature_types,
    )


def _is_dt_df(data: DataType) -> bool:
    return lazy_isinstance(data, "datatable", "Frame") or lazy_isinstance(
        data, "datatable", "DataTable"
    )


def _transform_dt_df(
    data: DataType,
    feature_names: Optional[FeatureNames],
    feature_types: Optional[FeatureTypes],
    meta: Optional[str] = None,
    meta_type: Optional[NumpyDType] = None,
) -> Tuple[np.ndarray, Optional[FeatureNames], Optional[FeatureTypes]]:
    """Validate feature names and types if data table"""
    _dt_type_mapper = {"bool": "bool", "int": "int", "real": "float"}
    _dt_type_mapper2 = {"bool": "i", "int": "int", "real": "float"}
    if meta and data.shape[1] > 1:
        raise ValueError("DataTable for meta info cannot have multiple columns")
    if meta:
        meta_type = "float" if meta_type is None else meta_type
        # below requires new dt version
        # extract first column
        data = data.to_numpy()[:, 0].astype(meta_type)
        return data, None, None

    data_types_names = tuple(lt.name for lt in data.ltypes)
    bad_fields = [
        data.names[i]
        for i, type_name in enumerate(data_types_names)
        if type_name not in _dt_type_mapper
    ]
    if bad_fields:
        msg = """DataFrame.types for data must be int, float or bool.
                Did not expect the data types in fields """
        raise ValueError(msg + ", ".join(bad_fields))

    if feature_names is None and meta is None:
        feature_names = data.names

        # always return stypes for dt ingestion
        if feature_types is not None:
            raise ValueError("DataTable has own feature types, cannot pass them in.")
        feature_types = np.vectorize(_dt_type_mapper2.get)(data_types_names).tolist()

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
        data, feature_names, feature_types, None, None
    )

    ptrs = (ctypes.c_void_p * data.ncols)()
    if hasattr(data, "internal") and hasattr(data.internal, "column"):
        # datatable>0.8.0
        for icol in range(data.ncols):
            col = data.internal.column(icol)
            ptr = col.data_pointer
            ptrs[icol] = ctypes.c_void_p(ptr)
    else:
        # datatable<=0.8.0
        from datatable.internal import (
            frame_column_data_r,  # pylint: disable=no-name-in-module
        )

        for icol in range(data.ncols):
            ptrs[icol] = frame_column_data_r(data, icol)

    # always return stypes for dt ingestion
    feature_type_strings = (ctypes.c_char_p * data.ncols)()
    for icol in range(data.ncols):
        feature_type_strings[icol] = ctypes.c_char_p(
            data.stypes[icol].name.encode("utf-8")
        )

    _warn_unused_missing(data, missing)
    handle = ctypes.c_void_p()
    _check_call(
        _LIB.XGDMatrixCreateFromDT(
            ptrs,
            feature_type_strings,
            c_bst_ulong(data.shape[0]),
            c_bst_ulong(data.shape[1]),
            ctypes.byref(handle),
            ctypes.c_int(nthread),
        )
    )
    return handle, feature_names, feature_types


def _is_arrow(data: DataType) -> bool:
    return lazy_isinstance(data, "pyarrow.lib", "Table") or lazy_isinstance(
        data, "pyarrow._dataset", "Dataset"
    )


def _arrow_transform(data: DataType) -> Any:
    import pandas as pd
    import pyarrow as pa
    from pyarrow.dataset import Dataset

    if isinstance(data, Dataset):
        raise TypeError("arrow Dataset is not supported.")

    data = cast(pa.Table, data)

    def type_mapper(dtype: pa.DataType) -> Optional[str]:
        """Maps pyarrow type to pandas arrow extension type."""
        if pa.types.is_int8(dtype):
            return pd.ArrowDtype(pa.int8())
        if pa.types.is_int16(dtype):
            return pd.ArrowDtype(pa.int16())
        if pa.types.is_int32(dtype):
            return pd.ArrowDtype(pa.int32())
        if pa.types.is_int64(dtype):
            return pd.ArrowDtype(pa.int64())
        if pa.types.is_uint8(dtype):
            return pd.ArrowDtype(pa.uint8())
        if pa.types.is_uint16(dtype):
            return pd.ArrowDtype(pa.uint16())
        if pa.types.is_uint32(dtype):
            return pd.ArrowDtype(pa.uint32())
        if pa.types.is_uint64(dtype):
            return pd.ArrowDtype(pa.uint64())
        if pa.types.is_float16(dtype):
            return pd.ArrowDtype(pa.float16())
        if pa.types.is_float32(dtype):
            return pd.ArrowDtype(pa.float32())
        if pa.types.is_float64(dtype):
            return pd.ArrowDtype(pa.float64())
        if pa.types.is_boolean(dtype):
            return pd.ArrowDtype(pa.bool_())
        return None

    # For common cases, this is zero-copy, can check with:
    # pa.total_allocated_bytes()
    df = data.to_pandas(types_mapper=type_mapper)
    return df


def _is_cudf_df(data: DataType) -> bool:
    return lazy_isinstance(data, "cudf.core.dataframe", "DataFrame")


def _get_cudf_cat_predicate() -> Callable[[Any], bool]:
    try:
        from cudf import CategoricalDtype

        def is_categorical_dtype(dtype: Any) -> bool:
            return isinstance(dtype, CategoricalDtype)

    except ImportError:
        try:
            from cudf.api.types import is_categorical_dtype  # type: ignore
        except ImportError:
            from cudf.utils.dtypes import is_categorical_dtype  # type: ignore

    return is_categorical_dtype


def _cudf_array_interfaces(data: DataType, cat_codes: list) -> bytes:
    """Extract CuDF __cuda_array_interface__.  This is special as it returns a new list
    of data and a list of array interfaces.  The data is list of categorical codes that
    caller can safely ignore, but have to keep their reference alive until usage of
    array interface is finished.

    """
    is_categorical_dtype = _get_cudf_cat_predicate()
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
        from cudf.api.types import is_bool_dtype
    except ImportError:
        from pandas.api.types import is_bool_dtype

    is_categorical_dtype = _get_cudf_cat_predicate()
    # Work around https://github.com/dmlc/xgboost/issues/10181
    if _is_cudf_ser(data):
        if is_bool_dtype(data.dtype):
            data = data.astype(np.uint8)
    else:
        data = data.astype(
            {col: np.uint8 for col in data.select_dtypes(include="bool")}
        )

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
        else:
            feature_names = list(data.columns.map(str))

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
    _check_call(
        _LIB.XGDMatrixCreateFromCudaColumnar(
            interfaces_str,
            make_jcargs(nthread=nthread, missing=missing),
            ctypes.byref(handle),
        )
    )
    return handle, feature_names, feature_types


def _is_cudf_ser(data: DataType) -> bool:
    return lazy_isinstance(data, "cudf.core.series", "Series")


def _is_cupy_alike(data: DataType) -> bool:
    return hasattr(data, "__cuda_array_interface__")


def _transform_cupy_array(data: DataType) -> CupyT:
    import cupy  # pylint: disable=import-error

    if not hasattr(data, "__cuda_array_interface__") and hasattr(data, "__array__"):
        data = cupy.array(data, copy=False)
    if _array_hasobject(data) or data.dtype in [cupy.bool_]:
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
            interface_str, config, ctypes.byref(handle)
        )
    )
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
    return "PyCapsule" in str(type(data)) and "dltensor" in str(data)


def _transform_dlpack(data: DataType) -> bool:
    from cupy import from_dlpack  # pylint: disable=E0401

    assert "used_dltensor" not in str(data)
    data = from_dlpack(data)
    return data


def _from_dlpack(
    data: DataType,
    missing: FloatCompatible,
    nthread: int,
    feature_names: Optional[FeatureNames],
    feature_types: Optional[FeatureTypes],
) -> DispatchedDataBackendReturnType:
    data = _transform_dlpack(data)
    return _from_cupy_array(data, missing, nthread, feature_names, feature_types)


def _is_uri(data: DataType) -> bool:
    return isinstance(data, (str, os.PathLike))


def _from_uri(
    data: DataType,
    missing: Optional[FloatCompatible],
    feature_names: Optional[FeatureNames],
    feature_types: Optional[FeatureTypes],
    data_split_mode: DataSplitMode = DataSplitMode.ROW,
) -> DispatchedDataBackendReturnType:
    _warn_unused_missing(data, missing)
    handle = ctypes.c_void_p()
    data = os.fspath(os.path.expanduser(data))
    args = {
        "uri": str(data),
        "data_split_mode": int(data_split_mode),
    }
    config = bytes(json.dumps(args), "utf-8")
    _check_call(_LIB.XGDMatrixCreateFromURI(config, ctypes.byref(handle)))
    return handle, feature_names, feature_types


def _is_list(data: DataType) -> bool:
    return isinstance(data, list)


def _from_list(
    data: Sequence,
    missing: FloatCompatible,
    n_threads: int,
    feature_names: Optional[FeatureNames],
    feature_types: Optional[FeatureTypes],
    data_split_mode: DataSplitMode = DataSplitMode.ROW,
) -> DispatchedDataBackendReturnType:
    array = np.array(data)
    _check_data_shape(data)
    return _from_numpy_array(
        array, missing, n_threads, feature_names, feature_types, data_split_mode
    )


def _is_tuple(data: DataType) -> bool:
    return isinstance(data, tuple)


def _from_tuple(
    data: Sequence,
    missing: FloatCompatible,
    n_threads: int,
    feature_names: Optional[FeatureNames],
    feature_types: Optional[FeatureTypes],
    data_split_mode: DataSplitMode = DataSplitMode.ROW,
) -> DispatchedDataBackendReturnType:
    return _from_list(
        data, missing, n_threads, feature_names, feature_types, data_split_mode
    )


def _is_iter(data: DataType) -> bool:
    return isinstance(data, DataIter)


def _has_array_protocol(data: DataType) -> bool:
    return hasattr(data, "__array__")


def _convert_unknown_data(data: DataType) -> DataType:
    warnings.warn(
        f"Unknown data type: {type(data)}, trying to convert it to csr_matrix",
        UserWarning,
    )
    try:
        import scipy.sparse
    except ImportError:
        return None

    try:
        data = scipy.sparse.csr_matrix(data)
    except Exception:  # pylint: disable=broad-except
        return None

    return data


def dispatch_data_backend(
    data: DataType,
    missing: FloatCompatible,  # Or Optional[Float]
    threads: int,
    feature_names: Optional[FeatureNames],
    feature_types: Optional[FeatureTypes],
    enable_categorical: bool = False,
    data_split_mode: DataSplitMode = DataSplitMode.ROW,
) -> DispatchedDataBackendReturnType:
    """Dispatch data for DMatrix."""
    if not _is_cudf_ser(data) and not _is_pandas_series(data):
        _check_data_shape(data)
    if is_scipy_csr(data):
        return _from_scipy_csr(
            data, missing, threads, feature_names, feature_types, data_split_mode
        )
    if is_scipy_csc(data):
        return _from_scipy_csc(
            data, missing, threads, feature_names, feature_types, data_split_mode
        )
    if is_scipy_coo(data):
        return _from_scipy_csr(
            data.tocsr(),
            missing,
            threads,
            feature_names,
            feature_types,
            data_split_mode,
        )
    if _is_np_array_like(data):
        return _from_numpy_array(
            data, missing, threads, feature_names, feature_types, data_split_mode
        )
    if _is_uri(data):
        return _from_uri(data, missing, feature_names, feature_types, data_split_mode)
    if _is_list(data):
        return _from_list(
            data, missing, threads, feature_names, feature_types, data_split_mode
        )
    if _is_tuple(data):
        return _from_tuple(
            data, missing, threads, feature_names, feature_types, data_split_mode
        )
    if _is_arrow(data):
        data = _arrow_transform(data)
    if _is_pandas_series(data):
        import pandas as pd

        data = pd.DataFrame(data)
    if _is_pandas_df(data):
        return _from_pandas_df(
            data,
            enable_categorical,
            missing,
            threads,
            feature_names,
            feature_types,
            data_split_mode,
        )
    if _is_cudf_df(data) or _is_cudf_ser(data):
        return _from_cudf_df(
            data, missing, threads, feature_names, feature_types, enable_categorical
        )
    if _is_cupy_alike(data):
        return _from_cupy_array(data, missing, threads, feature_names, feature_types)
    if _is_cupy_csr(data):
        raise TypeError("cupyx CSR is not supported yet.")
    if _is_cupy_csc(data):
        raise TypeError("cupyx CSC is not supported yet.")
    if _is_dlpack(data):
        return _from_dlpack(data, missing, threads, feature_names, feature_types)
    if _is_dt_df(data):
        _warn_unused_missing(data, missing)
        return _from_dt_df(
            data, missing, threads, feature_names, feature_types, enable_categorical
        )
    if _is_modin_df(data):
        return _from_pandas_df(
            data, enable_categorical, missing, threads, feature_names, feature_types
        )
    if _is_modin_series(data):
        return _from_pandas_series(
            data, missing, threads, enable_categorical, feature_names, feature_types
        )
    if _has_array_protocol(data):
        array = np.asarray(data)
        return _from_numpy_array(array, missing, threads, feature_names, feature_types)

    converted = _convert_unknown_data(data)
    if converted is not None:
        return _from_scipy_csr(
            converted, missing, threads, feature_names, feature_types
        )

    raise TypeError("Not supported type for data." + str(type(data)))


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
    data: Sequence, field: str, dtype: Optional[NumpyDType], handle: ctypes.c_void_p
) -> None:
    data_np = np.array(data)
    _meta_from_numpy(data_np, field, dtype, handle)


def _meta_from_tuple(
    data: Sequence, field: str, dtype: Optional[NumpyDType], handle: ctypes.c_void_p
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
    interface = bytes(json.dumps([data.__cuda_array_interface__], indent=2), "utf-8")
    _check_call(_LIB.XGDMatrixSetInfoFromInterface(handle, c_str(field), interface))


def _meta_from_cupy_array(data: DataType, field: str, handle: ctypes.c_void_p) -> None:
    data = _transform_cupy_array(data)
    interface = bytes(json.dumps([data.__cuda_array_interface__], indent=2), "utf-8")
    _check_call(_LIB.XGDMatrixSetInfoFromInterface(handle, c_str(field), interface))


def _meta_from_dt(
    data: DataType, field: str, dtype: Optional[NumpyDType], handle: ctypes.c_void_p
) -> None:
    data, _, _ = _transform_dt_df(data, None, None, field, dtype)
    _meta_from_numpy(data, field, dtype, handle)


def dispatch_meta_backend(
    matrix: DMatrix, data: DataType, name: str, dtype: Optional[NumpyDType] = None
) -> None:
    """Dispatch for meta info."""
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
    if _is_np_array_like(data):
        _meta_from_numpy(data, name, dtype, handle)
        return
    if _is_arrow(data):
        data = _arrow_transform(data)
    if _is_pandas_df(data):
        _meta_from_pandas_df(data, name, dtype=dtype, handle=handle)
        return
    if _is_pandas_series(data):
        _meta_from_pandas_series(data, name, dtype, handle)
        return
    if _is_dlpack(data):
        data = _transform_dlpack(data)
        _meta_from_cupy_array(data, name, handle)
        return
    if _is_cupy_alike(data):
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
        _meta_from_pandas_df(data, name, dtype=dtype, handle=handle)
        return
    if _is_modin_series(data):
        data = data.values.astype("float")
        assert len(data.shape) == 1 or data.shape[1] == 0 or data.shape[1] == 1
        _meta_from_numpy(data, name, dtype, handle)
        return
    if _has_array_protocol(data):
        # pyarrow goes here.
        array = np.asarray(data)
        _meta_from_numpy(array, name, dtype, handle)
        return
    raise TypeError("Unsupported type for " + name, str(type(data)))


class SingleBatchInternalIter(DataIter):  # pylint: disable=R0902
    """An iterator for single batch data to help creating device DMatrix.
    Transforming input directly to histogram with normal single batch data API
    can not access weight for sketching.  So this iterator acts as a staging
    area for meta info.

    """

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.it = 0  # pylint: disable=invalid-name

        # This does not necessarily increase memory usage as the data transformation
        # might use memory.
        super().__init__(release_data=False)

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
) -> TransformedData:
    if _is_cudf_df(data) or _is_cudf_ser(data):
        return _transform_cudf_df(
            data, feature_names, feature_types, enable_categorical
        )
    if _is_cupy_alike(data):
        data = _transform_cupy_array(data)
        return data, None, feature_names, feature_types
    if _is_dlpack(data):
        return _transform_dlpack(data), None, feature_names, feature_types
    if _is_list(data) or _is_tuple(data):
        data = np.array(data)
    if _is_np_array_like(data):
        data, _ = _ensure_np_dtype(data, data.dtype)
        return data, None, feature_names, feature_types
    if is_scipy_csr(data):
        data = transform_scipy_sparse(data, True)
        return data, None, feature_names, feature_types
    if is_scipy_csc(data):
        data = transform_scipy_sparse(data.tocsr(), True)
        return data, None, feature_names, feature_types
    if is_scipy_coo(data):
        data = transform_scipy_sparse(data.tocsr(), True)
        return data, None, feature_names, feature_types
    if _is_pandas_series(data):
        import pandas as pd

        data = pd.DataFrame(data)
    if _is_arrow(data):
        data = _arrow_transform(data)
    if _is_pandas_df(data):
        df, feature_names, feature_types = _transform_pandas_df(
            data, enable_categorical, feature_names, feature_types
        )
        return df, None, feature_names, feature_types
    raise TypeError("Value type is not supported for data iterator:" + str(type(data)))


def dispatch_proxy_set_data(
    proxy: _ProxyDMatrix,
    data: DataType,
    cat_codes: Optional[list],
) -> None:
    """Dispatch for QuantileDMatrix."""
    if not _is_cudf_ser(data) and not _is_pandas_series(data):
        _check_data_shape(data)

    if _is_cudf_df(data):
        # pylint: disable=W0212
        proxy._ref_data_from_cuda_columnar(data, cast(List, cat_codes))
        return
    if _is_cudf_ser(data):
        # pylint: disable=W0212
        proxy._ref_data_from_cuda_columnar(data, cast(List, cat_codes))
        return
    if _is_cupy_alike(data):
        proxy._ref_data_from_cuda_interface(data)  # pylint: disable=W0212
        return
    if _is_dlpack(data):
        data = _transform_dlpack(data)
        proxy._ref_data_from_cuda_interface(data)  # pylint: disable=W0212
        return
    # Host
    if isinstance(data, PandasTransformed):
        proxy._ref_data_from_pandas(data)  # pylint: disable=W0212
        return
    if _is_np_array_like(data):
        _check_data_shape(data)
        proxy._ref_data_from_array(data)  # pylint: disable=W0212
        return
    if is_scipy_csr(data):
        proxy._ref_data_from_csr(data)  # pylint: disable=W0212
        return

    err = TypeError("Value type is not supported for data iterator:" + str(type(data)))
    raise err
