# pylint: disable=too-many-arguments, too-many-branches, too-many-lines
# pylint: disable=too-many-return-statements
"""Data dispatching for DMatrix."""

import ctypes
import functools
import json
import os
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
    TypeGuard,
    Union,
)

import numpy as np

from ._data_utils import (
    AifType,
    Categories,
    DfCatAccessor,
    TransformedDf,
    _arrow_array_inf,
    _ensure_np_dtype,
    _is_df_cat,
    array_hasobject,
    array_interface,
    array_interface_dict,
    arrow_cat_inf,
    check_cudf_meta,
    cuda_array_interface,
    cuda_array_interface_dict,
    cudf_cat_inf,
    get_ref_categories,
    is_arrow_dict,
    pd_cat_inf,
)
from ._typing import (
    CupyT,
    DataType,
    FeatureNames,
    FeatureTypes,
    FloatCompatible,
    NumpyDType,
    PandasDType,
    PathLike,
    TransformedData,
    c_bst_ulong,
)
from .compat import (
    _is_arrow,
    _is_cudf_df,
    _is_cudf_pandas,
    _is_cudf_ser,
    _is_modin_df,
    _is_modin_series,
    _is_pandas_df,
    _is_pandas_series,
    _is_polars,
    _is_polars_lazyframe,
    _is_polars_series,
    import_pandas,
    import_polars,
    import_pyarrow,
    is_pyarrow_available,
    lazy_isinstance,
)
from .core import (
    _LIB,
    DataIter,
    DataSplitMode,
    DMatrix,
    _check_call,
    _ProxyDMatrix,
    c_str,
    make_jcargs,
)

if TYPE_CHECKING:
    import pyarrow as pa
    from pandas import DataFrame as PdDataFrame
    from pandas import Series as PdSeries


DispatchedDataBackendReturnType: TypeAlias = Tuple[
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
    *,
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
            array_interface(data.indptr),
            array_interface(data.indices),
            array_interface(data.data),
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
    *,
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
            array_interface(data.indptr),
            array_interface(data.indices),
            array_interface(data.data),
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


def _is_np_array_like(data: DataType) -> TypeGuard[np.ndarray]:
    return hasattr(data, "__array_interface__")


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
    *,
    data: np.ndarray,
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
            array_interface(data),
            make_jcargs(
                missing=float(missing),
                nthread=int(nthread),
                data_split_mode=int(data_split_mode),
            ),
            ctypes.byref(handle),
        )
    )
    return handle, feature_names, feature_types


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
    data: "PdDataFrame",
    meta: Optional[str],
    feature_names: Optional[FeatureNames],
    feature_types: Optional[FeatureTypes],
    enable_categorical: bool,
) -> Tuple[Optional[FeatureNames], Optional[FeatureTypes]]:
    """Handle feature info for pandas dataframe."""
    pd = import_pandas()

    # handle feature names
    if feature_names is None and meta is None:
        if isinstance(data.columns, pd.MultiIndex):
            feature_names = [" ".join([str(x) for x in i]) for i in data.columns]
        else:
            feature_names = list(data.columns.map(str))

    # handle feature types and dtype validation
    new_feature_types = []
    need_sparse_extension_warn = True
    for dtype in data.dtypes:
        if is_pd_sparse_dtype(dtype):
            new_feature_types.append(_pandas_dtype_mapper[dtype.subtype.name])
            if need_sparse_extension_warn:
                warnings.warn("Sparse arrays from pandas are converted into dense.")
                need_sparse_extension_warn = False
        elif (
            is_pd_cat_dtype(dtype) or is_pa_ext_categorical_dtype(dtype)
        ) and enable_categorical:
            new_feature_types.append(CAT_T)
        else:
            try:
                new_feature_types.append(_pandas_dtype_mapper[dtype.name])
            except KeyError:
                _invalid_dataframe_dtype(data)

    if feature_types is None and meta is None:
        feature_types = new_feature_types

    return feature_names, feature_types


def is_nullable_dtype(dtype: PandasDType) -> bool:
    """Whether dtype is a pandas nullable type."""

    from pandas.api.extensions import ExtensionDtype

    if not isinstance(dtype, ExtensionDtype):
        return False

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


@functools.cache
def _lazy_load_pd_is_cat() -> Callable[[PandasDType], bool]:
    pd = import_pandas()

    if hasattr(pd.util, "version") and hasattr(pd.util.version, "Version"):
        Version = pd.util.version.Version
        if Version(pd.__version__) >= Version("2.1.0"):
            from pandas import CategoricalDtype

            def pd_is_cat_210(dtype: PandasDType) -> bool:
                return isinstance(dtype, CategoricalDtype)

            return pd_is_cat_210
    from pandas.api.types import is_categorical_dtype  # type: ignore

    return is_categorical_dtype


def is_pd_cat_dtype(dtype: PandasDType) -> bool:
    """Wrapper for testing pandas category type."""
    is_cat = _lazy_load_pd_is_cat()
    return is_cat(dtype)


@functools.cache
def _lazy_load_pd_is_sparse() -> Callable[[PandasDType], bool]:
    pd = import_pandas()

    if hasattr(pd.util, "version") and hasattr(pd.util.version, "Version"):
        Version = pd.util.version.Version
        if Version(pd.__version__) >= Version("2.1.0"):
            from pandas import SparseDtype

            def pd_is_sparse_210(dtype: PandasDType) -> bool:
                return isinstance(dtype, SparseDtype)

            return pd_is_sparse_210

    from pandas.api.types import is_sparse  # type: ignore

    return is_sparse


def is_pd_sparse_dtype(dtype: PandasDType) -> bool:
    """Wrapper for testing pandas sparse type."""
    is_sparse = _lazy_load_pd_is_sparse()

    return is_sparse(dtype)


def pandas_pa_type(ser: Any) -> np.ndarray:
    """Handle pandas pyarrow extention."""
    pd = import_pandas()

    if TYPE_CHECKING:
        import pyarrow as pa
    else:
        pa = import_pyarrow()

    # No copy, callstack:
    # pandas.core.internals.managers.SingleBlockManager.array_values()
    # pandas.core.internals.blocks.EABackedBlock.values
    d_array: pd.arrays.ArrowExtensionArray = ser.array  # type: ignore
    # no copy in __arrow_array__
    # ArrowExtensionArray._data is a chunked array
    aa: "pa.ChunkedArray" = d_array.__arrow_array__()
    # combine_chunks takes the most significant amount of time
    chunk: "pa.Array" = aa.combine_chunks()
    # When there's null value, we have to use copy
    zero_copy = chunk.null_count == 0 and not pa.types.is_boolean(chunk.type)
    # Alternately, we can use chunk.buffers(), which returns a list of buffers and
    # we need to concatenate them ourselves.
    # FIXME(jiamingy): Is there a better way to access the arrow buffer along with
    # its mask?
    # Buffers from chunk.buffers() have the address attribute, but don't expose the
    # mask.
    arr: np.ndarray = chunk.to_numpy(zero_copy_only=zero_copy, writable=False)
    arr, _ = _ensure_np_dtype(arr, arr.dtype)
    return arr


@functools.cache
def _lazy_has_npdtypes() -> bool:
    return np.lib.NumpyVersion(np.__version__) > np.lib.NumpyVersion("1.25.0")


@functools.cache
def _lazy_load_pd_floats() -> tuple:
    from pandas import Float32Dtype, Float64Dtype

    return Float32Dtype, Float64Dtype


def pandas_transform_data(
    data: "PdDataFrame",
) -> List[Union[np.ndarray, DfCatAccessor]]:
    """Handle categorical dtype and extension types from pandas."""
    Float32Dtype, Float64Dtype = _lazy_load_pd_floats()

    result: List[Union[np.ndarray, DfCatAccessor]] = []
    np_dtypes = _lazy_has_npdtypes()

    def cat_codes(ser: "PdSeries") -> DfCatAccessor:
        return ser.cat

    def nu_type(ser: "PdSeries") -> np.ndarray:
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

    def oth_type(ser: "PdSeries") -> np.ndarray:
        # The dtypes module is added in 1.25.
        npdtypes = np_dtypes and isinstance(
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
            arr = data[col].values
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


class PandasTransformed(TransformedDf):
    """A storage class for transformed pandas DataFrame."""

    def __init__(
        self,
        columns: List[Union[np.ndarray, DfCatAccessor]],
        ref_categories: Optional[Categories],
    ) -> None:
        self.columns = columns

        aitfs: AifType = []

        # Get the array interface representation for each column.
        for col in self.columns:
            if _is_df_cat(col):
                # Categorical column
                jnames, jcodes, buf = pd_cat_inf(col.categories, col.codes)
                self.temporary_buffers.append(buf)
                aitfs.append((jnames, jcodes))
            else:
                assert isinstance(col, np.ndarray)
                inf = array_interface_dict(col)
                # Numeric column
                aitfs.append(inf)

        super().__init__(ref_categories=ref_categories, aitfs=aitfs)

    @property
    def shape(self) -> Tuple[int, int]:
        """Return shape of the transformed DataFrame."""
        if is_arrow_dict(self.columns[0]):
            # When input is arrow.
            n_samples = len(self.columns[0].indices)
        elif _is_df_cat(self.columns[0]):
            # When input is pandas.
            n_samples = self.columns[0].codes.shape[0]
        else:
            # Anything else, TypeGuard is ignored by mypy 1.15.0 for some reason
            n_samples = self.columns[0].shape[0]  # type: ignore
        return n_samples, len(self.columns)


def _transform_pandas_df(
    data: "PdDataFrame",
    enable_categorical: bool,
    feature_names: Optional[FeatureNames] = None,
    feature_types: Optional[Union[FeatureTypes, Categories]] = None,
    meta: Optional[str] = None,
) -> Tuple[PandasTransformed, Optional[FeatureNames], Optional[FeatureTypes]]:
    if meta and len(data.columns) > 1 and meta not in _matrix_meta:
        raise ValueError(f"DataFrame for {meta} cannot have multiple columns")

    feature_types, ref_categories = get_ref_categories(feature_types)
    feature_names, feature_types = pandas_feature_info(
        data, meta, feature_names, feature_types, enable_categorical
    )

    arrays = pandas_transform_data(data)
    return (
        PandasTransformed(arrays, ref_categories=ref_categories),
        feature_names,
        feature_types,
    )


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
    *,
    data: "PdDataFrame",
    enable_categorical: bool,
    missing: FloatCompatible,
    nthread: int,
    feature_names: Optional[FeatureNames],
    feature_types: Optional[Union[FeatureTypes, Categories]],
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


def _from_pandas_series(
    *,
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
        data=data.values.reshape(data.shape[0], 1).astype("float"),
        missing=missing,
        nthread=nthread,
        feature_names=feature_names,
        feature_types=feature_types,
    )


class ArrowTransformed(TransformedDf):
    """A storage class for transformed arrow table."""

    def __init__(
        self,
        columns: List[Union["pa.NumericArray", "pa.DictionaryArray"]],
        ref_categories: Optional[Categories] = None,
    ) -> None:
        self.columns = columns

        self.temporary_buffers: List[Tuple] = []

        if TYPE_CHECKING:
            import pyarrow as pa
        else:
            pa = import_pyarrow()

        aitfs: AifType = []

        def push_series(col: Union["pa.NumericArray", "pa.DictionaryArray"]) -> None:
            if isinstance(col, pa.DictionaryArray):
                cats = col.dictionary
                codes = col.indices
                if not isinstance(cats, (pa.StringArray, pa.LargeStringArray)):
                    raise TypeError(
                        "Only string-based categorical index is supported for arrow."
                    )
                jnames, jcodes, buf = arrow_cat_inf(cats, codes)
                self.temporary_buffers.append(buf)
                aitfs.append((jnames, jcodes))
            else:
                jdata = _arrow_array_inf(col)
                aitfs.append(jdata)

        for col in self.columns:
            push_series(col)

        super().__init__(ref_categories=ref_categories, aitfs=aitfs)

    @property
    def shape(self) -> Tuple[int, int]:
        """Return shape of the transformed DataFrame."""
        return len(self.columns[0]), len(self.columns)


def _transform_arrow_table(
    data: "pa.Table",
    enable_categorical: bool,
    feature_names: Optional[FeatureNames],
    feature_types: Optional[Union[FeatureTypes, Categories]],
) -> Tuple[ArrowTransformed, Optional[FeatureNames], Optional[FeatureTypes]]:
    if TYPE_CHECKING:
        import pyarrow as pa
    else:
        pa = import_pyarrow()

    t_names, t_types = _arrow_feature_info(data)
    feature_types, ref_categories = get_ref_categories(feature_types)

    if feature_names is None:
        feature_names = t_names
    if feature_types is None:
        feature_types = t_types

    columns = []
    for cname in feature_names:
        col0 = data.column(cname)
        col: Union["pa.NumericArray", "pa.DictionaryArray"] = col0.combine_chunks()
        if isinstance(col, pa.BooleanArray):
            col = col.cast(pa.int8())  # bit-compressed array, not supported.
        if is_arrow_dict(col) and not enable_categorical:
            # None because the function doesn't know how to get the type info from arrow
            # table.
            _invalid_dataframe_dtype(None)
        columns.append(col)

    df_t = ArrowTransformed(columns, ref_categories=ref_categories)
    return df_t, feature_names, feature_types


def _from_arrow_table(  # pylint: disable=too-many-positional-arguments
    data: DataType,
    enable_categorical: bool,
    missing: FloatCompatible,
    n_threads: int,
    feature_names: Optional[FeatureNames],
    feature_types: Optional[Union[FeatureTypes, Categories]],
    data_split_mode: DataSplitMode = DataSplitMode.ROW,
) -> DispatchedDataBackendReturnType:
    df_t, feature_names, feature_types = _transform_arrow_table(
        data, enable_categorical, feature_names, feature_types
    )
    handle = ctypes.c_void_p()
    _check_call(
        _LIB.XGDMatrixCreateFromColumnar(
            df_t.array_interface(),
            make_jcargs(
                nthread=n_threads, missing=missing, data_split_mode=data_split_mode
            ),
            ctypes.byref(handle),
        )
    )
    return handle, feature_names, feature_types


@functools.cache
def _arrow_dtype() -> Dict[DataType, str]:
    import pyarrow as pa

    mapping = {
        pa.int8(): "int",
        pa.int16(): "int",
        pa.int32(): "int",
        pa.int64(): "int",
        pa.uint8(): "int",
        pa.uint16(): "int",
        pa.uint32(): "int",
        pa.uint64(): "int",
        pa.float16(): "float",
        pa.float32(): "float",
        pa.float64(): "float",
        pa.bool_(): "i",
    }

    return mapping


def _arrow_feature_info(data: DataType) -> Tuple[List[str], List]:
    if TYPE_CHECKING:
        import pyarrow as pa
    else:
        pa = import_pyarrow()

    table: "pa.Table" = data
    names = table.column_names

    def map_type(name: str) -> str:
        col = table.column(name)
        if isinstance(col.type, pa.DictionaryType):
            return CAT_T  # pylint: disable=unreachable

        return _arrow_dtype()[col.type]

    types = list(map(map_type, names))
    return names, types


def _meta_from_arrow_table(
    data: DataType,
    name: str,
    dtype: Optional[NumpyDType],
    handle: ctypes.c_void_p,
) -> None:
    table: "pa.Table" = data
    _meta_from_pandas_df(table.to_pandas(), name=name, dtype=dtype, handle=handle)


def _check_pyarrow_for_polars() -> None:
    if not is_pyarrow_available():
        raise ImportError("`pyarrow` is required for polars.")


def _transform_polars_df(
    data: DataType,
    enable_categorical: bool,
    feature_names: Optional[FeatureNames],
    feature_types: Optional[Union[FeatureTypes, Categories]],
) -> Tuple[ArrowTransformed, Optional[FeatureNames], Optional[FeatureTypes]]:
    if _is_polars_lazyframe(data):
        df = data.collect()
        warnings.warn(
            "Using the default parameters for the polars `LazyFrame.collect`. Consider"
            " passing a realized `DataFrame` or `Series` instead.",
            UserWarning,
        )
    else:
        df = data

    _check_pyarrow_for_polars()
    table = df.to_arrow()
    return _transform_arrow_table(
        table, enable_categorical, feature_names, feature_types
    )


def _from_polars_df(  # pylint: disable=too-many-positional-arguments
    data: DataType,
    enable_categorical: bool,
    missing: FloatCompatible,
    n_threads: int,
    feature_names: Optional[FeatureNames],
    feature_types: Optional[Union[FeatureTypes, Categories]],
    data_split_mode: DataSplitMode = DataSplitMode.ROW,
) -> DispatchedDataBackendReturnType:
    df_t, feature_names, feature_types = _transform_polars_df(
        data, enable_categorical, feature_names, feature_types
    )
    handle = ctypes.c_void_p()
    _check_call(
        _LIB.XGDMatrixCreateFromColumnar(
            df_t.array_interface(),
            make_jcargs(
                nthread=n_threads, missing=missing, data_split_mode=data_split_mode
            ),
            ctypes.byref(handle),
        )
    )
    return handle, feature_names, feature_types


@functools.cache
def _lazy_load_cudf_is_cat() -> Callable[[Any], bool]:
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


@functools.cache
def _lazy_load_cudf_is_bool() -> Callable[[Any], bool]:
    from cudf.api.types import is_bool_dtype

    return is_bool_dtype


class CudfTransformed(TransformedDf):
    """A storage class for transformed cuDF dataframe."""

    def __init__(
        self,
        columns: List[Union["PdSeries", DfCatAccessor]],
        ref_categories: Optional[Categories],
    ) -> None:
        self.columns = columns
        # Buffers for temporary data that cannot be freed until the data is consumed by
        # the DMatrix or the booster.

        aitfs: AifType = []

        def push_series(ser: Any) -> None:
            if _is_df_cat(ser):
                cats, codes = ser.categories, ser.codes
                cats_ainf, codes_ainf, buf = cudf_cat_inf(cats, codes)
                self.temporary_buffers.append(buf)
                aitfs.append((cats_ainf, codes_ainf))
            else:
                # numeric column
                ainf = cuda_array_interface_dict(ser)
                aitfs.append(ainf)

        for col in self.columns:
            push_series(col)

        super().__init__(ref_categories=ref_categories, aitfs=aitfs)

    @property
    def shape(self) -> Tuple[int, int]:
        """Return shape of the transformed DataFrame."""
        if _is_df_cat(self.columns[0]):
            n_samples = self.columns[0].codes.shape[0]
        else:
            n_samples = self.columns[0].shape[0]  # type: ignore
        return n_samples, len(self.columns)


def _transform_cudf_df(
    data: DataType,
    feature_names: Optional[FeatureNames],
    feature_types: Optional[Union[FeatureTypes, Categories]],
    enable_categorical: bool,
) -> Tuple[
    CudfTransformed,
    Optional[FeatureNames],
    Optional[FeatureTypes],
]:
    is_bool_dtype = _lazy_load_cudf_is_bool()

    is_categorical_dtype = _lazy_load_cudf_is_cat()
    # Work around https://github.com/dmlc/xgboost/issues/10181
    if _is_cudf_ser(data):
        if is_bool_dtype(data.dtype):
            data = data.astype(np.uint8)
        dtypes = [data.dtype]
    else:
        data = data.astype(
            {col: np.uint8 for col in data.select_dtypes(include="bool")}
        )
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
    feature_types, ref_categories = get_ref_categories(feature_types)
    if feature_types is None:
        feature_types = []
        for dtype in dtypes:
            if is_categorical_dtype(dtype) and enable_categorical:
                feature_types.append(CAT_T)
            else:
                feature_types.append(_pandas_dtype_mapper[dtype.name])

    # handle categorical data
    result = []
    if _is_cudf_ser(data):
        # unlike pandas, cuDF uses NA for missing data.
        if is_categorical_dtype(data.dtype) and enable_categorical:
            result.append(data.cat)
        elif enable_categorical:
            raise ValueError(_ENABLE_CAT_ERR)
        else:
            result.append(data)
    else:
        for col, dtype in zip(data.columns, data.dtypes):
            series = data[col]
            if is_categorical_dtype(dtype) and enable_categorical:
                result.append(series.cat)
            elif is_categorical_dtype(dtype):
                raise ValueError(_ENABLE_CAT_ERR)
            else:
                result.append(series)

    return (
        CudfTransformed(result, ref_categories=ref_categories),
        feature_names,
        feature_types,
    )


def _from_cudf_df(
    *,
    data: DataType,
    missing: FloatCompatible,
    nthread: int,
    feature_names: Optional[FeatureNames],
    feature_types: Optional[Union[FeatureTypes, Categories]],
    enable_categorical: bool,
) -> DispatchedDataBackendReturnType:
    df, feature_names, feature_types = _transform_cudf_df(
        data, feature_names, feature_types, enable_categorical
    )
    handle = ctypes.c_void_p()
    _check_call(
        _LIB.XGDMatrixCreateFromCudaColumnar(
            df.array_interface(),
            make_jcargs(nthread=nthread, missing=missing),
            ctypes.byref(handle),
        )
    )
    return handle, feature_names, feature_types


def _is_cupy_alike(data: DataType) -> bool:
    return hasattr(data, "__cuda_array_interface__")


def _transform_cupy_array(data: DataType) -> CupyT:
    import cupy

    if not hasattr(data, "__cuda_array_interface__") and hasattr(data, "__array__"):
        data = cupy.array(data, copy=False)
    if array_hasobject(data) or data.dtype in [cupy.bool_]:
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
    interface_str = cuda_array_interface(data)
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


def _is_uri(data: DataType) -> TypeGuard[PathLike]:
    return isinstance(data, (str, os.PathLike))


def _from_uri(
    data: PathLike,
    missing: Optional[FloatCompatible],
    feature_names: Optional[FeatureNames],
    feature_types: Optional[FeatureTypes],
    data_split_mode: DataSplitMode = DataSplitMode.ROW,
) -> DispatchedDataBackendReturnType:
    _warn_unused_missing(data, missing)
    handle = ctypes.c_void_p()
    data = os.fspath(os.path.expanduser(data))
    config = make_jcargs(uri=str(data), data_split_mode=int(data_split_mode))
    _check_call(_LIB.XGDMatrixCreateFromURI(config, ctypes.byref(handle)))
    return handle, feature_names, feature_types


def _is_list(data: DataType) -> TypeGuard[list]:
    return isinstance(data, list)


def _from_list(
    *,
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
        data=array,
        missing=missing,
        nthread=n_threads,
        feature_names=feature_names,
        feature_types=feature_types,
        data_split_mode=data_split_mode,
    )


def _is_tuple(data: DataType) -> TypeGuard[tuple]:
    return isinstance(data, tuple)


def _from_tuple(
    *,
    data: Sequence,
    missing: FloatCompatible,
    n_threads: int,
    feature_names: Optional[FeatureNames],
    feature_types: Optional[FeatureTypes],
    data_split_mode: DataSplitMode = DataSplitMode.ROW,
) -> DispatchedDataBackendReturnType:
    return _from_list(
        data=data,
        missing=missing,
        n_threads=n_threads,
        feature_names=feature_names,
        feature_types=feature_types,
        data_split_mode=data_split_mode,
    )


def _is_iter(data: DataType) -> TypeGuard[DataIter]:
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
    *,
    data: DataType,
    missing: FloatCompatible,  # Or Optional[Float]
    threads: int,
    feature_names: Optional[FeatureNames],
    feature_types: Optional[Union[FeatureTypes, Categories]],
    enable_categorical: bool = False,
    data_split_mode: DataSplitMode = DataSplitMode.ROW,
) -> DispatchedDataBackendReturnType:
    """Dispatch data for DMatrix."""

    def check_cats(
        feature_types: Optional[Union[FeatureTypes, Categories]],
    ) -> TypeGuard[Optional[FeatureTypes]]:
        if isinstance(feature_types, Categories):
            raise ValueError(
                "Reference category is only supported by DataFrame inputs."
            )
        return True

    if (
        not _is_cudf_ser(data)
        and not _is_pandas_series(data)
        and not _is_polars_series(data)
    ):
        _check_data_shape(data)
    if is_scipy_csr(data):
        assert check_cats(feature_types)
        return _from_scipy_csr(
            data=data,
            missing=missing,
            nthread=threads,
            feature_names=feature_names,
            feature_types=feature_types,
            data_split_mode=data_split_mode,
        )
    if is_scipy_csc(data):
        assert check_cats(feature_types)
        return _from_scipy_csc(
            data=data,
            missing=missing,
            nthread=threads,
            feature_names=feature_names,
            feature_types=feature_types,
            data_split_mode=data_split_mode,
        )
    if is_scipy_coo(data):
        assert check_cats(feature_types)
        return _from_scipy_csr(
            data=data.tocsr(),
            missing=missing,
            nthread=threads,
            feature_names=feature_names,
            feature_types=feature_types,
            data_split_mode=data_split_mode,
        )
    if _is_np_array_like(data):
        assert check_cats(feature_types)
        return _from_numpy_array(
            data=data,
            missing=missing,
            nthread=threads,
            feature_names=feature_names,
            feature_types=feature_types,
            data_split_mode=data_split_mode,
        )
    if _is_uri(data):
        assert check_cats(feature_types)
        return _from_uri(data, missing, feature_names, feature_types, data_split_mode)
    if _is_list(data):
        assert check_cats(feature_types)
        return _from_list(
            data=data,
            missing=missing,
            n_threads=threads,
            feature_names=feature_names,
            feature_types=feature_types,
            data_split_mode=data_split_mode,
        )
    if _is_tuple(data):
        assert check_cats(feature_types)
        return _from_tuple(
            data=data,
            missing=missing,
            n_threads=threads,
            feature_names=feature_names,
            feature_types=feature_types,
            data_split_mode=data_split_mode,
        )
    if _is_polars_series(data):
        pl = import_polars()

        data = pl.DataFrame({data.name: data})
    if _is_polars(data):
        return _from_polars_df(
            data,
            enable_categorical,
            missing=missing,
            n_threads=threads,
            feature_names=feature_names,
            feature_types=feature_types,
            data_split_mode=data_split_mode,
        )
    if _is_arrow(data):
        return _from_arrow_table(
            data,
            enable_categorical,
            missing=missing,
            n_threads=threads,
            feature_names=feature_names,
            feature_types=feature_types,
            data_split_mode=data_split_mode,
        )
    if _is_cudf_pandas(data):
        data = data._fsproxy_fast  # pylint: disable=protected-access
    if _is_pandas_series(data):
        pd = import_pandas()

        data = pd.DataFrame(data)
    if _is_pandas_df(data):
        return _from_pandas_df(
            data=data,
            enable_categorical=enable_categorical,
            missing=missing,
            nthread=threads,
            feature_names=feature_names,
            feature_types=feature_types,
            data_split_mode=data_split_mode,
        )
    if _is_cudf_df(data) or _is_cudf_ser(data):
        return _from_cudf_df(
            data=data,
            missing=missing,
            nthread=threads,
            feature_names=feature_names,
            feature_types=feature_types,
            enable_categorical=enable_categorical,
        )
    if _is_cupy_alike(data):
        assert check_cats(feature_types)
        return _from_cupy_array(data, missing, threads, feature_names, feature_types)
    if _is_cupy_csr(data):
        raise TypeError("cupyx CSR is not supported yet.")
    if _is_cupy_csc(data):
        raise TypeError("cupyx CSC is not supported yet.")
    if _is_dlpack(data):
        assert check_cats(feature_types)
        return _from_dlpack(data, missing, threads, feature_names, feature_types)
    if _is_modin_series(data):
        pd = import_pandas()

        data = pd.DataFrame(data)
    if _is_modin_df(data):
        return _from_pandas_df(
            data=data,
            enable_categorical=enable_categorical,
            missing=missing,
            nthread=threads,
            feature_names=feature_names,
            feature_types=feature_types,
        )

    if _has_array_protocol(data):
        assert check_cats(feature_types)
        array = np.asarray(data)
        return _from_numpy_array(
            data=array,
            missing=missing,
            nthread=threads,
            feature_names=feature_names,
            feature_types=feature_types,
        )

    converted = _convert_unknown_data(data)
    if converted is not None:
        assert check_cats(feature_types)
        return _from_scipy_csr(
            data=converted,
            missing=missing,
            nthread=threads,
            feature_names=feature_names,
            feature_types=feature_types,
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
    interface_str = array_interface(data)
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
        interface = cuda_array_interface(data)
        _check_call(_LIB.XGDMatrixSetInfoFromInterface(handle, c_str(field), interface))


def _meta_from_cudf_series(data: DataType, field: str, handle: ctypes.c_void_p) -> None:
    check_cudf_meta(data, field)
    inf = cuda_array_interface(data)
    _check_call(_LIB.XGDMatrixSetInfoFromInterface(handle, c_str(field), inf))


def _meta_from_cupy_array(data: DataType, field: str, handle: ctypes.c_void_p) -> None:
    data = _transform_cupy_array(data)
    inf = cuda_array_interface(data)
    _check_call(_LIB.XGDMatrixSetInfoFromInterface(handle, c_str(field), inf))


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
        _meta_from_arrow_table(data, name, dtype, handle)
        return
    if _is_cudf_pandas(data):
        data = data._fsproxy_fast  # pylint: disable=protected-access
    if _is_polars(data):
        if _is_polars_lazyframe(data):
            data = data.collect()
        _check_pyarrow_for_polars()
        _meta_from_arrow_table(data.to_arrow(), name, dtype, handle)
        return
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
    if _is_cudf_ser(data):
        _meta_from_cudf_series(data, name, handle)
        return
    if _is_cudf_df(data):
        _meta_from_cudf_df(data, name, handle)
        return
    if _is_cupy_alike(data):
        _meta_from_cupy_array(data, name, handle)
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

    def next(self, input_data: Callable) -> bool:
        if self.it == 1:
            return False
        self.it += 1
        input_data(**self.kwargs)
        return True

    def reset(self) -> None:
        self.it = 0


def _proxy_transform(
    data: DataType,
    feature_names: Optional[FeatureNames],
    feature_types: Optional[FeatureTypes],
    enable_categorical: bool,
) -> TransformedData:
    if _is_cudf_pandas(data):
        data = data._fsproxy_fast  # pylint: disable=protected-access
    if _is_cudf_df(data) or _is_cudf_ser(data):
        return _transform_cudf_df(
            data, feature_names, feature_types, enable_categorical
        )
    if _is_cupy_alike(data):
        data = _transform_cupy_array(data)
        return data, feature_names, feature_types
    if _is_dlpack(data):
        return _transform_dlpack(data), feature_names, feature_types
    if _is_list(data) or _is_tuple(data):
        data = np.array(data)
    if _is_np_array_like(data):
        data, _ = _ensure_np_dtype(data, data.dtype)
        return data, feature_names, feature_types
    if is_scipy_csr(data):
        data = transform_scipy_sparse(data, True)
        return data, feature_names, feature_types
    if is_scipy_csc(data):
        data = transform_scipy_sparse(data.tocsr(), True)
        return data, feature_names, feature_types
    if is_scipy_coo(data):
        data = transform_scipy_sparse(data.tocsr(), True)
        return data, feature_names, feature_types
    if _is_polars(data):
        df_pl, feature_names, feature_types = _transform_polars_df(
            data, enable_categorical, feature_names, feature_types
        )
        return df_pl, feature_names, feature_types
    if _is_pandas_series(data):
        pd = import_pandas()

        data = pd.DataFrame(data)
    if _is_arrow(data):
        df_pa, feature_names, feature_types = _transform_arrow_table(
            data, enable_categorical, feature_names, feature_types
        )
        return df_pa, feature_names, feature_types
    if _is_pandas_df(data):
        df, feature_names, feature_types = _transform_pandas_df(
            data, enable_categorical, feature_names, feature_types
        )
        return df, feature_names, feature_types
    raise TypeError("Value type is not supported for data iterator:" + str(type(data)))


def is_on_cuda(data: Any) -> bool:
    """Whether the data is a CUDA-based data structure."""
    return any(
        p(data)
        for p in (
            _is_cudf_df,
            _is_cudf_ser,
            _is_cudf_pandas,
            _is_cupy_alike,
            _is_dlpack,
        )
    )


def dispatch_proxy_set_data(
    proxy: _ProxyDMatrix,
    data: DataType,
) -> None:
    """Dispatch for QuantileDMatrix."""
    if (
        not _is_cudf_ser(data)
        and not _is_pandas_series(data)
        and not _is_polars_series(data)
    ):
        _check_data_shape(data)

    if isinstance(data, CudfTransformed):
        # pylint: disable=W0212
        proxy._ref_data_from_cuda_columnar(data)
        return
    if _is_cupy_alike(data):
        proxy._ref_data_from_cuda_interface(data)  # pylint: disable=W0212
        return
    if _is_dlpack(data):
        data = _transform_dlpack(data)
        proxy._ref_data_from_cuda_interface(data)  # pylint: disable=W0212
        return
    # Host
    if isinstance(data, (ArrowTransformed, PandasTransformed)):
        proxy._ref_data_from_columnar(data)  # pylint: disable=W0212
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
