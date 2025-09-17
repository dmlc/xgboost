"""Helpers for interfacing array like objects."""

import copy
import ctypes
import json
from abc import ABC, abstractmethod
from functools import cache as fcache
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeAlias,
    TypedDict,
    TypeGuard,
    Union,
    cast,
    overload,
)

import numpy as np

from ._typing import (
    ArrowCatList,
    CNumericPtr,
    DataType,
    FeatureTypes,
    NumpyDType,
    NumpyOrCupy,
)
from .compat import import_cupy, import_pyarrow, lazy_isinstance

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa


# Used for accepting inputs for numpy and cupy arrays
class _ArrayLikeArg(Protocol):
    @property
    def __array_interface__(self) -> "ArrayInf": ...


class _CudaArrayLikeArg(Protocol):
    @property
    def __cuda_array_interface__(self) -> "CudaArrayInf": ...


ArrayInf = TypedDict(
    "ArrayInf",
    {
        "data": Tuple[int, bool],
        "typestr": str,
        "version": Literal[3],
        "strides": Optional[Tuple[int, ...]],
        "shape": Tuple[int, ...],
        "mask": Union["ArrayInf", None, _ArrayLikeArg],
    },
)

CudaArrayInf = TypedDict(
    "CudaArrayInf",
    {
        "data": Tuple[int, bool],
        "typestr": str,
        "version": Literal[3],
        "strides": Optional[Tuple[int, ...]],
        "shape": Tuple[int, ...],
        "mask": Union["ArrayInf", None, _ArrayLikeArg],
        "stream": int,
    },
)

StringArray = TypedDict("StringArray", {"offsets": ArrayInf, "values": ArrayInf})
CudaStringArray = TypedDict(
    "CudaStringArray", {"offsets": CudaArrayInf, "values": CudaArrayInf}
)


def array_hasobject(data: DataType) -> bool:
    """Whether the numpy array has object dtype."""
    return (
        hasattr(data, "dtype")
        and hasattr(data.dtype, "hasobject")
        and data.dtype.hasobject
    )


def cuda_array_interface_dict(data: _CudaArrayLikeArg) -> CudaArrayInf:
    """Returns a dictionary storing the CUDA array interface."""
    if array_hasobject(data):
        raise ValueError("Input data contains `object` dtype.  Expecting numeric data.")
    ainf = data.__cuda_array_interface__
    if "mask" in ainf:
        ainf["mask"] = ainf["mask"].__cuda_array_interface__  # type: ignore
    return ainf


def cuda_array_interface(data: _CudaArrayLikeArg) -> bytes:
    """Make cuda array interface str."""
    interface = cuda_array_interface_dict(data)
    interface_str = bytes(json.dumps(interface), "utf-8")
    return interface_str


def from_array_interface(interface: ArrayInf, zero_copy: bool = False) -> NumpyOrCupy:
    """Convert array interface to numpy or cupy array"""

    class Array:
        """Wrapper type for communicating with numpy and cupy."""

        _interface: Optional[ArrayInf] = None

        @property
        def __array_interface__(self) -> Optional[ArrayInf]:
            return self._interface

        @__array_interface__.setter
        def __array_interface__(self, interface: ArrayInf) -> None:
            self._interface = copy.copy(interface)
            # Convert some fields to tuple as required by numpy
            self._interface["shape"] = tuple(self._interface["shape"])
            self._interface["data"] = (
                self._interface["data"][0],
                self._interface["data"][1],
            )
            strides = self._interface.get("strides", None)
            if strides is not None:
                self._interface["strides"] = tuple(strides)

        @property
        def __cuda_array_interface__(self) -> Optional[ArrayInf]:
            return self.__array_interface__

        @__cuda_array_interface__.setter
        def __cuda_array_interface__(self, interface: ArrayInf) -> None:
            self.__array_interface__ = interface

    arr = Array()

    if "stream" in interface:
        # CUDA stream is presented, this is a __cuda_array_interface__.
        arr.__cuda_array_interface__ = interface
        out = import_cupy().array(arr, copy=not zero_copy)
    else:
        arr.__array_interface__ = interface
        out = np.array(arr, copy=not zero_copy)

    return out


# Default constant value for CUDA per-thread stream.
STREAM_PER_THREAD = 2


# Typing is not strict as there are subtle differences between CUDA array interface and
# array interface. We handle them uniformly for now.
def make_array_interface(
    ptr: Union[CNumericPtr, int],
    shape: Tuple[int, ...],
    dtype: Type[np.number],
    is_cuda: bool,
) -> ArrayInf:
    """Make an __(cuda)_array_interface__ from a pointer."""
    # Use an empty array to handle typestr and descr
    if is_cuda:
        empty = import_cupy().empty(shape=(0,), dtype=dtype)
        array = empty.__cuda_array_interface__  # pylint: disable=no-member
    else:
        empty = np.empty(shape=(0,), dtype=dtype)
        array = empty.__array_interface__  # pylint: disable=no-member

    if not isinstance(ptr, int):
        addr = ctypes.cast(ptr, ctypes.c_void_p).value
    else:
        addr = ptr
    length = int(np.prod(shape))
    # Handle empty dataset.
    assert addr is not None or length == 0

    if addr is None:
        return array

    array["data"] = (addr, True)
    if is_cuda and "stream" not in array:
        array["stream"] = STREAM_PER_THREAD
    array["shape"] = shape
    array["strides"] = None
    return array


def is_arrow_dict(data: Any) -> TypeGuard["pa.DictionaryArray"]:
    """Is this an arrow dictionary array?"""
    return lazy_isinstance(data, "pyarrow.lib", "DictionaryArray")


class DfCatAccessor(Protocol):
    """Protocol for pandas cat accessor."""

    @property
    def categories(  # pylint: disable=missing-function-docstring
        self,
    ) -> "pd.Index": ...

    @property
    def codes(self) -> "pd.Series": ...  # pylint: disable=missing-function-docstring

    @property
    def dtype(self) -> np.dtype: ...  # pylint: disable=missing-function-docstring

    @property
    def values(self) -> np.ndarray: ...  # pylint: disable=missing-function-docstring

    def to_arrow(  # pylint: disable=missing-function-docstring
        self,
    ) -> Union["pa.StringArray", "pa.IntegerArray"]: ...

    @property
    def __cuda_array_interface__(self) -> CudaArrayInf: ...

    @property
    def _column(self) -> Any: ...


def _is_df_cat(data: Any) -> TypeGuard[DfCatAccessor]:
    # Test pd.Series.cat, not pd.Series
    return hasattr(data, "categories") and hasattr(data, "codes")


@fcache
def _arrow_npdtype() -> Dict[Any, Type[np.number]]:
    import pyarrow as pa

    mapping: Dict[Any, Type[np.number]] = {
        pa.int8(): np.int8,
        pa.int16(): np.int16,
        pa.int32(): np.int32,
        pa.int64(): np.int64,
        pa.uint8(): np.uint8,
        pa.uint16(): np.uint16,
        pa.uint32(): np.uint32,
        pa.uint64(): np.uint64,
        pa.float16(): np.float16,
        pa.float32(): np.float32,
        pa.float64(): np.float64,
    }

    return mapping


@overload
def _arrow_buf_inf(address: int, typestr: str, size: int, stream: None) -> ArrayInf: ...


@overload
def _arrow_buf_inf(
    address: int, typestr: str, size: int, stream: int
) -> CudaArrayInf: ...


def _arrow_buf_inf(
    address: int, typestr: str, size: int, stream: Optional[int]
) -> Union[ArrayInf, CudaArrayInf]:
    if stream is not None:
        jcuaif: CudaArrayInf = {
            "data": (address, True),
            "typestr": typestr,
            "version": 3,
            "strides": None,
            "shape": (size,),
            "mask": None,
            "stream": stream,
        }
        return jcuaif

    jaif: ArrayInf = {
        "data": (address, True),
        "typestr": typestr,
        "version": 3,
        "strides": None,
        "shape": (size,),
        "mask": None,
    }
    return jaif


def _arrow_cat_names_inf(cats: "pa.StringArray") -> Tuple[StringArray, Any]:
    if not TYPE_CHECKING:
        pa = import_pyarrow()

    # FIXME(jiamingy): Account for offset, need to find an implementation that returns
    # offset > 0
    assert cats.offset == 0
    buffers: List[pa.Buffer] = cats.buffers()
    mask, offset, data = buffers
    assert offset.is_cpu

    off_len = len(cats) + 1

    def get_n_bytes(typ: Type) -> int:
        return off_len * (np.iinfo(typ).bits // 8)

    if offset.size == get_n_bytes(np.int64):
        if not isinstance(cats, pa.LargeStringArray):
            arrow_str_error = "Expecting a `pyarrow.Array`."
            raise TypeError(arrow_str_error + f" Got: {type(cats)}.")
        # Convert to 32bit integer, arrow recommends against the use of i64. Also,
        # XGBoost cannot handle large number of categories (> 2**31).
        i32cats = cats.cast(pa.string())
        mask, offset, data = i32cats.buffers()

    if offset.size != get_n_bytes(np.int32):
        raise TypeError(
            "Arrow dictionary type offsets is required to be 32-bit integer."
        )

    joffset = _arrow_buf_inf(offset.address, "<i4", off_len, None)
    jdata = _arrow_buf_inf(data.address, "|i1", data.size, None)
    # Categories should not have missing values.
    assert mask is None

    jnames: StringArray = {"offsets": joffset, "values": jdata}
    return jnames, (mask, offset, data)


def _arrow_array_inf(
    array: "pa.Array",
) -> ArrayInf:
    """Helper for handling categorical codes."""
    if not TYPE_CHECKING:
        pa = import_pyarrow()
    if not isinstance(array, pa.Array):  # pylint: disable=E0606
        raise TypeError(f"Invalid input type: {type(array)}")

    mask, data = array.buffers()
    jdata = make_array_interface(
        data.address,
        shape=(len(array),),
        dtype=_arrow_npdtype()[array.type],
        is_cuda=not data.is_cpu,
    )

    if mask is not None:
        jmask: Optional[ArrayInf] = {
            "data": (mask.address, True),
            "typestr": "<t1",
            "version": 3,
            "strides": None,
            "shape": (len(array),),
            "mask": None,
        }
        if not mask.is_cpu:
            jmask["stream"] = STREAM_PER_THREAD  # type: ignore
    else:
        jmask = None

    jdata["mask"] = jmask
    return jdata


def arrow_cat_inf(  # pylint: disable=too-many-locals
    cats: "pa.StringArray",
    codes: Union[_ArrayLikeArg, _CudaArrayLikeArg, "pa.IntegerArray"],
) -> Tuple[StringArray, ArrayInf, Tuple]:
    """Get the array interface representation of a string-based category array."""
    jnames, cats_tmp = _arrow_cat_names_inf(cats)
    jcodes = _arrow_array_inf(codes)

    return jnames, jcodes, (cats_tmp, None)


def _ensure_np_dtype(
    data: DataType, dtype: Optional[NumpyDType]
) -> Tuple[np.ndarray, Optional[NumpyDType]]:
    """Ensure the np array has correct type and is contiguous."""
    if array_hasobject(data) or data.dtype in [np.float16, np.bool_]:
        dtype = np.float32
        data = data.astype(dtype, copy=False)
    if not data.flags.aligned:
        data = np.require(data, requirements="A")
    return data, dtype


def array_interface_dict(data: np.ndarray) -> ArrayInf:
    """Returns an array interface from the input."""
    if array_hasobject(data):
        raise ValueError("Input data contains `object` dtype.  Expecting numeric data.")
    ainf = data.__array_interface__
    if "mask" in ainf:
        ainf["mask"] = ainf["mask"].__array_interface__
    return cast(ArrayInf, ainf)


def pd_cat_inf(  # pylint: disable=too-many-locals
    cats: DfCatAccessor, codes: "pd.Series"
) -> Tuple[Union[StringArray, ArrayInf], ArrayInf, Tuple]:
    """Get the array interface representation of pandas category accessor."""
    # pandas uses -1 to represent missing values for categorical features
    codes = codes.replace(-1, np.nan)

    if np.issubdtype(cats.dtype, np.floating) or np.issubdtype(cats.dtype, np.integer):
        # Numeric index type
        name_values_num = cats.values
        jarr_values = array_interface_dict(name_values_num)
        code_values = codes.values
        jarr_codes = array_interface_dict(code_values)
        return jarr_values, jarr_codes, (name_values_num, code_values)

    def npstr_to_arrow_strarr(strarr: np.ndarray) -> Tuple[np.ndarray, str]:
        """Convert a numpy string array to an arrow string array."""
        lenarr = np.vectorize(len)
        offsets = np.cumsum(
            np.concatenate([np.array([0], dtype=np.int64), lenarr(strarr)])
        )
        values = strarr.sum()
        assert "\0" not in values  # arrow string array doesn't need null terminal
        return offsets.astype(np.int32), values

    # String index type
    name_offsets, name_values = npstr_to_arrow_strarr(cats.values)
    name_offsets, _ = _ensure_np_dtype(name_offsets, np.int32)
    joffsets = array_interface_dict(name_offsets)
    bvalues = name_values.encode("utf-8")

    ptr = ctypes.c_void_p.from_buffer(ctypes.c_char_p(bvalues)).value
    assert ptr is not None

    jvalues: ArrayInf = {
        "data": (ptr, True),
        "typestr": "|i1",
        "shape": (len(name_values),),
        "strides": None,
        "version": 3,
        "mask": None,
    }
    jnames: StringArray = {"offsets": joffsets, "values": jvalues}

    code_values = codes.values
    jcodes = array_interface_dict(code_values)

    buf = (
        name_offsets,
        name_values,
        bvalues,
        code_values,
    )  # store temporary values
    return jnames, jcodes, buf


def array_interface(data: np.ndarray) -> bytes:
    """Make array interface str."""
    interface = array_interface_dict(data)
    interface_str = bytes(json.dumps(interface), "utf-8")
    return interface_str


def check_cudf_meta(data: _CudaArrayLikeArg, field: str) -> None:
    "Make sure no missing value in meta data."
    if (
        "mask" in data.__cuda_array_interface__
        and data.__cuda_array_interface__["mask"] is not None
    ):
        raise ValueError(f"Missing value is not allowed for: {field}")


class ArrowSchema(ctypes.Structure):
    """The Schema type from arrow C array."""

    _fields_ = [
        ("format", ctypes.c_char_p),
        ("name", ctypes.c_char_p),
        ("metadata", ctypes.c_char_p),
        ("flags", ctypes.c_int64),
        ("n_children", ctypes.c_int64),
        ("children", ctypes.POINTER(ctypes.c_void_p)),
        ("dictionary", ctypes.c_void_p),
        ("release", ctypes.c_void_p),
        ("private_data", ctypes.c_void_p),
    ]


class ArrowArray(ctypes.Structure):
    """The Array type from arrow C array."""


ArrowArray._fields_ = [  # pylint: disable=protected-access
    ("length", ctypes.c_int64),
    ("null_count", ctypes.c_int64),
    ("offset", ctypes.c_int64),
    ("n_buffers", ctypes.c_int64),
    ("n_children", ctypes.c_int64),
    ("buffers", ctypes.POINTER(ctypes.c_void_p)),
    ("children", ctypes.POINTER(ctypes.POINTER(ArrowArray))),
    ("dictionary", ctypes.POINTER(ArrowArray)),
    ("release", ctypes.c_void_p),
    ("private_data", ctypes.c_void_p),
]


class ArrowDeviceArray(ctypes.Structure):
    """The Array type from arrow C device array."""

    _fields_ = [
        ("array", ArrowArray),
        ("device_id", ctypes.c_int64),
        ("device_type", ctypes.c_int32),
        ("sync_event", ctypes.c_void_p),
        ("reserved", ctypes.c_int64 * 3),
    ]


PyCapsule_GetName = ctypes.pythonapi.PyCapsule_GetName
PyCapsule_GetName.restype = ctypes.c_char_p
PyCapsule_GetName.argtypes = [ctypes.py_object]


PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
PyCapsule_GetPointer.restype = ctypes.c_void_p
PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]


def wait_event(event_hdl: int) -> None:
    """Wait for CUDA event exported by arrow."""
    # cuda-python is a dependency of cuDF.
    from cuda.bindings import runtime as cudart

    event = ctypes.cast(event_hdl, ctypes.POINTER(ctypes.c_int64))
    (status,) = cudart.cudaStreamWaitEvent(
        STREAM_PER_THREAD,
        event.contents.value,
        cudart.cudaEventWaitDefault,
    )
    if status != cudart.cudaError_t.cudaSuccess:
        _, msg = cudart.cudaGetErrorString(status)
        raise ValueError(msg)


def cudf_cat_inf(  # pylint: disable=too-many-locals
    cats: DfCatAccessor, codes: "pd.Series"
) -> Tuple[Union[CudaArrayInf, CudaStringArray], ArrayInf, Tuple]:
    """Obtain the cuda array interface for cuDF categories."""
    cp = import_cupy()
    is_num_idx = cp.issubdtype(cats.dtype, cp.floating) or cp.issubdtype(
        cats.dtype, cp.integer
    )
    if is_num_idx:
        cats_ainf = cuda_array_interface_dict(cats)
        codes_ainf = cuda_array_interface_dict(codes)
        return cats_ainf, codes_ainf, (cats, codes)

    # pylint: disable=protected-access
    arrow_col = cats._column.to_pylibcudf(mode="read")
    # Tuple[types.CapsuleType, types.CapsuleType]
    schema, array = arrow_col.__arrow_c_device_array__()

    array_ptr = PyCapsule_GetPointer(array, PyCapsule_GetName(array))
    schema_ptr = PyCapsule_GetPointer(schema, PyCapsule_GetName(schema))

    # Cast to arrow array
    arrow_device_array = ctypes.cast(
        array_ptr, ctypes.POINTER(ArrowDeviceArray)
    ).contents
    wait_event(arrow_device_array.sync_event)
    assert arrow_device_array.device_type == 2  # 2 is CUDA

    arrow_array = arrow_device_array.array
    mask, offset, data = (
        arrow_array.buffers[0],
        arrow_array.buffers[1],
        arrow_array.buffers[2],
    )
    # Categories should not have missing values.
    assert mask is None
    assert arrow_array.n_children == 0
    assert arrow_array.n_buffers == 3
    assert arrow_array.offset == 0

    # Cast to ArrowSchema
    arrow_schema = ctypes.cast(schema_ptr, ctypes.POINTER(ArrowSchema)).contents
    assert arrow_schema.format in (b"u", b"U", b"vu")  # utf8, large utf8
    if arrow_schema.format in (b"u", b"vu"):
        joffset: CudaArrayInf = _arrow_buf_inf(
            offset, "<i4", arrow_array.length + 1, STREAM_PER_THREAD
        )
    elif arrow_schema.format == b"U":
        raise TypeError("Large string for category index (names) is not supported.")
    else:
        raise TypeError(
            "Unexpected type for category index. It's neither numeric nor string."
        )
    # 0 size for unknown
    jdata: CudaArrayInf = _arrow_buf_inf(data, "|i1", 0, STREAM_PER_THREAD)
    jnames: CudaStringArray = {
        "offsets": joffset,
        "values": jdata,
    }

    jcodes = cuda_array_interface_dict(codes)
    return jnames, jcodes, (arrow_col,)


class Categories:
    """An internal storage class for categories returned by the DMatrix and the
    Booster. This class is designed to be opaque. It is intended to be used exclusively
    by XGBoost as an intermediate storage for re-coding categorical data.

    The categories are saved along with the booster object. As a result, users don't
    need to preserve this class for re-coding. Use the booster model IO instead if you
    want to preserve the categories in a stable format.

    .. versionadded:: 3.1.0

    .. warning::

        This class is internal.

    .. code-block:: python

        Xy = xgboost.QuantileDMatrix(X, y, enable_categorical=True)
        booster = xgboost.train({}, Xy)

        categories = booster.get_categories() # Get categories

        # Use categories as a reference for re-coding
        Xy_new = xgboost.QuantileDMatrix(
            X_new, y_new, feature_types=categories, enable_categorical=True, ref=Xy
        )

        # Categories will be part of the `model.json`.
        booster.save_model("model.json")

    """

    def __init__(
        self,
        handle: Tuple[ctypes.c_void_p, Callable[[], None]],
        arrow_arrays: Optional[ArrowCatList],
    ) -> None:
        # The handle type is a bundle of the handle and the free call. Otherwise, we
        # will have to import the lib and checkcall inside the __del__ method from the
        # core module to avoid cyclic model dependency. Importing modules in __del__ can
        # result in Python abort if __del__ is called during exception handling
        # (interpreter is shutting down).
        self._handle, self._free = handle
        self._arrow_arrays = arrow_arrays

    def to_arrow(self) -> ArrowCatList:
        """Get the categories in the dataset. The results are stored in a list of
        (feature name, arrow array) pairs, with one array for each categorical
        feature. If a feature is numerical, then the corresponding column in the list is
        None. A value error will be raised if this container was created without the
        `export_to_arrow` option.

        """
        if self._arrow_arrays is None:
            raise ValueError(
                "The `export_to_arrow` option of the `get_categories` method"
                " is required."
            )
        return self._arrow_arrays

    def empty(self) -> bool:
        """Returns True if there's no category."""
        return self._handle.value is None

    def get_handle(self) -> int:
        """Internal method for retrieving the handle."""
        assert self._handle.value
        return self._handle.value

    def __del__(self) -> None:
        if self._handle.value is None:
            return
        self._free()


def get_ref_categories(
    feature_types: Optional[Union[FeatureTypes, Categories]],
) -> Tuple[Optional[FeatureTypes], Optional[Categories]]:
    """Get the optional reference categories from the `feature_types`. This is used by
    various `DMatrix` where the `feature_types` is reused for specifying the reference
    categories.

    """
    if isinstance(feature_types, Categories):
        ref_categories = feature_types
        feature_types = None
    else:
        ref_categories = None
    return feature_types, ref_categories


# Type schema for storing JSON-encoded array interface
AifType: TypeAlias = List[
    Union[
        # numeric column
        Union[ArrayInf, CudaArrayInf],
        # categorical column
        Tuple[
            # (cuda) numeric index | (cuda) string index
            Union[ArrayInf, CudaArrayInf, StringArray, CudaStringArray],
            Union[ArrayInf, CudaArrayInf],  # codes
        ],
    ]
]


class TransformedDf(ABC):
    """Internal class for storing transformed dataframe.

    Parameters
    ----------
    ref_categories :
        Optional reference categories used for re-coding.

    aitfs :
        Array interface for each column.

    """

    temporary_buffers: List[Tuple] = []

    def __init__(self, ref_categories: Optional[Categories], aitfs: AifType) -> None:
        self.ref_categories = ref_categories
        if ref_categories is not None and ref_categories.get_handle() is not None:
            aif = ref_categories.get_handle()
            self.ref_aif: Optional[int] = aif
        else:
            self.ref_aif = None

        self.aitfs = aitfs

    def array_interface(self) -> bytes:
        """Return a byte string for JSON encoded array interface."""
        if self.ref_categories is not None:
            ref_inf: dict = {"ref_categories": self.ref_aif, "columns": self.aitfs}
            inf = bytes(json.dumps(ref_inf), "utf-8")
        else:
            inf = bytes(json.dumps(self.aitfs), "utf-8")
        return inf

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the dataframe."""
