"""Helpers for interfacing array like objects."""

import copy
import ctypes
import functools
import json
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypedDict,
    TypeGuard,
    Union,
    cast,
    overload,
)

import numpy as np

from ._typing import CNumericPtr, DataType, NumpyDType, NumpyOrCupy
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
    def __cuda_array_interface__(self) -> "ArrayInf": ...


class TransformedDf(Protocol):
    """Protocol class for storing transformed dataframe."""

    def array_interface(self) -> bytes:
        """Get a JSON-encoded list of array interfaces."""

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the dataframe."""


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

StringArray = TypedDict("StringArray", {"offsets": ArrayInf, "values": ArrayInf})


def array_hasobject(data: DataType) -> bool:
    """Whether the numpy array has object dtype."""
    return (
        hasattr(data, "dtype")
        and hasattr(data.dtype, "hasobject")
        and data.dtype.hasobject
    )


def cuda_array_interface_dict(data: _CudaArrayLikeArg) -> ArrayInf:
    """Returns a dictionary storing the CUDA array interface."""
    if array_hasobject(data):
        raise ValueError("Input data contains `object` dtype.  Expecting numeric data.")
    ainf = data.__cuda_array_interface__
    if "mask" in ainf:
        ainf["mask"] = ainf["mask"].__cuda_array_interface__  # type: ignore
    return cast(ArrayInf, ainf)


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

    def to_arrow(  # pylint: disable=missing-function-docstring
        self,
    ) -> Union["pa.StringArray", "pa.IntegerArray"]: ...

    @property
    def __cuda_array_interface__(self) -> ArrayInf: ...


def _is_df_cat(data: Any) -> TypeGuard[DfCatAccessor]:
    # Test pd.Series.cat, not pd.Series
    return hasattr(data, "categories") and hasattr(data, "codes")


@functools.cache
def _arrow_typestr() -> Dict["pa.DataType", str]:
    import pyarrow as pa

    mapping = {
        pa.int8(): "<i1",
        pa.int16(): "<i2",
        pa.int32(): "<i4",
        pa.int64(): "<i8",
        pa.uint8(): "<u1",
        pa.uint16(): "<u2",
        pa.uint32(): "<u4",
        pa.uint64(): "<u8",
    }

    return mapping


def npstr_to_arrow_strarr(strarr: np.ndarray) -> Tuple[np.ndarray, str]:
    """Convert a numpy string array to an arrow string array."""
    lenarr = np.vectorize(len)
    offsets = np.cumsum(np.concatenate([np.array([0], dtype=np.int64), lenarr(strarr)]))
    values = strarr.sum()
    assert "\0" not in values  # arrow string array doesn't need null terminal
    return offsets.astype(np.int32), values


def _arrow_cat_inf(  # pylint: disable=too-many-locals
    cats: "pa.StringArray",
    codes: Union[_ArrayLikeArg, _CudaArrayLikeArg, "pa.IntegerArray"],
) -> Tuple[StringArray, ArrayInf, Tuple]:
    if not TYPE_CHECKING:
        pa = import_pyarrow()

    # FIXME(jiamingy): Account for offset, need to find an implementation that returns
    # offset > 0
    assert cats.offset == 0
    buffers: List[pa.Buffer] = cats.buffers()
    mask, offset, data = buffers
    assert offset.is_cpu

    off_len = len(cats) + 1
    if offset.size != off_len * (np.iinfo(np.int32).bits / 8):
        raise TypeError("Arrow dictionary type offsets is required to be 32 bit.")

    joffset: ArrayInf = {
        "data": (offset.address, True),
        "typestr": "<i4",
        "version": 3,
        "strides": None,
        "shape": (off_len,),
        "mask": None,
    }

    def make_buf_inf(buf: pa.Buffer, typestr: str) -> ArrayInf:
        return {
            "data": (buf.address, True),
            "typestr": typestr,
            "version": 3,
            "strides": None,
            "shape": (buf.size,),
            "mask": None,
        }

    jdata = make_buf_inf(data, "<i1")
    # Categories should not have missing values.
    assert mask is None

    jnames: StringArray = {"offsets": joffset, "values": jdata}

    def make_array_inf(
        array: Any,
    ) -> Tuple[ArrayInf, Optional[Tuple[pa.Buffer, pa.Buffer]]]:
        """Helper for handling categorical codes."""
        # Handle cuDF data
        if hasattr(array, "__cuda_array_interface__"):
            inf = cuda_array_interface_dict(array)
            return inf, None

        # Other types (like arrow itself) are not yet supported.
        raise TypeError("Invalid input type.")

    cats_tmp = (mask, offset, data)
    jcodes, codes_tmp = make_array_inf(codes)

    return jnames, jcodes, (cats_tmp, codes_tmp)


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


@overload
def array_interface_dict(data: np.ndarray) -> ArrayInf: ...


@overload
def array_interface_dict(
    data: DfCatAccessor,
) -> Tuple[StringArray, ArrayInf, Tuple]: ...


@overload
def array_interface_dict(
    data: "pa.DictionaryArray",
) -> Tuple[StringArray, ArrayInf, Tuple]: ...


def array_interface_dict(  # pylint: disable=too-many-locals
    data: Union[np.ndarray, DfCatAccessor],
) -> Union[ArrayInf, Tuple[StringArray, ArrayInf, Optional[Tuple]]]:
    """Returns an array interface from the input."""
    # Handle categorical values
    if _is_df_cat(data):
        cats = data.categories
        # pandas uses -1 to represent missing values for categorical features
        codes = data.codes.replace(-1, np.nan)

        if np.issubdtype(cats.dtype, np.floating) or np.issubdtype(
            cats.dtype, np.integer
        ):
            # Numeric index type
            name_values = cats.values
            jarr_values = array_interface_dict(name_values)
            code_values = codes.values
            jarr_codes = array_interface_dict(code_values)
            return jarr_values, jarr_codes, (name_values, code_values)

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

    # Handle numeric values
    assert isinstance(data, np.ndarray)
    if array_hasobject(data):
        raise ValueError("Input data contains `object` dtype.  Expecting numeric data.")
    ainf = data.__array_interface__
    if "mask" in ainf:
        ainf["mask"] = ainf["mask"].__array_interface__
    return cast(ArrayInf, ainf)


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


def cudf_cat_inf(
    cats: DfCatAccessor, codes: "pd.Series"
) -> Tuple[Union[ArrayInf, StringArray], ArrayInf, Tuple]:
    """Obtain the cuda array interface for cuDF categories."""
    cp = import_cupy()
    is_num_idx = cp.issubdtype(cats.dtype, cp.floating) or cp.issubdtype(
        cats.dtype, cp.integer
    )
    if is_num_idx:
        cats_ainf = cats.__cuda_array_interface__
        codes_ainf = cuda_array_interface_dict(codes)
        return cats_ainf, codes_ainf, (cats, codes)

    joffset, jdata, buf = _arrow_cat_inf(cats.to_arrow(), codes)
    return joffset, jdata, buf
