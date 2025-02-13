"""Helpers for interfacing array like objects."""

import copy
import ctypes
import json
from typing import Literal, Optional, Protocol, Tuple, Type, TypedDict, Union, cast

import numpy as np

from ._typing import CNumericPtr, DataType, NumpyOrCupy
from .compat import import_cupy


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


def array_hasobject(data: DataType) -> bool:
    """Whether the numpy array has object dtype."""
    return hasattr(data.dtype, "hasobject") and data.dtype.hasobject


def cuda_array_interface(data: DataType) -> bytes:
    """Make cuda array interface str."""
    if array_hasobject(data):
        raise ValueError("Input data contains `object` dtype.  Expecting numeric data.")
    interface = data.__cuda_array_interface__
    if "mask" in interface:
        interface["mask"] = interface["mask"].__cuda_array_interface__
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
    if is_cuda:
        array["stream"] = 2
    array["shape"] = shape
    array["strides"] = None
    return array


def array_interface_dict(data: np.ndarray) -> ArrayInf:
    """Convert array interface into a Python dictionary."""
    if array_hasobject(data):
        raise ValueError("Input data contains `object` dtype.  Expecting numeric data.")
    arrinf = data.__array_interface__
    if "mask" in arrinf:
        arrinf["mask"] = arrinf["mask"].__array_interface__
    return cast(ArrayInf, arrinf)


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
