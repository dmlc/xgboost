from abc import abstractproperty
import os
from typing import Union, Dict, Iterable, List, Tuple, Optional
from typing import SupportsIndex, Sized

import numpy as np
try:
    from numpy import typing as npt
    DTypeLike = npt.DTypeLike
except ImportError:
    DTypeLike = np.dtype        # type: ignore

try:
    from typing import Protocol
except ImportError:
    Protocol = object           # type: ignore


class NPArrayLike(Protocol, Sized):
    def __array__(self) -> np.ndarray:
        ...

    @abstractproperty
    def __array_interface__(self) -> Dict[str, Union[str, int, Dict, "NPArrayLike"]]:
        ...

    @abstractproperty
    def shape(self) -> Tuple[int, ...]:
        ...

    @abstractproperty
    def dtype(self) -> np.dtype:
        ...


class CuArrayLike(Protocol):
    @abstractproperty
    def __cuda_array_interface__(self) -> Dict[str, Union[str, int, Dict, "CuArrayLike"]]:
        ...

    @abstractproperty
    def shape(self) -> Tuple[int, ...]:
        ...

    @abstractproperty
    def dtype(self) -> np.dtype:
        ...

    def reshape(self, *s: SupportsIndex) -> "CuArrayLike":
        ...


class CSRLike(Protocol):
    @abstractproperty
    def shape(self) -> Tuple[int, ...]:
        ...

    @abstractproperty
    def dtype(self) -> np.dtype:
        ...

    indptr: NPArrayLike
    indices: NPArrayLike
    data: NPArrayLike


class DFLike(Protocol):
    @abstractproperty
    def values(self) -> NPArrayLike:
        ...

    @abstractproperty
    def shape(self) -> Tuple[int, ...]:
        ...

    @abstractproperty
    def dtype(self) -> np.dtype:
        ...

    @abstractproperty
    def dtypes(self) -> Tuple[np.dtype]:
        ...


class CuDFLike(Protocol, Iterable):
    @abstractproperty
    def values(self) -> CuArrayLike:
        ...

    @abstractproperty
    def __cuda_array_interface__(self) -> Dict[str, Union[str, int, Dict, "CuArrayLike"]]:
        ...

    @abstractproperty
    def shape(self) -> Tuple[int, ...]:
        ...

    @abstractproperty
    def name(self) -> str:
        ...

    @abstractproperty
    def dtype(self) -> np.dtype:
        ...

    @abstractproperty
    def dtypes(self) -> Tuple[np.dtype, ...]:
        ...

    def __getitem__(self, key: str) -> "CuDFLike":
        ...


FloatT = Union[float, np.float16, np.float32, np.float64]


array_like = Union[NPArrayLike, DFLike, CuArrayLike, CuDFLike, CSRLike]
NativeInput = Union[NPArrayLike, DFLike, CuArrayLike, CuDFLike, CSRLike, str, os.PathLike]


FeatureTypes = Optional[Union[List[str], List[DTypeLike]]]
