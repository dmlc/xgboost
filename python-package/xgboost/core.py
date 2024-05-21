# pylint: disable=too-many-arguments, too-many-branches, invalid-name
# pylint: disable=too-many-lines, too-many-locals
"""Core XGBoost Library."""
import copy
import ctypes
import json
import os
import re
import sys
import warnings
import weakref
from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import IntEnum, unique
from functools import wraps
from inspect import Parameter, signature
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
import scipy.sparse

from ._typing import (
    _T,
    ArrayLike,
    BoosterParam,
    CFloatPtr,
    CNumeric,
    CNumericPtr,
    CStrPptr,
    CStrPtr,
    CTypeT,
    DataType,
    FeatureInfo,
    FeatureNames,
    FeatureTypes,
    Integer,
    IterationRange,
    ModelIn,
    NumpyOrCupy,
    TransformedData,
    c_bst_ulong,
)
from .compat import PANDAS_INSTALLED, DataFrame, import_cupy, py_str
from .libpath import find_lib_path


class XGBoostError(ValueError):
    """Error thrown by xgboost trainer."""


@overload
def from_pystr_to_cstr(data: str) -> bytes: ...


@overload
def from_pystr_to_cstr(data: List[str]) -> ctypes.Array: ...


def from_pystr_to_cstr(data: Union[str, List[str]]) -> Union[bytes, ctypes.Array]:
    """Convert a Python str or list of Python str to C pointer

    Parameters
    ----------
    data
        str or list of str
    """

    if isinstance(data, str):
        return bytes(data, "utf-8")
    if isinstance(data, list):
        data_as_bytes: List[bytes] = [bytes(d, "utf-8") for d in data]
        pointers: ctypes.Array[ctypes.c_char_p] = (
            ctypes.c_char_p * len(data_as_bytes)
        )(*data_as_bytes)
        return pointers
    raise TypeError()


def from_cstr_to_pystr(data: CStrPptr, length: c_bst_ulong) -> List[str]:
    """Revert C pointer to Python str

    Parameters
    ----------
    data :
        pointer to data
    length :
        pointer to length of data
    """
    res = []
    for i in range(length.value):
        try:
            res.append(str(cast(bytes, data[i]).decode("ascii")))
        except UnicodeDecodeError:
            res.append(str(cast(bytes, data[i]).decode("utf-8")))
    return res


def make_jcargs(**kwargs: Any) -> bytes:
    "Make JSON-based arguments for C functions."
    return from_pystr_to_cstr(json.dumps(kwargs))


def _parse_eval_str(result: str) -> List[Tuple[str, float]]:
    """Parse an eval result string from the booster."""
    splited = result.split()[1:]
    # split up `test-error:0.1234`
    metric_score_str = [tuple(s.split(":")) for s in splited]
    # convert to float
    metric_score = [(n, float(s)) for n, s in metric_score_str]
    return metric_score


IterRange = TypeVar("IterRange", Optional[Tuple[int, int]], Tuple[int, int])


def _expect(expectations: Sequence[Type], got: Type) -> str:
    """Translate input error into string.

    Parameters
    ----------
    expectations :
        a list of expected value.
    got :
        actual input

    Returns
    -------
    msg: str
    """
    msg = "Expecting "
    for t in range(len(expectations) - 1):
        msg += str(expectations[t])
        msg += " or "
    msg += str(expectations[-1])
    msg += ".  Got " + str(got)
    return msg


def _log_callback(msg: bytes) -> None:
    """Redirect logs from native library into Python console"""
    smsg = py_str(msg)
    if smsg.find("WARNING:") != -1:
        warnings.warn(smsg, UserWarning)
        return
    print(smsg)


def _get_log_callback_func() -> Callable:
    """Wrap log_callback() method in ctypes callback type"""
    c_callback = ctypes.CFUNCTYPE(None, ctypes.c_char_p)
    return c_callback(_log_callback)


def _lib_version(lib: ctypes.CDLL) -> Tuple[int, int, int]:
    """Get the XGBoost version from native shared object."""
    major = ctypes.c_int()
    minor = ctypes.c_int()
    patch = ctypes.c_int()
    lib.XGBoostVersion(ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch))
    return major.value, minor.value, patch.value


def _py_version() -> str:
    """Get the XGBoost version from Python version file."""
    VERSION_FILE = os.path.join(os.path.dirname(__file__), "VERSION")
    with open(VERSION_FILE, encoding="ascii") as f:
        return f.read().strip()


def _register_log_callback(lib: ctypes.CDLL) -> None:
    lib.XGBGetLastError.restype = ctypes.c_char_p
    lib.callback = _get_log_callback_func()  # type: ignore
    if lib.XGBRegisterLogCallback(lib.callback) != 0:
        raise XGBoostError(lib.XGBGetLastError())


def _load_lib() -> ctypes.CDLL:
    """Load xgboost Library."""
    lib_paths = find_lib_path()
    if not lib_paths:
        # This happens only when building document.
        return None  # type: ignore
    try:
        pathBackup = os.environ["PATH"].split(os.pathsep)
    except KeyError:
        pathBackup = []
    lib_success = False
    os_error_list = []
    for lib_path in lib_paths:
        try:
            # needed when the lib is linked with non-system-available
            # dependencies
            os.environ["PATH"] = os.pathsep.join(
                pathBackup + [os.path.dirname(lib_path)]
            )
            lib = ctypes.cdll.LoadLibrary(lib_path)
            setattr(lib, "path", os.path.normpath(lib_path))
            lib_success = True
            break
        except OSError as e:
            os_error_list.append(str(e))
            continue
        finally:
            os.environ["PATH"] = os.pathsep.join(pathBackup)
    if not lib_success:
        libname = os.path.basename(lib_paths[0])
        raise XGBoostError(
            f"""
XGBoost Library ({libname}) could not be loaded.
Likely causes:
  * OpenMP runtime is not installed
    - vcomp140.dll or libgomp-1.dll for Windows
    - libomp.dylib for Mac OSX
    - libgomp.so for Linux and other UNIX-like OSes
    Mac OSX users: Run `brew install libomp` to install OpenMP runtime.

  * You are running 32-bit Python on a 64-bit OS

Error message(s): {os_error_list}
"""
        )
    _register_log_callback(lib)

    def parse(ver: str) -> Tuple[int, int, int]:
        """Avoid dependency on packaging (PEP 440)."""
        # 2.0.0-dev, 2.0.0, or 2.0.0rc1
        major, minor, patch = ver.split("-")[0].split(".")
        rc = patch.find("rc")
        if rc != -1:
            patch = patch[:rc]
        return int(major), int(minor), int(patch)

    libver = _lib_version(lib)
    pyver = parse(_py_version())

    # verify that we are loading the correct binary.
    if pyver != libver:
        pyver_str = ".".join((str(v) for v in pyver))
        libver_str = ".".join((str(v) for v in libver))
        msg = (
            "Mismatched version between the Python package and the native shared "
            f"""object.  Python package version: {pyver_str}. Shared object """
            f"""version: {libver_str}. Shared object is loaded from: {lib.path}.
Likely cause:
  * XGBoost is first installed with anaconda then upgraded with pip. To fix it """
            "please remove one of the installations."
        )
        raise ValueError(msg)

    return lib


# load the XGBoost library globally
_LIB = _load_lib()


def _check_call(ret: int) -> None:
    """Check the return value of C API call

    This function will raise exception when error occurs.
    Wrap every API call with this function

    Parameters
    ----------
    ret :
        return value from API calls
    """
    if ret != 0:
        raise XGBoostError(py_str(_LIB.XGBGetLastError()))


def _check_distributed_params(kwargs: Dict[str, Any]) -> None:
    """Validate parameters in distributed environments."""
    device = kwargs.get("device", None)
    if device and not isinstance(device, str):
        msg = "Invalid type for the `device` parameter"
        msg += _expect((str,), type(device))
        raise TypeError(msg)

    if device and device.find(":") != -1:
        raise ValueError(
            "Distributed training doesn't support selecting device ordinal as GPUs are"
            " managed by the distributed frameworks. use `device=cuda` or `device=gpu`"
            " instead."
        )

    if kwargs.get("booster", None) == "gblinear":
        raise NotImplementedError(
            f"booster `{kwargs['booster']}` is not supported for distributed training."
        )


def _validate_feature_info(
    feature_info: Sequence[str], n_features: int, is_column_split: bool, name: str
) -> List[str]:
    if isinstance(feature_info, str) or not isinstance(feature_info, Sequence):
        raise TypeError(
            f"Expecting a sequence of strings for {name}, got: {type(feature_info)}"
        )
    feature_info = list(feature_info)
    if len(feature_info) != n_features and n_features != 0 and not is_column_split:
        msg = (
            f"{name} must have the same length as the number of data columns, ",
            f"expected {n_features}, got {len(feature_info)}",
        )
        raise ValueError(msg)
    return feature_info


def build_info() -> dict:
    """Build information of XGBoost.  The returned value format is not stable. Also,
    please note that build time dependency is not the same as runtime dependency. For
    instance, it's possible to build XGBoost with older CUDA version but run it with the
    lastest one.

      .. versionadded:: 1.6.0

    """
    j_info = ctypes.c_char_p()
    _check_call(_LIB.XGBuildInfo(ctypes.byref(j_info)))
    assert j_info.value is not None
    res = json.loads(j_info.value.decode())  # pylint: disable=no-member
    res["libxgboost"] = _LIB.path
    return res


def _numpy2ctypes_type(dtype: Type[np.number]) -> Type[CNumeric]:
    _NUMPY_TO_CTYPES_MAPPING: Dict[Type[np.number], Type[CNumeric]] = {
        np.float32: ctypes.c_float,
        np.float64: ctypes.c_double,
        np.uint32: ctypes.c_uint,
        np.uint64: ctypes.c_uint64,
        np.int32: ctypes.c_int32,
        np.int64: ctypes.c_int64,
    }
    if np.intc is not np.int32:  # Windows
        _NUMPY_TO_CTYPES_MAPPING[np.intc] = _NUMPY_TO_CTYPES_MAPPING[np.int32]
    if dtype not in _NUMPY_TO_CTYPES_MAPPING:
        raise TypeError(
            f"Supported types: {_NUMPY_TO_CTYPES_MAPPING.keys()}, got: {dtype}"
        )
    return _NUMPY_TO_CTYPES_MAPPING[dtype]


def _array_hasobject(data: DataType) -> bool:
    return hasattr(data.dtype, "hasobject") and data.dtype.hasobject


def _cuda_array_interface(data: DataType) -> bytes:
    if _array_hasobject(data):
        raise ValueError("Input data contains `object` dtype.  Expecting numeric data.")
    interface = data.__cuda_array_interface__
    if "mask" in interface:
        interface["mask"] = interface["mask"].__cuda_array_interface__
    interface_str = bytes(json.dumps(interface), "utf-8")
    return interface_str


def ctypes2numpy(cptr: CNumericPtr, length: int, dtype: Type[np.number]) -> np.ndarray:
    """Convert a ctypes pointer array to a numpy array."""
    ctype: Type[CNumeric] = _numpy2ctypes_type(dtype)
    if not isinstance(cptr, ctypes.POINTER(ctype)):
        raise RuntimeError(f"expected {ctype} pointer")
    res = np.zeros(length, dtype=dtype)
    if not ctypes.memmove(res.ctypes.data, cptr, length * res.strides[0]):
        raise RuntimeError("memmove failed")
    return res


def ctypes2buffer(cptr: CStrPtr, length: int) -> bytearray:
    """Convert ctypes pointer to buffer type."""
    if not isinstance(cptr, ctypes.POINTER(ctypes.c_char)):
        raise RuntimeError("expected char pointer")
    res = bytearray(length)
    rptr = (ctypes.c_char * length).from_buffer(res)
    if not ctypes.memmove(rptr, cptr, length):
        raise RuntimeError("memmove failed")
    return res


def c_str(string: str) -> ctypes.c_char_p:
    """Convert a python string to cstring."""
    return ctypes.c_char_p(string.encode("utf-8"))


def c_array(
    ctype: Type[CTypeT], values: ArrayLike
) -> Union[ctypes.Array, ctypes._Pointer]:
    """Convert a python array to c array."""
    if isinstance(values, np.ndarray) and values.dtype.itemsize == ctypes.sizeof(ctype):
        return values.ctypes.data_as(ctypes.POINTER(ctype))
    return (ctype * len(values))(*values)


def from_array_interface(interface: dict) -> NumpyOrCupy:
    """Convert array interface to numpy or cupy array"""

    class Array:  # pylint: disable=too-few-public-methods
        """Wrapper type for communicating with numpy and cupy."""

        _interface: Optional[dict] = None

        @property
        def __array_interface__(self) -> Optional[dict]:
            return self._interface

        @__array_interface__.setter
        def __array_interface__(self, interface: dict) -> None:
            self._interface = copy.copy(interface)
            # converts some fields to tuple as required by numpy
            self._interface["shape"] = tuple(self._interface["shape"])
            self._interface["data"] = tuple(self._interface["data"])
            if self._interface.get("strides", None) is not None:
                self._interface["strides"] = tuple(self._interface["strides"])

        @property
        def __cuda_array_interface__(self) -> Optional[dict]:
            return self.__array_interface__

        @__cuda_array_interface__.setter
        def __cuda_array_interface__(self, interface: dict) -> None:
            self.__array_interface__ = interface

    arr = Array()

    if "stream" in interface:
        # CUDA stream is presented, this is a __cuda_array_interface__.
        arr.__cuda_array_interface__ = interface
        out = import_cupy().array(arr, copy=True)
    else:
        arr.__array_interface__ = interface
        out = np.array(arr, copy=True)

    return out


def make_array_interface(
    ptr: CNumericPtr, shape: Tuple[int, ...], dtype: Type[np.number], is_cuda: bool
) -> Dict[str, Union[int, tuple, None]]:
    """Make an __(cuda)_array_interface__ from a pointer."""
    # Use an empty array to handle typestr and descr
    if is_cuda:
        empty = import_cupy().empty(shape=(0,), dtype=dtype)
        array = empty.__cuda_array_interface__  # pylint: disable=no-member
    else:
        empty = np.empty(shape=(0,), dtype=dtype)
        array = empty.__array_interface__  # pylint: disable=no-member

    addr = ctypes.cast(ptr, ctypes.c_void_p).value
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


def _prediction_output(
    shape: CNumericPtr, dims: c_bst_ulong, predts: CFloatPtr, is_cuda: bool
) -> NumpyOrCupy:
    arr_shape = tuple(ctypes2numpy(shape, dims.value, np.uint64).flatten())
    array = from_array_interface(
        make_array_interface(predts, arr_shape, np.float32, is_cuda)
    )
    return array


class DataIter(ABC):  # pylint: disable=too-many-instance-attributes
    """The interface for user defined data iterator. The iterator facilitates
    distributed training, :py:class:`QuantileDMatrix`, and external memory support using
    :py:class:`DMatrix`. Most of time, users don't need to interact with this class
    directly.

    .. note::

        The class caches some intermediate results using the `data` input (predictor
        `X`) as key. Don't repeat the `X` for multiple batches with different meta data
        (like `label`), make a copy if necessary.

    Parameters
    ----------
    cache_prefix :
        Prefix to the cache files, only used in external memory.
    release_data :
        Whether the iterator should release the data during iteration. Set it to True if
        the data transformation (converting data to np.float32 type) is memory
        intensive. Otherwise, if the transformation is computation intensive then we can
        keep the cache.

    """

    def __init__(
        self, cache_prefix: Optional[str] = None, release_data: bool = True
    ) -> None:
        self.cache_prefix = cache_prefix

        self._handle = _ProxyDMatrix()
        self._exception: Optional[Exception] = None
        self._enable_categorical = False
        self._release = release_data
        # Stage data in Python until reset or next is called to avoid data being free.
        self._temporary_data: Optional[TransformedData] = None
        self._data_ref: Optional[weakref.ReferenceType] = None

    def get_callbacks(self, enable_categorical: bool) -> Tuple[Callable, Callable]:
        """Get callback functions for iterating in C. This is an internal function."""
        assert hasattr(self, "cache_prefix"), "__init__ is not called."
        self._reset_callback = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(
            self._reset_wrapper
        )
        self._next_callback = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_void_p,
        )(self._next_wrapper)
        self._enable_categorical = enable_categorical
        return self._reset_callback, self._next_callback

    @property
    def proxy(self) -> "_ProxyDMatrix":
        """Handle of DMatrix proxy."""
        return self._handle

    def _handle_exception(self, fn: Callable, dft_ret: _T) -> _T:
        if self._exception is not None:
            return dft_ret

        try:
            return fn()
        except Exception as e:  # pylint: disable=broad-except
            # Defer the exception in order to return 0 and stop the iteration.
            # Exception inside a ctype callback function has no effect except
            # for printing to stderr (doesn't stop the execution).
            tb = sys.exc_info()[2]
            # On dask, the worker is restarted and somehow the information is
            # lost.
            self._exception = e.with_traceback(tb)
        return dft_ret

    def reraise(self) -> None:
        """Reraise the exception thrown during iteration."""
        self._temporary_data = None
        if self._exception is not None:
            #  pylint 2.7.0 believes `self._exception` can be None even with `assert
            #  isinstace`
            exc = self._exception
            self._exception = None
            raise exc  # pylint: disable=raising-bad-type

    def __del__(self) -> None:
        assert self._temporary_data is None
        assert self._exception is None

    def _reset_wrapper(self, this: None) -> None:  # pylint: disable=unused-argument
        """A wrapper for user defined `reset` function."""
        # free the data
        if self._release:
            self._temporary_data = None
        self._handle_exception(self.reset, None)

    def _next_wrapper(self, this: None) -> int:  # pylint: disable=unused-argument
        """A wrapper for user defined `next` function.

        `this` is not used in Python.  ctypes can handle `self` of a Python
        member function automatically when converting it to c function
        pointer.

        """

        @require_keyword_args(True)
        def input_data(
            *,
            data: Any,
            feature_names: Optional[FeatureNames] = None,
            feature_types: Optional[FeatureTypes] = None,
            **kwargs: Any,
        ) -> None:
            from .data import _proxy_transform, dispatch_proxy_set_data

            # Reduce the amount of transformation that's needed for QuantileDMatrix.
            #
            # To construct the QDM, one needs 4 iterations on CPU, or 2 iterations on
            # GPU. If the QDM has only one batch of input (most of the cases), we can
            # avoid transforming the data repeatly.
            try:
                ref = weakref.ref(data)
            except TypeError:
                ref = None
            if (
                self._temporary_data is not None
                and ref is not None
                and ref is self._data_ref
            ):
                new, cat_codes, feature_names, feature_types = self._temporary_data
            else:
                new, cat_codes, feature_names, feature_types = _proxy_transform(
                    data,
                    feature_names,
                    feature_types,
                    self._enable_categorical,
                )
            # Stage the data, meta info are copied inside C++ MetaInfo.
            self._temporary_data = (new, cat_codes, feature_names, feature_types)
            dispatch_proxy_set_data(self.proxy, new, cat_codes)
            self.proxy.set_info(
                feature_names=feature_names,
                feature_types=feature_types,
                **kwargs,
            )
            self._data_ref = ref

        # Release the data before next batch is loaded.
        if self._release:
            self._temporary_data = None
        # pylint: disable=not-callable
        return self._handle_exception(lambda: self.next(input_data), 0)

    @abstractmethod
    def reset(self) -> None:
        """Reset the data iterator.  Prototype for user defined function."""
        raise NotImplementedError()

    @abstractmethod
    def next(self, input_data: Callable) -> int:
        """Set the next batch of data.

        Parameters
        ----------

        input_data:
            A function with same data fields like `data`, `label` with
            `xgboost.DMatrix`.

        Returns
        -------
        0 if there's no more batch, otherwise 1.

        """
        raise NotImplementedError()


# Notice for `require_keyword_args`
# Authors: Olivier Grisel
#          Gael Varoquaux
#          Andreas Mueller
#          Lars Buitinck
#          Alexandre Gramfort
#          Nicolas Tresegnie
#          Sylvain Marie
# License: BSD 3 clause
def require_keyword_args(
    error: bool,
) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
    """Decorator for methods that issues warnings for positional arguments

    Using the keyword-only argument syntax in pep 3102, arguments after the
    * will issue a warning or error when passed as a positional argument.

    Modified from sklearn utils.validation.

    Parameters
    ----------
    error :
        Whether to throw an error or raise a warning.
    """

    def throw_if(func: Callable[..., _T]) -> Callable[..., _T]:
        """Throw an error/warning if there are positional arguments after the asterisk.

        Parameters
        ----------
        f :
            function to check arguments on.

        """
        sig = signature(func)
        kwonly_args = []
        all_args = []

        for name, param in sig.parameters.items():
            if param.kind == Parameter.POSITIONAL_OR_KEYWORD:
                all_args.append(name)
            elif param.kind == Parameter.KEYWORD_ONLY:
                kwonly_args.append(name)

        @wraps(func)
        def inner_f(*args: Any, **kwargs: Any) -> _T:
            extra_args = len(args) - len(all_args)
            if not all_args and extra_args > 0:  # keyword argument only
                raise TypeError("Keyword argument is required.")

            if extra_args > 0:
                # ignore first 'self' argument for instance methods
                args_msg = [
                    f"{name}"
                    for name, _ in zip(kwonly_args[:extra_args], args[-extra_args:])
                ]
                # pylint: disable=consider-using-f-string
                msg = "Pass `{}` as keyword args.".format(", ".join(args_msg))
                if error:
                    raise TypeError(msg)
                warnings.warn(msg, FutureWarning)
            for k, arg in zip(sig.parameters, args):
                kwargs[k] = arg
            return func(**kwargs)

        return inner_f

    return throw_if


_deprecate_positional_args = require_keyword_args(False)


@unique
class DataSplitMode(IntEnum):
    """Supported data split mode for DMatrix."""

    ROW = 0
    COL = 1


class DMatrix:  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """Data Matrix used in XGBoost.

    DMatrix is an internal data structure that is used by XGBoost, which is optimized
    for both memory efficiency and training speed.  You can construct DMatrix from
    multiple different sources of data.

    """

    @_deprecate_positional_args
    def __init__(
        self,
        data: DataType,
        label: Optional[ArrayLike] = None,
        *,
        weight: Optional[ArrayLike] = None,
        base_margin: Optional[ArrayLike] = None,
        missing: Optional[float] = None,
        silent: bool = False,
        feature_names: Optional[FeatureNames] = None,
        feature_types: Optional[FeatureTypes] = None,
        nthread: Optional[int] = None,
        group: Optional[ArrayLike] = None,
        qid: Optional[ArrayLike] = None,
        label_lower_bound: Optional[ArrayLike] = None,
        label_upper_bound: Optional[ArrayLike] = None,
        feature_weights: Optional[ArrayLike] = None,
        enable_categorical: bool = False,
        data_split_mode: DataSplitMode = DataSplitMode.ROW,
    ) -> None:
        """Parameters
        ----------
        data :
            Data source of DMatrix. See :ref:`py-data` for a list of supported input
            types.
        label :
            Label of the training data.
        weight :
            Weight for each instance.

             .. note::

                 For ranking task, weights are per-group.  In ranking task, one weight
                 is assigned to each group (not each data point). This is because we
                 only care about the relative ordering of data points within each group,
                 so it doesn't make sense to assign weights to individual data points.

        base_margin :
            Global bias for each instance. See :doc:`/tutorials/intercept` for details.
        missing :
            Value in the input data which needs to be present as a missing value. If
            None, defaults to np.nan.
        silent :
            Whether print messages during construction
        feature_names :
            Set names for features.
        feature_types :

            Set types for features. If `data` is a DataFrame type and passing
            `enable_categorical=True`, the types will be deduced automatically
            from the column types.

            Otherwise, one can pass a list-like input with the same length as number
            of columns in `data`, with the following possible values:

            - "c", which represents categorical columns.
            - "q", which represents numeric columns.
            - "int", which represents integer columns.
            - "i", which represents boolean columns.

            Note that, while categorical types are treated differently from
            the rest for model fitting purposes, the other types do not influence
            the generated model, but have effects in other functionalities such as
            feature importances.

            For categorical features, the input is assumed to be preprocessed and
            encoded by the users. The encoding can be done via
            :py:class:`sklearn.preprocessing.OrdinalEncoder` or pandas dataframe
            `.cat.codes` method. This is useful when users want to specify categorical
            features without having to construct a dataframe as input.

        nthread :
            Number of threads to use for loading data when parallelization is
            applicable. If -1, uses maximum threads available on the system.
        group :
            Group size for all ranking group.
        qid :
            Query ID for data samples, used for ranking.
        label_lower_bound :
            Lower bound for survival training.
        label_upper_bound :
            Upper bound for survival training.
        feature_weights :
            Set feature weights for column sampling.
        enable_categorical :

            .. versionadded:: 1.3.0

            .. note:: This parameter is experimental

            Experimental support of specializing for categorical features.

            If passing 'True' and 'data' is a data frame (from supported libraries such
            as Pandas, Modin or cuDF), columns of categorical types will automatically
            be set to be of categorical type (feature_type='c') in the resulting
            DMatrix.

            If passing 'False' and 'data' is a data frame with categorical columns,
            it will result in an error being thrown.

            If 'data' is not a data frame, this argument is ignored.

            JSON/UBJSON serialization format is required for this.

        """
        if group is not None and qid is not None:
            raise ValueError("Either one of `group` or `qid` should be None.")

        self.missing = missing if missing is not None else np.nan
        self.nthread = nthread if nthread is not None else -1
        self.silent = silent

        if isinstance(data, ctypes.c_void_p):
            # Used for constructing DMatrix slice.
            self.handle = data
            return

        from .data import _is_iter, dispatch_data_backend

        if _is_iter(data):
            self._init_from_iter(data, enable_categorical)
            assert self.handle is not None
            return

        handle, feature_names, feature_types = dispatch_data_backend(
            data,
            missing=self.missing,
            threads=self.nthread,
            feature_names=feature_names,
            feature_types=feature_types,
            enable_categorical=enable_categorical,
            data_split_mode=data_split_mode,
        )
        assert handle is not None
        self.handle = handle

        self.set_info(
            label=label,
            weight=weight,
            base_margin=base_margin,
            group=group,
            qid=qid,
            label_lower_bound=label_lower_bound,
            label_upper_bound=label_upper_bound,
            feature_weights=feature_weights,
        )

        if feature_names is not None:
            self.feature_names = feature_names
        if feature_types is not None:
            self.feature_types = feature_types

    def _init_from_iter(self, iterator: DataIter, enable_categorical: bool) -> None:
        it = iterator
        args = {
            "missing": self.missing,
            "nthread": self.nthread,
            "cache_prefix": it.cache_prefix if it.cache_prefix else "",
        }
        args_cstr = from_pystr_to_cstr(json.dumps(args))
        handle = ctypes.c_void_p()
        reset_callback, next_callback = it.get_callbacks(enable_categorical)
        ret = _LIB.XGDMatrixCreateFromCallback(
            None,
            it.proxy.handle,
            reset_callback,
            next_callback,
            args_cstr,
            ctypes.byref(handle),
        )
        it.reraise()
        # delay check_call to throw intermediate exception first
        _check_call(ret)
        self.handle = handle

    def __del__(self) -> None:
        if hasattr(self, "handle"):
            assert self.handle is not None
            _check_call(_LIB.XGDMatrixFree(self.handle))
            del self.handle

    @_deprecate_positional_args
    def set_info(
        self,
        *,
        label: Optional[ArrayLike] = None,
        weight: Optional[ArrayLike] = None,
        base_margin: Optional[ArrayLike] = None,
        group: Optional[ArrayLike] = None,
        qid: Optional[ArrayLike] = None,
        label_lower_bound: Optional[ArrayLike] = None,
        label_upper_bound: Optional[ArrayLike] = None,
        feature_names: Optional[FeatureNames] = None,
        feature_types: Optional[FeatureTypes] = None,
        feature_weights: Optional[ArrayLike] = None,
    ) -> None:
        """Set meta info for DMatrix.  See doc string for :py:obj:`xgboost.DMatrix`."""
        from .data import dispatch_meta_backend

        if label is not None:
            self.set_label(label)
        if weight is not None:
            self.set_weight(weight)
        if base_margin is not None:
            self.set_base_margin(base_margin)
        if group is not None:
            self.set_group(group)
        if qid is not None:
            self.set_uint_info("qid", qid)
        if label_lower_bound is not None:
            self.set_float_info("label_lower_bound", label_lower_bound)
        if label_upper_bound is not None:
            self.set_float_info("label_upper_bound", label_upper_bound)
        if feature_names is not None:
            self.feature_names = feature_names
        if feature_types is not None:
            self.feature_types = feature_types
        if feature_weights is not None:
            dispatch_meta_backend(
                matrix=self, data=feature_weights, name="feature_weights"
            )

    def get_float_info(self, field: str) -> np.ndarray:
        """Get float property from the DMatrix.

        Parameters
        ----------
        field: str
            The field name of the information

        Returns
        -------
        info : array
            a numpy array of float information of the data
        """
        length = c_bst_ulong()
        ret = ctypes.POINTER(ctypes.c_float)()
        _check_call(
            _LIB.XGDMatrixGetFloatInfo(
                self.handle, c_str(field), ctypes.byref(length), ctypes.byref(ret)
            )
        )
        return ctypes2numpy(ret, length.value, np.float32)

    def get_uint_info(self, field: str) -> np.ndarray:
        """Get unsigned integer property from the DMatrix.

        Parameters
        ----------
        field: str
            The field name of the information

        Returns
        -------
        info : array
            a numpy array of unsigned integer information of the data
        """
        length = c_bst_ulong()
        ret = ctypes.POINTER(ctypes.c_uint)()
        _check_call(
            _LIB.XGDMatrixGetUIntInfo(
                self.handle, c_str(field), ctypes.byref(length), ctypes.byref(ret)
            )
        )
        return ctypes2numpy(ret, length.value, np.uint32)

    def set_float_info(self, field: str, data: ArrayLike) -> None:
        """Set float type property into the DMatrix.

        Parameters
        ----------
        field: str
            The field name of the information

        data: numpy array
            The array of data to be set
        """
        from .data import dispatch_meta_backend

        dispatch_meta_backend(self, data, field, "float")

    def set_float_info_npy2d(self, field: str, data: ArrayLike) -> None:
        """Set float type property into the DMatrix
           for numpy 2d array input

        Parameters
        ----------
        field: str
            The field name of the information

        data: numpy array
            The array of data to be set
        """
        from .data import dispatch_meta_backend

        dispatch_meta_backend(self, data, field, "float")

    def set_uint_info(self, field: str, data: ArrayLike) -> None:
        """Set uint type property into the DMatrix.

        Parameters
        ----------
        field: str
            The field name of the information

        data: numpy array
            The array of data to be set
        """
        from .data import dispatch_meta_backend

        dispatch_meta_backend(self, data, field, "uint32")

    def save_binary(self, fname: Union[str, os.PathLike], silent: bool = True) -> None:
        """Save DMatrix to an XGBoost buffer.  Saved binary can be later loaded
        by providing the path to :py:func:`xgboost.DMatrix` as input.

        Parameters
        ----------
        fname : string or os.PathLike
            Name of the output buffer file.
        silent : bool (optional; default: True)
            If set, the output is suppressed.
        """
        fname = os.fspath(os.path.expanduser(fname))
        _check_call(
            _LIB.XGDMatrixSaveBinary(self.handle, c_str(fname), ctypes.c_int(silent))
        )

    def set_label(self, label: ArrayLike) -> None:
        """Set label of dmatrix

        Parameters
        ----------
        label: array like
            The label information to be set into DMatrix
        """
        from .data import dispatch_meta_backend

        dispatch_meta_backend(self, label, "label", "float")

    def set_weight(self, weight: ArrayLike) -> None:
        """Set weight of each instance.

        Parameters
        ----------
        weight : array like
            Weight for each data point

            .. note:: For ranking task, weights are per-group.

                In ranking task, one weight is assigned to each group (not each
                data point). This is because we only care about the relative
                ordering of data points within each group, so it doesn't make
                sense to assign weights to individual data points.

        """
        from .data import dispatch_meta_backend

        dispatch_meta_backend(self, weight, "weight", "float")

    def set_base_margin(self, margin: ArrayLike) -> None:
        """Set base margin of booster to start from.

        This can be used to specify a prediction value of existing model to be
        base_margin However, remember margin is needed, instead of transformed
        prediction e.g. for logistic regression: need to put in value before
        logistic transformation see also example/demo.py

        Parameters
        ----------
        margin: array like
            Prediction margin of each datapoint

        """
        from .data import dispatch_meta_backend

        dispatch_meta_backend(self, margin, "base_margin", "float")

    def set_group(self, group: ArrayLike) -> None:
        """Set group size of DMatrix (used for ranking).

        Parameters
        ----------
        group : array like
            Group size of each group
        """
        from .data import dispatch_meta_backend

        dispatch_meta_backend(self, group, "group", "uint32")

    def get_label(self) -> np.ndarray:
        """Get the label of the DMatrix.

        Returns
        -------
        label : array
        """
        return self.get_float_info("label")

    def get_weight(self) -> np.ndarray:
        """Get the weight of the DMatrix.

        Returns
        -------
        weight : array
        """
        return self.get_float_info("weight")

    def get_base_margin(self) -> np.ndarray:
        """Get the base margin of the DMatrix.

        Returns
        -------
        base_margin
        """
        return self.get_float_info("base_margin")

    def get_group(self) -> np.ndarray:
        """Get the group of the DMatrix.

        Returns
        -------
        group
        """
        group_ptr = self.get_uint_info("group_ptr")
        return np.diff(group_ptr)

    def get_data(self) -> scipy.sparse.csr_matrix:
        """Get the predictors from DMatrix as a CSR matrix. This getter is mostly for
        testing purposes. If this is a quantized DMatrix then quantized values are
        returned instead of input values.

        .. versionadded:: 1.7.0

        """
        indptr = np.empty(self.num_row() + 1, dtype=np.uint64)
        indices = np.empty(self.num_nonmissing(), dtype=np.uint32)
        data = np.empty(self.num_nonmissing(), dtype=np.float32)

        c_indptr = indptr.ctypes.data_as(ctypes.POINTER(c_bst_ulong))
        c_indices = indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        c_data = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        config = from_pystr_to_cstr(json.dumps({}))

        _check_call(
            _LIB.XGDMatrixGetDataAsCSR(self.handle, config, c_indptr, c_indices, c_data)
        )
        ret = scipy.sparse.csr_matrix(
            (data, indices, indptr), shape=(self.num_row(), self.num_col())
        )
        return ret

    def get_quantile_cut(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get quantile cuts for quantization.

        .. versionadded:: 2.0.0

        """
        n_features = self.num_col()

        c_sindptr = ctypes.c_char_p()
        c_sdata = ctypes.c_char_p()
        config = make_jcargs()
        _check_call(
            _LIB.XGDMatrixGetQuantileCut(
                self.handle, config, ctypes.byref(c_sindptr), ctypes.byref(c_sdata)
            )
        )
        assert c_sindptr.value is not None
        assert c_sdata.value is not None

        i_indptr = json.loads(c_sindptr.value)
        indptr = from_array_interface(i_indptr)
        assert indptr.size == n_features + 1
        assert indptr.dtype == np.uint64

        i_data = json.loads(c_sdata.value)
        data = from_array_interface(i_data)
        assert data.size == indptr[-1]
        assert data.dtype == np.float32
        return indptr, data

    def num_row(self) -> int:
        """Get the number of rows in the DMatrix."""
        ret = c_bst_ulong()
        _check_call(_LIB.XGDMatrixNumRow(self.handle, ctypes.byref(ret)))
        return ret.value

    def num_col(self) -> int:
        """Get the number of columns (features) in the DMatrix."""
        ret = c_bst_ulong()
        _check_call(_LIB.XGDMatrixNumCol(self.handle, ctypes.byref(ret)))
        return ret.value

    def num_nonmissing(self) -> int:
        """Get the number of non-missing values in the DMatrix.

        .. versionadded:: 1.7.0

        """
        ret = c_bst_ulong()
        _check_call(_LIB.XGDMatrixNumNonMissing(self.handle, ctypes.byref(ret)))
        return ret.value

    def data_split_mode(self) -> DataSplitMode:
        """Get the data split mode of the DMatrix.

        .. versionadded:: 2.1.0

        """
        ret = c_bst_ulong()
        _check_call(_LIB.XGDMatrixDataSplitMode(self.handle, ctypes.byref(ret)))
        return DataSplitMode(ret.value)

    def slice(
        self, rindex: Union[List[int], np.ndarray], allow_groups: bool = False
    ) -> "DMatrix":
        """Slice the DMatrix and return a new DMatrix that only contains `rindex`.

        Parameters
        ----------
        rindex
            List of indices to be selected.
        allow_groups
            Allow slicing of a matrix with a groups attribute

        Returns
        -------
        res
            A new DMatrix containing only selected indices.
        """
        from .data import _maybe_np_slice

        handle = ctypes.c_void_p()

        rindex = _maybe_np_slice(rindex, dtype=np.int32)
        _check_call(
            _LIB.XGDMatrixSliceDMatrixEx(
                self.handle,
                c_array(ctypes.c_int, rindex),
                c_bst_ulong(len(rindex)),
                ctypes.byref(handle),
                ctypes.c_int(1 if allow_groups else 0),
            )
        )
        return DMatrix(handle)

    @property
    def feature_names(self) -> Optional[FeatureNames]:
        """Labels for features (column labels).

        Setting it to ``None`` resets existing feature names.

        """
        length = c_bst_ulong()
        sarr = ctypes.POINTER(ctypes.c_char_p)()
        _check_call(
            _LIB.XGDMatrixGetStrFeatureInfo(
                self.handle,
                c_str("feature_name"),
                ctypes.byref(length),
                ctypes.byref(sarr),
            )
        )
        feature_names = from_cstr_to_pystr(sarr, length)
        if not feature_names:
            return None
        return feature_names

    @feature_names.setter
    def feature_names(self, feature_names: Optional[FeatureNames]) -> None:
        if feature_names is None:
            _check_call(
                _LIB.XGDMatrixSetStrFeatureInfo(
                    self.handle, c_str("feature_name"), None, c_bst_ulong(0)
                )
            )
            return

        # validate feature name
        feature_names = _validate_feature_info(
            feature_names,
            self.num_col(),
            self.data_split_mode() == DataSplitMode.COL,
            "feature names",
        )
        if len(feature_names) != len(set(feature_names)):
            values, counts = np.unique(
                feature_names,
                return_index=False,
                return_inverse=False,
                return_counts=True,
            )
            duplicates = [name for name, cnt in zip(values, counts) if cnt > 1]
            raise ValueError(
                f"feature_names must be unique. Duplicates found: {duplicates}"
            )

        # prohibit the use symbols that may affect parsing. e.g. []<
        if not all(
            isinstance(f, str) and not any(x in f for x in ["[", "]", "<"])
            for f in feature_names
        ):
            raise ValueError(
                "feature_names must be string, and may not contain [, ] or <"
            )

        feature_names_bytes = [bytes(f, encoding="utf-8") for f in feature_names]
        c_feature_names = (ctypes.c_char_p * len(feature_names_bytes))(
            *feature_names_bytes
        )
        _check_call(
            _LIB.XGDMatrixSetStrFeatureInfo(
                self.handle,
                c_str("feature_name"),
                c_feature_names,
                c_bst_ulong(len(feature_names)),
            )
        )

    @property
    def feature_types(self) -> Optional[FeatureTypes]:
        """Type of features (column types).

        This is for displaying the results and categorical data support. See
        :py:class:`DMatrix` for details.

        Setting it to ``None`` resets existing feature types.

        """
        length = c_bst_ulong()
        sarr = ctypes.POINTER(ctypes.c_char_p)()
        _check_call(
            _LIB.XGDMatrixGetStrFeatureInfo(
                self.handle,
                c_str("feature_type"),
                ctypes.byref(length),
                ctypes.byref(sarr),
            )
        )
        res = from_cstr_to_pystr(sarr, length)
        if not res:
            return None
        return res

    @feature_types.setter
    def feature_types(self, feature_types: Optional[FeatureTypes]) -> None:
        if feature_types is None:
            # Reset
            _check_call(
                _LIB.XGDMatrixSetStrFeatureInfo(
                    self.handle, c_str("feature_type"), None, c_bst_ulong(0)
                )
            )
            return

        feature_types = _validate_feature_info(
            feature_types,
            self.num_col(),
            self.data_split_mode() == DataSplitMode.COL,
            "feature types",
        )

        feature_types_bytes = [bytes(f, encoding="utf-8") for f in feature_types]
        c_feature_types = (ctypes.c_char_p * len(feature_types_bytes))(
            *feature_types_bytes
        )
        _check_call(
            _LIB.XGDMatrixSetStrFeatureInfo(
                self.handle,
                c_str("feature_type"),
                c_feature_types,
                c_bst_ulong(len(feature_types)),
            )
        )


class _ProxyDMatrix(DMatrix):
    """A placeholder class when DMatrix cannot be constructed (QuantileDMatrix,
    inplace_predict).

    """

    def __init__(self) -> None:  # pylint: disable=super-init-not-called
        self.handle = ctypes.c_void_p()
        _check_call(_LIB.XGProxyDMatrixCreate(ctypes.byref(self.handle)))

    def _ref_data_from_cuda_interface(self, data: DataType) -> None:
        """Reference data from CUDA array interface."""
        interface = data.__cuda_array_interface__
        interface_str = bytes(json.dumps(interface), "utf-8")
        _check_call(
            _LIB.XGProxyDMatrixSetDataCudaArrayInterface(self.handle, interface_str)
        )

    def _ref_data_from_cuda_columnar(self, data: DataType, cat_codes: list) -> None:
        """Reference data from CUDA columnar format."""
        from .data import _cudf_array_interfaces

        interfaces_str = _cudf_array_interfaces(data, cat_codes)
        _check_call(_LIB.XGProxyDMatrixSetDataCudaColumnar(self.handle, interfaces_str))

    def _ref_data_from_array(self, data: np.ndarray) -> None:
        """Reference data from numpy array."""
        from .data import _array_interface

        _check_call(
            _LIB.XGProxyDMatrixSetDataDense(self.handle, _array_interface(data))
        )

    def _ref_data_from_pandas(self, data: DataType) -> None:
        """Reference data from a pandas DataFrame. The input is a PandasTransformed instance."""
        _check_call(
            _LIB.XGProxyDMatrixSetDataColumnar(self.handle, data.array_interface())
        )

    def _ref_data_from_csr(self, csr: scipy.sparse.csr_matrix) -> None:
        """Reference data from scipy csr."""
        from .data import _array_interface

        _LIB.XGProxyDMatrixSetDataCSR(
            self.handle,
            _array_interface(csr.indptr),
            _array_interface(csr.indices),
            _array_interface(csr.data),
            ctypes.c_size_t(csr.shape[1]),
        )


class QuantileDMatrix(DMatrix):
    """A DMatrix variant that generates quantilized data directly from input for the
    ``hist`` tree method. This DMatrix is primarily designed to save memory in training
    by avoiding intermediate storage. Set ``max_bin`` to control the number of bins
    during quantisation, which should be consistent with the training parameter
    ``max_bin``. When ``QuantileDMatrix`` is used for validation/test dataset, ``ref``
    should be another ``QuantileDMatrix``(or ``DMatrix``, but not recommended as it
    defeats the purpose of saving memory) constructed from training dataset.  See
    :py:obj:`xgboost.DMatrix` for documents on meta info.

    .. note::

        Do not use ``QuantileDMatrix`` as validation/test dataset without supplying a
        reference (the training dataset) ``QuantileDMatrix`` using ``ref`` as some
        information may be lost in quantisation.

    .. versionadded:: 1.7.0

    Parameters
    ----------
    max_bin :
        The number of histogram bin, should be consistent with the training parameter
        ``max_bin``.

    ref :
        The training dataset that provides quantile information, needed when creating
        validation/test dataset with ``QuantileDMatrix``. Supplying the training DMatrix
        as a reference means that the same quantisation applied to the training data is
        applied to the validation/test data

    """

    @_deprecate_positional_args
    def __init__(  # pylint: disable=super-init-not-called
        self,
        data: DataType,
        label: Optional[ArrayLike] = None,
        *,
        weight: Optional[ArrayLike] = None,
        base_margin: Optional[ArrayLike] = None,
        missing: Optional[float] = None,
        silent: bool = False,
        feature_names: Optional[FeatureNames] = None,
        feature_types: Optional[FeatureTypes] = None,
        nthread: Optional[int] = None,
        max_bin: Optional[int] = None,
        ref: Optional[DMatrix] = None,
        group: Optional[ArrayLike] = None,
        qid: Optional[ArrayLike] = None,
        label_lower_bound: Optional[ArrayLike] = None,
        label_upper_bound: Optional[ArrayLike] = None,
        feature_weights: Optional[ArrayLike] = None,
        enable_categorical: bool = False,
        data_split_mode: DataSplitMode = DataSplitMode.ROW,
    ) -> None:
        self.max_bin = max_bin
        self.missing = missing if missing is not None else np.nan
        self.nthread = nthread if nthread is not None else -1
        self._silent = silent  # unused, kept for compatibility

        if isinstance(data, ctypes.c_void_p):
            self.handle = data
            return

        if qid is not None and group is not None:
            raise ValueError(
                "Only one of the eval_qid or eval_group for each evaluation "
                "dataset should be provided."
            )
        if isinstance(data, DataIter):
            if any(
                info is not None
                for info in (
                    label,
                    weight,
                    base_margin,
                    feature_names,
                    feature_types,
                    group,
                    qid,
                    label_lower_bound,
                    label_upper_bound,
                    feature_weights,
                )
            ):
                raise ValueError(
                    "If data iterator is used as input, data like label should be "
                    "specified as batch argument."
                )

        self._init(
            data,
            ref=ref,
            label=label,
            weight=weight,
            base_margin=base_margin,
            group=group,
            qid=qid,
            label_lower_bound=label_lower_bound,
            label_upper_bound=label_upper_bound,
            feature_weights=feature_weights,
            feature_names=feature_names,
            feature_types=feature_types,
            enable_categorical=enable_categorical,
        )

    def _init(
        self,
        data: DataType,
        ref: Optional[DMatrix],
        enable_categorical: bool,
        **meta: Any,
    ) -> None:
        from .data import (
            SingleBatchInternalIter,
            _is_dlpack,
            _is_iter,
            _transform_dlpack,
        )

        if _is_dlpack(data):
            # We specialize for dlpack because cupy will take the memory from it so
            # it can't be transformed twice.
            data = _transform_dlpack(data)
        if _is_iter(data):
            it = data
        else:
            it = SingleBatchInternalIter(data=data, **meta)

        handle = ctypes.c_void_p()
        reset_callback, next_callback = it.get_callbacks(enable_categorical)
        if it.cache_prefix is not None:
            raise ValueError(
                "QuantileDMatrix doesn't cache data, remove the cache_prefix "
                "in iterator to fix this error."
            )

        config = make_jcargs(
            nthread=self.nthread, missing=self.missing, max_bin=self.max_bin
        )
        ret = _LIB.XGQuantileDMatrixCreateFromCallback(
            None,
            it.proxy.handle,
            ref.handle if ref is not None else ref,
            reset_callback,
            next_callback,
            config,
            ctypes.byref(handle),
        )
        it.reraise()
        # delay check_call to throw intermediate exception first
        _check_call(ret)
        self.handle = handle


class DeviceQuantileDMatrix(QuantileDMatrix):
    """Use `QuantileDMatrix` instead.

    .. deprecated:: 1.7.0

    .. versionadded:: 1.1.0

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn("Please use `QuantileDMatrix` instead.", FutureWarning)
        super().__init__(*args, **kwargs)


Objective = Callable[[np.ndarray, DMatrix], Tuple[np.ndarray, np.ndarray]]
Metric = Callable[[np.ndarray, DMatrix], Tuple[str, float]]


def _configure_metrics(params: BoosterParam) -> BoosterParam:
    if (
        isinstance(params, dict)
        and "eval_metric" in params
        and isinstance(params["eval_metric"], list)
    ):
        eval_metrics = params["eval_metric"]
        params.pop("eval_metric", None)
        params_list = list(params.items())
        for eval_metric in eval_metrics:
            params_list += [("eval_metric", eval_metric)]
        return params_list
    return params


class Booster:
    # pylint: disable=too-many-public-methods
    """A Booster of XGBoost.

    Booster is the model of xgboost, that contains low level routines for
    training, prediction and evaluation.
    """

    def __init__(
        self,
        params: Optional[BoosterParam] = None,
        cache: Optional[Sequence[DMatrix]] = None,
        model_file: Optional[Union["Booster", bytearray, os.PathLike, str]] = None,
    ) -> None:
        # pylint: disable=invalid-name
        """
        Parameters
        ----------
        params :
            Parameters for boosters.
        cache :
            List of cache items.
        model_file :
            Path to the model file if it's string or PathLike.
        """
        cache = cache if cache is not None else []
        for d in cache:
            if not isinstance(d, DMatrix):
                raise TypeError(f"invalid cache item: {type(d).__name__}", cache)

        dmats = c_array(ctypes.c_void_p, [d.handle for d in cache])
        self.handle: Optional[ctypes.c_void_p] = ctypes.c_void_p()
        _check_call(
            _LIB.XGBoosterCreate(
                dmats, c_bst_ulong(len(cache)), ctypes.byref(self.handle)
            )
        )
        for d in cache:
            # Validate feature only after the feature names are saved into booster.
            self._assign_dmatrix_features(d)

        if isinstance(model_file, Booster):
            assert self.handle is not None
            # We use the pickle interface for getting memory snapshot from
            # another model, and load the snapshot with this booster.
            state = model_file.__getstate__()
            handle = state["handle"]
            del state["handle"]
            ptr = (ctypes.c_char * len(handle)).from_buffer(handle)
            length = c_bst_ulong(len(handle))
            _check_call(_LIB.XGBoosterUnserializeFromBuffer(self.handle, ptr, length))
            self.__dict__.update(state)
        elif isinstance(model_file, (str, os.PathLike, bytearray)):
            self.load_model(model_file)
        elif model_file is None:
            pass
        else:
            raise TypeError("Unknown type:", model_file)

        params = params or {}
        params_processed = _configure_metrics(params.copy())
        params_processed = self._configure_constraints(params_processed)
        if isinstance(params_processed, list):
            params_processed.append(("validate_parameters", True))
        else:
            params_processed["validate_parameters"] = True

        self.set_param(params_processed or {})

    def _transform_monotone_constrains(
        self, value: Union[Dict[str, int], str, Tuple[int, ...]]
    ) -> Union[Tuple[int, ...], str]:
        if isinstance(value, str):
            return value
        if isinstance(value, tuple):
            return value

        constrained_features = set(value.keys())
        feature_names = self.feature_names or []
        if not constrained_features.issubset(set(feature_names)):
            raise ValueError(
                "Constrained features are not a subset of training data feature names"
            )

        return tuple(value.get(name, 0) for name in feature_names)

    def _transform_interaction_constraints(
        self, value: Union[Sequence[Sequence[str]], str]
    ) -> Union[str, List[List[int]]]:
        if isinstance(value, str):
            return value
        feature_idx_mapping = {
            name: idx for idx, name in enumerate(self.feature_names or [])
        }

        try:
            result = []
            for constraint in value:
                result.append(
                    [feature_idx_mapping[feature_name] for feature_name in constraint]
                )
            return result
        except KeyError as e:
            raise ValueError(
                "Constrained features are not a subset of training data feature names"
            ) from e

    def _configure_constraints(self, params: BoosterParam) -> BoosterParam:
        if isinstance(params, dict):
            # we must use list in the internal code as there can be multiple metrics
            # with the same parameter name `eval_metric` (same key for dictionary).
            params = list(params.items())
        for idx, param in enumerate(params):
            name, value = param
            if value is None:
                continue

            if name == "monotone_constraints":
                params[idx] = (name, self._transform_monotone_constrains(value))
            elif name == "interaction_constraints":
                params[idx] = (name, self._transform_interaction_constraints(value))

        return params

    def __del__(self) -> None:
        if hasattr(self, "handle") and self.handle is not None:
            _check_call(_LIB.XGBoosterFree(self.handle))
            self.handle = None

    def __getstate__(self) -> Dict:
        # can't pickle ctypes pointers, put model content in bytearray
        this = self.__dict__.copy()
        handle = this["handle"]
        if handle is not None:
            length = c_bst_ulong()
            cptr = ctypes.POINTER(ctypes.c_char)()
            _check_call(
                _LIB.XGBoosterSerializeToBuffer(
                    self.handle, ctypes.byref(length), ctypes.byref(cptr)
                )
            )
            buf = ctypes2buffer(cptr, length.value)
            this["handle"] = buf
        return this

    def __setstate__(self, state: Dict) -> None:
        # reconstruct handle from raw data
        handle = state["handle"]
        if handle is not None:
            buf = handle
            dmats = c_array(ctypes.c_void_p, [])
            handle = ctypes.c_void_p()
            _check_call(
                _LIB.XGBoosterCreate(dmats, c_bst_ulong(0), ctypes.byref(handle))
            )
            length = c_bst_ulong(len(buf))
            ptr = (ctypes.c_char * len(buf)).from_buffer(buf)
            _check_call(_LIB.XGBoosterUnserializeFromBuffer(handle, ptr, length))
            state["handle"] = handle
        self.__dict__.update(state)

    def __getitem__(self, val: Union[Integer, tuple, slice]) -> "Booster":
        """Get a slice of the tree-based model.

        .. versionadded:: 1.3.0

        """
        # convert to slice for all other types
        if isinstance(val, (np.integer, int)):
            val = slice(int(val), int(val + 1))
        if isinstance(val, type(Ellipsis)):
            val = slice(0, 0)
        if isinstance(val, tuple):
            raise ValueError("Only supports slicing through 1 dimension.")
        # All supported types are now slice
        # FIXME(jiamingy): Use `types.EllipsisType` once Python 3.10 is used.
        if not isinstance(val, slice):
            msg = _expect((int, slice, np.integer, type(Ellipsis)), type(val))
            raise TypeError(msg)

        if isinstance(val.start, type(Ellipsis)) or val.start is None:
            start = 0
        else:
            start = val.start
        if isinstance(val.stop, type(Ellipsis)) or val.stop is None:
            stop = 0
        else:
            stop = val.stop
            if stop < start:
                raise ValueError("Invalid slice", val)

        step = val.step if val.step is not None else 1

        c_start = ctypes.c_int(start)
        c_stop = ctypes.c_int(stop)
        c_step = ctypes.c_int(step)

        sliced_handle = ctypes.c_void_p()
        status = _LIB.XGBoosterSlice(
            self.handle, c_start, c_stop, c_step, ctypes.byref(sliced_handle)
        )
        if status == -2:
            raise IndexError("Layer index out of range")
        _check_call(status)

        sliced = Booster()
        _check_call(_LIB.XGBoosterFree(sliced.handle))
        sliced.handle = sliced_handle
        return sliced

    def __iter__(self) -> Generator["Booster", None, None]:
        """Iterator method for getting individual trees.

        .. versionadded:: 2.0.0

        """
        for i in range(0, self.num_boosted_rounds()):
            yield self[i]

    def save_config(self) -> str:
        """Output internal parameter configuration of Booster as a JSON
        string.

        .. versionadded:: 1.0.0

        """
        json_string = ctypes.c_char_p()
        length = c_bst_ulong()
        _check_call(
            _LIB.XGBoosterSaveJsonConfig(
                self.handle, ctypes.byref(length), ctypes.byref(json_string)
            )
        )
        assert json_string.value is not None
        result = json_string.value.decode()  # pylint: disable=no-member
        return result

    def load_config(self, config: str) -> None:
        """Load configuration returned by `save_config`.

        .. versionadded:: 1.0.0
        """
        assert isinstance(config, str)
        _check_call(_LIB.XGBoosterLoadJsonConfig(self.handle, c_str(config)))

    def __copy__(self) -> "Booster":
        return self.__deepcopy__(None)

    def __deepcopy__(self, _: Any) -> "Booster":
        """Return a copy of booster."""
        return Booster(model_file=self)

    def copy(self) -> "Booster":
        """Copy the booster object.

        Returns
        -------
        booster :
            A copied booster model
        """
        return copy.copy(self)

    def attr(self, key: str) -> Optional[str]:
        """Get attribute string from the Booster.

        Parameters
        ----------
        key :
            The key to get attribute from.

        Returns
        -------
        value :
            The attribute value of the key, returns None if attribute do not exist.
        """
        ret = ctypes.c_char_p()
        success = ctypes.c_int()
        _check_call(
            _LIB.XGBoosterGetAttr(
                self.handle, c_str(key), ctypes.byref(ret), ctypes.byref(success)
            )
        )
        if success.value != 0:
            value = ret.value
            assert value
            return py_str(value)
        return None

    def attributes(self) -> Dict[str, Optional[str]]:
        """Get attributes stored in the Booster as a dictionary.

        Returns
        -------
        result : dictionary of  attribute_name: attribute_value pairs of strings.
            Returns an empty dict if there's no attributes.
        """
        length = c_bst_ulong()
        sarr = ctypes.POINTER(ctypes.c_char_p)()
        _check_call(
            _LIB.XGBoosterGetAttrNames(
                self.handle, ctypes.byref(length), ctypes.byref(sarr)
            )
        )
        attr_names = from_cstr_to_pystr(sarr, length)
        return {n: self.attr(n) for n in attr_names}

    def set_attr(self, **kwargs: Optional[Any]) -> None:
        """Set the attribute of the Booster.

        Parameters
        ----------
        **kwargs
            The attributes to set. Setting a value to None deletes an attribute.
        """
        for key, value in kwargs.items():
            c_value = None
            if value is not None:
                c_value = c_str(str(value))
            _check_call(_LIB.XGBoosterSetAttr(self.handle, c_str(key), c_value))

    def _get_feature_info(self, field: str) -> Optional[FeatureInfo]:
        length = c_bst_ulong()
        sarr = ctypes.POINTER(ctypes.c_char_p)()
        if not hasattr(self, "handle") or self.handle is None:
            return None
        _check_call(
            _LIB.XGBoosterGetStrFeatureInfo(
                self.handle,
                c_str(field),
                ctypes.byref(length),
                ctypes.byref(sarr),
            )
        )
        feature_info = from_cstr_to_pystr(sarr, length)
        return feature_info if feature_info else None

    def _set_feature_info(self, features: Optional[FeatureInfo], field: str) -> None:
        if features is not None:
            assert isinstance(features, list)
            feature_info_bytes = [bytes(f, encoding="utf-8") for f in features]
            c_feature_info = (ctypes.c_char_p * len(feature_info_bytes))(
                *feature_info_bytes
            )
            _check_call(
                _LIB.XGBoosterSetStrFeatureInfo(
                    self.handle,
                    c_str(field),
                    c_feature_info,
                    c_bst_ulong(len(features)),
                )
            )
        else:
            _check_call(
                _LIB.XGBoosterSetStrFeatureInfo(
                    self.handle, c_str(field), None, c_bst_ulong(0)
                )
            )

    @property
    def feature_types(self) -> Optional[FeatureTypes]:
        """Feature types for this booster.  Can be directly set by input data or by
        assignment.  See :py:class:`DMatrix` for details.

        """
        return self._get_feature_info("feature_type")

    @feature_types.setter
    def feature_types(self, features: Optional[FeatureTypes]) -> None:
        self._set_feature_info(features, "feature_type")

    @property
    def feature_names(self) -> Optional[FeatureNames]:
        """Feature names for this booster.  Can be directly set by input data or by
        assignment.

        """
        return self._get_feature_info("feature_name")

    @feature_names.setter
    def feature_names(self, features: Optional[FeatureNames]) -> None:
        self._set_feature_info(features, "feature_name")

    def set_param(
        self,
        params: Union[Dict, Iterable[Tuple[str, Any]], str],
        value: Optional[str] = None,
    ) -> None:
        """Set parameters into the Booster.

        Parameters
        ----------
        params :
           list of key,value pairs, dict of key to value or simply str key
        value :
           value of the specified parameter, when params is str key
        """
        if isinstance(params, Mapping):
            params = params.items()
        elif isinstance(params, str) and value is not None:
            params = [(params, value)]
        for key, val in cast(Iterable[Tuple[str, str]], params):
            if isinstance(val, np.ndarray):
                val = val.tolist()
            if val is not None:
                _check_call(
                    _LIB.XGBoosterSetParam(self.handle, c_str(key), c_str(str(val)))
                )

    def update(
        self, dtrain: DMatrix, iteration: int, fobj: Optional[Objective] = None
    ) -> None:
        """Update for one iteration, with objective function calculated
        internally.  This function should not be called directly by users.

        Parameters
        ----------
        dtrain :
            Training data.
        iteration :
            Current iteration number.
        fobj :
            Customized objective function.

        """
        if not isinstance(dtrain, DMatrix):
            raise TypeError(f"invalid training matrix: {type(dtrain).__name__}")
        self._assign_dmatrix_features(dtrain)

        if fobj is None:
            _check_call(
                _LIB.XGBoosterUpdateOneIter(
                    self.handle, ctypes.c_int(iteration), dtrain.handle
                )
            )
        else:
            pred = self.predict(dtrain, output_margin=True, training=True)
            grad, hess = fobj(pred, dtrain)
            self.boost(dtrain, iteration=iteration, grad=grad, hess=hess)

    def boost(
        self, dtrain: DMatrix, iteration: int, grad: NumpyOrCupy, hess: NumpyOrCupy
    ) -> None:
        """Boost the booster for one iteration with customized gradient statistics.
        Like :py:func:`xgboost.Booster.update`, this function should not be called
        directly by users.

        Parameters
        ----------
        dtrain :
            The training DMatrix.
        grad :
            The first order of gradient.
        hess :
            The second order of gradient.

        """
        from .data import (
            _array_interface,
            _cuda_array_interface,
            _ensure_np_dtype,
            _is_cupy_alike,
        )

        self._assign_dmatrix_features(dtrain)

        def is_flatten(array: NumpyOrCupy) -> bool:
            return len(array.shape) == 1 or array.shape[1] == 1

        def array_interface(array: NumpyOrCupy) -> bytes:
            # Can we check for __array_interface__ instead of a specific type instead?
            msg = (
                "Expecting `np.ndarray` or `cupy.ndarray` for gradient and hessian."
                f" Got: {type(array)}"
            )
            if not isinstance(array, np.ndarray) and not _is_cupy_alike(array):
                raise TypeError(msg)

            n_samples = dtrain.num_row()
            if array.shape[0] != n_samples and is_flatten(array):
                warnings.warn(
                    "Since 2.1.0, the shape of the gradient and hessian is required to"
                    " be (n_samples, n_targets) or (n_samples, n_classes).",
                    FutureWarning,
                )
                array = array.reshape(n_samples, array.size // n_samples)

            if isinstance(array, np.ndarray):
                array, _ = _ensure_np_dtype(array, array.dtype)
                interface = _array_interface(array)
            elif _is_cupy_alike(array):
                interface = _cuda_array_interface(array)
            else:
                raise TypeError(msg)

            return interface

        _check_call(
            _LIB.XGBoosterTrainOneIter(
                self.handle,
                dtrain.handle,
                iteration,
                array_interface(grad),
                array_interface(hess),
            )
        )

    def eval_set(
        self,
        evals: Sequence[Tuple[DMatrix, str]],
        iteration: int = 0,
        feval: Optional[Metric] = None,
        output_margin: bool = True,
    ) -> str:
        # pylint: disable=invalid-name
        """Evaluate a set of data.

        Parameters
        ----------
        evals :
            List of items to be evaluated.
        iteration :
            Current iteration.
        feval :
            Custom evaluation function.

        Returns
        -------
        result: str
            Evaluation result string.
        """
        for d in evals:
            if not isinstance(d[0], DMatrix):
                raise TypeError(f"expected DMatrix, got {type(d[0]).__name__}")
            if not isinstance(d[1], str):
                raise TypeError(f"expected string, got {type(d[1]).__name__}")
            self._assign_dmatrix_features(d[0])

        dmats = c_array(ctypes.c_void_p, [d[0].handle for d in evals])
        evnames = c_array(ctypes.c_char_p, [c_str(d[1]) for d in evals])
        msg = ctypes.c_char_p()
        _check_call(
            _LIB.XGBoosterEvalOneIter(
                self.handle,
                ctypes.c_int(iteration),
                dmats,
                evnames,
                c_bst_ulong(len(evals)),
                ctypes.byref(msg),
            )
        )
        assert msg.value is not None
        res = msg.value.decode()  # pylint: disable=no-member
        if feval is not None:
            for dmat, evname in evals:
                feval_ret = feval(
                    self.predict(dmat, training=False, output_margin=output_margin),
                    dmat,
                )
                if isinstance(feval_ret, list):
                    for name, val in feval_ret:
                        # pylint: disable=consider-using-f-string
                        res += "\t%s-%s:%f" % (evname, name, val)
                else:
                    name, val = feval_ret
                    # pylint: disable=consider-using-f-string
                    res += "\t%s-%s:%f" % (evname, name, val)
        return res

    def eval(self, data: DMatrix, name: str = "eval", iteration: int = 0) -> str:
        """Evaluate the model on mat.

        Parameters
        ----------
        data :
            The dmatrix storing the input.

        name :
            The name of the dataset.

        iteration :
            The current iteration number.

        Returns
        -------
        result: str
            Evaluation result string.
        """
        self._assign_dmatrix_features(data)
        return self.eval_set([(data, name)], iteration)

    # pylint: disable=too-many-function-args
    def predict(
        self,
        data: DMatrix,
        output_margin: bool = False,
        pred_leaf: bool = False,
        pred_contribs: bool = False,
        approx_contribs: bool = False,
        pred_interactions: bool = False,
        validate_features: bool = True,
        training: bool = False,
        iteration_range: IterationRange = (0, 0),
        strict_shape: bool = False,
    ) -> np.ndarray:
        """Predict with data.  The full model will be used unless `iteration_range` is
        specified, meaning user have to either slice the model or use the
        ``best_iteration`` attribute to get prediction from best model returned from
        early stopping.

        .. note::

            See :doc:`Prediction </prediction>` for issues like thread safety and a
            summary of outputs from this function.

        Parameters
        ----------
        data :
            The dmatrix storing the input.

        output_margin :
            Whether to output the raw untransformed margin value.

        pred_leaf :
            When this option is on, the output will be a matrix of (nsample,
            ntrees) with each record indicating the predicted leaf index of
            each sample in each tree.  Note that the leaf index of a tree is
            unique per tree, so you may find leaf 1 in both tree 1 and tree 0.

        pred_contribs :
            When this is True the output will be a matrix of size (nsample,
            nfeats + 1) with each record indicating the feature contributions
            (SHAP values) for that prediction. The sum of all feature
            contributions is equal to the raw untransformed margin value of the
            prediction. Note the final column is the bias term.

        approx_contribs :
            Approximate the contributions of each feature.  Used when ``pred_contribs`` or
            ``pred_interactions`` is set to True.  Changing the default of this parameter
            (False) is not recommended.

        pred_interactions :
            When this is True the output will be a matrix of size (nsample,
            nfeats + 1, nfeats + 1) indicating the SHAP interaction values for
            each pair of features. The sum of each row (or column) of the
            interaction values equals the corresponding SHAP value (from
            pred_contribs), and the sum of the entire matrix equals the raw
            untransformed margin value of the prediction. Note the last row and
            column correspond to the bias term.

        validate_features :
            When this is True, validate that the Booster's and data's
            feature_names are identical.  Otherwise, it is assumed that the
            feature_names are the same.

        training :
            Whether the prediction value is used for training.  This can effect `dart`
            booster, which performs dropouts during training iterations but use all trees
            for inference. If you want to obtain result with dropouts, set this parameter
            to `True`.  Also, the parameter is set to true when obtaining prediction for
            custom objective function.

            .. versionadded:: 1.0.0

        iteration_range :
            Specifies which layer of trees are used in prediction.  For example, if a
            random forest is trained with 100 rounds.  Specifying `iteration_range=(10,
            20)`, then only the forests built during [10, 20) (half open set) rounds are
            used in this prediction.

            .. versionadded:: 1.4.0

        strict_shape :
            When set to True, output shape is invariant to whether classification is used.
            For both value and margin prediction, the output shape is (n_samples,
            n_groups), n_groups == 1 when multi-class is not used.  Default to False, in
            which case the output shape can be (n_samples, ) if multi-class is not used.

            .. versionadded:: 1.4.0

        Returns
        -------
        prediction : numpy array

        """
        if not isinstance(data, DMatrix):
            raise TypeError("Expecting data to be a DMatrix object, got: ", type(data))
        if validate_features:
            fn = data.feature_names
            self._validate_features(fn)
        args = {
            "type": 0,
            "training": training,
            "iteration_begin": int(iteration_range[0]),
            "iteration_end": int(iteration_range[1]),
            "strict_shape": strict_shape,
        }

        def assign_type(t: int) -> None:
            if args["type"] != 0:
                raise ValueError("One type of prediction at a time.")
            args["type"] = t

        if output_margin:
            assign_type(1)
        if pred_contribs:
            assign_type(2 if not approx_contribs else 3)
        if pred_interactions:
            assign_type(4 if not approx_contribs else 5)
        if pred_leaf:
            assign_type(6)
        preds = ctypes.POINTER(ctypes.c_float)()
        shape = ctypes.POINTER(c_bst_ulong)()
        dims = c_bst_ulong()
        _check_call(
            _LIB.XGBoosterPredictFromDMatrix(
                self.handle,
                data.handle,
                from_pystr_to_cstr(json.dumps(args)),
                ctypes.byref(shape),
                ctypes.byref(dims),
                ctypes.byref(preds),
            )
        )
        return _prediction_output(shape, dims, preds, False)

    # pylint: disable=too-many-statements
    def inplace_predict(
        self,
        data: DataType,
        iteration_range: IterationRange = (0, 0),
        predict_type: str = "value",
        missing: float = np.nan,
        validate_features: bool = True,
        base_margin: Any = None,
        strict_shape: bool = False,
    ) -> NumpyOrCupy:
        """Run prediction in-place when possible, Unlike :py:meth:`predict` method,
        inplace prediction does not cache the prediction result.

        Calling only ``inplace_predict`` in multiple threads is safe and lock
        free.  But the safety does not hold when used in conjunction with other
        methods. E.g. you can't train the booster in one thread and perform
        prediction in the other.

        .. note::

            If the device ordinal of the input data doesn't match the one configured for
            the booster, data will be copied to the booster device.

        .. code-block:: python

            booster.set_param({"device": "cuda:0"})
            booster.inplace_predict(cupy_array)

            booster.set_param({"device": "cpu"})
            booster.inplace_predict(numpy_array)

        .. versionadded:: 1.1.0

        Parameters
        ----------
        data :
            The input data.
        iteration_range :
            See :py:meth:`predict` for details.
        predict_type :
            * `value` Output model prediction values.
            * `margin` Output the raw untransformed margin value.
        missing :
            See :py:obj:`xgboost.DMatrix` for details.
        validate_features:
            See :py:meth:`xgboost.Booster.predict` for details.
        base_margin:
            See :py:obj:`xgboost.DMatrix` for details.

            .. versionadded:: 1.4.0

        strict_shape:
            See :py:meth:`xgboost.Booster.predict` for details.

            .. versionadded:: 1.4.0

        Returns
        -------
        prediction : numpy.ndarray/cupy.ndarray
            The prediction result.  When input data is on GPU, prediction result is
            stored in a cupy array.

        """
        preds = ctypes.POINTER(ctypes.c_float)()

        # once caching is supported, we can pass id(data) as cache id.
        args = make_jcargs(
            type=1 if predict_type == "margin" else 0,
            training=False,
            iteration_begin=int(iteration_range[0]),
            iteration_end=int(iteration_range[1]),
            missing=missing,
            strict_shape=strict_shape,
            cache_id=0,
        )
        shape = ctypes.POINTER(c_bst_ulong)()
        dims = c_bst_ulong()

        if base_margin is not None:
            proxy: Optional[_ProxyDMatrix] = _ProxyDMatrix()
            assert proxy is not None
            proxy.set_info(base_margin=base_margin)
            p_handle = proxy.handle
        else:
            proxy = None
            p_handle = ctypes.c_void_p()
        assert proxy is None or isinstance(proxy, _ProxyDMatrix)

        from .data import (
            PandasTransformed,
            _array_interface,
            _arrow_transform,
            _is_arrow,
            _is_cudf_df,
            _is_cupy_alike,
            _is_list,
            _is_np_array_like,
            _is_pandas_df,
            _is_pandas_series,
            _is_tuple,
            _transform_pandas_df,
        )

        enable_categorical = True
        if _is_arrow(data):
            data = _arrow_transform(data)
        if _is_pandas_series(data):
            import pandas as pd

            data = pd.DataFrame(data)
        if _is_pandas_df(data):
            data, fns, _ = _transform_pandas_df(data, enable_categorical)
            if validate_features:
                self._validate_features(fns)
        if _is_list(data) or _is_tuple(data):
            data = np.array(data)

        if validate_features:
            if not hasattr(data, "shape"):
                raise TypeError(
                    "`shape` attribute is required when `validate_features` is True."
                )
            if len(data.shape) != 1 and self.num_features() != data.shape[1]:
                raise ValueError(
                    f"Feature shape mismatch, expected: {self.num_features()}, "
                    f"got {data.shape[1]}"
                )

        if _is_np_array_like(data):
            from .data import _ensure_np_dtype

            data, _ = _ensure_np_dtype(data, data.dtype)
            _check_call(
                _LIB.XGBoosterPredictFromDense(
                    self.handle,
                    _array_interface(data),
                    args,
                    p_handle,
                    ctypes.byref(shape),
                    ctypes.byref(dims),
                    ctypes.byref(preds),
                )
            )
            return _prediction_output(shape, dims, preds, False)
        if isinstance(data, PandasTransformed):
            _check_call(
                _LIB.XGBoosterPredictFromColumnar(
                    self.handle,
                    data.array_interface(),
                    args,
                    p_handle,
                    ctypes.byref(shape),
                    ctypes.byref(dims),
                    ctypes.byref(preds),
                )
            )
            return _prediction_output(shape, dims, preds, False)
        if isinstance(data, scipy.sparse.csr_matrix):
            from .data import transform_scipy_sparse

            data = transform_scipy_sparse(data, True)
            _check_call(
                _LIB.XGBoosterPredictFromCSR(
                    self.handle,
                    _array_interface(data.indptr),
                    _array_interface(data.indices),
                    _array_interface(data.data),
                    c_bst_ulong(data.shape[1]),
                    args,
                    p_handle,
                    ctypes.byref(shape),
                    ctypes.byref(dims),
                    ctypes.byref(preds),
                )
            )
            return _prediction_output(shape, dims, preds, False)
        if _is_cupy_alike(data):
            from .data import _transform_cupy_array

            data = _transform_cupy_array(data)
            interface_str = _cuda_array_interface(data)
            _check_call(
                _LIB.XGBoosterPredictFromCudaArray(
                    self.handle,
                    interface_str,
                    args,
                    p_handle,
                    ctypes.byref(shape),
                    ctypes.byref(dims),
                    ctypes.byref(preds),
                )
            )
            return _prediction_output(shape, dims, preds, True)
        if _is_cudf_df(data):
            from .data import _cudf_array_interfaces, _transform_cudf_df

            data, cat_codes, fns, _ = _transform_cudf_df(
                data, None, None, enable_categorical
            )
            interfaces_str = _cudf_array_interfaces(data, cat_codes)
            if validate_features:
                self._validate_features(fns)
            _check_call(
                _LIB.XGBoosterPredictFromCudaColumnar(
                    self.handle,
                    interfaces_str,
                    args,
                    p_handle,
                    ctypes.byref(shape),
                    ctypes.byref(dims),
                    ctypes.byref(preds),
                )
            )
            return _prediction_output(shape, dims, preds, True)

        raise TypeError(
            "Data type:" + str(type(data)) + " not supported by inplace prediction."
        )

    def save_model(self, fname: Union[str, os.PathLike]) -> None:
        """Save the model to a file.

        The model is saved in an XGBoost internal format which is universal among the
        various XGBoost interfaces. Auxiliary attributes of the Python Booster object
        (such as feature_names) are only saved when using JSON or UBJSON (default)
        format. See :doc:`Model IO </tutorials/saving_model>` for more info.

        .. code-block:: python

          model.save_model("model.json")
          # or
          model.save_model("model.ubj")

        Parameters
        ----------
        fname :
            Output file name

        """
        if isinstance(fname, (str, os.PathLike)):  # assume file name
            fname = os.fspath(os.path.expanduser(fname))
            _check_call(_LIB.XGBoosterSaveModel(self.handle, c_str(fname)))
        else:
            raise TypeError("fname must be a string or os PathLike")

    def save_raw(self, raw_format: str = "ubj") -> bytearray:
        """Save the model to a in memory buffer representation instead of file.

        The model is saved in an XGBoost internal format which is universal among the
        various XGBoost interfaces. Auxiliary attributes of the Python Booster object
        (such as feature_names) are only saved when using JSON or UBJSON (default)
        format. See :doc:`Model IO </tutorials/saving_model>` for more info.

        Parameters
        ----------
        raw_format :
            Format of output buffer. Can be `json`, `ubj` or `deprecated`.

        Returns
        -------
        An in memory buffer representation of the model
        """
        length = c_bst_ulong()
        cptr = ctypes.POINTER(ctypes.c_char)()
        config = make_jcargs(format=raw_format)
        _check_call(
            _LIB.XGBoosterSaveModelToBuffer(
                self.handle, config, ctypes.byref(length), ctypes.byref(cptr)
            )
        )
        return ctypes2buffer(cptr, length.value)

    def load_model(self, fname: ModelIn) -> None:
        """Load the model from a file or a bytearray.

        The model is saved in an XGBoost internal format which is universal among the
        various XGBoost interfaces. Auxiliary attributes of the Python Booster object
        (such as feature_names) are only saved when using JSON or UBJSON (default)
        format. See :doc:`Model IO </tutorials/saving_model>` for more info.

        .. code-block:: python

          model.load_model("model.json")
          # or
          model.load_model("model.ubj")

        Parameters
        ----------
        fname :
            Input file name or memory buffer(see also save_raw)

        """
        if isinstance(fname, (str, os.PathLike)):
            # assume file name, cannot use os.path.exist to check, file can be
            # from URL.
            fname = os.fspath(os.path.expanduser(fname))
            _check_call(_LIB.XGBoosterLoadModel(self.handle, c_str(fname)))
        elif isinstance(fname, bytearray):
            buf = fname
            length = c_bst_ulong(len(buf))
            ptr = (ctypes.c_char * len(buf)).from_buffer(buf)
            _check_call(_LIB.XGBoosterLoadModelFromBuffer(self.handle, ptr, length))
        else:
            raise TypeError("Unknown file type: ", fname)

    @property
    def best_iteration(self) -> int:
        """The best iteration during training."""
        best = self.attr("best_iteration")
        if best is not None:
            return int(best)

        raise AttributeError(
            "`best_iteration` is only defined when early stopping is used."
        )

    @best_iteration.setter
    def best_iteration(self, iteration: int) -> None:
        self.set_attr(best_iteration=iteration)

    @property
    def best_score(self) -> float:
        """The best evaluation score during training."""
        best = self.attr("best_score")
        if best is not None:
            return float(best)

        raise AttributeError(
            "`best_score` is only defined when early stopping is used."
        )

    @best_score.setter
    def best_score(self, score: int) -> None:
        self.set_attr(best_score=score)

    def num_boosted_rounds(self) -> int:
        """Get number of boosted rounds.  For gblinear this is reset to 0 after
        serializing the model.

        """
        rounds = ctypes.c_int()
        assert self.handle is not None
        _check_call(_LIB.XGBoosterBoostedRounds(self.handle, ctypes.byref(rounds)))
        return rounds.value

    def num_features(self) -> int:
        """Number of features in booster."""
        features = c_bst_ulong()
        assert self.handle is not None
        _check_call(_LIB.XGBoosterGetNumFeature(self.handle, ctypes.byref(features)))
        return features.value

    def dump_model(
        self,
        fout: Union[str, os.PathLike],
        fmap: Union[str, os.PathLike] = "",
        with_stats: bool = False,
        dump_format: str = "text",
    ) -> None:
        """Dump model into a text or JSON file.  Unlike :py:meth:`save_model`, the
        output format is primarily used for visualization or interpretation,
        hence it's more human readable but cannot be loaded back to XGBoost.

        Parameters
        ----------
        fout :
            Output file name.
        fmap :
            Name of the file containing feature map names.
        with_stats :
            Controls whether the split statistics are output.
        dump_format :
            Format of model dump file. Can be 'text' or 'json'.
        """
        if isinstance(fout, (str, os.PathLike)):
            fout = os.fspath(os.path.expanduser(fout))
            # pylint: disable=consider-using-with
            fout_obj = open(fout, "w", encoding="utf-8")
            need_close = True
        else:
            fout_obj = fout
            need_close = False
        ret = self.get_dump(fmap, with_stats, dump_format)
        if dump_format == "json":
            fout_obj.write("[\n")
            for i, val in enumerate(ret):
                fout_obj.write(val)
                if i < len(ret) - 1:
                    fout_obj.write(",\n")
            fout_obj.write("\n]")
        else:
            for i, val in enumerate(ret):
                fout_obj.write(f"booster[{i}]:\n")
                fout_obj.write(val)
        if need_close:
            fout_obj.close()

    def get_dump(
        self,
        fmap: Union[str, os.PathLike] = "",
        with_stats: bool = False,
        dump_format: str = "text",
    ) -> List[str]:
        """Returns the model dump as a list of strings.  Unlike :py:meth:`save_model`,
        the output format is primarily used for visualization or interpretation, hence
        it's more human readable but cannot be loaded back to XGBoost.

        Parameters
        ----------
        fmap :
            Name of the file containing feature map names.
        with_stats :
            Controls whether the split statistics are output.
        dump_format :
            Format of model dump. Can be 'text', 'json' or 'dot'.

        """
        fmap = os.fspath(os.path.expanduser(fmap))
        length = c_bst_ulong()
        sarr = ctypes.POINTER(ctypes.c_char_p)()
        _check_call(
            _LIB.XGBoosterDumpModelEx(
                self.handle,
                c_str(fmap),
                ctypes.c_int(with_stats),
                c_str(dump_format),
                ctypes.byref(length),
                ctypes.byref(sarr),
            )
        )
        res = from_cstr_to_pystr(sarr, length)
        return res

    def get_fscore(
        self, fmap: Union[str, os.PathLike] = ""
    ) -> Dict[str, Union[float, List[float]]]:
        """Get feature importance of each feature.

        .. note:: Zero-importance features will not be included

           Keep in mind that this function does not include zero-importance feature, i.e.
           those features that have not been used in any split conditions.

        Parameters
        ----------
        fmap :
           The name of feature map file
        """

        return self.get_score(fmap, importance_type="weight")

    def get_score(
        self, fmap: Union[str, os.PathLike] = "", importance_type: str = "weight"
    ) -> Dict[str, Union[float, List[float]]]:
        """Get feature importance of each feature.
        For tree model Importance type can be defined as:

        * 'weight': the number of times a feature is used to split the data across all trees.
        * 'gain': the average gain across all splits the feature is used in.
        * 'cover': the average coverage across all splits the feature is used in.
        * 'total_gain': the total gain across all splits the feature is used in.
        * 'total_cover': the total coverage across all splits the feature is used in.

        .. note::

           For linear model, only "weight" is defined and it's the normalized coefficients
           without bias.

        .. note:: Zero-importance features will not be included

           Keep in mind that this function does not include zero-importance feature, i.e.
           those features that have not been used in any split conditions.

        Parameters
        ----------
        fmap :
           The name of feature map file.
        importance_type :
            One of the importance types defined above.

        Returns
        -------
        A map between feature names and their scores.  When `gblinear` is used for
        multi-class classification the scores for each feature is a list with length
        `n_classes`, otherwise they're scalars.
        """
        fmap = os.fspath(os.path.expanduser(fmap))
        features = ctypes.POINTER(ctypes.c_char_p)()
        scores = ctypes.POINTER(ctypes.c_float)()
        n_out_features = c_bst_ulong()
        out_dim = c_bst_ulong()
        shape = ctypes.POINTER(c_bst_ulong)()

        _check_call(
            _LIB.XGBoosterFeatureScore(
                self.handle,
                make_jcargs(importance_type=importance_type, feature_map=fmap),
                ctypes.byref(n_out_features),
                ctypes.byref(features),
                ctypes.byref(out_dim),
                ctypes.byref(shape),
                ctypes.byref(scores),
            )
        )
        features_arr = from_cstr_to_pystr(features, n_out_features)
        scores_arr = _prediction_output(shape, out_dim, scores, False)

        results: Dict[str, Union[float, List[float]]] = {}
        if len(scores_arr.shape) > 1 and scores_arr.shape[1] > 1:
            for feat, score in zip(features_arr, scores_arr):
                results[feat] = [float(s) for s in score]
        else:
            for feat, score in zip(features_arr, scores_arr):
                results[feat] = float(score)
        return results

    # pylint: disable=too-many-statements
    def trees_to_dataframe(self, fmap: Union[str, os.PathLike] = "") -> DataFrame:
        """Parse a boosted tree model text dump into a pandas DataFrame structure.

        This feature is only defined when the decision tree model is chosen as base
        learner (`booster in {gbtree, dart}`). It is not defined for other base learner
        types, such as linear learners (`booster=gblinear`).

        Parameters
        ----------
        fmap :
           The name of feature map file.
        """
        # pylint: disable=too-many-locals
        fmap = os.fspath(os.path.expanduser(fmap))
        if not PANDAS_INSTALLED:
            raise ImportError(
                (
                    "pandas must be available to use this method."
                    "Install pandas before calling again."
                )
            )
        booster = json.loads(self.save_config())["learner"]["gradient_booster"]["name"]
        if booster not in {"gbtree", "dart"}:
            raise ValueError(f"This method is not defined for Booster type {booster}")

        tree_ids = []
        node_ids = []
        fids = []
        splits: List[Union[float, str]] = []
        categories: List[Union[Optional[float], List[str]]] = []
        y_directs: List[Union[float, str]] = []
        n_directs: List[Union[float, str]] = []
        missings: List[Union[float, str]] = []
        gains = []
        covers = []

        trees = self.get_dump(fmap, with_stats=True)
        for i, tree in enumerate(trees):
            for line in tree.split("\n"):
                arr = line.split("[")
                # Leaf node
                if len(arr) == 1:
                    # Last element of line.split is an empty string
                    if arr == [""]:
                        continue
                    # parse string
                    parse = arr[0].split(":")
                    stats = re.split("=|,", parse[1])

                    # append to lists
                    tree_ids.append(i)
                    node_ids.append(int(re.findall(r"\b\d+\b", parse[0])[0]))
                    fids.append("Leaf")
                    splits.append(float("NAN"))
                    categories.append(float("NAN"))
                    y_directs.append(float("NAN"))
                    n_directs.append(float("NAN"))
                    missings.append(float("NAN"))
                    gains.append(float(stats[1]))
                    covers.append(float(stats[3]))
                # Not a Leaf Node
                else:
                    # parse string
                    fid = arr[1].split("]")
                    if fid[0].find("<") != -1:
                        # numerical
                        parse = fid[0].split("<")
                        splits.append(float(parse[1]))
                        categories.append(None)
                    elif fid[0].find(":{") != -1:
                        # categorical
                        parse = fid[0].split(":")
                        cats = parse[1][1:-1]  # strip the {}
                        cats_split = cats.split(",")
                        splits.append(float("NAN"))
                        categories.append(cats_split if cats_split else None)
                    else:
                        raise ValueError("Failed to parse model text dump.")
                    stats = re.split("=|,", fid[1])

                    # append to lists
                    tree_ids.append(i)
                    node_ids.append(int(re.findall(r"\b\d+\b", arr[0])[0]))
                    fids.append(parse[0])
                    str_i = str(i)
                    y_directs.append(str_i + "-" + stats[1])
                    n_directs.append(str_i + "-" + stats[3])
                    missings.append(str_i + "-" + stats[5])
                    gains.append(float(stats[7]))
                    covers.append(float(stats[9]))

        ids = [str(t_id) + "-" + str(n_id) for t_id, n_id in zip(tree_ids, node_ids)]
        df = DataFrame(
            {
                "Tree": tree_ids,
                "Node": node_ids,
                "ID": ids,
                "Feature": fids,
                "Split": splits,
                "Yes": y_directs,
                "No": n_directs,
                "Missing": missings,
                "Gain": gains,
                "Cover": covers,
                "Category": categories,
            }
        )

        if callable(getattr(df, "sort_values", None)):
            # pylint: disable=no-member
            return df.sort_values(["Tree", "Node"]).reset_index(drop=True)
        # pylint: disable=no-member
        return df.sort(["Tree", "Node"]).reset_index(drop=True)

    def _assign_dmatrix_features(self, data: DMatrix) -> None:
        if data.num_row() == 0:
            return

        fn = data.feature_names
        ft = data.feature_types

        if self.feature_names is None:
            self.feature_names = fn
        if self.feature_types is None:
            self.feature_types = ft

        self._validate_features(fn)

    def _validate_features(self, feature_names: Optional[FeatureNames]) -> None:
        if self.feature_names is None:
            return

        if feature_names is None and self.feature_names is not None:
            raise ValueError(
                "training data did not have the following fields: "
                + ", ".join(self.feature_names)
            )

        if self.feature_names != feature_names:
            dat_missing = set(cast(FeatureNames, self.feature_names)) - set(
                cast(FeatureNames, feature_names)
            )
            my_missing = set(cast(FeatureNames, feature_names)) - set(
                cast(FeatureNames, self.feature_names)
            )

            msg = "feature_names mismatch: {0} {1}"

            if dat_missing:
                msg += (
                    "\nexpected "
                    + ", ".join(str(s) for s in dat_missing)
                    + " in input data"
                )

            if my_missing:
                msg += (
                    "\ntraining data did not have the following fields: "
                    + ", ".join(str(s) for s in my_missing)
                )

            raise ValueError(msg.format(self.feature_names, feature_names))

    def get_split_value_histogram(
        self,
        feature: str,
        fmap: Union[os.PathLike, str] = "",
        bins: Optional[int] = None,
        as_pandas: bool = True,
    ) -> Union[np.ndarray, DataFrame]:
        """Get split value histogram of a feature

        Parameters
        ----------
        feature :
            The name of the feature.
        fmap:
            The name of feature map file.
        bin :
            The maximum number of bins.
            Number of bins equals number of unique split values n_unique,
            if bins == None or bins > n_unique.
        as_pandas :
            Return pd.DataFrame when pandas is installed.
            If False or pandas is not installed, return numpy ndarray.

        Returns
        -------
        a histogram of used splitting values for the specified feature
        either as numpy array or pandas DataFrame.
        """
        xgdump = self.get_dump(fmap=fmap)
        values = []
        # pylint: disable=consider-using-f-string
        regexp = re.compile(r"\[{0}<([\d.Ee+-]+)\]".format(feature))
        for val in xgdump:
            m = re.findall(regexp, val)
            values.extend([float(x) for x in m])

        n_unique = len(np.unique(values))
        bins = max(min(n_unique, bins) if bins is not None else n_unique, 1)

        nph = np.histogram(values, bins=bins)
        nph_stacked = np.column_stack((nph[1][1:], nph[0]))
        nph_stacked = nph_stacked[nph_stacked[:, 1] > 0]

        if nph_stacked.size == 0:
            ft = self.feature_types
            fn = self.feature_names
            if fn is None:
                # Let xgboost generate the feature names.
                fn = [f"f{i}" for i in range(self.num_features())]
            try:
                index = fn.index(feature)
                feature_t: Optional[str] = cast(List[str], ft)[index]
            except (ValueError, AttributeError, TypeError):
                # None.index: attr err, None[0]: type err, fn.index(-1): value err
                feature_t = None
            if feature_t == "c":  # categorical
                raise ValueError(
                    "Split value historgam doesn't support categorical split."
                )

        if as_pandas and PANDAS_INSTALLED:
            return DataFrame(nph_stacked, columns=["SplitValue", "Count"])
        if as_pandas and not PANDAS_INSTALLED:
            warnings.warn(
                "Returning histogram as ndarray"
                " (as_pandas == True, but pandas is not installed).",
                UserWarning,
            )
        return nph_stacked
