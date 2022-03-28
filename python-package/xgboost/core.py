# pylint: disable=too-many-arguments, too-many-branches, invalid-name
# pylint: disable=too-many-lines, too-many-locals
"""Core XGBoost Library."""
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import List, Optional, Any, Union, Dict, TypeVar
from typing import Callable, Tuple, cast, Sequence, Type, Iterable
import ctypes
import os
import re
import sys
import json
import warnings
from functools import wraps
from inspect import signature, Parameter

import numpy as np
import scipy.sparse

from .compat import STRING_TYPES, DataFrame, py_str, PANDAS_INSTALLED
from .libpath import find_lib_path
from ._typing import (
    CStrPptr,
    c_bst_ulong,
    CNumeric,
    DataType,
    CNumericPtr,
    CStrPtr,
    CTypeT,
    ArrayLike,
    CFloatPtr,
    NumpyOrCupy,
    FeatureNames,
    _T,
    CupyT,
)


class XGBoostError(ValueError):
    """Error thrown by xgboost trainer."""


def from_pystr_to_cstr(data: Union[str, List[str]]) -> Union[bytes, CStrPptr]:
    """Convert a Python str or list of Python str to C pointer

    Parameters
    ----------
    data
        str or list of str
    """

    if isinstance(data, str):
        return bytes(data, "utf-8")
    if isinstance(data, list):
        pointers: ctypes.pointer = (ctypes.c_char_p * len(data))()
        data_as_bytes = [bytes(d, 'utf-8') for d in data]
        pointers[:] = data_as_bytes
        return pointers
    raise TypeError()


def from_cstr_to_pystr(data: CStrPptr, length: c_bst_ulong) -> List[str]:
    """Revert C pointer to Python str

    Parameters
    ----------
    data : ctypes pointer
        pointer to data
    length : ctypes pointer
        pointer to length of data
    """
    res = []
    for i in range(length.value):
        try:
            res.append(str(data[i].decode('ascii')))  # type: ignore
        except UnicodeDecodeError:
            res.append(str(data[i].decode('utf-8')))  # type: ignore
    return res


IterRange = TypeVar("IterRange", Optional[Tuple[int, int]], Tuple[int, int])


def _convert_ntree_limit(
    booster: "Booster",
    ntree_limit: Optional[int],
    iteration_range: IterRange
) -> IterRange:
    if ntree_limit is not None and ntree_limit != 0:
        warnings.warn(
            "ntree_limit is deprecated, use `iteration_range` or model "
            "slicing instead.",
            UserWarning
        )
        if iteration_range is not None and iteration_range[1] != 0:
            raise ValueError(
                "Only one of `iteration_range` and `ntree_limit` can be non zero."
            )
        num_parallel_tree, _ = _get_booster_layer_trees(booster)
        num_parallel_tree = max([num_parallel_tree, 1])
        iteration_range = (0, ntree_limit // num_parallel_tree)
    return iteration_range


def _expect(expectations: Sequence[Type], got: Type) -> str:
    """Translate input error into string.

    Parameters
    ----------
    expectations: sequence
        a list of expected value.
    got:
        actual input

    Returns
    -------
    msg: str
    """
    msg = 'Expecting '
    for t in range(len(expectations) - 1):
        msg += str(expectations[t])
        msg += ' or '
    msg += str(expectations[-1])
    msg += '.  Got ' + str(got)
    return msg


def _log_callback(msg: bytes) -> None:
    """Redirect logs from native library into Python console"""
    print(py_str(msg))


def _get_log_callback_func() -> Callable:
    """Wrap log_callback() method in ctypes callback type"""
    c_callback = ctypes.CFUNCTYPE(None, ctypes.c_char_p)
    return c_callback(_log_callback)


def _load_lib() -> ctypes.CDLL:
    """Load xgboost Library."""
    lib_paths = find_lib_path()
    if not lib_paths:
        # This happens only when building document.
        return None  # type: ignore
    try:
        pathBackup = os.environ['PATH'].split(os.pathsep)
    except KeyError:
        pathBackup = []
    lib_success = False
    os_error_list = []
    for lib_path in lib_paths:
        try:
            # needed when the lib is linked with non-system-available
            # dependencies
            os.environ['PATH'] = os.pathsep.join(
                pathBackup + [os.path.dirname(lib_path)])
            lib = ctypes.cdll.LoadLibrary(lib_path)
            lib_success = True
        except OSError as e:
            os_error_list.append(str(e))
            continue
        finally:
            os.environ['PATH'] = os.pathsep.join(pathBackup)
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
""")
    lib.XGBGetLastError.restype = ctypes.c_char_p
    lib.callback = _get_log_callback_func()  # type: ignore
    if lib.XGBRegisterLogCallback(lib.callback) != 0:
        raise XGBoostError(lib.XGBGetLastError())
    return lib


# load the XGBoost library globally
_LIB = _load_lib()


def _check_call(ret: int) -> None:
    """Check the return value of C API call

    This function will raise exception when error occurs.
    Wrap every API call with this function

    Parameters
    ----------
    ret : int
        return value from API calls
    """
    if ret != 0:
        raise XGBoostError(py_str(_LIB.XGBGetLastError()))


def _has_categorical(booster: "Booster", data: DataType) -> bool:
    """Check whether the booster and input data for prediction contain categorical data.

    """
    from .data import _is_pandas_df, _is_cudf_df
    if _is_pandas_df(data) or _is_cudf_df(data):
        ft = booster.feature_types
        if ft is None:
            enable_categorical = False
        else:
            enable_categorical = any(f == "c" for f in ft)
    else:
        enable_categorical = False
    return enable_categorical


def build_info() -> dict:
    """Build information of XGBoost.  The returned value format is not stable. Also, please
    note that build time dependency is not the same as runtime dependency. For instance,
    it's possible to build XGBoost with older CUDA version but run it with the lastest
    one.

      .. versionadded:: 1.6.0

    """
    j_info = ctypes.c_char_p()
    _check_call(_LIB.XGBuildInfo(ctypes.byref(j_info)))
    assert j_info.value is not None
    res = json.loads(j_info.value.decode())  # pylint: disable=no-member
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


def _cuda_array_interface(data: DataType) -> bytes:
    assert (
        data.dtype.hasobject is False
    ), "Input data contains `object` dtype.  Expecting numeric data."
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


def ctypes2cupy(cptr: CNumericPtr, length: int, dtype: Type[np.number]) -> CupyT:
    """Convert a ctypes pointer array to a cupy array."""
    # pylint: disable=import-error
    import cupy
    from cupy.cuda.memory import MemoryPointer
    from cupy.cuda.memory import UnownedMemory

    CUPY_TO_CTYPES_MAPPING = {cupy.float32: ctypes.c_float, cupy.uint32: ctypes.c_uint}
    if dtype not in CUPY_TO_CTYPES_MAPPING:
        raise RuntimeError(f"Supported types: {CUPY_TO_CTYPES_MAPPING.keys()}")
    addr = ctypes.cast(cptr, ctypes.c_void_p).value
    # pylint: disable=c-extension-no-member,no-member
    device = cupy.cuda.runtime.pointerGetAttributes(addr).device
    # The owner field is just used to keep the memory alive with ref count.  As
    # unowned's life time is scoped within this function we don't need that.
    unownd = UnownedMemory(
        addr, length * ctypes.sizeof(CUPY_TO_CTYPES_MAPPING[dtype]), owner=None
    )
    memptr = MemoryPointer(unownd, 0)
    # pylint: disable=unexpected-keyword-arg
    mem = cupy.ndarray((length,), dtype=dtype, memptr=memptr)
    assert mem.device.id == device
    arr = cupy.array(mem, copy=True)
    return arr


def ctypes2buffer(cptr: CStrPtr, length: int) -> bytearray:
    """Convert ctypes pointer to buffer type."""
    if not isinstance(cptr, ctypes.POINTER(ctypes.c_char)):
        raise RuntimeError('expected char pointer')
    res = bytearray(length)
    rptr = (ctypes.c_char * length).from_buffer(res)
    if not ctypes.memmove(rptr, cptr, length):
        raise RuntimeError('memmove failed')
    return res


def c_str(string: str) -> ctypes.c_char_p:
    """Convert a python string to cstring."""
    return ctypes.c_char_p(string.encode('utf-8'))


def c_array(ctype: Type[CTypeT], values: ArrayLike) -> ctypes.Array:
    """Convert a python string to c array."""
    if isinstance(values, np.ndarray) and values.dtype.itemsize == ctypes.sizeof(ctype):
        return (ctype * len(values)).from_buffer_copy(values)
    return (ctype * len(values))(*values)


def _prediction_output(
    shape: CNumericPtr,
    dims: c_bst_ulong,
    predts: CFloatPtr,
    is_cuda: bool
) -> NumpyOrCupy:
    arr_shape = ctypes2numpy(shape, dims.value, np.uint64)
    length = int(np.prod(arr_shape))
    if is_cuda:
        arr_predict = ctypes2cupy(predts, length, np.float32)
    else:
        arr_predict = ctypes2numpy(predts, length, np.float32)
    arr_predict = arr_predict.reshape(arr_shape)
    return arr_predict


class DataIter(ABC):  # pylint: disable=too-many-instance-attributes
    """The interface for user defined data iterator.

    Parameters
    ----------
    cache_prefix:
        Prefix to the cache files, only used in external memory.  It can be either an URI
        or a file path.

    """
    _T = TypeVar("_T")

    def __init__(self, cache_prefix: Optional[str] = None) -> None:
        self.cache_prefix = cache_prefix

        self._handle = _ProxyDMatrix()
        self._exception: Optional[Exception] = None
        self._enable_categorical = False
        self._allow_host = True
        # Stage data in Python until reset or next is called to avoid data being free.
        self._temporary_data: Optional[Tuple[Any, Any]] = None

    def get_callbacks(
        self, allow_host: bool, enable_categorical: bool
    ) -> Tuple[Callable, Callable]:
        """Get callback functions for iterating in C."""
        assert hasattr(self, "cache_prefix"), "__init__ is not called."
        self._reset_callback = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(
            self._reset_wrapper
        )
        self._next_callback = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_void_p,
        )(self._next_wrapper)
        self._allow_host = allow_host
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
        self._temporary_data = None
        self._handle_exception(self.reset, None)

    def _next_wrapper(self, this: None) -> int:  # pylint: disable=unused-argument
        """A wrapper for user defined `next` function.

        `this` is not used in Python.  ctypes can handle `self` of a Python
        member function automatically when converting it to c function
        pointer.

        """
        @_deprecate_positional_args
        def data_handle(
            data: Any,
            *,
            feature_names: FeatureNames = None,
            feature_types: Optional[List[str]] = None,
            **kwargs: Any,
        ) -> None:
            from .data import dispatch_proxy_set_data
            from .data import _proxy_transform

            new, cat_codes, feature_names, feature_types = _proxy_transform(
                data,
                feature_names,
                feature_types,
                self._enable_categorical,
            )
            # Stage the data, meta info are copied inside C++ MetaInfo.
            self._temporary_data = (new, cat_codes)
            dispatch_proxy_set_data(self.proxy, new, cat_codes, self._allow_host)
            self.proxy.set_info(
                feature_names=feature_names,
                feature_types=feature_types,
                **kwargs,
            )
        # pylint: disable=not-callable
        return self._handle_exception(lambda: self.next(data_handle), 0)

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


# Notice for `_deprecate_positional_args`
# Authors: Olivier Grisel
#          Gael Varoquaux
#          Andreas Mueller
#          Lars Buitinck
#          Alexandre Gramfort
#          Nicolas Tresegnie
#          Sylvain Marie
# License: BSD 3 clause
def _deprecate_positional_args(f: Callable[..., _T]) -> Callable[..., _T]:
    """Decorator for methods that issues warnings for positional arguments

    Using the keyword-only argument syntax in pep 3102, arguments after the
    * will issue a warning when passed as a positional argument.

    Modified from sklearn utils.validation.

    Parameters
    ----------
    f : function
        function to check arguments on
    """
    sig = signature(f)
    kwonly_args = []
    all_args = []

    for name, param in sig.parameters.items():
        if param.kind == Parameter.POSITIONAL_OR_KEYWORD:
            all_args.append(name)
        elif param.kind == Parameter.KEYWORD_ONLY:
            kwonly_args.append(name)

    @wraps(f)
    def inner_f(*args: Any, **kwargs: Any) -> _T:
        extra_args = len(args) - len(all_args)
        if extra_args > 0:
            # ignore first 'self' argument for instance methods
            args_msg = [
                f"{name}" for name, _ in zip(
                    kwonly_args[:extra_args], args[-extra_args:]
                )
            ]
            # pylint: disable=consider-using-f-string
            warnings.warn(
                "Pass `{}` as keyword args.  Passing these as positional "
                "arguments will be considered as error in future releases.".
                format(", ".join(args_msg)), FutureWarning
            )
        for k, arg in zip(sig.parameters, args):
            kwargs[k] = arg
        return f(**kwargs)

    return inner_f


class DMatrix:  # pylint: disable=too-many-instance-attributes
    """Data Matrix used in XGBoost.

    DMatrix is an internal data structure that is used by XGBoost,
    which is optimized for both memory efficiency and training speed.
    You can construct DMatrix from multiple different sources of data.
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
        feature_names: FeatureNames = None,
        feature_types: Optional[List[str]] = None,
        nthread: Optional[int] = None,
        group: Optional[ArrayLike] = None,
        qid: Optional[ArrayLike] = None,
        label_lower_bound: Optional[ArrayLike] = None,
        label_upper_bound: Optional[ArrayLike] = None,
        feature_weights: Optional[ArrayLike] = None,
        enable_categorical: bool = False,
    ) -> None:
        """Parameters
        ----------
        data : os.PathLike/string/numpy.array/scipy.sparse/pd.DataFrame/
               dt.Frame/cudf.DataFrame/cupy.array/dlpack
            Data source of DMatrix.
            When data is string or os.PathLike type, it represents the path
            libsvm format txt file, csv file (by specifying uri parameter
            'path_to_csv?format=csv'), or binary file that xgboost can read
            from.
        label : array_like
            Label of the training data.
        weight : array_like
            Weight for each instance.

            .. note:: For ranking task, weights are per-group.

                In ranking task, one weight is assigned to each group (not each
                data point). This is because we only care about the relative
                ordering of data points within each group, so it doesn't make
                sense to assign weights to individual data points.

        base_margin: array_like
            Base margin used for boosting from existing model.
        missing : float, optional
            Value in the input data which needs to be present as a missing
            value. If None, defaults to np.nan.
        silent : boolean, optional
            Whether print messages during construction
        feature_names : list, optional
            Set names for features.
        feature_types :

            Set types for features.  When `enable_categorical` is set to `True`, string
            "c" represents categorical data type.

        nthread : integer, optional
            Number of threads to use for loading data when parallelization is
            applicable. If -1, uses maximum threads available on the system.
        group : array_like
            Group size for all ranking group.
        qid : array_like
            Query ID for data samples, used for ranking.
        label_lower_bound : array_like
            Lower bound for survival training.
        label_upper_bound : array_like
            Upper bound for survival training.
        feature_weights : array_like, optional
            Set feature weights for column sampling.
        enable_categorical: boolean, optional

            .. versionadded:: 1.3.0

            .. note:: This parameter is experimental

            Experimental support of specializing for categorical features.  Do not set
            to True unless you are interested in development. Also, JSON/UBJSON
            serialization format is required.

        """
        if group is not None and qid is not None:
            raise ValueError("Either one of `group` or `qid` should be None.")

        self.missing = missing if missing is not None else np.nan
        self.nthread = nthread if nthread is not None else -1
        self.silent = silent

        # force into void_p, mac need to pass things in as void_p
        if data is None:
            self.handle: Optional[ctypes.c_void_p] = None
            return

        from .data import dispatch_data_backend, _is_iter

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
        reset_callback, next_callback = it.get_callbacks(
            True, enable_categorical
        )
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
        if hasattr(self, "handle") and self.handle:
            _check_call(_LIB.XGDMatrixFree(self.handle))
            self.handle = None

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
        feature_names: FeatureNames = None,
        feature_types: Optional[List[str]] = None,
        feature_weights: Optional[ArrayLike] = None
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
            self.set_uint_info('qid', qid)
        if label_lower_bound is not None:
            self.set_float_info('label_lower_bound', label_lower_bound)
        if label_upper_bound is not None:
            self.set_float_info('label_upper_bound', label_upper_bound)
        if feature_names is not None:
            self.feature_names = feature_names
        if feature_types is not None:
            self.feature_types = feature_types
        if feature_weights is not None:
            dispatch_meta_backend(matrix=self, data=feature_weights,
                                  name='feature_weights')

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
        _check_call(_LIB.XGDMatrixGetFloatInfo(self.handle,
                                               c_str(field),
                                               ctypes.byref(length),
                                               ctypes.byref(ret)))
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
        _check_call(_LIB.XGDMatrixGetUIntInfo(self.handle,
                                              c_str(field),
                                              ctypes.byref(length),
                                              ctypes.byref(ret)))
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
        dispatch_meta_backend(self, data, field, 'float')

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
        dispatch_meta_backend(self, data, field, 'float')

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
        dispatch_meta_backend(self, data, field, 'uint32')

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
        _check_call(_LIB.XGDMatrixSaveBinary(self.handle,
                                             c_str(fname),
                                             ctypes.c_int(silent)))

    def set_label(self, label: ArrayLike) -> None:
        """Set label of dmatrix

        Parameters
        ----------
        label: array like
            The label information to be set into DMatrix
        """
        from .data import dispatch_meta_backend
        dispatch_meta_backend(self, label, 'label', 'float')

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
        dispatch_meta_backend(self, weight, 'weight', 'float')

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
        dispatch_meta_backend(self, margin, 'base_margin', 'float')

    def set_group(self, group: ArrayLike) -> None:
        """Set group size of DMatrix (used for ranking).

        Parameters
        ----------
        group : array like
            Group size of each group
        """
        from .data import dispatch_meta_backend
        dispatch_meta_backend(self, group, 'group', 'uint32')

    def get_label(self) -> np.ndarray:
        """Get the label of the DMatrix.

        Returns
        -------
        label : array
        """
        return self.get_float_info('label')

    def get_weight(self) -> np.ndarray:
        """Get the weight of the DMatrix.

        Returns
        -------
        weight : array
        """
        return self.get_float_info('weight')

    def get_base_margin(self) -> np.ndarray:
        """Get the base margin of the DMatrix.

        Returns
        -------
        base_margin
        """
        return self.get_float_info('base_margin')

    def get_group(self) -> np.ndarray:
        """Get the group of the DMatrix.

        Returns
        -------
        group
        """
        group_ptr = self.get_uint_info("group_ptr")
        return np.diff(group_ptr)

    def num_row(self) -> int:
        """Get the number of rows in the DMatrix.

        Returns
        -------
        number of rows : int
        """
        ret = c_bst_ulong()
        _check_call(_LIB.XGDMatrixNumRow(self.handle,
                                         ctypes.byref(ret)))
        return ret.value

    def num_col(self) -> int:
        """Get the number of columns (features) in the DMatrix.

        Returns
        -------
        number of columns : int
        """
        ret = c_bst_ulong()
        _check_call(_LIB.XGDMatrixNumCol(self.handle, ctypes.byref(ret)))
        return ret.value

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

        res = DMatrix(None)
        res.handle = ctypes.c_void_p()
        rindex = _maybe_np_slice(rindex, dtype=np.int32)
        _check_call(
            _LIB.XGDMatrixSliceDMatrixEx(
                self.handle,
                c_array(ctypes.c_int, rindex),
                c_bst_ulong(len(rindex)),
                ctypes.byref(res.handle),
                ctypes.c_int(1 if allow_groups else 0),
            )
        )
        return res

    @property
    def feature_names(self) -> Optional[List[str]]:
        """Get feature names (column labels).

        Returns
        -------
        feature_names : list or None
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
    def feature_names(self, feature_names: FeatureNames) -> None:
        """Set feature names (column labels).

        Parameters
        ----------
        feature_names : list or None
            Labels for features. None will reset existing feature names
        """
        if feature_names is not None:
            # validate feature name
            try:
                if not isinstance(feature_names, str):
                    feature_names = list(feature_names)
                else:
                    feature_names = [feature_names]
            except TypeError:
                feature_names = [feature_names]

            if len(feature_names) != len(set(feature_names)):
                raise ValueError('feature_names must be unique')
            if len(feature_names) != self.num_col() and self.num_col() != 0:
                msg = ("feature_names must have the same length as data, ",
                       f"expected {self.num_col()}, got {len(feature_names)}")
                raise ValueError(msg)
            # prohibit to use symbols may affect to parse. e.g. []<
            if not all(isinstance(f, str) and
                       not any(x in f for x in set(('[', ']', '<')))
                       for f in feature_names):
                raise ValueError('feature_names must be string, and may not contain [, ] or <')
            feature_names_bytes = [bytes(f, encoding='utf-8') for f in feature_names]
            c_feature_names = (ctypes.c_char_p *
                               len(feature_names_bytes))(*feature_names_bytes)
            _check_call(_LIB.XGDMatrixSetStrFeatureInfo(
                self.handle, c_str('feature_name'),
                c_feature_names,
                c_bst_ulong(len(feature_names))))
        else:
            # reset feature_types also
            _check_call(_LIB.XGDMatrixSetStrFeatureInfo(
                self.handle,
                c_str('feature_name'),
                None,
                c_bst_ulong(0)))
            self.feature_types = None

    @property
    def feature_types(self) -> Optional[List[str]]:
        """Get feature types (column types).

        Returns
        -------
        feature_types : list or None
        """
        length = c_bst_ulong()
        sarr = ctypes.POINTER(ctypes.c_char_p)()
        _check_call(_LIB.XGDMatrixGetStrFeatureInfo(self.handle,
                                                    c_str('feature_type'),
                                                    ctypes.byref(length),
                                                    ctypes.byref(sarr)))
        res = from_cstr_to_pystr(sarr, length)
        if not res:
            return None
        return res

    @feature_types.setter
    def feature_types(self, feature_types: Optional[Union[List[str], str]]) -> None:
        """Set feature types (column types).

        This is for displaying the results and categorical data support.  See doc string
        of :py:obj:`xgboost.DMatrix` for details.

        Parameters
        ----------
        feature_types : list or None
            Labels for features. None will reset existing feature names

        """
        # For compatibility reason this function wraps single str input into a list.  But
        # we should not promote such usage since other than visualization, the field is
        # also used for specifying categorical data type.
        if feature_types is not None:
            if not isinstance(feature_types, (list, str)):
                raise TypeError(
                    'feature_types must be string or list of strings')
            if isinstance(feature_types, str):
                # single string will be applied to all columns
                feature_types = [feature_types] * self.num_col()
            try:
                if not isinstance(feature_types, str):
                    feature_types = list(feature_types)
                else:
                    feature_types = [feature_types]
            except TypeError:
                feature_types = [feature_types]
            feature_types_bytes = [bytes(f, encoding='utf-8')
                               for f in feature_types]
            c_feature_types = (ctypes.c_char_p *
                               len(feature_types_bytes))(*feature_types_bytes)
            _check_call(_LIB.XGDMatrixSetStrFeatureInfo(
                self.handle, c_str('feature_type'),
                c_feature_types,
                c_bst_ulong(len(feature_types))))

            if len(feature_types) != self.num_col() and self.num_col() != 0:
                msg = 'feature_types must have the same length as data'
                raise ValueError(msg)
        else:
            # Reset.
            _check_call(_LIB.XGDMatrixSetStrFeatureInfo(
                self.handle,
                c_str('feature_type'),
                None,
                c_bst_ulong(0)))


class _ProxyDMatrix(DMatrix):
    """A placeholder class when DMatrix cannot be constructed (DeviceQuantileDMatrix,
    inplace_predict).

    """

    def __init__(self) -> None:  # pylint: disable=super-init-not-called
        self.handle = ctypes.c_void_p()
        _check_call(_LIB.XGProxyDMatrixCreate(ctypes.byref(self.handle)))

    def _set_data_from_cuda_interface(self, data: DataType) -> None:
        """Set data from CUDA array interface."""
        interface = data.__cuda_array_interface__
        interface_str = bytes(json.dumps(interface, indent=2), "utf-8")
        _check_call(
            _LIB.XGProxyDMatrixSetDataCudaArrayInterface(self.handle, interface_str)
        )

    def _set_data_from_cuda_columnar(self, data: DataType, cat_codes: list) -> None:
        """Set data from CUDA columnar format."""
        from .data import _cudf_array_interfaces

        interfaces_str = _cudf_array_interfaces(data, cat_codes)
        _check_call(_LIB.XGProxyDMatrixSetDataCudaColumnar(self.handle, interfaces_str))

    def _set_data_from_array(self, data: np.ndarray) -> None:
        """Set data from numpy array."""
        from .data import _array_interface

        _check_call(
            _LIB.XGProxyDMatrixSetDataDense(self.handle, _array_interface(data))
        )

    def _set_data_from_csr(self, csr: scipy.sparse.csr_matrix) -> None:
        """Set data from scipy csr"""
        from .data import _array_interface

        _LIB.XGProxyDMatrixSetDataCSR(
            self.handle,
            _array_interface(csr.indptr),
            _array_interface(csr.indices),
            _array_interface(csr.data),
            ctypes.c_size_t(csr.shape[1]),
        )


class DeviceQuantileDMatrix(DMatrix):
    """Device memory Data Matrix used in XGBoost for training with tree_method='gpu_hist'. Do
    not use this for test/validation tasks as some information may be lost in
    quantisation. This DMatrix is primarily designed to save memory in training from
    device memory inputs by avoiding intermediate storage. Set max_bin to control the
    number of bins during quantisation.  See doc string in :py:obj:`xgboost.DMatrix` for
    documents on meta info.

    You can construct DeviceQuantileDMatrix from cupy/cudf/dlpack.

    .. versionadded:: 1.1.0

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
        feature_names: FeatureNames = None,
        feature_types: Optional[List[str]] = None,
        nthread: Optional[int] = None,
        max_bin: int = 256,
        group: Optional[ArrayLike] = None,
        qid: Optional[ArrayLike] = None,
        label_lower_bound: Optional[ArrayLike] = None,
        label_upper_bound: Optional[ArrayLike] = None,
        feature_weights: Optional[ArrayLike] = None,
        enable_categorical: bool = False,
    ) -> None:
        self.max_bin = max_bin
        self.missing = missing if missing is not None else np.nan
        self.nthread = nthread if nthread is not None else 1
        self._silent = silent  # unused, kept for compatibility

        if isinstance(data, ctypes.c_void_p):
            self.handle = data
            return

        if qid is not None and group is not None:
            raise ValueError(
                'Only one of the eval_qid or eval_group for each evaluation '
                'dataset should be provided.'
            )

        self._init(
            data,
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

    def _init(self, data: DataType, enable_categorical: bool, **meta: Any) -> None:
        from .data import (
            _is_dlpack,
            _transform_dlpack,
            _is_iter,
            SingleBatchInternalIter,
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
        reset_callback, next_callback = it.get_callbacks(False, enable_categorical)
        if it.cache_prefix is not None:
            raise ValueError(
                "DeviceQuantileDMatrix doesn't cache data, remove the cache_prefix "
                "in iterator to fix this error."
            )
        ret = _LIB.XGDeviceQuantileDMatrixCreateFromCallback(
            None,
            it.proxy.handle,
            reset_callback,
            next_callback,
            ctypes.c_float(self.missing),
            ctypes.c_int(self.nthread),
            ctypes.c_int(self.max_bin),
            ctypes.byref(handle),
        )
        it.reraise()
        # delay check_call to throw intermediate exception first
        _check_call(ret)
        self.handle = handle


Objective = Callable[[np.ndarray, DMatrix], Tuple[np.ndarray, np.ndarray]]
Metric = Callable[[np.ndarray, DMatrix], Tuple[str, float]]


def _get_booster_layer_trees(model: "Booster") -> Tuple[int, int]:
    """Get number of trees added to booster per-iteration.  This function will be removed
    once `best_ntree_limit` is dropped in favor of `best_iteration`.  Returns
    `num_parallel_tree` and `num_groups`.

    """
    config = json.loads(model.save_config())
    booster = config["learner"]["gradient_booster"]["name"]
    if booster == "gblinear":
        num_parallel_tree = 0
    elif booster == "dart":
        num_parallel_tree = int(
            config["learner"]["gradient_booster"]["gbtree"]["gbtree_model_param"][
                "num_parallel_tree"
            ]
        )
    elif booster == "gbtree":
        try:
            num_parallel_tree = int(
                config["learner"]["gradient_booster"]["gbtree_model_param"][
                    "num_parallel_tree"
                ]
            )
        except KeyError:
            num_parallel_tree = int(
                config["learner"]["gradient_booster"]["gbtree_train_param"][
                    "num_parallel_tree"
                ]
            )
    else:
        raise ValueError(f"Unknown booster: {booster}")
    num_groups = int(config["learner"]["learner_model_param"]["num_class"])
    return num_parallel_tree, num_groups


def _configure_metrics(params: Union[Dict, List]) -> Union[Dict, List]:
    if (
        isinstance(params, dict)
        and "eval_metric" in params
        and isinstance(params["eval_metric"], list)
    ):
        params = dict((k, v) for k, v in params.items())
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
        params: Optional[Dict] = None,
        cache: Optional[Sequence[DMatrix]] = None,
        model_file: Optional[Union["Booster", bytearray, os.PathLike, str]] = None
    ) -> None:
        # pylint: disable=invalid-name
        """
        Parameters
        ----------
        params : dict
            Parameters for boosters.
        cache : list
            List of cache items.
        model_file : string/os.PathLike/Booster/bytearray
            Path to the model file if it's string or PathLike.
        """
        cache = cache if cache is not None else []
        for d in cache:
            if not isinstance(d, DMatrix):
                raise TypeError(f'invalid cache item: {type(d).__name__}', cache)

        dmats = c_array(ctypes.c_void_p, [d.handle for d in cache])
        self.handle: Optional[ctypes.c_void_p] = ctypes.c_void_p()
        _check_call(_LIB.XGBoosterCreate(dmats, c_bst_ulong(len(cache)),
                                         ctypes.byref(self.handle)))
        for d in cache:
            # Validate feature only after the feature names are saved into booster.
            self._validate_features(d)

        if isinstance(model_file, Booster):
            assert self.handle is not None
            # We use the pickle interface for getting memory snapshot from
            # another model, and load the snapshot with this booster.
            state = model_file.__getstate__()
            handle = state['handle']
            del state['handle']
            ptr = (ctypes.c_char * len(handle)).from_buffer(handle)
            length = c_bst_ulong(len(handle))
            _check_call(
                _LIB.XGBoosterUnserializeFromBuffer(self.handle, ptr, length))
            self.__dict__.update(state)
        elif isinstance(model_file, (STRING_TYPES, os.PathLike, bytearray)):
            self.load_model(model_file)
        elif model_file is None:
            pass
        else:
            raise TypeError('Unknown type:', model_file)

        params = params or {}
        params_processed = _configure_metrics(params.copy())
        params_processed = self._configure_constraints(params_processed)
        if isinstance(params_processed, list):
            params_processed.append(("validate_parameters", True))
        else:
            params_processed["validate_parameters"] = True

        self.set_param(params_processed or {})

    def _transform_monotone_constrains(
        self, value: Union[Dict[str, int], str]
    ) -> Union[Tuple[int, ...], str]:
        if isinstance(value, str):
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

    def _configure_constraints(self, params: Union[List, Dict]) -> Union[List, Dict]:
        if isinstance(params, dict):
            value = params.get("monotone_constraints")
            if value is not None:
                params["monotone_constraints"] = self._transform_monotone_constrains(
                    value
                )

            value = params.get("interaction_constraints")
            if value is not None:
                params[
                    "interaction_constraints"
                ] = self._transform_interaction_constraints(value)
        elif isinstance(params, list):
            for idx, param in enumerate(params):
                name, value = param
                if not value:
                    continue

                if name == "monotone_constraints":
                    params[idx] = (name, self._transform_monotone_constrains(value))
                elif name == "interaction_constraints":
                    params[idx] = (name, self._transform_interaction_constraints(value))

        return params

    def __del__(self) -> None:
        if hasattr(self, 'handle') and self.handle is not None:
            _check_call(_LIB.XGBoosterFree(self.handle))
            self.handle = None

    def __getstate__(self) -> Dict:
        # can't pickle ctypes pointers, put model content in bytearray
        this = self.__dict__.copy()
        handle = this['handle']
        if handle is not None:
            length = c_bst_ulong()
            cptr = ctypes.POINTER(ctypes.c_char)()
            _check_call(_LIB.XGBoosterSerializeToBuffer(self.handle,
                                                        ctypes.byref(length),
                                                        ctypes.byref(cptr)))
            buf = ctypes2buffer(cptr, length.value)
            this["handle"] = buf
        return this

    def __setstate__(self, state: Dict) -> None:
        # reconstruct handle from raw data
        handle = state['handle']
        if handle is not None:
            buf = handle
            dmats = c_array(ctypes.c_void_p, [])
            handle = ctypes.c_void_p()
            _check_call(_LIB.XGBoosterCreate(
                dmats, c_bst_ulong(0), ctypes.byref(handle)))
            length = c_bst_ulong(len(buf))
            ptr = (ctypes.c_char * len(buf)).from_buffer(buf)
            _check_call(
                _LIB.XGBoosterUnserializeFromBuffer(handle, ptr, length))
            state['handle'] = handle
        self.__dict__.update(state)

    def __getitem__(self, val: Union[int, tuple, slice]) -> "Booster":
        if isinstance(val, int):
            val = slice(val, val+1)
        if isinstance(val, tuple):
            raise ValueError('Only supports slicing through 1 dimension.')
        if not isinstance(val, slice):
            msg = _expect((int, slice), type(val))
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
                raise ValueError('Invalid slice', val)

        step = val.step if val.step is not None else 1

        c_start = ctypes.c_int(start)
        c_stop = ctypes.c_int(stop)
        c_step = ctypes.c_int(step)

        sliced_handle = ctypes.c_void_p()
        status = _LIB.XGBoosterSlice(
            self.handle, c_start, c_stop, c_step, ctypes.byref(sliced_handle)
        )
        if status == -2:
            raise IndexError('Layer index out of range')
        _check_call(status)

        sliced = Booster()
        _check_call(_LIB.XGBoosterFree(sliced.handle))
        sliced.handle = sliced_handle
        return sliced

    def save_config(self) -> str:
        '''Output internal parameter configuration of Booster as a JSON
        string.

        .. versionadded:: 1.0.0
        '''
        json_string = ctypes.c_char_p()
        length = c_bst_ulong()
        _check_call(_LIB.XGBoosterSaveJsonConfig(
            self.handle,
            ctypes.byref(length),
            ctypes.byref(json_string)))
        assert json_string.value is not None
        result = json_string.value.decode()  # pylint: disable=no-member
        return result

    def load_config(self, config: str) -> None:
        '''Load configuration returned by `save_config`.

        .. versionadded:: 1.0.0
        '''
        assert isinstance(config, str)
        _check_call(_LIB.XGBoosterLoadJsonConfig(
            self.handle,
            c_str(config)))

    def __copy__(self) -> "Booster":
        return self.__deepcopy__(None)

    def __deepcopy__(self, _: Any) -> "Booster":
        '''Return a copy of booster.'''
        return Booster(model_file=self)

    def copy(self) -> "Booster":
        """Copy the booster object.

        Returns
        -------
        booster: `Booster`
            a copied booster model
        """
        return self.__copy__()

    def attr(self, key: str) -> Optional[str]:
        """Get attribute string from the Booster.

        Parameters
        ----------
        key : str
            The key to get attribute from.

        Returns
        -------
        value : str
            The attribute value of the key, returns None if attribute do not exist.
        """
        ret = ctypes.c_char_p()
        success = ctypes.c_int()
        _check_call(_LIB.XGBoosterGetAttr(
            self.handle, c_str(key), ctypes.byref(ret), ctypes.byref(success)))
        if success.value != 0:
            return py_str(ret.value)
        return None

    def attributes(self) -> Dict[str, str]:
        """Get attributes stored in the Booster as a dictionary.

        Returns
        -------
        result : dictionary of  attribute_name: attribute_value pairs of strings.
            Returns an empty dict if there's no attributes.
        """
        length = c_bst_ulong()
        sarr = ctypes.POINTER(ctypes.c_char_p)()
        _check_call(_LIB.XGBoosterGetAttrNames(self.handle,
                                               ctypes.byref(length),
                                               ctypes.byref(sarr)))
        attr_names = from_cstr_to_pystr(sarr, length)
        return {n: self.attr(n) for n in attr_names}

    def set_attr(self, **kwargs: Optional[str]) -> None:
        """Set the attribute of the Booster.

        Parameters
        ----------
        **kwargs
            The attributes to set. Setting a value to None deletes an attribute.
        """
        for key, value in kwargs.items():
            if value is not None:
                if not isinstance(value, STRING_TYPES):
                    raise ValueError("Set Attr only accepts string values")
                value = c_str(str(value))
            _check_call(_LIB.XGBoosterSetAttr(
                self.handle, c_str(key), value))

    def _get_feature_info(self, field: str) -> Optional[List[str]]:
        length = c_bst_ulong()
        sarr = ctypes.POINTER(ctypes.c_char_p)()
        if not hasattr(self, "handle") or self.handle is None:
            return None
        _check_call(
            _LIB.XGBoosterGetStrFeatureInfo(
                self.handle, c_str(field), ctypes.byref(length), ctypes.byref(sarr),
            )
        )
        feature_info = from_cstr_to_pystr(sarr, length)
        return feature_info if feature_info else None

    def _set_feature_info(self, features: Optional[List[str]], field: str) -> None:
        if features is not None:
            assert isinstance(features, list)
            feature_info_bytes = [bytes(f, encoding="utf-8") for f in features]
            c_feature_info = (ctypes.c_char_p * len(feature_info_bytes))(*feature_info_bytes)
            _check_call(
                _LIB.XGBoosterSetStrFeatureInfo(
                    self.handle, c_str(field), c_feature_info, c_bst_ulong(len(features))
                )
            )
        else:
            _check_call(
                _LIB.XGBoosterSetStrFeatureInfo(
                    self.handle, c_str(field), None, c_bst_ulong(0)
                )
            )

    @property
    def feature_types(self) -> Optional[List[str]]:
        """Feature types for this booster.  Can be directly set by input data or by
        assignment.

        """
        return self._get_feature_info("feature_type")

    @feature_types.setter
    def feature_types(self, features: Optional[List[str]]) -> None:
        self._set_feature_info(features, "feature_type")

    @property
    def feature_names(self) -> Optional[List[str]]:
        """Feature names for this booster.  Can be directly set by input data or by
        assignment.

        """
        return self._get_feature_info("feature_name")

    @feature_names.setter
    def feature_names(self, features: FeatureNames) -> None:
        self._set_feature_info(features, "feature_name")

    def set_param(
        self,
        params: Union[Dict, Iterable[Tuple[str, Any]], str],
        value: Optional[str] = None
    ) -> None:
        """Set parameters into the Booster.

        Parameters
        ----------
        params: dict/list/str
           list of key,value pairs, dict of key to value or simply str key
        value: optional
           value of the specified parameter, when params is str key
        """
        if isinstance(params, Mapping):
            params = params.items()
        elif isinstance(params, STRING_TYPES) and value is not None:
            params = [(params, value)]
        for key, val in params:
            if val is not None:
                _check_call(_LIB.XGBoosterSetParam(self.handle, c_str(key),
                                                   c_str(str(val))))

    def update(
        self, dtrain: DMatrix, iteration: int, fobj: Optional[Objective] = None
    ) -> None:
        """Update for one iteration, with objective function calculated
        internally.  This function should not be called directly by users.

        Parameters
        ----------
        dtrain : DMatrix
            Training data.
        iteration : int
            Current iteration number.
        fobj : function
            Customized objective function.

        """
        if not isinstance(dtrain, DMatrix):
            raise TypeError(f"invalid training matrix: {type(dtrain).__name__}")
        self._validate_features(dtrain)

        if fobj is None:
            _check_call(_LIB.XGBoosterUpdateOneIter(self.handle,
                                                    ctypes.c_int(iteration),
                                                    dtrain.handle))
        else:
            pred = self.predict(dtrain, output_margin=True, training=True)
            grad, hess = fobj(pred, dtrain)
            self.boost(dtrain, grad, hess)

    def boost(self, dtrain: DMatrix, grad: np.ndarray, hess: np.ndarray) -> None:
        """Boost the booster for one iteration, with customized gradient
        statistics.  Like :py:func:`xgboost.Booster.update`, this
        function should not be called directly by users.

        Parameters
        ----------
        dtrain :
            The training DMatrix.
        grad :
            The first order of gradient.
        hess :
            The second order of gradient.

        """
        if len(grad) != len(hess):
            raise ValueError(
                f"grad / hess length mismatch: {len(grad)} / {len(hess)}"
            )
        if not isinstance(dtrain, DMatrix):
            raise TypeError(f"invalid training matrix: {type(dtrain).__name__}")
        self._validate_features(dtrain)

        _check_call(_LIB.XGBoosterBoostOneIter(self.handle, dtrain.handle,
                                               c_array(ctypes.c_float, grad),
                                               c_array(ctypes.c_float, hess),
                                               c_bst_ulong(len(grad))))

    def eval_set(
        self,
        evals: Sequence[Tuple[DMatrix, str]],
        iteration: int = 0,
        feval: Optional[Metric] = None,
        output_margin: bool = True
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
            if not isinstance(d[1], STRING_TYPES):
                raise TypeError(f"expected string, got {type(d[1]).__name__}")
            self._validate_features(d[0])

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
                    self.predict(dmat, training=False, output_margin=output_margin), dmat
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

    def eval(self, data: DMatrix, name: str = 'eval', iteration: int = 0) -> str:
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
        self._validate_features(data)
        return self.eval_set([(data, name)], iteration)

    # pylint: disable=too-many-function-args
    def predict(
        self,
        data: DMatrix,
        output_margin: bool = False,
        ntree_limit: int = 0,
        pred_leaf: bool = False,
        pred_contribs: bool = False,
        approx_contribs: bool = False,
        pred_interactions: bool = False,
        validate_features: bool = True,
        training: bool = False,
        iteration_range: Tuple[int, int] = (0, 0),
        strict_shape: bool = False,
    ) -> np.ndarray:
        """Predict with data.  The full model will be used unless `iteration_range` is specified,
        meaning user have to either slice the model or use the ``best_iteration``
        attribute to get prediction from best model returned from early stopping.

        .. note::

            See :doc:`Prediction </prediction>` for issues like thread safety and a
            summary of outputs from this function.

        Parameters
        ----------
        data :
            The dmatrix storing the input.

        output_margin :
            Whether to output the raw untransformed margin value.

        ntree_limit :
            Deprecated, use `iteration_range` instead.

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
            raise TypeError('Expecting data to be a DMatrix object, got: ', type(data))
        if validate_features:
            self._validate_features(data)
        iteration_range = _convert_ntree_limit(self, ntree_limit, iteration_range)
        args = {
            "type": 0,
            "training": training,
            "iteration_begin": iteration_range[0],
            "iteration_end": iteration_range[1],
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
                ctypes.byref(preds)
            )
        )
        return _prediction_output(shape, dims, preds, False)

    def inplace_predict(
        self,
        data: DataType,
        iteration_range: Tuple[int, int] = (0, 0),
        predict_type: str = "value",
        missing: float = np.nan,
        validate_features: bool = True,
        base_margin: Any = None,
        strict_shape: bool = False
    ) -> NumpyOrCupy:
        """Run prediction in-place, Unlike :py:meth:`predict` method, inplace prediction does not
        cache the prediction result.

        Calling only ``inplace_predict`` in multiple threads is safe and lock
        free.  But the safety does not hold when used in conjunction with other
        methods. E.g. you can't train the booster in one thread and perform
        prediction in the other.

        .. code-block:: python

            booster.set_param({'predictor': 'gpu_predictor'})
            booster.inplace_predict(cupy_array)

            booster.set_param({'predictor': 'cpu_predictor})
            booster.inplace_predict(numpy_array)

        .. versionadded:: 1.1.0

        Parameters
        ----------
        data : numpy.ndarray/scipy.sparse.csr_matrix/cupy.ndarray/
               cudf.DataFrame/pd.DataFrame
            The input data, must not be a view for numpy array.  Set
            ``predictor`` to ``gpu_predictor`` for running prediction on CuPy
            array or CuDF DataFrame.
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
            The prediction result.  When input data is on GPU, prediction
            result is stored in a cupy array.

        """
        preds = ctypes.POINTER(ctypes.c_float)()

        # once caching is supported, we can pass id(data) as cache id.
        args = {
            "type": 0,
            "training": False,
            "iteration_begin": iteration_range[0],
            "iteration_end": iteration_range[1],
            "missing": missing,
            "strict_shape": strict_shape,
            "cache_id": 0,
        }
        if predict_type == "margin":
            args["type"] = 1
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

        from .data import (
            _is_pandas_df,
            _transform_pandas_df,
            _is_cudf_df,
            _is_cupy_array,
            _array_interface,
        )
        enable_categorical = _has_categorical(self, data)
        if _is_pandas_df(data):
            data, _, _ = _transform_pandas_df(data, enable_categorical)

        if isinstance(data, np.ndarray):
            from .data import _ensure_np_dtype
            data, _ = _ensure_np_dtype(data, data.dtype)
            _check_call(
                _LIB.XGBoosterPredictFromDense(
                    self.handle,
                    _array_interface(data),
                    from_pystr_to_cstr(json.dumps(args)),
                    p_handle,
                    ctypes.byref(shape),
                    ctypes.byref(dims),
                    ctypes.byref(preds),
                )
            )
            return _prediction_output(shape, dims, preds, False)
        if isinstance(data, scipy.sparse.csr_matrix):
            csr = data
            _check_call(
                _LIB.XGBoosterPredictFromCSR(
                    self.handle,
                    _array_interface(csr.indptr),
                    _array_interface(csr.indices),
                    _array_interface(csr.data),
                    ctypes.c_size_t(csr.shape[1]),
                    from_pystr_to_cstr(json.dumps(args)),
                    p_handle,
                    ctypes.byref(shape),
                    ctypes.byref(dims),
                    ctypes.byref(preds),
                )
            )
            return _prediction_output(shape, dims, preds, False)
        if _is_cupy_array(data):
            from .data import _transform_cupy_array

            data = _transform_cupy_array(data)
            interface_str = _cuda_array_interface(data)
            _check_call(
                _LIB.XGBoosterPredictFromCudaArray(
                    self.handle,
                    interface_str,
                    from_pystr_to_cstr(json.dumps(args)),
                    p_handle,
                    ctypes.byref(shape),
                    ctypes.byref(dims),
                    ctypes.byref(preds),
                )
            )
            return _prediction_output(shape, dims, preds, True)
        if _is_cudf_df(data):
            from .data import _cudf_array_interfaces, _transform_cudf_df
            data, cat_codes, _, _ = _transform_cudf_df(
                data, None, None, enable_categorical
            )
            interfaces_str = _cudf_array_interfaces(data, cat_codes)
            _check_call(
                _LIB.XGBoosterPredictFromCudaColumnar(
                    self.handle,
                    interfaces_str,
                    from_pystr_to_cstr(json.dumps(args)),
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
        (such as feature_names) will not be saved when using binary format.  To save
        those attributes, use JSON/UBJ instead. See :doc:`Model IO
        </tutorials/saving_model>` for more info.

        .. code-block:: python

          model.save_model("model.json")
          # or
          model.save_model("model.ubj")

        Parameters
        ----------
        fname : string or os.PathLike
            Output file name

        """
        if isinstance(fname, (STRING_TYPES, os.PathLike)):  # assume file name
            fname = os.fspath(os.path.expanduser(fname))
            _check_call(_LIB.XGBoosterSaveModel(
                self.handle, c_str(fname)))
        else:
            raise TypeError("fname must be a string or os PathLike")

    def save_raw(self, raw_format: str = "deprecated") -> bytearray:
        """Save the model to a in memory buffer representation instead of file.

        Parameters
        ----------
        raw_format :
            Format of output buffer. Can be `json`, `ubj` or `deprecated`.  Right now
            the default is `deprecated` but it will be changed to `ubj` (univeral binary
            json) in the future.

        Returns
        -------
        An in memory buffer representation of the model
        """
        length = c_bst_ulong()
        cptr = ctypes.POINTER(ctypes.c_char)()
        config = from_pystr_to_cstr(json.dumps({"format": raw_format}))
        _check_call(
            _LIB.XGBoosterSaveModelToBuffer(
                self.handle, config, ctypes.byref(length), ctypes.byref(cptr)
            )
        )
        return ctypes2buffer(cptr, length.value)

    def load_model(self, fname: Union[str, bytearray, os.PathLike]) -> None:
        """Load the model from a file or bytearray. Path to file can be local
        or as an URI.

        The model is loaded from XGBoost format which is universal among the various
        XGBoost interfaces. Auxiliary attributes of the Python Booster object (such as
        feature_names) will not be loaded when using binary format.  To save those
        attributes, use JSON/UBJ instead.  See :doc:`Model IO </tutorials/saving_model>`
        for more info.

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
            _check_call(_LIB.XGBoosterLoadModel(
                self.handle, c_str(fname)))
        elif isinstance(fname, bytearray):
            buf = fname
            length = c_bst_ulong(len(buf))
            ptr = (ctypes.c_char * len(buf)).from_buffer(buf)
            _check_call(_LIB.XGBoosterLoadModelFromBuffer(self.handle, ptr,
                                                          length))
        else:
            raise TypeError('Unknown file type: ', fname)

        if self.attr("best_iteration") is not None:
            self.best_iteration = int(self.attr("best_iteration"))  # type: ignore
        if self.attr("best_score") is not None:
            self.best_score = float(self.attr("best_score"))  # type: ignore
        if self.attr("best_ntree_limit") is not None:
            self.best_ntree_limit = int(self.attr("best_ntree_limit"))  # type: ignore

    def num_boosted_rounds(self) -> int:
        '''Get number of boosted rounds.  For gblinear this is reset to 0 after
        serializing the model.

        '''
        rounds = ctypes.c_int()
        assert self.handle is not None
        _check_call(_LIB.XGBoosterBoostedRounds(self.handle, ctypes.byref(rounds)))
        return rounds.value

    def num_features(self) -> int:
        '''Number of features in booster.'''
        features = c_bst_ulong()
        assert self.handle is not None
        _check_call(_LIB.XGBoosterGetNumFeature(self.handle, ctypes.byref(features)))
        return features.value

    def dump_model(self, fout: Union[str, os.PathLike], fmap: Union[str, os.PathLike] = '',
                   with_stats: bool = False, dump_format: str = "text") -> None:
        """Dump model into a text or JSON file.  Unlike :py:meth:`save_model`, the
        output format is primarily used for visualization or interpretation,
        hence it's more human readable but cannot be loaded back to XGBoost.

        Parameters
        ----------
        fout : string or os.PathLike
            Output file name.
        fmap : string or os.PathLike, optional
            Name of the file containing feature map names.
        with_stats : bool, optional
            Controls whether the split statistics are output.
        dump_format : string, optional
            Format of model dump file. Can be 'text' or 'json'.
        """
        if isinstance(fout, (STRING_TYPES, os.PathLike)):
            fout = os.fspath(os.path.expanduser(fout))
            # pylint: disable=consider-using-with
            fout_obj = open(fout, 'w', encoding="utf-8")
            need_close = True
        else:
            fout_obj = fout
            need_close = False
        ret = self.get_dump(fmap, with_stats, dump_format)
        if dump_format == 'json':
            fout_obj.write('[\n')
            for i, _ in enumerate(ret):
                fout_obj.write(ret[i])
                if i < len(ret) - 1:
                    fout_obj.write(",\n")
            fout_obj.write('\n]')
        else:
            for i, _ in enumerate(ret):
                fout_obj.write(f"booster[{i}]:\n")
                fout_obj.write(ret[i])
        if need_close:
            fout_obj.close()

    def get_dump(
        self,
        fmap: Union[str, os.PathLike] = "",
        with_stats: bool = False,
        dump_format: str = "text"
    ) -> List[str]:
        """Returns the model dump as a list of strings.  Unlike :py:meth:`save_model`, the output
        format is primarily used for visualization or interpretation, hence it's more
        human readable but cannot be loaded back to XGBoost.

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
        _check_call(_LIB.XGBoosterDumpModelEx(self.handle,
                                              c_str(fmap),
                                              ctypes.c_int(with_stats),
                                              c_str(dump_format),
                                              ctypes.byref(length),
                                              ctypes.byref(sarr)))
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

        return self.get_score(fmap, importance_type='weight')

    def get_score(
        self, fmap: Union[str, os.PathLike] = '', importance_type: str = 'weight'
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
        fmap:
           The name of feature map file.
        importance_type:
            One of the importance types defined above.

        Returns
        -------
        A map between feature names and their scores.  When `gblinear` is used for
        multi-class classification the scores for each feature is a list with length
        `n_classes`, otherwise they're scalars.
        """
        fmap = os.fspath(os.path.expanduser(fmap))
        args = from_pystr_to_cstr(
            json.dumps({"importance_type": importance_type, "feature_map": fmap})
        )
        features = ctypes.POINTER(ctypes.c_char_p)()
        scores = ctypes.POINTER(ctypes.c_float)()
        n_out_features = c_bst_ulong()
        out_dim = c_bst_ulong()
        shape = ctypes.POINTER(c_bst_ulong)()

        _check_call(
            _LIB.XGBoosterFeatureScore(
                self.handle,
                args,
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
    def trees_to_dataframe(self, fmap: Union[str, os.PathLike] = '') -> DataFrame:
        """Parse a boosted tree model text dump into a pandas DataFrame structure.

        This feature is only defined when the decision tree model is chosen as base
        learner (`booster in {gbtree, dart}`). It is not defined for other base learner
        types, such as linear learners (`booster=gblinear`).

        Parameters
        ----------
        fmap: str or os.PathLike (optional)
           The name of feature map file.
        """
        # pylint: disable=too-many-locals
        fmap = os.fspath(os.path.expanduser(fmap))
        if not PANDAS_INSTALLED:
            raise ImportError(('pandas must be available to use this method.'
                               'Install pandas before calling again.'))
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
            for line in tree.split('\n'):
                arr = line.split('[')
                # Leaf node
                if len(arr) == 1:
                    # Last element of line.split is an empty string
                    if arr == ['']:
                        continue
                    # parse string
                    parse = arr[0].split(':')
                    stats = re.split('=|,', parse[1])

                    # append to lists
                    tree_ids.append(i)
                    node_ids.append(int(re.findall(r'\b\d+\b', parse[0])[0]))
                    fids.append('Leaf')
                    splits.append(float('NAN'))
                    categories.append(float('NAN'))
                    y_directs.append(float('NAN'))
                    n_directs.append(float('NAN'))
                    missings.append(float('NAN'))
                    gains.append(float(stats[1]))
                    covers.append(float(stats[3]))
                # Not a Leaf Node
                else:
                    # parse string
                    fid = arr[1].split(']')
                    if fid[0].find("<") != -1:
                        # numerical
                        parse = fid[0].split('<')
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
                    stats = re.split('=|,', fid[1])

                    # append to lists
                    tree_ids.append(i)
                    node_ids.append(int(re.findall(r'\b\d+\b', arr[0])[0]))
                    fids.append(parse[0])
                    str_i = str(i)
                    y_directs.append(str_i + '-' + stats[1])
                    n_directs.append(str_i + '-' + stats[3])
                    missings.append(str_i + '-' + stats[5])
                    gains.append(float(stats[7]))
                    covers.append(float(stats[9]))

        ids = [str(t_id) + '-' + str(n_id) for t_id, n_id in zip(tree_ids, node_ids)]
        df = DataFrame({'Tree': tree_ids, 'Node': node_ids, 'ID': ids,
                        'Feature': fids, 'Split': splits, 'Yes': y_directs,
                        'No': n_directs, 'Missing': missings, 'Gain': gains,
                        'Cover': covers, "Category": categories})

        if callable(getattr(df, 'sort_values', None)):
            # pylint: disable=no-member
            return df.sort_values(['Tree', 'Node']).reset_index(drop=True)
        # pylint: disable=no-member
        return df.sort(['Tree', 'Node']).reset_index(drop=True)

    def _validate_features(self, data: DMatrix) -> None:
        """
        Validate Booster and data's feature_names are identical.
        Set feature_names and feature_types from DMatrix
        """
        if data.num_row() == 0:
            return

        if self.feature_names is None:
            self.feature_names = data.feature_names
            self.feature_types = data.feature_types
        if data.feature_names is None and self.feature_names is not None:
            raise ValueError(
                "training data did not have the following fields: " +
                ", ".join(self.feature_names)
            )
        # Booster can't accept data with different feature names
        if self.feature_names != data.feature_names:
            dat_missing = set(self.feature_names) - set(data.feature_names)
            my_missing = set(data.feature_names) - set(self.feature_names)

            msg = 'feature_names mismatch: {0} {1}'

            if dat_missing:
                msg += ('\nexpected ' + ', '.join(
                    str(s) for s in dat_missing) + ' in input data')

            if my_missing:
                msg += ('\ntraining data did not have the following fields: ' +
                        ', '.join(str(s) for s in my_missing))

            raise ValueError(msg.format(self.feature_names, data.feature_names))

    def get_split_value_histogram(
        self,
        feature: str,
        fmap: Union[os.PathLike, str] = '',
        bins: Optional[int] = None,
        as_pandas: bool = True
    ) -> Union[np.ndarray, DataFrame]:
        """Get split value histogram of a feature

        Parameters
        ----------
        feature: str
            The name of the feature.
        fmap: str or os.PathLike (optional)
            The name of feature map file.
        bin: int, default None
            The maximum number of bins.
            Number of bins equals number of unique split values n_unique,
            if bins == None or bins > n_unique.
        as_pandas: bool, default True
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
        for i, _ in enumerate(xgdump):
            m = re.findall(regexp, xgdump[i])
            values.extend([float(x) for x in m])

        n_unique = len(np.unique(values))
        bins = max(min(n_unique, bins) if bins is not None else n_unique, 1)

        nph = np.histogram(values, bins=bins)
        nph = np.column_stack((nph[1][1:], nph[0]))
        nph = nph[nph[:, 1] > 0]

        if nph.size == 0:
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
            return DataFrame(nph, columns=['SplitValue', 'Count'])
        if as_pandas and not PANDAS_INSTALLED:
            warnings.warn(
                "Returning histogram as ndarray"
                " (as_pandas == True, but pandas is not installed).",
                UserWarning
            )
        return nph
