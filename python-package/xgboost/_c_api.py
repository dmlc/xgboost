"""Low-level ctypes bridge for the XGBoost C API."""

import ctypes
import json
import os
import warnings
from typing import Any, Callable, List, Tuple, Union, cast, overload

from ._typing import CStrPptr, c_bst_ulong
from .compat import py_str
from .libpath import find_lib_path


class XGBoostError(ValueError):
    """Error thrown by xgboost trainer."""


@overload
def from_pystr_to_cstr(data: str) -> bytes: ...


@overload
def from_pystr_to_cstr(data: List[str]) -> ctypes.Array: ...


def from_pystr_to_cstr(data: Union[str, List[str]]) -> Union[bytes, ctypes.Array]:
    """Convert a Python str or list of Python str to C pointer."""
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
    """Revert C pointer to Python str."""
    res = []
    for i in range(length.value):
        try:
            res.append(str(cast(bytes, data[i]).decode("ascii")))
        except UnicodeDecodeError:
            res.append(str(cast(bytes, data[i]).decode("utf-8")))
    return res


def make_jcargs(**kwargs: Any) -> bytes:
    """Make JSON-based arguments for C functions."""
    return from_pystr_to_cstr(json.dumps(kwargs))


def _log_callback(msg: bytes) -> None:
    """Redirect logs from native library into Python console."""
    smsg = py_str(msg)
    if smsg.find("WARNING:") != -1:
        # Stacklevel:
        # 1: This line
        # 2: XGBoost C functions like `_LIB.XGBoosterTrainOneIter`.
        # 3: The Python function that calls the C function.
        warnings.warn(smsg, UserWarning, stacklevel=3)
        return
    print(smsg)


def _get_log_callback_func() -> Callable:
    """Wrap log_callback() method in ctypes callback type."""
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
    version_file = os.path.join(os.path.dirname(__file__), "VERSION")
    with open(version_file, encoding="ascii") as f:
        return f.read().strip()


def _register_log_callback(lib: ctypes.CDLL) -> None:
    lib.XGBGetLastError.restype = ctypes.c_char_p
    lib.callback = _get_log_callback_func()  # type: ignore
    if lib.XGBRegisterLogCallback(lib.callback) != 0:
        raise XGBoostError(lib.XGBGetLastError())


def _parse_version(ver: str) -> Tuple[Tuple[int, int, int], str]:
    """Avoid dependency on packaging (PEP 440)."""
    # 2.0.0-dev, 2.0.0, 2.0.0.post1, or 2.0.0rc1
    if ver.find("post") != -1:
        major, minor, patch = ver.split(".")[:-1]
        postfix = ver.split(".")[-1]
    elif "-dev" in ver:
        major, minor, patch = ver.split("-")[0].split(".")
        postfix = "dev"
    else:
        major, minor, patch = ver.split(".")
        rc = patch.find("rc")
        if rc != -1:
            postfix = patch[rc:]
            patch = patch[:rc]
        else:
            postfix = ""

    return (int(major), int(minor), int(patch)), postfix


def _load_lib() -> ctypes.CDLL:
    """Load xgboost library."""
    lib_paths = find_lib_path()
    if not lib_paths:
        # This happens only when building document.
        return None  # type: ignore
    try:
        path_backup = os.environ["PATH"].split(os.pathsep)
    except KeyError:
        path_backup = []
    lib_success = False
    os_error_list = []
    for lib_path in lib_paths:
        try:
            # needed when the lib is linked with non-system-available
            # dependencies
            os.environ["PATH"] = os.pathsep.join(
                path_backup + [os.path.dirname(lib_path)]
            )
            lib = ctypes.cdll.LoadLibrary(lib_path)
            setattr(lib, "path", os.path.normpath(lib_path))
            lib_success = True
            break
        except OSError as e:
            os_error_list.append(str(e))
            continue
        finally:
            os.environ["PATH"] = os.pathsep.join(path_backup)
    if not lib_success:
        libname = os.path.basename(lib_paths[0])
        raise XGBoostError(f"""
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
    _register_log_callback(lib)

    libver = _lib_version(lib)
    pyver, _ = _parse_version(_py_version())

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
    """Check the return value of C API call."""
    if ret != 0:
        raise XGBoostError(py_str(_LIB.XGBGetLastError()))


def c_str(string: str) -> ctypes.c_char_p:
    """Convert a python string to cstring."""
    return ctypes.c_char_p(string.encode("utf-8"))
