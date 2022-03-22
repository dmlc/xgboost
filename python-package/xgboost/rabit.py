"""Distributed XGBoost Rabit related API."""
import ctypes
from enum import IntEnum, unique
import pickle
from typing import Any, TypeVar, Callable, Optional, cast, List, Union

import numpy as np

from .core import _LIB, c_str, STRING_TYPES, _check_call


def _init_rabit() -> None:
    """internal library initializer."""
    if _LIB is not None:
        _LIB.RabitGetRank.restype = ctypes.c_int
        _LIB.RabitGetWorldSize.restype = ctypes.c_int
        _LIB.RabitIsDistributed.restype = ctypes.c_int
        _LIB.RabitVersionNumber.restype = ctypes.c_int


def init(args: Optional[List[bytes]] = None) -> None:
    """Initialize the rabit library with arguments"""
    if args is None:
        args = []
    arr = (ctypes.c_char_p * len(args))()
    arr[:] = cast(List[Union[ctypes.c_char_p, bytes, None, int]], args)
    _LIB.RabitInit(len(arr), arr)


def finalize() -> None:
    """Finalize the process, notify tracker everything is done."""
    _LIB.RabitFinalize()


def get_rank() -> int:
    """Get rank of current process.

    Returns
    -------
    rank : int
        Rank of current process.
    """
    ret = _LIB.RabitGetRank()
    return ret


def get_world_size() -> int:
    """Get total number workers.

    Returns
    -------
    n : int
        Total number of process.
    """
    ret = _LIB.RabitGetWorldSize()
    return ret


def is_distributed() -> int:
    '''If rabit is distributed.'''
    is_dist = _LIB.RabitIsDistributed()
    return is_dist


def tracker_print(msg: Any) -> None:
    """Print message to the tracker.

    This function can be used to communicate the information of
    the progress to the tracker

    Parameters
    ----------
    msg : str
        The message to be printed to tracker.
    """
    if not isinstance(msg, STRING_TYPES):
        msg = str(msg)
    is_dist = _LIB.RabitIsDistributed()
    if is_dist != 0:
        _check_call(_LIB.RabitTrackerPrint(c_str(msg)))
    else:
        print(msg.strip(), flush=True)


def get_processor_name() -> bytes:
    """Get the processor name.

    Returns
    -------
    name : str
        the name of processor(host)
    """
    mxlen = 256
    length = ctypes.c_ulong()
    buf = ctypes.create_string_buffer(mxlen)
    _LIB.RabitGetProcessorName(buf, ctypes.byref(length), mxlen)
    return buf.value


T = TypeVar("T")                # pylint:disable=invalid-name


def broadcast(data: T, root: int) -> T:
    """Broadcast object from one node to all other nodes.

    Parameters
    ----------
    data : any type that can be pickled
        Input data, if current rank does not equal root, this can be None
    root : int
        Rank of the node to broadcast data from.

    Returns
    -------
    object : int
        the result of broadcast.
    """
    rank = get_rank()
    length = ctypes.c_ulong()
    if root == rank:
        assert data is not None, 'need to pass in data when broadcasting'
        s = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        length.value = len(s)
    # run first broadcast
    _check_call(_LIB.RabitBroadcast(ctypes.byref(length),
                                    ctypes.sizeof(ctypes.c_ulong), root))
    if root != rank:
        dptr = (ctypes.c_char * length.value)()
        # run second
        _check_call(_LIB.RabitBroadcast(ctypes.cast(dptr, ctypes.c_void_p),
                                        length.value, root))
        data = pickle.loads(dptr.raw)
        del dptr
    else:
        _check_call(_LIB.RabitBroadcast(ctypes.cast(ctypes.c_char_p(s), ctypes.c_void_p),
                                        length.value, root))
        del s
    return data


# enumeration of dtypes
DTYPE_ENUM__ = {
    np.dtype('int8'): 0,
    np.dtype('uint8'): 1,
    np.dtype('int32'): 2,
    np.dtype('uint32'): 3,
    np.dtype('int64'): 4,
    np.dtype('uint64'): 5,
    np.dtype('float32'): 6,
    np.dtype('float64'): 7
}


@unique
class Op(IntEnum):
    '''Supported operations for rabit.'''
    MAX = 0
    MIN = 1
    SUM = 2
    OR = 3


def allreduce(                  # pylint:disable=invalid-name
    data: np.ndarray, op: Op, prepare_fun: Optional[Callable[[np.ndarray], None]] = None
) -> np.ndarray:
    """Perform allreduce, return the result.

    Parameters
    ----------
    data :
        Input data.
    op :
        Reduction operators, can be MIN, MAX, SUM, BITOR
    prepare_fun :
        Lazy preprocessing function, if it is not None, prepare_fun(data)
        will be called by the function before performing allreduce, to initialize the data
        If the result of Allreduce can be recovered directly,
        then prepare_fun will NOT be called

    Returns
    -------
    result :
        The result of allreduce, have same shape as data

    Notes
    -----
    This function is not thread-safe.
    """
    if not isinstance(data, np.ndarray):
        raise Exception('allreduce only takes in numpy.ndarray')
    buf = data.ravel()
    if buf.base is data.base:
        buf = buf.copy()
    if buf.dtype not in DTYPE_ENUM__:
        raise Exception(f"data type {buf.dtype} not supported")
    if prepare_fun is None:
        _check_call(_LIB.RabitAllreduce(buf.ctypes.data_as(ctypes.c_void_p),
                                        buf.size, DTYPE_ENUM__[buf.dtype],
                                        int(op), None, None))
    else:
        func_ptr = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

        def pfunc(_: Any) -> None:
            """prepare function."""
            fn = cast(Callable[[np.ndarray], None], prepare_fun)
            fn(data)
        _check_call(_LIB.RabitAllreduce(buf.ctypes.data_as(ctypes.c_void_p),
                                        buf.size, DTYPE_ENUM__[buf.dtype],
                                        op, func_ptr(pfunc), None))
    return buf


def version_number() -> int:
    """Returns version number of current stored model.

    This means how many calls to CheckPoint we made so far.

    Returns
    -------
    version : int
        Version number of currently stored model
    """
    ret = _LIB.RabitVersionNumber()
    return ret


# initialization script
_init_rabit()
