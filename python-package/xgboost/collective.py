"""XGBoost collective communication related API."""
import ctypes
import logging
import pickle
from enum import IntEnum, unique
from typing import Any, Optional, cast, List, Union

import numpy as np

from ._typing import _T
from .core import _LIB, _check_call, c_str, py_str

LOGGER = logging.getLogger("[xgboost.collective]")


def init(args: Optional[List[bytes]] = None) -> None:
    """Initialize the collective library with arguments"""
    if args is None:
        args = []
    arr = (ctypes.c_char_p * len(args))()
    arr[:] = cast(List[Union[ctypes.c_char_p, bytes, None, int]], args)
    _LIB.XGCommunicatorInit(len(arr), arr)


def finalize() -> None:
    """Finalize the process, notify tracker everything is done."""
    _LIB.XGCommunicatorFinalize()


def get_rank() -> int:
    """Get rank of current process.

    Returns
    -------
    rank : int
        Rank of current process.
    """
    ret = _LIB.XGCommunicatorGetRank()
    return ret


def get_world_size() -> int:
    """Get total number workers.

    Returns
    -------
    n : int
        Total number of process.
    """
    ret = _LIB.XGCommunicatorGetWorldSize()
    return ret


def is_distributed() -> int:
    """If the collective communicator is distributed."""
    is_dist = _LIB.XGCommunicatorIsDistributed()
    return is_dist


def communicator_print(msg: Any) -> None:
    """Print message to the communicator.

    This function can be used to communicate the information of
    the progress to the communicator.

    Parameters
    ----------
    msg : str
        The message to be printed to the communicator.
    """
    if not isinstance(msg, str):
        msg = str(msg)
    is_dist = _LIB.XGCommunicatorIsDistributed()
    if is_dist != 0:
        _check_call(_LIB.XGCommunicatorPrint(c_str(msg)))
    else:
        print(msg.strip(), flush=True)


def get_processor_name() -> str:
    """Get the processor name.

    Returns
    -------
    name : str
        the name of processor(host)
    """
    name_str = ctypes.c_char_p()
    _check_call(_LIB.XGCommunicatorGetProcessorName(ctypes.byref(name_str)))
    value = name_str.value
    assert value
    return py_str(value)


def broadcast(data: _T, root: int) -> _T:
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
    _check_call(_LIB.XGCommunicatorBroadcast(ctypes.byref(length),
                                             ctypes.sizeof(ctypes.c_ulong), root))
    if root != rank:
        dptr = (ctypes.c_char * length.value)()
        # run second
        _check_call(_LIB.XGCommunicatorBroadcast(ctypes.cast(dptr, ctypes.c_void_p),
                                                 length.value, root))
        data = pickle.loads(dptr.raw)
        del dptr
    else:
        _check_call(_LIB.XGCommunicatorBroadcast(ctypes.cast(ctypes.c_char_p(s), ctypes.c_void_p),
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
    """Supported operations for rabit."""
    MAX = 0
    MIN = 1
    SUM = 2


def allreduce(  # pylint:disable=invalid-name
        data: np.ndarray, op: Op
) -> np.ndarray:
    """Perform allreduce, return the result.

    Parameters
    ----------
    data :
        Input data.
    op :
        Reduction operators, can be MAX or SUM

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
    _check_call(_LIB.XGCommunicatorAllreduce(buf.ctypes.data_as(ctypes.c_void_p),
                                             buf.size, DTYPE_ENUM__[buf.dtype],
                                             int(op), None, None))
    return buf


class CommunicatorContext:
    """A context controlling collective communicator initialization and finalization."""

    def __init__(self, args: List[bytes] = None) -> None:
        if args is None:
            args = []
        self.args = args

    def __enter__(self) -> None:
        init(self.args)
        assert is_distributed()
        LOGGER.debug("-------------- communicator say hello ------------------")

    def __exit__(self, *args: List) -> None:
        finalize()
        LOGGER.debug("--------------- communicator say bye ------------------")
