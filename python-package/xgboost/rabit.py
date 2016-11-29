# coding: utf-8
# pylint: disable= invalid-name

"""Distributed XGBoost Rabit related API."""
from __future__ import absolute_import
import sys
import ctypes
import numpy as np

from .core import _LIB, c_str, STRING_TYPES
from .compat import pickle


def _init_rabit():
    """internal library initializer."""
    if _LIB is not None:
        _LIB.RabitGetRank.restype = ctypes.c_int
        _LIB.RabitGetWorldSize.restype = ctypes.c_int
        _LIB.RabitIsDistributed.restype = ctypes.c_int
        _LIB.RabitVersionNumber.restype = ctypes.c_int


def init(args=None):
    """Initialize the rabit library with arguments"""
    if args is None:
        args = []
    arr = (ctypes.c_char_p * len(args))()
    arr[:] = args
    _LIB.RabitInit(len(arr), arr)


def finalize():
    """Finalize the process, notify tracker everything is done."""
    _LIB.RabitFinalize()


def get_rank():
    """Get rank of current process.

    Returns
    -------
    rank : int
        Rank of current process.
    """
    ret = _LIB.RabitGetRank()
    return ret


def get_world_size():
    """Get total number workers.

    Returns
    -------
    n : int
        Total number of process.
    """
    ret = _LIB.RabitGetWorldSize()
    return ret


def tracker_print(msg):
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
        _LIB.RabitTrackerPrint(c_str(msg))
    else:
        sys.stdout.write(msg)
        sys.stdout.flush()


def get_processor_name():
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


def broadcast(data, root):
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
    _LIB.RabitBroadcast(ctypes.byref(length),
                        ctypes.sizeof(ctypes.c_ulong), root)
    if root != rank:
        dptr = (ctypes.c_char * length.value)()
        # run second
        _LIB.RabitBroadcast(ctypes.cast(dptr, ctypes.c_void_p),
                            length.value, root)
        data = pickle.loads(dptr.raw)
        del dptr
    else:
        _LIB.RabitBroadcast(ctypes.cast(ctypes.c_char_p(s), ctypes.c_void_p),
                            length.value, root)
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


def allreduce(data, op, prepare_fun=None):
    """Perform allreduce, return the result.

    Parameters
    ----------
    data: numpy array
        Input data.
    op: int
        Reduction operators, can be MIN, MAX, SUM, BITOR
    prepare_fun: function
        Lazy preprocessing function, if it is not None, prepare_fun(data)
        will be called by the function before performing allreduce, to initialize the data
        If the result of Allreduce can be recovered directly,
        then prepare_fun will NOT be called

    Returns
    -------
    result : array_like
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
        raise Exception('data type %s not supported' % str(buf.dtype))
    if prepare_fun is None:
        _LIB.RabitAllreduce(buf.ctypes.data_as(ctypes.c_void_p),
                            buf.size, DTYPE_ENUM__[buf.dtype],
                            op, None, None)
    else:
        func_ptr = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

        def pfunc(_):
            """prepare function."""
            prepare_fun(data)
        _LIB.RabitAllreduce(buf.ctypes.data_as(ctypes.c_void_p),
                            buf.size, DTYPE_ENUM__[buf.dtype],
                            op, func_ptr(pfunc), None)
    return buf


def version_number():
    """Returns version number of current stored model.

    This means how many calls to CheckPoint we made so far.

    Returns
    -------
    version : int
        Version number of currently stored model
    """
    ret = _LIB.RabitVersionNumber()
    return ret


# intialization script
_init_rabit()
