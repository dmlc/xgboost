"""
Reliable Allreduce and Broadcast Library.

Author: Tianqi Chen
"""
# pylint: disable=unused-argument,invalid-name,global-statement,dangerous-default-value,
import cPickle as pickle
import ctypes
import os
import sys
import warnings
import numpy as np

# version information about the doc
__version__ = '1.0'

_LIB = None

def _find_lib_path(dll_name):
    """Find the rabit dynamic library files.

    Returns
    -------
    lib_path: list(string)
       List of all found library path to rabit
    """
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    # make pythonpack hack: copy this directory one level upper for setup.py
    dll_path = [curr_path,
                os.path.join(curr_path, '../lib/'),
                os.path.join(curr_path, './lib/')]
    if os.name == 'nt':
        dll_path = [os.path.join(p, dll_name) for p in dll_path]
    else:
        dll_path = [os.path.join(p, dll_name) for p in dll_path]
    lib_path = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]
    #From github issues, most of installation errors come from machines w/o compilers
    if len(lib_path) == 0 and not os.environ.get('XGBOOST_BUILD_DOC', False):
        raise RuntimeError(
            'Cannot find Rabit Libarary in the candicate path, ' +
            'did you install compilers and run build.sh in root path?\n'
            'List of candidates:\n' + ('\n'.join(dll_path)))
    return lib_path

# load in xgboost library
def _loadlib(lib='standard', lib_dll=None):
    """Load rabit library."""
    global _LIB
    if _LIB is not None:
        warnings.warn('rabit.int call was ignored because it has'\
                          ' already been initialized', level=2)
        return

    if lib_dll is not None:
        _LIB = lib_dll
        return

    if lib == 'standard':
        dll_name = 'librabit'
    else:
        dll_name = 'librabit_' + lib

    if os.name == 'nt':
        dll_name += '.dll'
    else:
        dll_name += '.so'

    _LIB = ctypes.cdll.LoadLibrary(_find_lib_path(dll_name)[0])
    _LIB.RabitGetRank.restype = ctypes.c_int
    _LIB.RabitGetWorldSize.restype = ctypes.c_int
    _LIB.RabitVersionNumber.restype = ctypes.c_int

def _unloadlib():
    """Unload rabit library."""
    global _LIB
    del _LIB
    _LIB = None

# reduction operators
MAX = 0
MIN = 1
SUM = 2
BITOR = 3

def init(args=None, lib='standard', lib_dll=None):
    """Intialize the rabit module, call this once before using anything.

    Parameters
    ----------
    args: list of str, optional
        The list of arguments used to initialized the rabit
        usually you need to pass in sys.argv.
        Defaults to sys.argv when it is None.
    lib: {'standard', 'mock', 'mpi'}, optional
        Type of library we want to load
        When cdll is specified
    lib_dll: ctypes.DLL, optional
        The DLL object used as lib.
        When this is presented argument lib will be ignored.
    """
    if args is None:
        args = sys.argv
    _loadlib(lib, lib_dll)
    arr = (ctypes.c_char_p * len(args))()
    arr[:] = args
    _LIB.RabitInit(len(args), arr)

def finalize():
    """Finalize the rabit engine.

    Call this function after you finished all jobs.
    """
    _LIB.RabitFinalize()
    _unloadlib()

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
    if not isinstance(msg, str):
        msg = str(msg)
    _LIB.RabitTrackerPrint(ctypes.c_char_p(msg).encode('utf-8'))

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
    np.dtype('int8') : 0,
    np.dtype('uint8') : 1,
    np.dtype('int32') : 2,
    np.dtype('uint32') : 3,
    np.dtype('int64') : 4,
    np.dtype('uint64') : 5,
    np.dtype('float32') : 6,
    np.dtype('float64') : 7
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
        will be called by the function before performing allreduce, to intialize the data
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
        def pfunc(args):
            """prepare function."""
            prepare_fun(data)
        _LIB.RabitAllreduce(buf.ctypes.data_as(ctypes.c_void_p),
                            buf.size, DTYPE_ENUM__[buf.dtype],
                            op, func_ptr(pfunc), None)
    return buf


def _load_model(ptr, length):
    """
    Internal function used by the module,
    unpickle a model from a buffer specified by ptr, length
    Arguments:
        ptr: ctypes.POINTER(ctypes._char)
            pointer to the memory region of buffer
        length: int
            the length of buffer
    """
    data = (ctypes.c_char * length).from_address(ctypes.addressof(ptr.contents))
    return pickle.loads(data.raw)

def load_checkpoint(with_local=False):
    """Load latest check point.

    Parameters
    ----------
    with_local: bool, optional
        whether the checkpoint contains local model

    Returns
    -------
    tuple : tuple
        if with_local: return (version, gobal_model, local_model)
        else return (version, gobal_model)
        if returned version == 0, this means no model has been CheckPointed
        and global_model, local_model returned will be None
    """
    gptr = ctypes.POINTER(ctypes.c_char)()
    global_len = ctypes.c_ulong()
    if with_local:
        lptr = ctypes.POINTER(ctypes.c_char)()
        local_len = ctypes.c_ulong()
        version = _LIB.RabitLoadCheckPoint(
            ctypes.byref(gptr),
            ctypes.byref(global_len),
            ctypes.byref(lptr),
            ctypes.byref(local_len))
        if version == 0:
            return (version, None, None)
        return (version,
                _load_model(gptr, global_len.value),
                _load_model(lptr, local_len.value))
    else:
        version = _LIB.RabitLoadCheckPoint(
            ctypes.byref(gptr),
            ctypes.byref(global_len),
            None, None)
        if version == 0:
            return (version, None)
        return (version,
                _load_model(gptr, global_len.value))

def checkpoint(global_model, local_model=None):
    """Checkpoint the model.

    This means we finished a stage of execution.
    Every time we call check point, there is a version number which will increase by one.

    Parameters
    ----------
    global_model: anytype that can be pickled
        globally shared model/state when calling this function,
        the caller need to gauranttees that global_model is the same in all nodes

    local_model: anytype that can be pickled
       Local model, that is specific to current node/rank.
       This can be None when no local state is needed.

    Notes
    -----
    local_model requires explicit replication of the model for fault-tolerance.
    This will bring replication cost in checkpoint function.
    while global_model do not need explicit replication.
    It is recommended to use global_model if possible.
    """
    sglobal = pickle.dumps(global_model)
    if local_model is None:
        _LIB.RabitCheckPoint(sglobal, len(sglobal), None, 0)
        del sglobal
    else:
        slocal = pickle.dumps(local_model)
        _LIB.RabitCheckPoint(sglobal, len(sglobal), slocal, len(slocal))
        del slocal
        del sglobal

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
