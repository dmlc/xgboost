"""
Python interface for rabit
  Reliable Allreduce and Broadcast Library
Author: Tianqi Chen
"""
import cPickle as pickle
import ctypes
import os
import sys
import warnings
import numpy as np

if os.name == 'nt':
    WRAPPER_PATH = os.path.dirname(__file__) + '\\..\\windows\\x64\\Release\\rabit_wrapper%s.dll'
else:
    WRAPPER_PATH = os.path.dirname(__file__) + '/librabit_wrapper%s.so'
rbtlib = None

# load in xgboost library
def loadlib__(lib = 'standard'):    
    global rbtlib
    if rbtlib != None:
        warnings.Warn('rabit.int call was ignored because it has already been initialized', level = 2)
        return
    if lib == 'standard':
        rbtlib = ctypes.cdll.LoadLibrary(WRAPPER_PATH % '')
    elif lib == 'mock':
        rbtlib = ctypes.cdll.LoadLibrary(WRAPPER_PATH % '_mock')
    elif lib == 'mpi':
        rbtlib = ctypes.cdll.LoadLibrary(WRAPPER_PATH % '_mpi')
    else:
        raise Exception('unknown rabit lib %s, can be standard, mock, mpi' % lib)
    rbtlib.RabitGetRank.restype = ctypes.c_int
    rbtlib.RabitGetWorldSize.restype = ctypes.c_int
    rbtlib.RabitVersionNumber.restype = ctypes.c_int

def unloadlib__():
    global rbtlib
    del rbtlib
    rbtlib = None

# reduction operators
MAX = 0
MIN = 1
SUM = 2
BITOR = 3

def check_err__():    
    """
    reserved function used to check error    
    """
    return

def init(args = sys.argv, lib = 'standard'):
    """
    intialize the rabit module, call this once before using anything
    Arguments:
        args: list(string) [default=sys.argv]
           the list of arguments used to initialized the rabit
           usually you need to pass in sys.argv
        with_mock: boolean [default=False]
            Whether initialize the mock test module
    """
    loadlib__(lib)
    arr = (ctypes.c_char_p * len(args))()
    arr[:] = args
    rbtlib.RabitInit(len(args), arr)
    check_err__()

def finalize():
    """
    finalize the rabit engine, call this function after you finished all jobs 
    """
    rbtlib.RabitFinalize()
    check_err__()
    unloadlib__()

def get_rank():
    """
    Returns rank of current process
    """
    ret = rbtlib.RabitGetRank()
    check_err__()
    return ret

def get_world_size():
    """
    Returns get total number of process
    """
    ret = rbtlib.RabitGetWorldSize()
    check_err__()
    return ret

def tracker_print(msg):
    """
    print message to the tracker
    this function can be used to communicate the information of the progress
    to the tracker
    """
    if not isinstance(msg, str):
        msg = str(msg)
    rbtlib.RabitTrackerPrint(ctypes.c_char_p(msg).encode('utf-8'))
    check_err__()

def get_processor_name():
    """
    Returns the name of processor(host)
    """
    mxlen = 256
    length = ctypes.c_ulong()
    buf = ctypes.create_string_buffer(mxlen)
    rbtlib.RabitGetProcessorName(buf, ctypes.byref(length),
                                 mxlen)
    check_err__()
    return buf.value

def broadcast(data, root):
    """
    broadcast object from one node to all other nodes
    this function will return the broadcasted object

    Example: the following example broadcast hello from rank 0 to all other nodes
    ```python
    rabit.init()
    n = 3
    rank = rabit.get_rank()
    s = None
    if rank == 0:
        s = {'hello world':100, 2:3}
    print '@node[%d] before-broadcast: s=\"%s\"' % (rank, str(s))
    s = rabit.broadcast(s, 0)
    print '@node[%d] after-broadcast: s=\"%s\"' % (rank, str(s))
    rabit.finalize()
    ```
    
    Arguments:
        data: anytype that can be pickled
              input data, if current rank does not equal root, this can be None
        root: int
              rank of the node to broadcast data from
    Returns:
        the result of broadcast
    """
    rank = get_rank()
    length = ctypes.c_ulong()
    if root == rank:
        assert data is not None, 'need to pass in data when broadcasting'
        s = pickle.dumps(data, protocol = pickle.HIGHEST_PROTOCOL)
        length.value = len(s)
    # run first broadcast
    rbtlib.RabitBroadcast(ctypes.byref(length),
                          ctypes.sizeof(ctypes.c_ulong),
                          root)    
    check_err__()
    if root != rank:
        dptr = (ctypes.c_char * length.value)()
        # run second
        rbtlib.RabitBroadcast(ctypes.cast(dptr, ctypes.c_void_p),
                              length.value, root)
        check_err__()
        data = pickle.loads(dptr.raw)
        del dptr
    else:
        rbtlib.RabitBroadcast(ctypes.cast(ctypes.c_char_p(s), ctypes.c_void_p),
                              length.value, root)
        check_err__()
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

def allreduce(data, op, prepare_fun = None):
    """
    perform allreduce, return the result, this function is not thread-safe
    Arguments:
        data: numpy ndarray
           input data 
        op: int
            reduction operators, can be MIN, MAX, SUM, BITOR
        prepare_fun: lambda data
            Lazy preprocessing function, if it is not None, prepare_fun(data)
            will be called by the function before performing allreduce, to intialize the data
            If the result of Allreduce can be recovered directly, then prepare_fun will NOT be called
    Returns:
        the result of allreduce, have same shape as data
    """
    if not isinstance(data, np.ndarray):
        raise Exception('allreduce only takes in numpy.ndarray')
    buf = data.ravel()
    if buf.base is data.base:
        buf = buf.copy()
    if buf.dtype not in DTYPE_ENUM__:
        raise Exception('data type %s not supported' % str(buf.dtype))
    if prepare_fun is None:
        rbtlib.RabitAllreduce(buf.ctypes.data_as(ctypes.c_void_p),
                              buf.size, DTYPE_ENUM__[buf.dtype],
                              op, None, None)
    else:
        PFUNC = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
        def pfunc(args):
            prepare_fun(data)
        rbtlib.RabitAllreduce(buf.ctypes.data_as(ctypes.c_void_p),
                              buf.size, DTYPE_ENUM__[buf.dtype],
                              op, PFUNC(pfunc), None)               
    check_err__()
    return buf


def load_model__(ptr, length):
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

def load_checkpoint(with_local = False):
    """
    load latest check point
    Arguments:
        with_local: boolean [default = False]
            whether the checkpoint contains local model
    Returns: 
        if with_local: return (version, gobal_model, local_model)
        else return (version, gobal_model)
        if returned version == 0, this means no model has been CheckPointed
        and global_model, local_model returned will be None
    """
    gp = ctypes.POINTER(ctypes.c_char)()
    global_len = ctypes.c_ulong()
    if with_local:
        lp = ctypes.POINTER(ctypes.c_char)()
        local_len = ctypes.c_ulong()
        version = rbtlib.RabitLoadCheckPoint(
            ctypes.byref(gp),
            ctypes.byref(global_len),
            ctypes.byref(lp),
            ctypes.byref(local_len))
        check_err__()
        if version == 0:
            return (version, None, None)
        return (version,
                load_model__(gp, global_len.value),
                load_model__(lp, local_len.value))
    else:
        version = rbtlib.RabitLoadCheckPoint(
            ctypes.byref(gp),
            ctypes.byref(global_len),
            None, None)
        check_err__()
        if version == 0:
            return (version, None)
        return (version,
                load_model__(gp, global_len.value))
    
def checkpoint(global_model, local_model = None):
    """
    checkpoint the model, meaning we finished a stage of execution
    every time we call check point, there is a version number which will increase by one    

    Arguments:
        global_model: anytype that can be pickled
            globally shared model/state when calling this function,
            the caller need to gauranttees that global_model is the same in all nodes
        local_model: anytype that can be pickled
            local model, that is specific to current node/rank.
            This can be None when no local state is needed.
            local_model requires explicit replication of the model for fault-tolerance,
            which will bring replication cost in checkpoint function,
            while global_model do not need explicit replication.
            It is recommended to use global_model if possible
    """
    sg = pickle.dumps(global_model)
    if local_model is None:
        rbtlib.RabitCheckPoint(sg, len(sg), None, 0)
        check_err__()
        del sg;
    else:
        sl = pickle.dumps(local_model)
        rbtlib.RabitCheckPoint(sg, len(sg), sl, len(sl))
        check_err__()
        del sl; del sg;

def version_number():
    """
    Returns version number of current stored model,
    which means how many calls to CheckPoint we made so far
    """
    ret = rbtlib.RabitVersionNumber()
    check_err__()
    return ret
