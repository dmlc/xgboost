"""
Python interface for rabit
  Reliable Allreduce and Broadcast Library
Author: Tianqi Chen
"""
import cPickle as pickle
import ctypes
import os
import sys
import numpy as np

if os.name == 'nt':
    assert False, "Rabit windows is not yet compiled"
else:
    RABIT_PATH = os.path.dirname(__file__)+'/librabit_wrapper.so'

# load in xgboost library
rbtlib = ctypes.cdll.LoadLibrary(RABIT_PATH)
rbtlib.RabitGetRank.restype = ctypes.c_int
rbtlib.RabitGetWorldSize.restype = ctypes.c_int

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

def init(args = sys.argv):
    """
    intialize the rabit module, call this once before using anything
    Arguments:
        args: list(string)
              the list of arguments used to initialized the rabit
              usually you need to pass in sys.argv
    """
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
    ret = rbtlib.RabitGetWorlSize()
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
        dptr = (ctypes.c_char * length.value)()
        dptr[:] = s
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
    return  pickle.loads(dptr.value)

def allreduce(data, op, prepare_fun = None):
    """
    perform allreduce, return the result, this function is not thread-safe
    Arguments:
        data: numpy ndarray
           input data 
        op: reduction operators, can be MIN, MAX, SUM, BITOR
        prepare_fun: lambda :
            Lazy preprocessing function, if it is not None, prepare_fun()
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
    if buf.dtype is np.dtype('int8'):
        dtype = 0
    elif buf.dtype is np.dtype('uint8'):
        dtype = 1
    elif buf.dtype is np.dtype('int32'):
        dtype = 2
    elif buf.dtype is np.dtype('uint32'):
        dtype = 3
    elif buf.dtype is np.dtype('int64'):
        dtype = 4
    elif buf.dtype is np.dtype('uint64'):
        dtype = 5
    elif buf.dtype is np.dtype('float32'):
        dtype = 6
    elif buf.dtype is np.dtype('float64'):
        dtype = 7
    else:
        raise Exception('data type %s not supported' % str(buf.dtype))
    rbtlib.RabitAllreduce(buf.ctypes.data_as(ctypes.c_void_p),
                          buf.size, dtype, op, None, None);
    check_err__()
    return buf
