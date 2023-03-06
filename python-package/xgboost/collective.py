"""XGBoost collective communication related API."""
import ctypes
import json
import logging
import pickle
from enum import IntEnum, unique
from typing import Any, Dict, List

import numpy as np

from ._typing import _T
from .core import _LIB, _check_call, c_str, from_pystr_to_cstr, py_str

LOGGER = logging.getLogger("[xgboost.collective]")


def init(**args: Any) -> None:
    """Initialize the collective library with arguments.

    Parameters
    ----------
    args: Dict[str, Any]
        Keyword arguments representing the parameters and their values.

        Accepted parameters:
          - xgboost_communicator: The type of the communicator. Can be set as an environment
            variable.
            * rabit: Use Rabit. This is the default if the type is unspecified.
            * federated: Use the gRPC interface for Federated Learning.
        Only applicable to the Rabit communicator (these are case sensitive):
          -- rabit_tracker_uri: Hostname of the tracker.
          -- rabit_tracker_port: Port number of the tracker.
          -- rabit_task_id: ID of the current task, can be used to obtain deterministic rank
             assignment.
          -- rabit_world_size: Total number of workers.
          -- rabit_hadoop_mode: Enable Hadoop support.
          -- rabit_tree_reduce_minsize: Minimal size for tree reduce.
          -- rabit_reduce_ring_mincount: Minimal count to perform ring reduce.
          -- rabit_reduce_buffer: Size of the reduce buffer.
          -- rabit_bootstrap_cache: Size of the bootstrap cache.
          -- rabit_debug: Enable debugging.
          -- rabit_timeout: Enable timeout.
          -- rabit_timeout_sec: Timeout in seconds.
          -- rabit_enable_tcp_no_delay: Enable TCP no delay on Unix platforms.
        Only applicable to the Rabit communicator (these are case-sensitive, and can be set as
        environment variables):
          -- DMLC_TRACKER_URI: Hostname of the tracker.
          -- DMLC_TRACKER_PORT: Port number of the tracker.
          -- DMLC_TASK_ID: ID of the current task, can be used to obtain deterministic rank
             assignment.
          -- DMLC_ROLE: Role of the current task, "worker" or "server".
          -- DMLC_NUM_ATTEMPT: Number of attempts after task failure.
          -- DMLC_WORKER_CONNECT_RETRY: Number of retries to connect to the tracker.
        Only applicable to the Federated communicator (use upper case for environment variables, use
        lower case for runtime configuration):
          -- federated_server_address: Address of the federated server.
          -- federated_world_size: Number of federated workers.
          -- federated_rank: Rank of the current worker.
          -- federated_server_cert: Server certificate file path. Only needed for the SSL mode.
          -- federated_client_key: Client key file path. Only needed for the SSL mode.
          -- federated_client_cert: Client certificate file path. Only needed for the SSL mode.
    """
    config = from_pystr_to_cstr(json.dumps(args))
    _check_call(_LIB.XGCommunicatorInit(config))


def finalize() -> None:
    """Finalize the communicator."""
    _check_call(_LIB.XGCommunicatorFinalize())


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
        _check_call(_LIB.XGCommunicatorPrint(c_str(msg.strip())))
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
        assert data is not None, "need to pass in data when broadcasting"
        s = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        length.value = len(s)
    # run first broadcast
    _check_call(
        _LIB.XGCommunicatorBroadcast(
            ctypes.byref(length), ctypes.sizeof(ctypes.c_ulong), root
        )
    )
    if root != rank:
        dptr = (ctypes.c_char * length.value)()
        # run second
        _check_call(
            _LIB.XGCommunicatorBroadcast(
                ctypes.cast(dptr, ctypes.c_void_p), length.value, root
            )
        )
        data = pickle.loads(dptr.raw)
        del dptr
    else:
        _check_call(
            _LIB.XGCommunicatorBroadcast(
                ctypes.cast(ctypes.c_char_p(s), ctypes.c_void_p), length.value, root
            )
        )
        del s
    return data


# enumeration of dtypes
DTYPE_ENUM__ = {
    np.dtype("int8"): 0,
    np.dtype("uint8"): 1,
    np.dtype("int32"): 2,
    np.dtype("uint32"): 3,
    np.dtype("int64"): 4,
    np.dtype("uint64"): 5,
    np.dtype("float32"): 6,
    np.dtype("float64"): 7,
}


@unique
class Op(IntEnum):
    """Supported operations for allreduce."""

    MAX = 0
    MIN = 1
    SUM = 2
    BITWISE_AND = 3
    BITWISE_OR = 4
    BITWISE_XOR = 5


def allreduce(data: np.ndarray, op: Op) -> np.ndarray:  # pylint:disable=invalid-name
    """Perform allreduce, return the result.

    Parameters
    ----------
    data :
        Input data.
    op :
        Reduction operator.

    Returns
    -------
    result :
        The result of allreduce, have same shape as data

    Notes
    -----
    This function is not thread-safe.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("allreduce only takes in numpy.ndarray")
    buf = data.ravel()
    if buf.base is data.base:
        buf = buf.copy()
    if buf.dtype not in DTYPE_ENUM__:
        raise TypeError(f"data type {buf.dtype} not supported")
    _check_call(
        _LIB.XGCommunicatorAllreduce(
            buf.ctypes.data_as(ctypes.c_void_p),
            buf.size,
            DTYPE_ENUM__[buf.dtype],
            int(op),
            None,
            None,
        )
    )
    return buf


class CommunicatorContext:
    """A context controlling collective communicator initialization and finalization."""

    def __init__(self, **args: Any) -> None:
        self.args = args

    def __enter__(self) -> Dict[str, Any]:
        init(**self.args)
        assert is_distributed()
        LOGGER.debug("-------------- communicator say hello ------------------")
        return self.args

    def __exit__(self, *args: List) -> None:
        finalize()
        LOGGER.debug("--------------- communicator say bye ------------------")
