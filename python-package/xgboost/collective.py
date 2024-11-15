"""XGBoost collective communication related API."""

import ctypes
import logging
import os
import pickle
from dataclasses import dataclass
from enum import IntEnum, unique
from typing import Any, Dict, Optional, TypeAlias, Union

import numpy as np

from ._typing import _T
from .core import _LIB, _check_call, build_info, c_str, make_jcargs, py_str

LOGGER = logging.getLogger("[xgboost.collective]")


_ArgVals: TypeAlias = Optional[Union[int, str]]
_Args: TypeAlias = Dict[str, _ArgVals]


@dataclass
class Config:
    """User configuration for the communicator context. This is used for easier
    integration with distributed frameworks. Users of the collective module can pass the
    parameters directly into tracker and the communicator.

    .. versionadded:: 3.0

    Attributes
    ----------
    retry : See `dmlc_retry` in :py:meth:`init`.

    timeout :
        See `dmlc_timeout` in :py:meth:`init`. This is only used for communicators, not
        the tracker. They are different parameters since the timeout for tracker limits
        only the time for starting and finalizing the communication group, whereas the
        timeout for communicators limits the time used for collective operations.

    tracker_host_ip : See :py:class:`~xgboost.tracker.RabitTracker`.

    tracker_port : See :py:class:`~xgboost.tracker.RabitTracker`.

    tracker_timeout : See :py:class:`~xgboost.tracker.RabitTracker`.

    """

    retry: Optional[int] = None
    timeout: Optional[int] = None

    tracker_host_ip: Optional[str] = None
    tracker_port: Optional[int] = None
    tracker_timeout: Optional[int] = None

    def get_comm_config(self, args: _Args) -> _Args:
        """Update the arguments for the communicator."""
        if self.retry is not None:
            args["dmlc_retry"] = self.retry
        if self.timeout is not None:
            args["dmlc_timeout"] = self.timeout
        return args


def init(**args: _ArgVals) -> None:
    """Initialize the collective library with arguments.

    Parameters
    ----------
    args :
        Keyword arguments representing the parameters and their values.

        Accepted parameters:
          - dmlc_communicator: The type of the communicator.
            * rabit: Use Rabit. This is the default if the type is unspecified.
            * federated: Use the gRPC interface for Federated Learning.

        Only applicable to the Rabit communicator:
          - dmlc_tracker_uri: Hostname of the tracker.
          - dmlc_tracker_port: Port number of the tracker.
          - dmlc_task_id: ID of the current task, can be used to obtain deterministic
          - dmlc_retry: The number of retry when handling network errors.
          - dmlc_timeout: Timeout in seconds.
          - dmlc_nccl_path: Path to load (dlopen) nccl for GPU-based communication.

        Only applicable to the Federated communicator:
          - federated_server_address: Address of the federated server.
          - federated_world_size: Number of federated workers.
          - federated_rank: Rank of the current worker.
          - federated_server_cert: Server certificate file path. Only needed for the SSL
            mode.
          - federated_client_key: Client key file path. Only needed for the SSL mode.
          - federated_client_cert: Client certificate file path. Only needed for the SSL
            mode.

        Use upper case for environment variables, use lower case for runtime configuration.

    """
    _check_call(_LIB.XGCommunicatorInit(make_jcargs(**args)))


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
    # Run first broadcast
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
def _map_dtype(dtype: np.dtype) -> int:
    dtype_map = {
        np.dtype("float16"): 0,
        np.dtype("float32"): 1,
        np.dtype("float64"): 2,
        np.dtype("int8"): 4,
        np.dtype("int16"): 5,
        np.dtype("int32"): 6,
        np.dtype("int64"): 7,
        np.dtype("uint8"): 8,
        np.dtype("uint16"): 9,
        np.dtype("uint32"): 10,
        np.dtype("uint64"): 11,
    }
    try:
        dtype_map.update({np.dtype("float128"): 3})
    except TypeError:  # float128 doesn't exist on the system
        pass

    if dtype not in dtype_map:
        raise TypeError(f"data type {dtype} is not supported on the current platform.")

    return dtype_map[dtype]


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
    buf = data.ravel().copy()
    _check_call(
        _LIB.XGCommunicatorAllreduce(
            buf.ctypes.data_as(ctypes.c_void_p),
            buf.size,
            _map_dtype(buf.dtype),
            int(op),
        )
    )
    return buf


def signal_error() -> None:
    """Kill the process."""
    _check_call(_LIB.XGCommunicatorSignalError())


class CommunicatorContext:
    """A context controlling collective communicator initialization and finalization."""

    def __init__(self, **args: _ArgVals) -> None:
        self.args = args
        key = "dmlc_nccl_path"
        if args.get(key, None) is not None:
            return

        binfo = build_info()
        if not binfo["USE_DLOPEN_NCCL"]:
            return

        try:
            # PyPI package of NCCL.
            from nvidia.nccl import lib

            # There are two versions of nvidia-nccl, one is from PyPI, another one from
            # nvidia-pyindex. We support only the first one as the second one is too old
            # (2.9.8 as of writing).
            if lib.__file__ is not None:
                dirname: Optional[str] = os.path.dirname(lib.__file__)
            else:
                dirname = None

            if dirname:
                path = os.path.join(dirname, "libnccl.so.2")
                self.args[key] = path
        except ImportError:
            pass

    def __enter__(self) -> _Args:
        init(**self.args)
        assert is_distributed()
        LOGGER.debug("-------------- communicator say hello ------------------")
        return self.args

    def __exit__(self, *args: Any) -> None:
        finalize()
        LOGGER.debug("--------------- communicator say bye ------------------")
