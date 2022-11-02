"""Compatibility shim for xgboost.rabit; to be removed in 2.0"""
import logging
import warnings
from enum import IntEnum, unique
from typing import Any, TypeVar, Callable, Optional, List

import numpy as np

from . import collective

LOGGER = logging.getLogger("[xgboost.rabit]")


def _deprecation_warning() -> str:
    return (
        "The xgboost.rabit submodule is marked as deprecated in 1.7 and will be removed "
        "in 2.0. Please use xgboost.collective instead."
    )


def init(args: Optional[List[bytes]] = None) -> None:
    """Initialize the rabit library with arguments"""
    warnings.warn(_deprecation_warning(), FutureWarning)
    parsed = {}
    if args:
        for arg in args:
            kv = arg.decode().split('=')
            if len(kv) == 2:
                parsed[kv[0]] = kv[1]
    collective.init(**parsed)


def finalize() -> None:
    """Finalize the process, notify tracker everything is done."""
    collective.finalize()


def get_rank() -> int:
    """Get rank of current process.
    Returns
    -------
    rank : int
        Rank of current process.
    """
    return collective.get_rank()


def get_world_size() -> int:
    """Get total number workers.
    Returns
    -------
    n : int
        Total number of process.
    """
    return collective.get_world_size()


def is_distributed() -> int:
    """If rabit is distributed."""
    return collective.is_distributed()


def tracker_print(msg: Any) -> None:
    """Print message to the tracker.
    This function can be used to communicate the information of
    the progress to the tracker
    Parameters
    ----------
    msg : str
        The message to be printed to tracker.
    """
    collective.communicator_print(msg)


def get_processor_name() -> bytes:
    """Get the processor name.
    Returns
    -------
    name : str
        the name of processor(host)
    """
    return collective.get_processor_name().encode()


T = TypeVar("T")  # pylint:disable=invalid-name


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
    return collective.broadcast(data, root)


@unique
class Op(IntEnum):
    """Supported operations for rabit."""
    MAX = 0
    MIN = 1
    SUM = 2
    OR = 3


def allreduce(  # pylint:disable=invalid-name
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
    if prepare_fun is None:
        return collective.allreduce(data, collective.Op(op))
    raise Exception("preprocessing function is no longer supported")


def version_number() -> int:
    """Returns version number of current stored model.
    This means how many calls to CheckPoint we made so far.
    Returns
    -------
    version : int
        Version number of currently stored model
    """
    return 0


class RabitContext:
    """A context controlling rabit initialization and finalization."""

    def __init__(self, args: List[bytes] = None) -> None:
        if args is None:
            args = []
        self.args = args

    def __enter__(self) -> None:
        init(self.args)
        assert is_distributed()
        LOGGER.warning(_deprecation_warning())
        LOGGER.debug("-------------- rabit say hello ------------------")

    def __exit__(self, *args: List) -> None:
        finalize()
        LOGGER.debug("--------------- rabit say bye ------------------")
