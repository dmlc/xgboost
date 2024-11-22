"""Tracker for XGBoost collective."""

import ctypes
import json
import socket
from enum import IntEnum, unique
from typing import Dict, Optional, Union

from .core import _LIB, _check_call, _deprecate_positional_args, make_jcargs


def get_family(addr: str) -> int:
    """Get network family from address."""
    return socket.getaddrinfo(addr, None)[0][0]


class RabitTracker:
    """Tracker for the collective used in XGBoost, acting as a coordinator between
    workers.

    Parameters
    ----------

    n_workers:

        The total number of workers in the communication group.

    host_ip:

        The IP address of the tracker node. XGBoost can try to guess one by probing with
        sockets. But it's best to explicitly pass an address.

    port:

        The port this tracker should listen to. XGBoost can query an available port from
        the OS, this configuration is useful for restricted network environments.

    sortby:

        How to sort the workers for rank assignment. The default is host, but users can
        set the `DMLC_TASK_ID` via arguments of :py:meth:`~xgboost.collective.init` and
        obtain deterministic rank assignment through sorting by task name. Available
        options are:

          - host
          - task

    timeout :

        Timeout for constructing (bootstrap) and shutting down the communication group,
        doesn't apply to communication when the group is up and running.

        The timeout value should take the time of data loading and pre-processing into
        account, due to potential lazy execution. By default the Tracker doesn't have
        any timeout to avoid pre-mature aborting.

        The :py:meth:`.wait_for` method has a different timeout parameter that can stop
        the tracker even if the tracker is still being used. A value error is raised
        when timeout is reached.

    Examples
    --------

    .. code-block:: python

        from xgboost.tracker import RabitTracker
        from xgboost import collective as coll

        tracker = RabitTracker(host_ip="127.0.0.1", n_workers=2)
        tracker.start()

        with coll.CommunicatorContext(**tracker.worker_args()):
            ret = coll.broadcast("msg", 0)
            assert str(ret) == "msg"

    """

    @unique
    class _SortBy(IntEnum):
        HOST = 0
        TASK = 1

    @_deprecate_positional_args
    def __init__(  # pylint: disable=too-many-arguments
        self,
        n_workers: int,
        host_ip: Optional[str],
        port: int = 0,
        *,
        sortby: str = "host",
        timeout: int = 0,
    ) -> None:

        handle = ctypes.c_void_p()
        if sortby not in ("host", "task"):
            raise ValueError("Expecting either 'host' or 'task' for sortby.")
        if host_ip is not None:
            get_family(host_ip)  # use python socket to stop early for invalid address
        args = make_jcargs(
            host=host_ip,
            n_workers=n_workers,
            port=port,
            dmlc_communicator="rabit",
            sortby=self._SortBy.HOST if sortby == "host" else self._SortBy.TASK,
            timeout=int(timeout),
        )
        _check_call(_LIB.XGTrackerCreate(args, ctypes.byref(handle)))
        self.handle = handle

    def free(self) -> None:
        """Internal function for testing."""
        if hasattr(self, "handle"):
            handle = self.handle
            del self.handle
            _check_call(_LIB.XGTrackerFree(handle))

    def __del__(self) -> None:
        self.free()

    def start(self) -> None:
        """Start the tracker. Once started, the client still need to call the
        :py:meth:`wait_for` method in order to wait for it to finish (think of it as a
        thread).

        """
        _check_call(_LIB.XGTrackerRun(self.handle, make_jcargs()))

    def wait_for(self, timeout: Optional[int] = None) -> None:
        """Wait for the tracker to finish all the work and shutdown. When timeout is
        reached, a value error is raised. By default we don't have timeout since we
        don't know how long it takes for the model to finish training.

        """
        _check_call(_LIB.XGTrackerWaitFor(self.handle, make_jcargs(timeout=timeout)))

    def worker_args(self) -> Dict[str, Union[str, int]]:
        """Get arguments for workers."""
        c_env = ctypes.c_char_p()
        _check_call(_LIB.XGTrackerWorkerArgs(self.handle, ctypes.byref(c_env)))
        assert c_env.value is not None
        env = json.loads(c_env.value)
        return env
