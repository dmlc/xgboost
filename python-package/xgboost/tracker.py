"""
This script is a variant of dmlc-core/dmlc_tracker/tracker.py,
which is a specialized version for xgboost tasks.
"""

import argparse
import ctypes
import json
import logging
import socket
import sys
from enum import IntEnum, unique
from typing import Dict, List, Optional, Tuple, Union

from .core import _LIB, _check_call, make_jcargs

_RingMap = Dict[int, Tuple[int, int]]
_TreeMap = Dict[int, List[int]]


class RabitTracker:
    """Tracker for the collective used in XGBoost, acting as a coordinator between
    workers.

    Parameters
    ..........
    sortby:
        How to sort the workers for rank assignment. The default is host, but users can
        set the `DMLC_TASK_ID` via RABIT initialization arguments and obtain
        deterministic rank assignment. Available options are:
          - host
          - task

    timeout :
        Timeout for constructing the communication group and waiting for the tracker to
        shutdown when it's instructed to, doesn't not apply to communication when
        tracking is running. The join method has a different timeout parameter that can
        stop the tracker even if the tracker is still being used. A value error is
        raised when timeout is reached.

    """

    @unique
    class _SortBy(IntEnum):
        HOST = 0
        TASK = 1

    def __init__(  # pylint: disable=too-many-arguments
        self,
        host_ip: str | None,
        n_workers: int,
        port: int = 0,
        sortby: str = "host",
        timeout: int = 300,
    ) -> None:

        handle = ctypes.c_void_p()
        if sortby not in ("host", "task"):
            raise ValueError("Expecting either 'host' or 'task' for sortby.")
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

    def __del__(self) -> None:
        if hasattr(self, "handle"):
            _check_call(_LIB.XGTrackerFree(self.handle))

    def start(self) -> None:
        """Start the tracker. Once started, the client still need to call the
        :py:meth:`join` method in order to wait for it to finish (think of it as a
        thread).

        """
        _check_call(_LIB.XGTrackerRun(self.handle, make_jcargs()))

    def join(self, timeout: Optional[int] = None) -> None:
        """Wait for the tracker to finish all the work and shutdown. When timeout is
        reached, a value error is raised. By default we don't have timeout since we
        don't know how long it takes for the model to finish training.

        """
        _check_call(_LIB.XGTrackerWait(self.handle, make_jcargs(timeout=timeout)))

    def worker_envs(self) -> Dict[str, Union[str, int]]:
        """Get arguments for workers."""
        c_env = ctypes.c_char_p()
        _check_call(_LIB.XGTrackerWorkerArgs(self.handle, ctypes.byref(c_env)))
        assert c_env.value is not None
        env = json.loads(c_env.value)
        return env

def start_rabit_tracker(args: argparse.Namespace) -> None:
    """Standalone function to start rabit tracker.

    Parameters
    ----------
    args: arguments to start the rabit tracker.
    """
    envs = {"n_workers": args.num_workers}
    rabit = RabitTracker(host_ip=get_host_ip(args.host_ip), n_workers=args.num_workers)
    envs.update(rabit.worker_envs())
    rabit.start()
    sys.stdout.write("DMLC_TRACKER_ENV_START\n")
    # simply write configuration to stdout
    for k, v in envs.items():
        sys.stdout.write(f"{k}={v}\n")
    sys.stdout.write("DMLC_TRACKER_ENV_END\n")
    sys.stdout.flush()
    rabit.join()


def main() -> None:
    """Main function if tracker is executed in standalone mode."""
    parser = argparse.ArgumentParser(description="Rabit Tracker start.")
    parser.add_argument(
        "--num-workers",
        required=True,
        type=int,
        help="Number of worker process to be launched.",
    )
    parser.add_argument(
        "--num-servers",
        default=0,
        type=int,
        help="Number of server process to be launched. Only used in PS jobs.",
    )
    parser.add_argument(
        "--host-ip",
        default=None,
        type=str,
        help=(
            "Host IP addressed, this is only needed "
            + "if the host IP cannot be automatically guessed."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        type=str,
        choices=["INFO", "DEBUG"],
        help="Logging level of the logger.",
    )
    args = parser.parse_args()

    fmt = "%(asctime)s %(levelname)s %(message)s"
    if args.log_level == "INFO":
        level = logging.INFO
    elif args.log_level == "DEBUG":
        level = logging.DEBUG
    else:
        raise RuntimeError(f"Unknown logging level {args.log_level}")

    logging.basicConfig(format=fmt, level=level)

    if args.num_servers == 0:
        start_rabit_tracker(args)
    else:
        raise RuntimeError("Do not yet support start ps tracker in standalone mode.")


if __name__ == "__main__":
    main()
