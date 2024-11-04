"""Utilities for the XGBoost Dask interface."""

import logging
from typing import Any, Dict, Optional, Tuple

import distributed

from ..collective import Config

LOGGER = logging.getLogger("[xgboost.dask]")


def get_n_threads(local_param: Dict[str, Any], worker: "distributed.Worker") -> int:
    """Get the number of threads from a worker and the user-supplied parameters."""
    # dask worker nthreads, "state" is available in 2022.6.1
    dwnt = worker.state.nthreads if hasattr(worker, "state") else worker.nthreads
    n_threads = None
    for p in ["nthread", "n_jobs"]:
        if local_param.get(p, None) is not None and local_param.get(p, dwnt) != dwnt:
            LOGGER.info("Overriding `nthreads` defined in dask worker.")
            n_threads = local_param[p]
            break
    if n_threads == 0 or n_threads is None:
        n_threads = dwnt
    return n_threads


def get_address_from_user(
    dconfig: Optional[Dict[str, Any]], coll_config: Config
) -> Tuple[Optional[str], int]:
    """Get the tracker address from the optional user configuration.

    Parameters
    ----------
    dconfig :
        Dask global configuration.

    coll_config :
        Collective configuration.

    Returns
    -------
    The IP address along with the port number.

    """

    valid_config = ["scheduler_address"]

    host_ip = None
    port = 0

    if dconfig is not None:
        for k in dconfig:
            if k not in valid_config:
                raise ValueError(f"Unknown configuration: {k}")
    else:
        dconfig = {}

    host_ip = dconfig.get("scheduler_address", None)
    if host_ip is not None and host_ip.startswith("[") and host_ip.endswith("]"):
        # convert dask bracket format to proper IPv6 address.
        host_ip = host_ip[1:-1]
    if host_ip is not None:
        try:
            host_ip, port = distributed.comm.get_address_host_port(host_ip)
        except ValueError:
            pass

    if coll_config is None:
        coll_config = Config()
    if coll_config.tracker_host is not None:
        if host_ip is not None and coll_config.tracker_host != host_ip:
            raise ValueError(
                "Conflicting host IP addresses from the dask configuration and the "
                f"collective configuration: {host_ip} v.s. {coll_config.tracker_host}."
            )
        host_ip = coll_config.tracker_host
    if coll_config.tracker_port is not None:
        if (
            port != 0
            and port is not None
            and coll_config.tracker_port != 0
            and port != coll_config.tracker_port
        ):
            raise ValueError(
                "Conflicting ports from the dask configuration and the "
                f"collective configuration: {port} v.s. {coll_config.tracker_port}."
            )
        port = coll_config.tracker_port

    return host_ip, port
