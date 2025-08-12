# pylint: disable=invalid-name
"""Utilities for the XGBoost Dask interface."""

import logging
import warnings
from functools import cache as fcache
from typing import Any, Dict, Optional, Tuple

import dask
import distributed
from packaging.version import Version
from packaging.version import parse as parse_version

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
    dconfig: Optional[Dict[str, Any]], coll_cfg: Config
) -> Tuple[Optional[str], int]:
    """Get the tracker address from the optional user configuration.

    Parameters
    ----------
    dconfig :
        Dask global configuration.

    coll_cfg :
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
            warnings.warn(
                (
                    "Use `coll_cfg` instead of the Dask global configuration store"
                    f" for the XGBoost tracker configuration: {k}."
                ),
                FutureWarning,
            )
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

    if coll_cfg is None:
        coll_cfg = Config()
    if coll_cfg.tracker_host_ip is not None:
        if host_ip is not None and coll_cfg.tracker_host_ip != host_ip:
            raise ValueError(
                "Conflicting host IP addresses from the dask configuration and the "
                f"collective configuration: {host_ip} v.s. {coll_cfg.tracker_host_ip}."
            )
        host_ip = coll_cfg.tracker_host_ip
    if coll_cfg.tracker_port is not None:
        if (
            port != 0
            and port is not None
            and coll_cfg.tracker_port != 0
            and port != coll_cfg.tracker_port
        ):
            raise ValueError(
                "Conflicting ports from the dask configuration and the "
                f"collective configuration: {port} v.s. {coll_cfg.tracker_port}."
            )
        port = coll_cfg.tracker_port

    return host_ip, port


@fcache
def _DASK_VERSION() -> Version:
    return parse_version(dask.__version__)


@fcache
def _DASK_2024_12_1() -> bool:
    return _DASK_VERSION() >= parse_version("2024.12.1")


@fcache
def _DASK_2025_3_0() -> bool:
    return _DASK_VERSION() >= parse_version("2025.3.0")
