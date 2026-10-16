"""Utilities for the XGBoost Dask interface."""

import logging
from functools import cache as fcache
from typing import Any, Dict

import dask
import distributed
from packaging.version import Version
from packaging.version import parse as parse_version

LOGGER = logging.getLogger("[xgboost.dask]")


def get_n_threads(local_param: Dict[str, Any], worker: "distributed.Worker") -> int:
    """Get the number of threads from a worker and the user-supplied parameters."""
    # dask worker nthreads
    dwnt = worker.state.nthreads
    n_threads = None
    for p in ["nthread", "n_jobs"]:
        if local_param.get(p, None) is not None and local_param.get(p, dwnt) != dwnt:
            LOGGER.info("Overriding `nthreads` defined in dask worker.")
            n_threads = local_param[p]
            break
    if n_threads == 0 or n_threads is None:
        n_threads = dwnt
    return n_threads


@fcache
def _DASK_VERSION() -> Version:
    return parse_version(dask.__version__)


@fcache
def _DASK_2024_12_1() -> bool:
    return _DASK_VERSION() >= parse_version("2024.12.1")


@fcache
def _DASK_2025_3_0() -> bool:
    return _DASK_VERSION() >= parse_version("2025.3.0")
