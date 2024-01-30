"""Utilities for the XGBoost Dask interface."""

import logging
from typing import TYPE_CHECKING, Any, Dict

LOGGER = logging.getLogger("[xgboost.dask]")


if TYPE_CHECKING:
    import distributed


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
