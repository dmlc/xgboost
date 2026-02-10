"""Shared fixtures for Dask tests."""

from __future__ import annotations

import os
from typing import Any, Dict, Generator

import pytest
from distributed import Client, LocalCluster


@pytest.fixture(scope="session")
def client_kwargs(request: pytest.FixtureRequest) -> Dict[str, Any]:
    n_threads = os.cpu_count()
    assert n_threads is not None
    kwargs: Dict[str, Any] = {
        "n_workers": 2,
        "threads_per_worker": max(1, n_threads // 2),
        "dashboard_address": ":0",
    }
    if hasattr(request, "param"):
        kwargs.update(request.param)
    return kwargs


@pytest.fixture(scope="session")
def client(client_kwargs: Dict[str, Any]) -> Generator[Client, None, None]:
    with LocalCluster(**client_kwargs) as dask_cluster:
        with Client(dask_cluster) as dask_client:
            yield dask_client


@pytest.fixture(autouse=True)
def client_as_current(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    for name in ("client", "client_one_worker"):
        if name in request.fixturenames:
            dask_client = request.getfixturevalue(name)
            with dask_client.as_current():
                yield
            return
    yield


@pytest.fixture(scope="session")
def client_one_worker() -> Generator[Client, None, None]:
    n_threads = os.cpu_count()
    assert n_threads is not None
    with LocalCluster(
        n_workers=1, threads_per_worker=max(1, n_threads), dashboard_address=":0"
    ) as dask_cluster:
        with Client(dask_cluster) as dask_client:
            yield dask_client
