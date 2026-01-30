"""Shared fixtures for Dask tests."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict, Generator, Iterator

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
def cluster(client_kwargs: Dict[str, Any]) -> Generator[LocalCluster, None, None]:
    with LocalCluster(**client_kwargs) as dask_cluster:
        yield dask_cluster


@pytest.fixture(scope="session")
def client(cluster: LocalCluster) -> Generator[Client, None, None]:
    with Client(cluster) as dask_client:
        yield dask_client


@pytest.fixture(scope="session")
def client_one_worker() -> Generator[Client, None, None]:
    n_threads = os.cpu_count()
    assert n_threads is not None
    with LocalCluster(
        n_workers=1, threads_per_worker=max(1, n_threads), dashboard_address=":0"
    ) as dask_cluster:
        with Client(dask_cluster) as dask_client:
            yield dask_client


@pytest.fixture
def client_factory() -> Any:
    @contextmanager
    def _factory(**kwargs: Any) -> Iterator[Client]:
        with LocalCluster(**kwargs) as dask_cluster:
            with Client(dask_cluster) as dask_client:
                yield dask_client

    return _factory


@pytest.fixture
def client_from_cluster() -> Any:
    @contextmanager
    def _factory(cluster: LocalCluster) -> Iterator[Client]:
        with Client(cluster) as dask_client:
            yield dask_client

    return _factory
