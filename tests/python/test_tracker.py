import re
import sys

import numpy as np
import pytest

import xgboost as xgb
from xgboost import RabitTracker, collective
from xgboost import testing as tm

if sys.platform.startswith("win"):
    pytest.skip("Skipping dask tests on Windows", allow_module_level=True)


def test_rabit_tracker():
    tracker = RabitTracker(host_ip="127.0.0.1", n_workers=1)
    tracker.start(1)
    with xgb.collective.CommunicatorContext(**tracker.worker_envs()):
        ret = xgb.collective.broadcast("test1234", 0)
        assert str(ret) == "test1234"


def run_rabit_ops(client, n_workers):
    from xgboost.dask import CommunicatorContext, _get_dask_config, _get_rabit_args

    workers = tm.get_client_workers(client)
    rabit_args = client.sync(_get_rabit_args, len(workers), _get_dask_config(), client)
    assert not collective.is_distributed()
    n_workers_from_dask = len(workers)
    assert n_workers == n_workers_from_dask

    def local_test(worker_id):
        with CommunicatorContext(**rabit_args):
            a = 1
            assert collective.is_distributed()
            a = np.array([a])
            reduced = collective.allreduce(a, collective.Op.SUM)
            assert reduced[0] == n_workers

            worker_id = np.array([worker_id])
            reduced = collective.allreduce(worker_id, collective.Op.MAX)
            assert reduced == n_workers - 1

            return 1

    futures = client.map(local_test, range(len(workers)), workers=workers)
    results = client.gather(futures)
    assert sum(results) == n_workers


@pytest.mark.skipif(**tm.no_dask())
def test_rabit_ops():
    from distributed import Client, LocalCluster

    n_workers = 3
    with LocalCluster(n_workers=n_workers) as cluster:
        with Client(cluster) as client:
            run_rabit_ops(client, n_workers)


@pytest.mark.skipif(**tm.no_ipv6())
@pytest.mark.skipif(**tm.no_dask())
def test_rabit_ops_ipv6():
    import dask
    from distributed import Client, LocalCluster

    n_workers = 3
    with dask.config.set({"xgboost.scheduler_address": "[::1]"}):
        with LocalCluster(n_workers=n_workers, host="[::1]") as cluster:
            with Client(cluster) as client:
                run_rabit_ops(client, n_workers)


def test_rank_assignment() -> None:
    from distributed import Client, LocalCluster

    def local_test(worker_id):
        with xgb.dask.CommunicatorContext(**args) as ctx:
            task_id = ctx["DMLC_TASK_ID"]
            matched = re.search(".*-([0-9]).*", task_id)
            rank = xgb.collective.get_rank()
            # As long as the number of workers is lesser than 10, rank and worker id
            # should be the same
            assert rank == int(matched.group(1))

    with LocalCluster(n_workers=8) as cluster:
        with Client(cluster) as client:
            workers = tm.get_client_workers(client)
            args = client.sync(
                xgb.dask._get_rabit_args,
                len(workers),
                None,
                client,
            )

            futures = client.map(local_test, range(len(workers)), workers=workers)
            client.gather(futures)
