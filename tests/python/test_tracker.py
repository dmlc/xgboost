from xgboost import RabitTracker
import xgboost as xgb
import pytest
import testing as tm
import numpy as np
import sys

if sys.platform.startswith("win"):
    pytest.skip("Skipping dask tests on Windows", allow_module_level=True)


def test_rabit_tracker():
    tracker = RabitTracker(host_ip='127.0.0.1', n_workers=1)
    tracker.start(1)
    worker_env = tracker.worker_envs()
    rabit_env = []
    for k, v in worker_env.items():
        rabit_env.append(f"{k}={v}".encode())
    xgb.rabit.init(rabit_env)
    ret = xgb.rabit.broadcast('test1234', 0)
    assert str(ret) == 'test1234'
    xgb.rabit.finalize()


def run_rabit_ops(client, n_workers):
    from test_with_dask import _get_client_workers
    from xgboost.dask import RabitContext, _get_rabit_args
    from xgboost import rabit

    workers = _get_client_workers(client)
    rabit_args = client.sync(_get_rabit_args, len(workers), None, client)
    assert not rabit.is_distributed()
    n_workers_from_dask = len(workers)
    assert n_workers == n_workers_from_dask

    def local_test(worker_id):
        with RabitContext(rabit_args):
            a = 1
            assert rabit.is_distributed()
            a = np.array([a])
            reduced = rabit.allreduce(a, rabit.Op.SUM)
            assert reduced[0] == n_workers

            worker_id = np.array([worker_id])
            reduced = rabit.allreduce(worker_id, rabit.Op.MAX)
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
