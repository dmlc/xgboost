from xgboost import RabitTracker
import xgboost as xgb
import pytest
import testing as tm
import numpy as np


def test_rabit_tracker():
    tracker = RabitTracker(hostIP='127.0.0.1', nslave=1)
    tracker.start(1)
    rabit_env = [
        str.encode('DMLC_TRACKER_URI=127.0.0.1'),
        str.encode('DMLC_TRACKER_PORT=9091'),
        str.encode('DMLC_TASK_ID=0')]
    xgb.rabit.init(rabit_env)
    ret = xgb.rabit.broadcast('test1234', 0)
    assert str(ret) == 'test1234'
    xgb.rabit.finalize()


def run_rabit_ops(client, n_workers):
    from xgboost.dask import RabitContext, _get_rabit_args, _get_client_workers
    from xgboost import rabit

    workers = list(_get_client_workers(client).keys())
    rabit_args = _get_rabit_args(workers, client)
    assert not rabit.is_distributed()

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
