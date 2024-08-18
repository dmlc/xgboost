import re
import sys
from functools import partial, update_wrapper
from typing import Dict, Union

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies

import xgboost as xgb
from xgboost import RabitTracker, collective
from xgboost import testing as tm


def test_rabit_tracker():
    tracker = RabitTracker(host_ip="127.0.0.1", n_workers=1)
    tracker.start()
    with collective.CommunicatorContext(**tracker.worker_args()):
        ret = collective.broadcast("test1234", 0)
        assert str(ret) == "test1234"


@pytest.mark.skipif(**tm.not_linux())
def test_socket_error():
    tracker = RabitTracker(host_ip="127.0.0.1", n_workers=2)
    tracker.start()
    env = tracker.worker_args()
    env["dmlc_tracker_port"] = 0
    env["dmlc_retry"] = 1
    with pytest.raises(ValueError, match="Failed to bootstrap the communication."):
        with collective.CommunicatorContext(**env):
            pass
    with pytest.raises(ValueError):
        tracker.free()


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



def run_allreduce(pool, n_workers: int) -> None:
    tracker = RabitTracker(host_ip="127.0.0.1", n_workers=n_workers)
    tracker.start()
    args = tracker.worker_args()

    def local_test(worker_id: int, rabit_args: Dict[str, Union[str, int]]) -> None:
        x = np.full(shape=(1024 * 1024 * 32), fill_value=1.0)
        with collective.CommunicatorContext(**rabit_args):
            k = np.asarray([1.0])
            for i in range(128):
                m = collective.allreduce(k, collective.Op.SUM)
                assert m == n_workers

            y = collective.allreduce(x, collective.Op.SUM)
            np.testing.assert_allclose(y, np.full_like(y, fill_value=float(n_workers)))

    fn = update_wrapper(partial(local_test, rabit_args=args), local_test)
    results = pool.map(fn, range(n_workers))
    for r in results:
        assert r is None


@pytest.mark.skipif(**tm.no_loky())
def test_allreduce() -> None:
    from loky import get_reusable_executor

    n_workers = 4
    n_trials = 2
    for _ in range(n_trials):
        with get_reusable_executor(max_workers=n_workers) as pool:
            run_allreduce(pool, n_workers)


def run_broadcast(pool, n_workers: int) -> None:
    tracker = RabitTracker(host_ip="127.0.0.1", n_workers=n_workers)
    tracker.start()
    args = tracker.worker_args()

    def local_test(worker_id: int, rabit_args: Dict[str, Union[str, int]]):
        with collective.CommunicatorContext(**rabit_args):
            res = collective.broadcast(17, 0)
            return res

    fn = update_wrapper(partial(local_test, rabit_args=args), local_test)
    results = pool.map(fn, range(n_workers))
    np.testing.assert_allclose(np.array(list(results)), 17)


@pytest.mark.skipif(**tm.no_loky())
def test_broadcast():
    from loky import get_reusable_executor

    n_workers = 4
    n_trials = 2

    for _ in range(n_trials):
        with get_reusable_executor(max_workers=n_workers) as pool:
            run_broadcast(pool, n_workers)


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


@pytest.mark.skipif(**tm.no_dask())
def test_rank_assignment() -> None:
    from distributed import Client, LocalCluster

    def local_test(worker_id):
        with xgb.dask.CommunicatorContext(**args) as ctx:
            task_id = ctx["DMLC_TASK_ID"]
            matched = re.search(".*-([0-9]).*", task_id)
            rank = collective.get_rank()
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


ops_strategy = strategies.lists(
    strategies.sampled_from(["broadcast", "allreduce_max", "allreduce_sum"])
)


@pytest.mark.skipif(**tm.no_loky())
@given(ops=ops_strategy, size=strategies.integers(2**4, 2**16))
@settings(
    deadline=None,
    print_blob=True,
    max_examples=10,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_ops_restart_comm(ops, size) -> None:
    from loky import get_reusable_executor

    n_workers = 8

    def local_test(w: int, rabit_args: Dict[str, Union[str, int]]) -> None:
        a = np.arange(0, n_workers)
        with collective.CommunicatorContext(**rabit_args):
            for op in ops:
                if op == "broadcast":
                    b = collective.broadcast(a, root=1)
                    np.testing.assert_allclose(b, a)
                elif op == "allreduce_max":
                    b = collective.allreduce(a, collective.Op.MAX)
                    np.testing.assert_allclose(b, a)
                elif op == "allreduce_sum":
                    b = collective.allreduce(a, collective.Op.SUM)
                    np.testing.assert_allclose(a * n_workers, b)
                else:
                    raise ValueError()

    with get_reusable_executor(max_workers=n_workers) as pool:
        tracker = RabitTracker(host_ip="127.0.0.1", n_workers=n_workers)
        tracker.start()
        args = tracker.worker_args()

        fn = update_wrapper(partial(local_test, rabit_args=args), local_test)
        results = pool.map(fn, range(n_workers))

        for r in results:
            assert r is None


@pytest.mark.skipif(**tm.no_loky())
def test_ops_reuse_comm() -> None:
    from loky import get_reusable_executor

    rng = np.random.default_rng(1994)
    n_examples = 10
    ops = rng.choice(
        ["broadcast", "allreduce_sum", "allreduce_max"], size=n_examples
    ).tolist()

    n_workers = 8
    n_trials = 8

    def local_test(w: int, rabit_args: Dict[str, Union[str, int]]) -> None:
        a = np.arange(0, n_workers)

        with collective.CommunicatorContext(**rabit_args):
            for op in ops:
                if op == "broadcast":
                    b = collective.broadcast(a, root=1)
                    assert np.allclose(b, a)
                elif op == "allreduce_max":
                    c = np.full_like(a, collective.get_rank())
                    b = collective.allreduce(c, collective.Op.MAX)
                    assert np.allclose(b, n_workers - 1), b
                elif op == "allreduce_sum":
                    b = collective.allreduce(a, collective.Op.SUM)
                    assert np.allclose(a * 8, b)
                else:
                    raise ValueError()

    with get_reusable_executor(max_workers=n_workers) as pool:
        for _ in range(n_trials):
            tracker = RabitTracker(host_ip="127.0.0.1", n_workers=n_workers)
            tracker.start()
            args = tracker.worker_args()

            fn = update_wrapper(partial(local_test, rabit_args=args), local_test)
            results = pool.map(fn, range(n_workers))
            for r in results:
                assert r is None
