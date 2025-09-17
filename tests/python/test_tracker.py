import re
from functools import partial, update_wrapper
from platform import system
from typing import Dict, Union

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies

from xgboost import RabitTracker, collective
from xgboost import testing as tm


def test_rabit_tracker() -> None:
    tracker = RabitTracker(host_ip="127.0.0.1", n_workers=1)
    tracker.start()
    args = tracker.worker_args()
    port = args["dmlc_tracker_port"]
    with collective.CommunicatorContext(**tracker.worker_args()):
        ret = collective.broadcast("test1234", 0)
        assert str(ret) == "test1234"

    if system() == "Windows":
        pytest.skip("Windows is not supported.")

    with pytest.raises(ValueError, match="Failed to bind socket"):
        # Port is already being used
        RabitTracker(host_ip="127.0.0.1", port=port, n_workers=1)


@pytest.mark.skipif(**tm.not_linux())
def test_wait() -> None:
    tracker = RabitTracker(host_ip="127.0.0.1", n_workers=2)
    tracker.start()

    with pytest.raises(ValueError, match="Timeout waiting for the tracker"):
        tracker.wait_for(1)

    with pytest.raises(ValueError, match="Failed to accept"):
        tracker.free()


@pytest.mark.skipif(**tm.not_linux())
def test_socket_error() -> None:
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


def run_rabit_ops(pool, n_workers: int, address: str) -> None:
    tracker = RabitTracker(host_ip=address, n_workers=n_workers)
    tracker.start()
    args = tracker.worker_args()

    def local_test(worker_id: int, rabit_args: dict) -> int:
        with collective.CommunicatorContext(**rabit_args):
            a = 1
            assert collective.is_distributed()
            arr = np.array([a])
            reduced = collective.allreduce(arr, collective.Op.SUM)
            assert reduced[0] == n_workers

            arr = np.array([worker_id])
            reduced = collective.allreduce(arr, collective.Op.MAX)
            assert reduced == n_workers - 1

            return 1

    fn = update_wrapper(partial(local_test, rabit_args=args), local_test)
    results = pool.map(fn, range(n_workers))
    assert sum(results) == n_workers


@pytest.mark.skipif(**tm.no_loky())
def test_rabit_ops():
    from loky import get_reusable_executor

    n_workers = 4
    with get_reusable_executor(max_workers=n_workers) as pool:
        run_rabit_ops(pool, n_workers, "127.0.0.1")


@pytest.mark.skipif(**tm.no_ipv6())
@pytest.mark.skipif(**tm.no_loky())
def test_rabit_ops_ipv6():
    from loky import get_reusable_executor

    n_workers = 4
    with get_reusable_executor(max_workers=n_workers) as pool:
        run_rabit_ops(pool, n_workers, "::1")


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


@pytest.mark.skipif(**tm.no_dask())
def test_rank_assignment() -> None:
    from distributed import Client, LocalCluster

    from xgboost import dask as dxgb
    from xgboost.testing.dask import get_rabit_args

    def local_test(worker_id):
        with dxgb.CommunicatorContext(**args) as ctx:
            task_id = ctx["DMLC_TASK_ID"]
            matched = re.search(".*-([0-9]).*", task_id)
            rank = collective.get_rank()
            # As long as the number of workers is lesser than 10, rank and worker id
            # should be the same
            assert rank == int(matched.group(1))

    with LocalCluster(n_workers=8) as cluster:
        with Client(cluster) as client:
            workers = tm.dask.get_client_workers(client)
            args = get_rabit_args(client, len(workers))
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
