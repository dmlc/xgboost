import multiprocessing
import socket
import sys
from threading import Thread

import numpy as np
import pytest

import xgboost as xgb
from xgboost import RabitTracker, build_info, federated
from xgboost import testing as tm


def run_rabit_worker(rabit_env, world_size):
    with xgb.collective.CommunicatorContext(**rabit_env):
        assert xgb.collective.get_world_size() == world_size
        assert xgb.collective.is_distributed()
        assert xgb.collective.get_processor_name() == socket.gethostname()
        ret = xgb.collective.broadcast("test1234", 0)
        assert str(ret) == "test1234"
        ret = xgb.collective.allreduce(np.asarray([1, 2, 3]), xgb.collective.Op.SUM)
        assert np.array_equal(ret, np.asarray([2, 4, 6]))


def test_rabit_communicator() -> None:
    world_size = 2
    tracker = RabitTracker(host_ip="127.0.0.1", n_workers=world_size)
    tracker.start()
    workers = []
    for _ in range(world_size):
        worker = multiprocessing.Process(
            target=run_rabit_worker, args=(tracker.worker_args(), world_size)
        )
        workers.append(worker)
        worker.start()
    for worker in workers:
        worker.join()
        assert worker.exitcode == 0


def run_federated_worker(port: int, world_size: int, rank: int) -> None:
    with xgb.collective.CommunicatorContext(
        dmlc_communicator="federated",
        federated_server_address=f"localhost:{port}",
        federated_world_size=world_size,
        federated_rank=rank,
    ):
        assert xgb.collective.get_world_size() == world_size
        assert xgb.collective.is_distributed()
        assert xgb.collective.get_processor_name() == f"rank:{rank}"
        bret = xgb.collective.broadcast("test1234", 0)
        assert str(bret) == "test1234"
        aret = xgb.collective.allreduce(np.asarray([1, 2, 3]), xgb.collective.Op.SUM)
        assert np.array_equal(aret, np.asarray([2, 4, 6]))


@pytest.mark.skipif(**tm.skip_win())
def test_federated_communicator():
    if not build_info()["USE_FEDERATED"]:
        pytest.skip("XGBoost not built with federated learning enabled")

    port = 9091
    world_size = 2
    tracker = multiprocessing.Process(
        target=federated.run_federated_server,
        kwargs={"port": port, "n_workers": world_size, "blocking": False},
    )
    tracker.start()
    if not tracker.is_alive():
        raise Exception("Error starting Federated Learning server")

    workers = []
    for rank in range(world_size):
        worker = multiprocessing.Process(
            target=run_federated_worker, args=(port, world_size, rank)
        )
        workers.append(worker)
        worker.start()
    for worker in workers:
        worker.join()
        assert worker.exitcode == 0
