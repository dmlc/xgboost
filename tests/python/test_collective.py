import multiprocessing
import socket
import sys
import time

import numpy as np
import pytest

import xgboost as xgb
from xgboost import RabitTracker, build_info, federated

if sys.platform.startswith("win"):
    pytest.skip("Skipping collective tests on Windows", allow_module_level=True)


def run_rabit_worker(rabit_env, world_size):
    with xgb.collective.CommunicatorContext(**rabit_env):
        assert xgb.collective.get_world_size() == world_size
        assert xgb.collective.is_distributed()
        assert xgb.collective.get_processor_name() == socket.gethostname()
        ret = xgb.collective.broadcast('test1234', 0)
        assert str(ret) == 'test1234'
        ret = xgb.collective.allreduce(np.asarray([1, 2, 3]), xgb.collective.Op.SUM)
        assert np.array_equal(ret, np.asarray([2, 4, 6]))


def test_rabit_communicator():
    world_size = 2
    tracker = RabitTracker(host_ip='127.0.0.1', n_workers=world_size)
    tracker.start(world_size)
    workers = []
    for _ in range(world_size):
        worker = multiprocessing.Process(target=run_rabit_worker,
                                         args=(tracker.worker_envs(), world_size))
        workers.append(worker)
        worker.start()
    for worker in workers:
        worker.join()
        assert worker.exitcode == 0


# TODO(rongou): remove this once we remove the rabit api.
def run_rabit_api_worker(rabit_env, world_size):
    with xgb.rabit.RabitContext(rabit_env):
        assert xgb.rabit.get_world_size() == world_size
        assert xgb.rabit.is_distributed()
        assert xgb.rabit.get_processor_name().decode() == socket.gethostname()
        ret = xgb.rabit.broadcast('test1234', 0)
        assert str(ret) == 'test1234'
        ret = xgb.rabit.allreduce(np.asarray([1, 2, 3]), xgb.rabit.Op.SUM)
        assert np.array_equal(ret, np.asarray([2, 4, 6]))


# TODO(rongou): remove this once we remove the rabit api.
def test_rabit_api():
    world_size = 2
    tracker = RabitTracker(host_ip='127.0.0.1', n_workers=world_size)
    tracker.start(world_size)
    rabit_env = []
    for k, v in tracker.worker_envs().items():
        rabit_env.append(f"{k}={v}".encode())
    workers = []
    for _ in range(world_size):
        worker = multiprocessing.Process(target=run_rabit_api_worker,
                                         args=(rabit_env, world_size))
        workers.append(worker)
        worker.start()
    for worker in workers:
        worker.join()
        assert worker.exitcode == 0


def run_federated_worker(port, world_size, rank):
    with xgb.collective.CommunicatorContext(xgboost_communicator='federated',
                                            federated_server_address=f'localhost:{port}',
                                            federated_world_size=world_size,
                                            federated_rank=rank):
        assert xgb.collective.get_world_size() == world_size
        assert xgb.collective.is_distributed()
        assert xgb.collective.get_processor_name() == f'rank{rank}'
        ret = xgb.collective.broadcast('test1234', 0)
        assert str(ret) == 'test1234'
        ret = xgb.collective.allreduce(np.asarray([1, 2, 3]), xgb.collective.Op.SUM)
        assert np.array_equal(ret, np.asarray([2, 4, 6]))


def test_federated_communicator():
    if not build_info()["USE_FEDERATED"]:
        pytest.skip("XGBoost not built with federated learning enabled")

    port = 9091
    world_size = 2
    server = multiprocessing.Process(target=xgb.federated.run_federated_server, args=(port, world_size))
    server.start()
    time.sleep(1)
    if not server.is_alive():
        raise Exception("Error starting Federated Learning server")

    workers = []
    for rank in range(world_size):
        worker = multiprocessing.Process(target=run_federated_worker,
                                         args=(port, world_size, rank))
        workers.append(worker)
        worker.start()
    for worker in workers:
        worker.join()
        assert worker.exitcode == 0
    server.terminate()
