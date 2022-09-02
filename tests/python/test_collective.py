import multiprocessing
import socket
import sys

import numpy as np
import pytest

import xgboost as xgb
from xgboost import RabitTracker
from xgboost import collective

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
