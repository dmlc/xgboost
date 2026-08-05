import socket
from dataclasses import asdict

import numpy as np
import pytest
import xgboost as xgb
from xgboost import RabitTracker
from xgboost import testing as tm
from xgboost.collective import Config


def run_rabit_worker(rabit_env: dict, world_size: int) -> int:
    with xgb.collective.CommunicatorContext(**rabit_env):
        assert xgb.collective.get_world_size() == world_size
        assert xgb.collective.is_distributed()
        assert xgb.collective.get_processor_name() == socket.gethostname()
        ret = xgb.collective.broadcast("test1234", 0)
        assert str(ret) == "test1234"
        reduced = xgb.collective.allreduce(np.asarray([1, 2, 3]), xgb.collective.Op.SUM)
        assert np.array_equal(reduced, np.asarray([2, 4, 6]))
    return 0


@pytest.mark.skipif(**tm.no_loky())
def test_rabit_communicator() -> None:
    from loky import get_reusable_executor

    world_size = 2
    tracker = RabitTracker(host_ip="127.0.0.1", n_workers=world_size)
    tracker.start()
    workers = []

    with get_reusable_executor(max_workers=world_size) as pool:
        for _ in range(world_size):
            worker = pool.submit(
                run_rabit_worker, rabit_env=tracker.worker_args(), world_size=world_size
            )
            workers.append(worker)

        for worker in workers:
            assert worker.result() == 0


def test_config_serialization() -> None:
    cfg = Config(retry=1, timeout=2, tracker_host_ip="127.0.0.1", tracker_port=None)
    cfg1 = Config(**asdict(cfg))
    assert cfg == cfg1
