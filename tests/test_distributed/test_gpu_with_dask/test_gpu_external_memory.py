"""Copyright 2024-2025, XGBoost contributors"""

from functools import partial, update_wrapper
from typing import Any

import pytest
from dask_cuda import LocalCUDACluster
from distributed import Client

import xgboost as xgb
from xgboost import collective as coll
from xgboost import testing as tm
from xgboost.testing.dask import check_external_memory, get_rabit_args
from xgboost.tracker import RabitTracker


@pytest.mark.parametrize("is_qdm", [True, False])
def test_external_memory(is_qdm: bool) -> None:
    n_workers = 2
    with LocalCUDACluster(n_workers=2) as cluster:
        with Client(cluster) as client:
            args = get_rabit_args(client, 2)
            futs = client.map(
                check_external_memory,
                range(n_workers),
                n_workers=n_workers,
                device="cuda",
                comm_args=args,
                is_qdm=is_qdm,
            )
            client.gather(futs)


@pytest.mark.skipif(**tm.no_loky())
def test_extmem_qdm_distributed() -> None:
    from loky import get_reusable_executor

    n_samples_per_batch = 2048
    n_features = 128
    n_batches = 8

    def do_train(ordinal: int) -> None:
        it = tm.IteratorForTest(
            *tm.make_batches(n_samples_per_batch, n_features, n_batches, use_cupy=True),
            cache="cache",
            on_host=True,
        )

        Xy = xgb.ExtMemQuantileDMatrix(it)
        results: dict[str, Any] = {}
        booster = xgb.train(
            {"device": f"cuda:{ordinal}"},
            num_boost_round=2,
            dtrain=Xy,
            evals=[(Xy, "Train")],
            evals_result=results,
        )
        assert tm.non_increasing(results["Train"]["rmse"])

    tracker = RabitTracker(host_ip="127.0.0.1", n_workers=2)
    tracker.start()
    args = tracker.worker_args()

    def local_test(worker_id: int, rabit_args: dict) -> None:
        import cupy as cp

        cp.cuda.runtime.setDevice(worker_id)

        with coll.CommunicatorContext(**rabit_args, DMLC_TASK_ID=str(worker_id)):
            assert coll.get_rank() == worker_id
            do_train(coll.get_rank())

    n_workers = 2
    fn = update_wrapper(partial(local_test, rabit_args=args), local_test)
    with get_reusable_executor(max_workers=n_workers) as pool:
        results = pool.map(fn, range(n_workers))
