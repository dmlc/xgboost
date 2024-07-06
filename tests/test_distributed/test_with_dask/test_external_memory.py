from typing import List, cast

import numpy as np
from distributed import Client, Scheduler, Worker, get_worker
from distributed.utils_test import gen_cluster

import xgboost as xgb
from xgboost import testing as tm
from xgboost.compat import concat


def run_external_memory(worker_id: int, n_workers: int, comm_args: dict) -> None:
    n_samples_per_batch = 32
    n_features = 4
    n_batches = 16
    use_cupy = False

    n_threads = get_worker().state.nthreads
    with xgb.collective.CommunicatorContext(dmlc_communicator="rabit", **comm_args):
        it = tm.IteratorForTest(
            *tm.make_batches(
                n_samples_per_batch,
                n_features,
                n_batches,
                use_cupy,
                random_state=worker_id,
            ),
            cache="cache",
        )
        Xy = xgb.DMatrix(it, nthread=n_threads)
        results: xgb.callback.TrainingCallback.EvalsLog = {}
        booster = xgb.train(
            {"tree_method": "hist", "nthread": n_threads},
            Xy,
            evals=[(Xy, "Train")],
            num_boost_round=32,
            evals_result=results,
        )
        assert tm.non_increasing(cast(List[float], results["Train"]["rmse"]))

    lx, ly, lw = [], [], []
    for i in range(n_workers):
        x, y, w = tm.make_batches(
            n_samples_per_batch,
            n_features,
            n_batches,
            use_cupy,
            random_state=i,
        )
        lx.extend(x)
        ly.extend(y)
        lw.extend(w)

    X = concat(lx)
    yconcat = concat(ly)
    wconcat = concat(lw)
    Xy = xgb.DMatrix(X, yconcat, wconcat, nthread=n_threads)

    results_local: xgb.callback.TrainingCallback.EvalsLog = {}
    booster = xgb.train(
        {"tree_method": "hist", "nthread": n_threads},
        Xy,
        evals=[(Xy, "Train")],
        num_boost_round=32,
        evals_result=results_local,
    )
    np.testing.assert_allclose(
        results["Train"]["rmse"], results_local["Train"]["rmse"], rtol=1e-4
    )


@gen_cluster(client=True)
async def test_external_memory(
    client: Client, s: Scheduler, a: Worker, b: Worker
) -> None:
    workers = tm.get_client_workers(client)
    args = await client.sync(
        xgb.dask._get_rabit_args,
        len(workers),
        None,
        client,
    )
    n_workers = len(workers)

    futs = client.map(
        run_external_memory, range(n_workers), n_workers=n_workers, comm_args=args
    )
    await client.gather(futs)
