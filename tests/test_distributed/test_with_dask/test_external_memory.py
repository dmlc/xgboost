"""Copyright 2024-2026, XGBoost contributors"""

import pytest
from distributed import Client, Scheduler, Worker
from distributed.utils_test import gen_cluster
from xgboost.testing.dask import check_external_memory, get_rabit_args


@pytest.mark.parametrize("is_qdm", [True, False])
@gen_cluster(client=True)
async def test_external_memory(
    client: Client, s: Scheduler, a: Worker, b: Worker, is_qdm: bool
) -> None:
    workers = [a.address, b.address]
    n_workers = len(workers)
    args = await get_rabit_args(client, n_workers)

    futs = [
        client.submit(
            check_external_memory,
            worker_id,
            n_workers=n_workers,
            device="cpu",
            comm_args=args,
            is_qdm=is_qdm,
            workers=[worker],
            allow_other_workers=False,
        )
        for worker_id, worker in enumerate(workers)
    ]
    await client.gather(futs)
