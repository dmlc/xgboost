"""Copyright 2024, XGBoost contributors"""

import pytest
from distributed import Client, Scheduler, Worker
from distributed.utils_test import gen_cluster

from xgboost import testing as tm
from xgboost.testing.dask import check_external_memory, get_rabit_args


@pytest.mark.parametrize("is_qdm", [True, False])
@gen_cluster(client=True)
async def test_external_memory(
    client: Client, s: Scheduler, a: Worker, b: Worker, is_qdm: bool
) -> None:
    workers = tm.dask.get_client_workers(client)
    n_workers = len(workers)
    args = await get_rabit_args(client, n_workers)

    futs = client.map(
        check_external_memory,
        range(n_workers),
        n_workers=n_workers,
        device="cpu",
        comm_args=args,
        is_qdm=is_qdm,
    )
    await client.gather(futs)
