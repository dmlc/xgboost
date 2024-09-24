"""Copyright 2024, XGBoost contributors"""

import pytest
from dask_cuda import LocalCUDACluster
from distributed import Client

import xgboost as xgb
from xgboost.testing.dask import check_external_memory


@pytest.mark.parametrize("is_qdm", [True, False])
def test_external_memory(is_qdm: bool) -> None:
    n_workers = 2
    with LocalCUDACluster(n_workers=2) as cluster:
        with Client(cluster) as client:
            args = client.sync(
                xgb.dask._get_rabit_args,
                2,
                None,
                client,
            )

            futs = client.map(
                check_external_memory,
                range(n_workers),
                n_workers=n_workers,
                device="cuda",
                comm_args=args,
                is_qdm=is_qdm,
            )
            client.gather(futs)
