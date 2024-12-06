"""Copyright 2024, XGBoost contributors"""

import dask
import pytest
from distributed import Client

from xgboost.testing import dask as dtm


@pytest.mark.filterwarnings("error")
def test_no_group_split(local_cuda_client: Client) -> None:
    with dask.config.set(
        {
            "array.backend": "cupy",
            "dataframe.backend": "cudf",
        }
    ):
        dtm.check_no_group_split(local_cuda_client, "cuda")
