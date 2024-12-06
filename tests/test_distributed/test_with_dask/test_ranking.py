"""Copyright 2019-2024, XGBoost contributors"""

import os
from typing import Generator

import numpy as np
import pytest
import scipy.sparse
from dask import dataframe as dd
from distributed import Client, LocalCluster

from xgboost import dask as dxgb
from xgboost import testing as tm
from xgboost.testing import dask as dtm


@pytest.fixture(scope="module")
def cluster() -> Generator:
    n_threads = os.cpu_count()
    assert n_threads is not None
    with LocalCluster(
        n_workers=2, threads_per_worker=n_threads // 2, dashboard_address=":0"
    ) as dask_cluster:
        yield dask_cluster


@pytest.fixture
def client(cluster: LocalCluster) -> Generator:
    with Client(cluster) as dask_client:
        yield dask_client


def test_dask_ranking(client: Client) -> None:
    dpath = "demo/"
    mq2008 = tm.data.get_mq2008(dpath)
    data = []
    for d in mq2008:
        if isinstance(d, scipy.sparse.csr_matrix):
            d[d == 0] = np.inf
            d = d.toarray()
            d[d == 0] = np.nan
            d[np.isinf(d)] = 0
            data.append(dd.from_array(d, chunksize=32))
        else:
            data.append(dd.from_array(d, chunksize=32))

    (
        x_train,
        y_train,
        qid_train,
        x_test,
        y_test,
        qid_test,
        x_valid,
        y_valid,
        qid_valid,
    ) = data
    qid_train = qid_train.astype(np.uint32)
    qid_valid = qid_valid.astype(np.uint32)
    qid_test = qid_test.astype(np.uint32)

    rank = dxgb.DaskXGBRanker(
        n_estimators=2500,
        eval_metric=["ndcg"],
        early_stopping_rounds=10,
        allow_group_split=True,
    )
    rank.fit(
        x_train,
        y_train,
        qid=qid_train,
        eval_set=[(x_test, y_test), (x_train, y_train)],
        eval_qid=[qid_test, qid_train],
        verbose=True,
    )
    assert rank.n_features_in_ == 46
    assert rank.best_score > 0.98


@pytest.mark.filterwarnings("error")
def test_no_group_split(client: Client) -> None:
    dtm.check_no_group_split(client, "cpu")
