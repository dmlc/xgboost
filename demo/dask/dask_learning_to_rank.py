"""
Learning to rank with the Dask Interface
========================================

  .. versionadded:: 3.0.0

This is a demonstration of using XGBoost for learning to rank tasks using the
MSLR_10k_letor dataset. For more infomation about the dataset, please visit its
`description page <https://www.microsoft.com/en-us/research/project/mslr/>`_.

See :ref:`ltr-dist` for a general description for distributed learning to rank and
:ref:`ltr-dask` for Dask-specific features.

"""

from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from typing import Generator

import dask
import numpy as np
from dask import dataframe as dd
from distributed import Client, LocalCluster, wait
from sklearn.datasets import load_svmlight_file

from xgboost import dask as dxgb


def load_mslr_10k(
    device: str, data_path: str, cache_path: str
) -> tuple[dd.DataFrame, dd.DataFrame, dd.DataFrame]:
    """Load the MSLR10k dataset from data_path and save parquet files in the cache_path."""
    root_path = os.path.expanduser(args.data)
    cache_path = os.path.expanduser(args.cache)

    # Use only the Fold1 for demo:
    # Train,      Valid, Test
    # {S1,S2,S3}, S4,    S5
    fold = 1

    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
        fold_path = os.path.join(root_path, f"Fold{fold}")
        train_path = os.path.join(fold_path, "train.txt")
        valid_path = os.path.join(fold_path, "vali.txt")
        test_path = os.path.join(fold_path, "test.txt")

        X_train, y_train, qid_train = load_svmlight_file(
            train_path, query_id=True, dtype=np.float32
        )
        columns = [f"f{i}" for i in range(X_train.shape[1])]
        X_train = dd.from_array(X_train.toarray(), columns=columns)
        y_train = y_train.astype(np.int32)
        qid_train = qid_train.astype(np.int32)

        X_train["y"] = dd.from_array(y_train)
        X_train["qid"] = dd.from_array(qid_train)
        X_train.to_parquet(os.path.join(cache_path, "train"), engine="pyarrow")

        X_valid, y_valid, qid_valid = load_svmlight_file(
            valid_path, query_id=True, dtype=np.float32
        )
        X_valid = dd.from_array(X_valid.toarray(), columns=columns)
        y_valid = y_valid.astype(np.int32)
        qid_valid = qid_valid.astype(np.int32)

        X_valid["y"] = dd.from_array(y_valid)
        X_valid["qid"] = dd.from_array(qid_valid)
        X_valid.to_parquet(os.path.join(cache_path, "valid"), engine="pyarrow")

        X_test, y_test, qid_test = load_svmlight_file(
            test_path, query_id=True, dtype=np.float32
        )

        X_test = dd.from_array(X_test.toarray(), columns=columns)
        y_test = y_test.astype(np.int32)
        qid_test = qid_test.astype(np.int32)

        X_test["y"] = dd.from_array(y_test)
        X_test["qid"] = dd.from_array(qid_test)
        X_test.to_parquet(os.path.join(cache_path, "test"), engine="pyarrow")

    df_train = dd.read_parquet(
        os.path.join(cache_path, "train"), calculate_divisions=True
    )
    df_valid = dd.read_parquet(
        os.path.join(cache_path, "valid"), calculate_divisions=True
    )
    df_test = dd.read_parquet(
        os.path.join(cache_path, "test"), calculate_divisions=True
    )

    return df_train, df_valid, df_test


def ranking_demo(client: Client, args: argparse.Namespace) -> None:
    """Learning to rank with data sorted locally."""
    df_tr, df_va, _ = load_mslr_10k(args.device, args.data, args.cache)

    X_train: dd.DataFrame = df_tr[df_tr.columns.difference(["y", "qid"])]
    y_train = df_tr[["y", "qid"]]
    Xy_train = dxgb.DaskQuantileDMatrix(client, X_train, y_train.y, qid=y_train.qid)

    X_valid: dd.DataFrame = df_va[df_va.columns.difference(["y", "qid"])]
    y_valid = df_va[["y", "qid"]]
    Xy_valid = dxgb.DaskQuantileDMatrix(
        client, X_valid, y_valid.y, qid=y_valid.qid, ref=Xy_train
    )
    # Upon training, you will see a performance warning about sorting data based on
    # query groups.
    dxgb.train(
        client,
        {"objective": "rank:ndcg", "device": args.device},
        Xy_train,
        evals=[(Xy_train, "Train"), (Xy_valid, "Valid")],
        num_boost_round=100,
    )


def ranking_wo_split_demo(client: Client, args: argparse.Namespace) -> None:
    """Learning to rank with data partitioned according to query groups."""
    df_tr, df_va, df_te = load_mslr_10k(args.device, args.data, args.cache)

    X_tr = df_tr[df_tr.columns.difference(["y", "qid"])]
    X_va = df_va[df_va.columns.difference(["y", "qid"])]

    # `allow_group_split=False` makes sure data is partitioned according to the query
    # groups.
    ltr = dxgb.DaskXGBRanker(allow_group_split=False, device=args.device)
    ltr.client = client
    ltr = ltr.fit(
        X_tr,
        df_tr.y,
        qid=df_tr.qid,
        eval_set=[(X_tr, df_tr.y), (X_va, df_va.y)],
        eval_qid=[df_tr.qid, df_va.qid],
        verbose=True,
    )

    df_te = df_te.persist()
    wait([df_te])

    X_te = df_te[df_te.columns.difference(["y", "qid"])]
    predt = ltr.predict(X_te)
    y = client.compute(df_te.y)
    wait([predt, y])


@contextmanager
def gen_client(device: str) -> Generator[Client, None, None]:
    match device:
        case "cuda":
            from dask_cuda import LocalCUDACluster

            with LocalCUDACluster() as cluster:
                with Client(cluster) as client:
                    with dask.config.set(
                        {
                            "array.backend": "cupy",
                            "dataframe.backend": "cudf",
                        }
                    ):
                        yield client
        case "cpu":
            with LocalCluster() as cluster:
                with Client(cluster) as client:
                    yield client


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demonstration of learning to rank using XGBoost."
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Root directory of the MSLR-WEB10K data.",
        required=True,
    )
    parser.add_argument(
        "--cache",
        type=str,
        help="Directory for caching processed data.",
        required=True,
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Flag to indicate query groups should not be split.",
    )
    args = parser.parse_args()

    with gen_client(args.device) as client:
        if args.no_split:
            ranking_wo_split_demo(client, args)
        else:
            ranking_demo(client, args)
