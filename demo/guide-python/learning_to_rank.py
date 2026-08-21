"""
Getting started with learning to rank
=====================================

  .. versionadded:: 2.0.0

This is a demonstration of using XGBoost for learning to rank tasks using the
MSLR_10k_letor dataset. For more infomation about the dataset, please visit its
`description page <https://www.microsoft.com/en-us/research/project/mslr/>`_.

This demo contains a basic example of using XGBoost to train on relevance degrees.

For an overview of learning to rank in XGBoost, please see :doc:`Learning to Rank
</tutorials/learning_to_rank>`.
"""

from __future__ import annotations

import argparse
import os
import pickle as pkl

import numpy as np
import xgboost as xgb
from sklearn.datasets import load_svmlight_file
from xgboost.testing.data import RelDataCV


def load_mslr_10k(data_path: str, cache_path: str) -> RelDataCV:
    """Load the MSLR10k dataset from data_path and cache a pickle object in cache_path.

    Returns
    -------

    A list of tuples [(X, y, qid), ...].

    """
    root_path = os.path.expanduser(data_path)
    cacheroot_path = os.path.expanduser(cache_path)
    cache_path = os.path.join(cacheroot_path, "MSLR_10K_LETOR.pkl")

    # Use only the Fold1 for demo:
    # Train,      Valid, Test
    # {S1,S2,S3}, S4,    S5

    if not os.path.exists(cache_path):
        fold_path = os.path.join(root_path, "Fold1")
        train_path = os.path.join(fold_path, "train.txt")
        test_path = os.path.join(fold_path, "test.txt")
        X_train, y_train, qid_train = load_svmlight_file(
            train_path, query_id=True, dtype=np.float32
        )
        y_train = y_train.astype(np.int32)
        qid_train = qid_train.astype(np.int32)

        X_test, y_test, qid_test = load_svmlight_file(
            test_path, query_id=True, dtype=np.float32
        )
        y_test = y_test.astype(np.int32)
        qid_test = qid_test.astype(np.int32)

        data = RelDataCV(
            train=(X_train, y_train, qid_train),
            test=(X_test, y_test, qid_test),
            max_rel=4,
        )

        with open(cache_path, "wb") as fd:
            pkl.dump(data, fd)

    with open(cache_path, "rb") as fd:
        data = pkl.load(fd)

    return data


def ranking_demo(cli_args: argparse.Namespace) -> None:
    """Demonstration for learning to rank with relevance degree."""
    data = load_mslr_10k(cli_args.data, cli_args.cache)

    # Sort data according to query index
    X_train, y_train, qid_train = data.train
    sorted_idx = np.argsort(qid_train)
    X_train = X_train[sorted_idx]
    y_train = y_train[sorted_idx]
    qid_train = qid_train[sorted_idx]

    X_test, y_test, qid_test = data.test
    sorted_idx = np.argsort(qid_test)
    X_test = X_test[sorted_idx]
    y_test = y_test[sorted_idx]
    qid_test = qid_test[sorted_idx]

    ranker = xgb.XGBRanker(
        tree_method="hist",
        device="cuda",
        lambdarank_pair_method="topk",
        lambdarank_num_pair_per_sample=13,
        eval_metric=["ndcg@1", "ndcg@8"],
    )
    ranker.fit(
        X_train,
        y_train,
        qid=qid_train,
        eval_set=[(X_test, y_test)],
        eval_qid=[qid_test],
        verbose=True,
    )


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
    args = parser.parse_args()

    ranking_demo(args)
