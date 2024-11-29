"""
Getting started with learning to rank
=====================================

  .. versionadded:: 2.0.0

This is a demonstration of using XGBoost for learning to rank tasks using the
MSLR_10k_letor dataset. For more infomation about the dataset, please visit its
`description page <https://www.microsoft.com/en-us/research/project/mslr/>`_.

This is a two-part demo, the first one contains a basic example of using XGBoost to
train on relevance degree, and the second part simulates click data and enable the
position debiasing training.

For an overview of learning to rank in XGBoost, please see :doc:`Learning to Rank
</tutorials/learning_to_rank>`.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle as pkl

import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file

import xgboost as xgb
from xgboost.testing.data import RelDataCV, simulate_clicks, sort_ltr_samples


def load_mslr_10k(data_path: str, cache_path: str) -> RelDataCV:
    """Load the MSLR10k dataset from data_path and cache a pickle object in cache_path.

    Returns
    -------

    A list of tuples [(X, y, qid), ...].

    """
    root_path = os.path.expanduser(args.data)
    cacheroot_path = os.path.expanduser(args.cache)
    cache_path = os.path.join(cacheroot_path, "MSLR_10K_LETOR.pkl")

    # Use only the Fold1 for demo:
    # Train,      Valid, Test
    # {S1,S2,S3}, S4,    S5
    fold = 1

    if not os.path.exists(cache_path):
        fold_path = os.path.join(root_path, f"Fold{fold}")
        train_path = os.path.join(fold_path, "train.txt")
        valid_path = os.path.join(fold_path, "vali.txt")
        test_path = os.path.join(fold_path, "test.txt")
        X_train, y_train, qid_train = load_svmlight_file(
            train_path, query_id=True, dtype=np.float32
        )
        y_train = y_train.astype(np.int32)
        qid_train = qid_train.astype(np.int32)

        X_valid, y_valid, qid_valid = load_svmlight_file(
            valid_path, query_id=True, dtype=np.float32
        )
        y_valid = y_valid.astype(np.int32)
        qid_valid = qid_valid.astype(np.int32)

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


def ranking_demo(args: argparse.Namespace) -> None:
    """Demonstration for learning to rank with relevance degree."""
    data = load_mslr_10k(args.data, args.cache)

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


def click_data_demo(args: argparse.Namespace) -> None:
    """Demonstration for learning to rank with click data."""
    data = load_mslr_10k(args.data, args.cache)
    train, test = simulate_clicks(data)
    assert test is not None

    assert train.X.shape[0] == train.click.size
    assert test.X.shape[0] == test.click.size
    assert test.score.dtype == np.float32
    assert test.click.dtype == np.int32

    X_train, clicks_train, y_train, qid_train = sort_ltr_samples(
        train.X,
        train.y,
        train.qid,
        train.click,
        train.pos,
    )
    X_test, clicks_test, y_test, qid_test = sort_ltr_samples(
        test.X,
        test.y,
        test.qid,
        test.click,
        test.pos,
    )

    class ShowPosition(xgb.callback.TrainingCallback):
        def after_iteration(
            self,
            model: xgb.Booster,
            epoch: int,
            evals_log: xgb.callback.TrainingCallback.EvalsLog,
        ) -> bool:
            config = json.loads(model.save_config())
            ti_plus = np.array(config["learner"]["objective"]["ti+"])
            tj_minus = np.array(config["learner"]["objective"]["tj-"])
            df = pd.DataFrame({"ti+": ti_plus, "tj-": tj_minus})
            print(df)
            return False

    ranker = xgb.XGBRanker(
        n_estimators=512,
        tree_method="hist",
        device="cuda",
        learning_rate=0.01,
        reg_lambda=1.5,
        subsample=0.8,
        sampling_method="gradient_based",
        # LTR specific parameters
        objective="rank:ndcg",
        # - Enable bias estimation
        lambdarank_unbiased=True,
        # - normalization (1 / (norm + 1))
        lambdarank_bias_norm=1,
        # - Focus on the top 12 documents
        lambdarank_num_pair_per_sample=12,
        lambdarank_pair_method="topk",
        ndcg_exp_gain=True,
        eval_metric=["ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10"],
        callbacks=[ShowPosition()],
    )
    ranker.fit(
        X_train,
        clicks_train,
        qid=qid_train,
        eval_set=[(X_test, y_test), (X_test, clicks_test)],
        eval_qid=[qid_test, qid_test],
        verbose=True,
    )
    ranker.predict(X_test)


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
    click_data_demo(args)
