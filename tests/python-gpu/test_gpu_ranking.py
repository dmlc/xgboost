import os
from typing import Dict

import numpy as np
import pytest

import xgboost
from xgboost import testing as tm

pytestmark = tm.timeout(30)


def comp_training_with_rank_objective(
    dtrain: xgboost.DMatrix,
    dtest: xgboost.DMatrix,
    rank_objective: str,
    metric_name: str,
    tolerance: float = 1e-02,
) -> None:
    """Internal method that trains the dataset using the rank objective on GPU and CPU,
    evaluates the metric and determines if the delta between the metric is within the
    tolerance level.

    """
    # specify validations set to watch performance
    watchlist = [(dtest, "eval"), (dtrain, "train")]

    params = {
        "booster": "gbtree",
        "tree_method": "gpu_hist",
        "gpu_id": 0,
        "predictor": "gpu_predictor",
    }

    num_trees = 100
    check_metric_improvement_rounds = 10

    evals_result: Dict[str, Dict] = {}
    params["objective"] = rank_objective
    params["eval_metric"] = metric_name
    bst = xgboost.train(
        params,
        dtrain,
        num_boost_round=num_trees,
        early_stopping_rounds=check_metric_improvement_rounds,
        evals=watchlist,
        evals_result=evals_result,
    )
    gpu_scores = evals_result["train"][metric_name][-1]

    evals_result = {}

    cpu_params = {
        "booster": "gbtree",
        "tree_method": "hist",
        "gpu_id": -1,
        "predictor": "cpu_predictor",
    }
    cpu_params["objective"] = rank_objective
    cpu_params["eval_metric"] = metric_name
    bstc = xgboost.train(
        cpu_params,
        dtrain,
        num_boost_round=num_trees,
        early_stopping_rounds=check_metric_improvement_rounds,
        evals=watchlist,
        evals_result=evals_result,
    )
    cpu_scores = evals_result["train"][metric_name][-1]

    info = (rank_objective, metric_name)
    assert np.allclose(gpu_scores, cpu_scores, tolerance, tolerance), info
    assert np.allclose(bst.best_score, bstc.best_score, tolerance, tolerance), info

    evals_result_weighted: Dict[str, Dict] = {}
    dtest.set_weight(np.ones((dtest.get_group().size,)))
    dtrain.set_weight(np.ones((dtrain.get_group().size,)))
    watchlist = [(dtest, "eval"), (dtrain, "train")]
    bst_w = xgboost.train(
        params,
        dtrain,
        num_boost_round=num_trees,
        early_stopping_rounds=check_metric_improvement_rounds,
        evals=watchlist,
        evals_result=evals_result_weighted,
    )
    weighted_metric = evals_result_weighted["train"][metric_name][-1]

    tolerance = 1e-5
    assert np.allclose(bst_w.best_score, bst.best_score, tolerance, tolerance)
    assert np.allclose(weighted_metric, gpu_scores, tolerance, tolerance)


@pytest.mark.parametrize(
    "objective,metric",
    [
        ("rank:pairwise", "auc"),
        ("rank:pairwise", "ndcg"),
        ("rank:pairwise", "map"),
        ("rank:ndcg", "auc"),
        ("rank:ndcg", "ndcg"),
        ("rank:ndcg", "map"),
        ("rank:map", "auc"),
        ("rank:map", "ndcg"),
        ("rank:map", "map"),
    ],
)
def test_with_mq2008(objective, metric) -> None:
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
    ) = tm.data.get_mq2008(os.path.join(os.path.join(tm.demo_dir(__file__), "rank")))

    if metric.find("map") != -1 or objective.find("map") != -1:
        y_train[y_train <= 1] = 0.0
        y_train[y_train > 1] = 1.0
        y_test[y_test <= 1] = 0.0
        y_test[y_test > 1] = 1.0

    dtrain = xgboost.DMatrix(x_train, y_train, qid=qid_train)
    dtest = xgboost.DMatrix(x_test, y_test, qid=qid_test)

    comp_training_with_rank_objective(dtrain, dtest, objective, metric)
