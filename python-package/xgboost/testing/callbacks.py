# pylint: disable=too-many-locals
"""Tests for callback functions."""

import json
from itertools import product
from typing import Dict, List, Tuple

from ..callback import LearningRateScheduler
from ..core import Booster, DMatrix
from ..training import cv, train
from .utils import Device


def run_eta_decay(
    tree_method: str, dtrain: DMatrix, dtest: DMatrix, device: Device
) -> None:
    """Test learning rate scheduler, used by both CPU and GPU tests."""
    scheduler = LearningRateScheduler

    watchlist = [(dtest, "eval"), (dtrain, "train")]
    num_round = 4

    # learning_rates as a list
    # init eta with 0 to check whether learning_rates work
    param = {
        "max_depth": 2,
        "eta": 0,
        "objective": "binary:logistic",
        "eval_metric": "error",
        "tree_method": tree_method,
        "device": device,
    }
    evals_result: Dict[str, Dict] = {}
    bst = train(
        param,
        dtrain,
        num_round,
        evals=watchlist,
        callbacks=[scheduler([0.8, 0.7, 0.6, 0.5])],
        evals_result=evals_result,
    )
    eval_errors_0 = list(map(float, evals_result["eval"]["error"]))
    assert isinstance(bst, Booster)
    # validation error should decrease, if eta > 0
    assert eval_errors_0[0] > eval_errors_0[-1]

    # init learning_rate with 0 to check whether learning_rates work
    param = {
        "max_depth": 2,
        "learning_rate": 0,
        "objective": "binary:logistic",
        "eval_metric": "error",
        "tree_method": tree_method,
        "device": device,
    }
    evals_result = {}

    bst = train(
        param,
        dtrain,
        num_round,
        evals=watchlist,
        callbacks=[scheduler([0.8, 0.7, 0.6, 0.5])],
        evals_result=evals_result,
    )
    eval_errors_1 = list(map(float, evals_result["eval"]["error"]))
    assert isinstance(bst, Booster)
    # validation error should decrease, if learning_rate > 0
    assert eval_errors_1[0] > eval_errors_1[-1]

    # check if learning_rates override default value of eta/learning_rate
    param = {
        "max_depth": 2,
        "objective": "binary:logistic",
        "eval_metric": "error",
        "tree_method": tree_method,
        "device": device,
    }
    evals_result = {}
    bst = train(
        param,
        dtrain,
        num_round,
        evals=watchlist,
        callbacks=[scheduler([0, 0, 0, 0])],
        evals_result=evals_result,
    )
    eval_errors_2 = list(map(float, evals_result["eval"]["error"]))
    assert isinstance(bst, Booster)
    # validation error should not decrease, if eta/learning_rate = 0
    assert eval_errors_2[0] == eval_errors_2[-1]

    # learning_rates as a customized decay function
    def eta_decay(ithround: int, num_boost_round: int = num_round) -> float:
        return num_boost_round / (ithround + 1)

    evals_result = {}
    bst = train(
        param,
        dtrain,
        num_round,
        evals=watchlist,
        callbacks=[scheduler(eta_decay)],
        evals_result=evals_result,
    )
    eval_errors_3 = list(map(float, evals_result["eval"]["error"]))

    assert isinstance(bst, Booster)

    assert eval_errors_3[0] == eval_errors_2[0]

    for i in range(1, len(eval_errors_0)):
        assert eval_errors_3[i] != eval_errors_2[i]

    cv(param, dtrain, num_round, callbacks=[scheduler(eta_decay)])


def tree_methods_objs() -> List[Tuple[str, str]]:
    """Test parameters for the leaf output test."""
    return list(
        product(
            ["approx", "hist"],
            [
                "binary:logistic",
                "reg:absoluteerror",
                "reg:quantileerror",
            ],
        )
    )


def run_eta_decay_leaf_output(
    tree_method: str, objective: str, dtrain: DMatrix, dtest: DMatrix, device: Device
) -> None:
    """check decay has effect on leaf output."""
    num_round = 4
    scheduler = LearningRateScheduler

    watchlist = [(dtest, "eval"), (dtrain, "train")]

    param = {
        "max_depth": 2,
        "objective": objective,
        "eval_metric": "error",
        "tree_method": tree_method,
        "device": device,
    }
    if objective == "reg:quantileerror":
        param["quantile_alpha"] = 0.3

    def eta_decay_0(i: int) -> float:
        return num_round / (i + 1)

    bst0 = train(
        param,
        dtrain,
        num_round,
        evals=watchlist,
        callbacks=[scheduler(eta_decay_0)],
    )

    def eta_decay_1(i: int) -> float:
        if i > 1:
            return 5.0
        return num_round / (i + 1)

    bst1 = train(
        param,
        dtrain,
        num_round,
        evals=watchlist,
        callbacks=[scheduler(eta_decay_1)],
    )
    bst_json0 = bst0.save_raw(raw_format="json")
    bst_json1 = bst1.save_raw(raw_format="json")

    j0 = json.loads(bst_json0)
    j1 = json.loads(bst_json1)

    tree_2th_0 = j0["learner"]["gradient_booster"]["model"]["trees"][2]
    tree_2th_1 = j1["learner"]["gradient_booster"]["model"]["trees"][2]
    assert tree_2th_0["base_weights"] == tree_2th_1["base_weights"]
    assert tree_2th_0["split_conditions"] == tree_2th_1["split_conditions"]

    tree_3th_0 = j0["learner"]["gradient_booster"]["model"]["trees"][3]
    tree_3th_1 = j1["learner"]["gradient_booster"]["model"]["trees"][3]
    assert tree_3th_0["base_weights"] != tree_3th_1["base_weights"]
    assert tree_3th_0["split_conditions"] != tree_3th_1["split_conditions"]
