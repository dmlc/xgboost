import numpy as np
import pandas as pd
import pytest

import xgboost as xgb
from xgboost.testing.interaction_constraints import (
    run_interaction_constraints,
    training_accuracy,
)


class TestGPUInteractionConstraints:
    @pytest.mark.parametrize("tree_method", ["hist", "approx"])
    def test_interaction_constraints(self, tree_method: str) -> None:
        run_interaction_constraints(tree_method=tree_method, device="cuda")

    @pytest.mark.parametrize("tree_method", ["hist", "approx"])
    def test_training_accuracy(self, tree_method: str) -> None:
        dpath = "demo/data/"
        training_accuracy(tree_method=tree_method, dpath=dpath, device="cuda")

    # case where different number of features can occur in the evaluator
    def test_issue_8730(self):
        X = pd.DataFrame(
            zip(range(0, 100), range(200, 300), range(300, 400), range(400, 500)),
            columns=["A", "B", "C", "D"],
        )
        y = np.array([*([0] * 50), *([1] * 50)])
        dm = xgb.DMatrix(X, label=y)

        params = {
            "eta": 0.16095019509249486,
            "min_child_weight": 1,
            "subsample": 0.688567929338029,
            "colsample_bynode": 0.7,
            "gamma": 5.666579817418348e-06,
            "lambda": 0.14943712232059794,
            "grow_policy": "depthwise",
            "max_depth": 3,
            "tree_method": "hist",
            "device": "cuda",
            "interaction_constraints": [["A", "B"], ["B", "D", "C"], ["C", "D"]],
            "objective": "count:poisson",
            "eval_metric": "poisson-nloglik",
            "verbosity": 0,
        }

        xgb.train(params, dm, num_boost_round=100)
