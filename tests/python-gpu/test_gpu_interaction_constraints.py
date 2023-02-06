import sys

import numpy as np
import pandas as pd

import xgboost as xgb

sys.path.append("tests/python")
# Don't import the test class, otherwise they will run twice.
import test_interaction_constraints as test_ic  # noqa

rng = np.random.RandomState(1994)


class TestGPUInteractionConstraints:
    cputest = test_ic.TestInteractionConstraints()

    def test_interaction_constraints(self):
        self.cputest.run_interaction_constraints(tree_method="gpu_hist")

    def test_training_accuracy(self):
        self.cputest.training_accuracy(tree_method="gpu_hist")

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
            "tree_method": "gpu_hist",
            "interaction_constraints": [["A", "B"], ["B", "D", "C"], ["C", "D"]],
            "objective": "count:poisson",
            "eval_metric": "poisson-nloglik",
            "verbosity": 0,
        }

        xgb.train(params, dm, num_boost_round=100)
