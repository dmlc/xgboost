import json
from typing import Any, Dict, Set

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

    def test_hist_multi_interaction_constraints(self) -> None:
        run_interaction_constraints(tree_method="hist", device="cuda", n_targets=3)

    @pytest.mark.parametrize("n_targets", [1, 3])
    def test_branch_constraints(self, n_targets: int) -> None:
        constraints = [[0, 1], [1, 3, 4]]
        rng = np.random.default_rng(0)
        X = rng.normal(size=(5000, 5))
        y = X[:, 0] * X[:, 1] * X[:, 3] + X[:, 0] * X[:, 4] + X[:, 1] * X[:, 2]
        params = {
            "device": "cuda",
            "tree_method": "hist",
            "grow_policy": "depthwise",
            "max_depth": 4,
            "interaction_constraints": str(constraints),
        }
        if n_targets > 1:
            y = np.column_stack([y, y + X[:, 2], y - X[:, 3]])
            params["multi_strategy"] = "multi_output_tree"
        booster = xgb.train(params, xgb.DMatrix(X, y), num_boost_round=20)

        sets = [set(s) for s in constraints]

        # Both the CPU rule and the GPU's union rule require each split to share a
        # constraint set with a feature used above it on the branch.
        def check(node: Dict[str, Any], used: Set[int]) -> None:
            if "leaf" in node:
                return
            fidx = int(node["split"][1:])
            assert (
                not used or fidx in used or any(fidx in s and used & s for s in sets)
            ), (fidx, used)
            for child in node["children"]:
                check(child, used | {fidx})

        for tree in booster.get_dump(dump_format="json"):
            check(json.loads(tree), set())

    @pytest.mark.parametrize("tree_method", ["hist", "approx"])
    def test_training_accuracy(self, tree_method: str) -> None:
        dpath = "demo/data/"
        training_accuracy(tree_method=tree_method, dpath=dpath, device="cuda")

    # case where different number of features can occur in the evaluator
    def test_issue_8730(self) -> None:
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
