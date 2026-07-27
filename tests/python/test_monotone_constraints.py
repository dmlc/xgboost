from typing import Type

import pytest
import xgboost as xgb
from xgboost import testing as tm
from xgboost.testing.monotone_constraints import (
    is_correctly_constrained,
    run_monotone_constraints,
    run_multi_output_monotone,
    run_parent_gain,
    training_dset,
    x,
    y,
)


class TestMonotoneConstraints:
    def test_monotone_constraints_tuple(self) -> None:
        params_for_constrained = {"monotone_constraints": (1, -1)}
        constrained = xgb.train(params_for_constrained, training_dset)
        assert is_correctly_constrained(constrained)

    @pytest.mark.parametrize("fmt", [dict, list])
    def test_monotone_constraints_feature_names(self, fmt: Type) -> None:
        # next check monotonicity when initializing monotone_constraints by feature names
        params = {
            "tree_method": "hist",
            "grow_policy": "lossguide",
            "monotone_constraints": {"feature_0": 1, "feature_1": -1},
        }

        if fmt is list:
            params = list(params.items())  # type: ignore

        with pytest.raises(ValueError):
            xgb.train(params, training_dset)

        feature_names = ["feature_0", "feature_2"]
        training_dset_w_feature_names = xgb.DMatrix(
            x, label=y, feature_names=feature_names
        )

        with pytest.raises(ValueError):
            xgb.train(params, training_dset_w_feature_names)

        feature_names = ["feature_0", "feature_1"]
        training_dset_w_feature_names = xgb.DMatrix(
            x, label=y, feature_names=feature_names
        )

        constrained_learner = xgb.train(params, training_dset_w_feature_names)

        assert is_correctly_constrained(constrained_learner, feature_names)

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_training_accuracy(self) -> None:
        from sklearn.metrics import accuracy_score

        dpath = "demo/data/"
        dtrain = xgb.DMatrix(dpath + "agaricus.txt.train?indexing_mode=1&format=libsvm")
        dtest = xgb.DMatrix(dpath + "agaricus.txt.test?indexing_mode=1&format=libsvm")
        params = {
            "eta": 1,
            "max_depth": 6,
            "objective": "binary:logistic",
            "tree_method": "hist",
            "monotone_constraints": "(1, 0)",
        }
        num_boost_round = 5

        params["grow_policy"] = "lossguide"
        bst = xgb.train(params, dtrain, num_boost_round)
        pred_dtest = bst.predict(dtest) < 0.5
        assert accuracy_score(dtest.get_label(), pred_dtest) < 0.1

        params["grow_policy"] = "depthwise"
        bst = xgb.train(params, dtrain, num_boost_round)
        pred_dtest = bst.predict(dtest) < 0.5
        assert accuracy_score(dtest.get_label(), pred_dtest) < 0.1


@pytest.mark.parametrize(
    "tree_method,policy",
    [
        # exact only supports depthwise growth.
        ("exact", "depthwise"),
        ("hist", "depthwise"),
        ("approx", "depthwise"),
        ("hist", "lossguide"),
        ("approx", "lossguide"),
    ],
)
def test_monotone_constraints(tree_method: str, policy: str) -> None:
    run_monotone_constraints("cpu", tree_method, policy)


@pytest.mark.parametrize("multi_strategy", ["one_output_per_tree", "multi_output_tree"])
def test_parent_gain(multi_strategy: str) -> None:
    run_parent_gain("cpu", multi_strategy)


@pytest.mark.parametrize("policy", ["depthwise", "lossguide"])
def test_vector_leaf_monotone(policy: str) -> None:
    run_monotone_constraints("cpu", "hist", policy, multi_strategy="multi_output_tree")


@pytest.mark.parametrize("multi_strategy", ["one_output_per_tree", "multi_output_tree"])
@pytest.mark.parametrize("policy", ["depthwise", "lossguide"])
def test_deep_monotone(policy: str, multi_strategy: str) -> None:
    run_multi_output_monotone("cpu", policy, multi_strategy)
