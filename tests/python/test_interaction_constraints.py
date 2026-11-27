import numpy as np
import pytest

import xgboost as xgb
from xgboost import testing as tm
from xgboost.testing.interaction_constraints import (
    run_interaction_constraints,
    training_accuracy,
)


class TestInteractionConstraints:
    def test_exact_interaction_constraints(self) -> None:
        run_interaction_constraints(tree_method="exact", device="cpu")

    def test_hist_interaction_constraints(self) -> None:
        run_interaction_constraints(tree_method="hist", device="cpu")

    def test_approx_interaction_constraints(self) -> None:
        run_interaction_constraints(tree_method="approx", device="cpu")

    def test_hist_multi_interaction_constraints(self) -> None:
        run_interaction_constraints(tree_method="hist", device="cpu", n_targets=3)

    def test_interaction_constraints_feature_names(self) -> None:
        with pytest.raises(ValueError):
            constraints = [("feature_0", "feature_1")]
            run_interaction_constraints(
                tree_method="exact", device="cpu", interaction_constraints=constraints
            )

        with pytest.raises(ValueError):
            constraints = [("feature_0", "feature_3")]
            feature_names = ["feature_0", "feature_1", "feature_2"]
            run_interaction_constraints(
                tree_method="exact",
                device="cpu",
                feature_names=feature_names,
                interaction_constraints=constraints,
            )

        constraints = [("feature_0", "feature_1")]
        feature_names = ["feature_0", "feature_1", "feature_2"]
        run_interaction_constraints(
            tree_method="exact",
            device="cpu",
            feature_names=feature_names,
            interaction_constraints=constraints,
        )

        constraints_lst = [["feature_0", "feature_1"], ["feature_2"]]
        feature_names = ["feature_0", "feature_1", "feature_2"]
        run_interaction_constraints(
            tree_method="exact",
            device="cpu",
            feature_names=feature_names,
            interaction_constraints=constraints_lst,
        )

    @pytest.mark.parametrize("tree_method", ["exact", "hist", "approx"])
    def test_reconfigure_interaction_constraints(self, tree_method: str) -> None:
        class Tighten(xgb.callback.TrainingCallback):
            def after_iteration(self, model, epoch: int, evals_log) -> bool:
                if epoch == 0:
                    model.set_param({"interaction_constraints": "[[0], [1]]"})
                return False

        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        y = np.array([0, 1, 3, 4], dtype=np.float32)
        dtrain = xgb.DMatrix(X, label=y, nthread=1)
        params = {
            "tree_method": tree_method,
            "interaction_constraints": "[[0, 1]]",
            "max_depth": 2,
            "learning_rate": 0.5,
            "reg_lambda": 0.0,
            "base_score": 0.0,
            "nthread": 1,
        }
        booster = xgb.train(params, dtrain, num_boost_round=2, callbacks=[Tighten()])
        np.testing.assert_allclose(
            booster.predict(dtrain, iteration_range=(0, 1)), y / 2
        )
        # The second tree can only split on f0, so its leaves average y/2
        # within each f0 group and apply the learning rate again.
        np.testing.assert_allclose(
            booster.predict(dtrain, iteration_range=(1, 2)),
            [0.125, 0.125, 0.875, 0.875],
        )

    @pytest.mark.parametrize("tree_method", ["exact", "hist", "approx"])
    def test_more_features_than_rows(self, tree_method: str) -> None:
        # Only the last feature separates the two rows. It must remain a split
        # candidate at the root when there are fewer rows than features.
        X = np.array([[0, 0, 0], [0, 0, 1]], dtype=np.float32)
        y = np.array([-1, 1], dtype=np.float32)
        Xy = xgb.DMatrix(X, label=y)
        params = {
            "tree_method": tree_method,
            "interaction_constraints": "[[0, 1, 2]]",
            "max_depth": 1,
            "learning_rate": 1.0,
            "reg_lambda": 0.0,
            "base_score": 0.0,
        }
        booster = xgb.train(params, Xy, num_boost_round=1)
        np.testing.assert_allclose(booster.predict(Xy), y)

    @pytest.mark.skipif(**tm.no_sklearn())
    @pytest.mark.parametrize("tree_method", ["hist", "approx", "exact"])
    def test_hist_training_accuracy(self, tree_method: str) -> None:
        dpath = "demo/data/"
        training_accuracy(tree_method=tree_method, dpath=dpath, device="cpu")
