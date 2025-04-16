import pytest

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

    @pytest.mark.skipif(**tm.no_sklearn())
    @pytest.mark.parametrize("tree_method", ["hist", "approx", "exact"])
    def test_hist_training_accuracy(self, tree_method: str) -> None:
        dpath = "demo/data/"
        training_accuracy(tree_method=tree_method, dpath=dpath, device="cpu")
