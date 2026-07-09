import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from xgboost import testing as tm
from xgboost.testing.parse_tree import (
    run_split_value_histograms,
    run_tree_to_df_categorical,
)

pytestmark = pytest.mark.skipif(**tm.no_pandas())


dpath = "demo/data/"
rng = np.random.RandomState(1994)


class TestTreesToDataFrame:
    def build_model(self, max_depth, num_round):
        dtrain, _ = tm.load_agaricus(__file__)
        param = {"max_depth": max_depth, "objective": "binary:logistic", "verbosity": 1}
        num_round = num_round
        bst = xgb.train(param, dtrain, num_round)
        return bst

    def parse_dumped_model(self, booster, item_to_get, splitter):
        item_to_get += "="
        txt_dump = booster.get_dump(with_stats=True)
        tree_list = [tree.split("/n") for tree in txt_dump]
        split_trees = [tree[0].split(item_to_get)[1:] for tree in tree_list]
        res = sum(
            [float(line.split(splitter)[0]) for tree in split_trees for line in tree]
        )
        return res

    def test_trees_to_dataframe(self):
        bst = self.build_model(max_depth=5, num_round=10)
        gain_from_dump = self.parse_dumped_model(
            booster=bst, item_to_get="gain", splitter=","
        )
        cover_from_dump = self.parse_dumped_model(
            booster=bst, item_to_get="cover", splitter="\n"
        )
        # method being tested
        df = bst.trees_to_dataframe()

        # test for equality of gains
        gain_from_df = df[df.Feature != "Leaf"][["Gain"]].sum()
        assert np.allclose(gain_from_dump, gain_from_df)

        # test for equality of covers
        cover_from_df = df.Cover.sum()
        assert np.allclose(cover_from_dump, cover_from_df)

        # Structural checks on the JSON-derived frame: branch ids are well-formed and
        # reference existing (reachable) nodes, and the missing direction always follows
        # one of the two children.
        all_ids = set(df["ID"])
        non_leaf = df[df.Feature != "Leaf"]
        assert non_leaf["Yes"].isin(all_ids).all()
        assert non_leaf["No"].isin(all_ids).all()
        assert non_leaf["Missing"].isin(all_ids).all()
        assert (
            (non_leaf["Missing"] == non_leaf["Yes"])
            | (non_leaf["Missing"] == non_leaf["No"])
        ).all()
        # Numerical splits have a real threshold; leaves have a missing split.
        assert non_leaf["Split"].notna().all()
        assert df[df.Feature == "Leaf"]["Split"].isna().all()

    def test_tree_to_df_mixed(self) -> None:
        """Mixed numerical + categorical model: check ``Category`` column semantics."""
        X, y = tm.make_categorical(256, 8, 17, onehot=False, cat_ratio=0.5)
        Xy = xgb.DMatrix(X, y, enable_categorical=True)
        bst = xgb.train({"tree_method": "hist"}, Xy, num_boost_round=4)
        df = bst.trees_to_dataframe()

        saw_numerical = saw_categorical = False
        for _, x in df.iterrows():
            if x["Feature"] == "Leaf":
                # Leaves have no category and no split threshold.
                assert x["Category"] is None
                assert pd.isna(x["Split"])
            elif isinstance(x["Category"], list):
                # Categorical split: missing threshold, non-empty integer codes.
                saw_categorical = True
                assert pd.isna(x["Split"])
                assert len(x["Category"]) >= 1
            else:
                # Numerical split: ``Category`` stays ``None`` and split is a number.
                saw_numerical = True
                assert x["Category"] is None
                assert pd.notna(x["Split"])
        assert saw_numerical and saw_categorical

    def test_tree_to_df_categorical(self) -> None:
        run_tree_to_df_categorical("approx", "cpu")

    def test_tree_to_df_indicator(self, tmp_path) -> None:
        """Test trees_to_dataframe with indicator (boolean) features."""
        n_samples = 200
        n_features = 5
        X_int = rng.randint(0, 2, size=(n_samples, n_features))
        y = np.logical_xor(X_int[:, 0], X_int[:, 1]).astype(np.float32)
        X = X_int.astype(np.float32)
        # Use `i` as indicator
        dtrain = xgb.DMatrix(X, label=y, feature_types=["i"] * n_features)

        bst = xgb.train(
            {"max_depth": 3, "objective": "binary:logistic", "verbosity": 0},
            dtrain,
            num_boost_round=5,
        )
        df = bst.trees_to_dataframe()

        # Basic structure checks
        assert "Tree" in df.columns
        assert "Feature" in df.columns
        assert "Gain" in df.columns
        assert "Cover" in df.columns
        assert len(df) > 0

        # Indicator nodes should have NaN splits; missing defaults to no-direction
        non_leaf = df[df.Feature != "Leaf"]
        assert len(non_leaf) > 0
        assert non_leaf["Split"].isna().all()
        assert (non_leaf["Missing"] == non_leaf["No"]).all()

    def test_split_value_histograms(self):
        run_split_value_histograms("approx", "cpu")
