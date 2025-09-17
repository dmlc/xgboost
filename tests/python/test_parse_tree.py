import numpy as np
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

    def test_tree_to_df_categorical(self) -> None:
        run_tree_to_df_categorical("approx", "cpu")

    def test_split_value_histograms(self):
        run_split_value_histograms("approx", "cpu")
