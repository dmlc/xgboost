import numpy as np
import pytest

import xgboost as xgb
from xgboost import testing as tm

pytestmark = pytest.mark.skipif(**tm.no_pandas())


dpath = 'demo/data/'
rng = np.random.RandomState(1994)


class TestTreesToDataFrame:
    def build_model(self, max_depth, num_round):
        dtrain, _ = tm.load_agaricus(__file__)
        param = {'max_depth': max_depth, 'objective': 'binary:logistic',
                 'verbosity': 1}
        num_round = num_round
        bst = xgb.train(param, dtrain, num_round)
        return bst

    def parse_dumped_model(self, booster, item_to_get, splitter):
        item_to_get += '='
        txt_dump = booster.get_dump(with_stats=True)
        tree_list = [tree.split('/n') for tree in txt_dump]
        split_trees = [tree[0].split(item_to_get)[1:] for tree in tree_list]
        res = sum([float(line.split(splitter)[0])
                   for tree in split_trees for line in tree])
        return res

    def test_trees_to_dataframe(self):
        bst = self.build_model(max_depth=5, num_round=10)
        gain_from_dump = self.parse_dumped_model(booster=bst,
                                                 item_to_get='gain',
                                                 splitter=',')
        cover_from_dump = self.parse_dumped_model(booster=bst,
                                                  item_to_get='cover',
                                                  splitter='\n')
        # method being tested
        df = bst.trees_to_dataframe()

        # test for equality of gains
        gain_from_df = df[df.Feature != 'Leaf'][['Gain']].sum()
        assert np.allclose(gain_from_dump, gain_from_df)

        # test for equality of covers
        cover_from_df = df.Cover.sum()
        assert np.allclose(cover_from_dump, cover_from_df)

    def run_tree_to_df_categorical(self, tree_method: str) -> None:
        X, y = tm.make_categorical(100, 10, 31, False)
        Xy = xgb.DMatrix(X, y, enable_categorical=True)
        booster = xgb.train({"tree_method": tree_method}, Xy, num_boost_round=10)
        df = booster.trees_to_dataframe()
        for _, x in df.iterrows():
            if x["Feature"] != "Leaf":
                assert len(x["Category"]) >= 1

    def test_tree_to_df_categorical(self) -> None:
        self.run_tree_to_df_categorical("approx")

    def run_split_value_histograms(self, tree_method) -> None:
        X, y = tm.make_categorical(1000, 10, 13, False)
        reg = xgb.XGBRegressor(tree_method=tree_method, enable_categorical=True)
        reg.fit(X, y)

        with pytest.raises(ValueError, match="doesn't"):
            reg.get_booster().get_split_value_histogram("3", bins=5)

    def test_split_value_histograms(self):
        self.run_split_value_histograms("approx")
