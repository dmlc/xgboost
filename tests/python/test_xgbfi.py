import numpy as np
import xgboost as xgb
import testing as tm
import unittest
import os

dpath = 'demo/data/'
dtrain = xgb.DMatrix(dpath + 'agaricus.txt.train')
dtest = xgb.DMatrix(dpath + 'agaricus.txt.test')

rng = np.random.RandomState(1994)

tm._skip_if_no_pandas()


class TestXgbfi(unittest.TestCase):
    def test_xgbfi(self):
        def create_feature_map(fmap_filename, features):
            outfile = open(fmap_filename, 'w')
            for i, feat in enumerate(features):
                outfile.write('{0}\t{1}\tq\n'.format(i, feat))
            outfile.close()

        param = {'silent': 1, 'objective': 'binary:logistic',
                 'booster': 'gblinear', 'alpha': 0.0001, 'lambda': 1, 'nthread': 1}
        num_round = 1
        bst = xgb.train(param, dtrain, num_round)
        df_fi = bst.get_feature_interactions()
        assert df_fi.shape[0] == 0  # gblinear not supported

        param = {'silent': 1, 'objective': 'binary:logistic',
                 'booster': 'gbtree', 'alpha': 0.0001, 'lambda': 1, 'nthread': 1}
        num_round = 1
        bst = xgb.train(param, dtrain, num_round)
        df_fi = bst.get_feature_interactions()
        assert df_fi.shape[0] > 0

        param = {'silent': 1, 'objective': 'binary:logistic', 'max_depth': 5,
                 'booster': 'gbtree', 'alpha': 0.0001, 'lambda': 1, 'nthread': 1}
        num_round = 10
        bst = xgb.train(param, dtrain, num_round)
        fnames = "|".join(["FEAT{}".format(x) for x in range(128)])
        df_fi = bst.get_feature_interactions(fmap=fnames)
        assert 'FEAT' in df_fi.iloc[0, 0]

        create_feature_map('xgbfi.fmap', ["FEAT{}".format(x) for x in range(128)])
        df_fi = bst.get_feature_interactions(fmap='xgbfi.fmap')
        assert 'FEAT' in df_fi.iloc[0, 0]
        os.remove('xgbfi.fmap')

        # max_fi_depth=2, max_tree_depth=-1, max_deepening=-1, ntrees=-1
        df_fi = bst.get_feature_interactions()
        assert df_fi.shape[0] == 43

        # max_fi_depth=2, max_tree_depth=5, max_deepening=-1, ntrees=10
        df_fi = bst.get_feature_interactions(2, 5, -1, 10)
        assert df_fi.shape[0] == 43

        # max_fi_depth=-1, max_tree_depth=-1, max_deepening=-1, ntrees=-1
        df_fi = bst.get_feature_interactions(-1, -1, -1, -1)
        assert df_fi.shape[0] == 55

        # max_fi_depth=-1, max_tree_depth=-1, max_deepening=-1, ntrees=1
        df_fi = bst.get_feature_interactions(-1, -1, -1, 1)
        assert df_fi.shape[0] == 31

        # max_fi_depth=0, max_tree_depth=-1, max_deepening=0, ntrees=1
        df_fi = bst.get_feature_interactions(0, -1, 0, 1)
        assert df_fi.shape[0] == 1

        # max_fi_depth=-1, max_tree_depth=-1, max_deepening=0, ntrees=1
        df_fi = bst.get_feature_interactions(-1, -1, 0, 1)
        assert df_fi.shape[0] == 10

        # max_fi_depth=0, max_tree_depth=-1, max_deepening=1, ntrees=1
        df_fi = bst.get_feature_interactions(0, -1, 1, 1)
        assert df_fi.shape[0] == 3

        # max_fi_depth=0, max_tree_depth=1, max_deepening=1, ntrees=1
        df_fi = bst.get_feature_interactions(0, 1, 1, 1)
        assert df_fi.shape[0] == 1

        # max_fi_depth=0, max_tree_depth=1, max_deepening=0, ntrees=3
        df_fi = bst.get_feature_interactions(0, 1, 0, 3)
        assert df_fi.shape[0] == 1 and df_fi.loc[0, 'fscore'] == 3

        # max_fi_depth=0, max_tree_depth=1, max_deepening=-1, ntrees=5
        df_fi = bst.get_feature_interactions(0, 1, -1, 5)
        assert df_fi.shape[0] == 1 and df_fi.loc[0, 'fscore'] == 5
