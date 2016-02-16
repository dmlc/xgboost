import xgboost as xgb
import numpy as np
from sklearn.datasets import load_digits
from sklearn.cross_validation import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error
import unittest

rng = np.random.RandomState(1994)


class TestCrossValidation(unittest.TestCase):
    def test_cv(self):
        digits = load_digits(3)
        X = digits['data']
        y = digits['target']
        dm = xgb.DMatrix(X, label=y)
        
        params = {
            'max_depth': 2,
            'eta': 1,
            'silent': 1,
            'objective':
            'multi:softprob',
            'num_class': 3
        }

        seed = 2016
        nfolds = 5
        skf = StratifiedKFold(y, n_folds=nfolds, shuffle=True, random_state=seed)

        import pandas as pd
        cv1 = xgb.cv(params, dm, num_boost_round=10, nfold=nfolds, seed=seed)
        cv2 = xgb.cv(params, dm, num_boost_round=10, folds=skf, seed=seed)
        cv3 = xgb.cv(params, dm, num_boost_round=10, nfold=nfolds, stratified=True, seed=seed)
        assert cv1.shape[0] == cv2.shape[0] and cv2.shape[0] == cv3.shape[0]
        assert cv2.iloc[-1,0] == cv3.iloc[-1,0]

