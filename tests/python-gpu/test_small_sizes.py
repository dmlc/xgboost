'''Make sure XGBoost works when having less data than GPUs.'''
import numpy as np
import xgboost as xgb
import unittest
import pytest

np.random.seed(0)


class TestGPU(unittest.TestCase):
    params = {
        'learning_rate': 0.1,
        'max_depth': 6,
        'min_child_weight': 1,
        'n_estimators': 20,
        'objective': 'binary:logistic',
        'reg_alpha': 0,
        'reg_lambda': 1,
        'scale_pos_weight': 1,
        'seed': 1234,
        'tree_method': 'gpu_hist',
        'n_gpus': -1,
        'gpu_id': 0,
        'verbosity': '3'
    }
    N_COLS = 128
    MAX_ROWS = 16               # 16 GPUs should be a lot
    DIFF = 2

    @pytest.mark.mgpu
    def test_eval_larger_than_train(self):
        for n_rows in range(1, self.MAX_ROWS):
            X = np.random.randn(n_rows, self.N_COLS)
            y = np.random.randint(0, 2, size=n_rows)
            d_train = xgb.DMatrix(X, label=y)

            X_eval = np.random.randn(n_rows+self.DIFF, self.N_COLS)
            y_eval = np.random.randint(0, 2, size=n_rows+self.DIFF)
            d_eval = xgb.DMatrix(X_eval, label=y_eval)

            xgb.train(params=self.params,
                      dtrain=d_train, evals=[(d_eval, "eval")])

    @pytest.mark.mgpu
    def test_train_larger_than_eval(self):
        for n_rows in range(1, self.MAX_ROWS):
            X = np.random.randn(n_rows+self.DIFF, self.N_COLS)
            y = np.random.randint(0, 2, size=n_rows+self.DIFF)
            d_train = xgb.DMatrix(X, label=y)

            X_eval = np.random.randn(n_rows, self.N_COLS)
            y_eval = np.random.randint(0, 2, size=n_rows)
            d_eval = xgb.DMatrix(X_eval, label=y_eval)

            xgb.train(params=self.params,
                      dtrain=d_train, evals=[(d_eval, "eval")])
