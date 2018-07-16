import xgboost as xgb
import testing as tm
import numpy as np
import unittest

rng = np.random.RandomState(1337)

class TestEvalMetrics(unittest.TestCase):
    xgb_params_01 = {
        'silent': 1,
        'nthread': 1,
        'eval_metric': 'error'
    }

    xgb_params_02 = {
        'silent': 1,
        'nthread': 1,
        'eval_metric': ['error']
    }

    xgb_params_03 = {
        'silent': 1,
        'nthread': 1,
        'eval_metric': ['rmse', 'error']
    }

    xgb_params_04 = {
        'silent': 1,
        'nthread': 1,
        'eval_metric': ['error', 'rmse']
    }

    xgb_params_05 = {
        'silent': 1,
        'nthread': 1,
        'eval_metric': ['error']
    }

    def evalerror_01(self, preds, dtrain):
        labels = dtrain.get_label()
        return 'error', float(sum(labels != (preds > 0.0))) / len(labels)

    def evalerror_02(self, preds, dtrain):
        labels = dtrain.get_label()
        return [('error', float(sum(labels != (preds > 0.0))) / len(labels))]

    def evalerror_03(self, preds, dtrain):
        tm._skip_if_no_sklearn()
        from sklearn.metrics import mean_squared_error

        labels = dtrain.get_label()
        return [('rmse', mean_squared_error(labels, preds)),
                ('error', float(sum(labels != (preds > 0.0))) / len(labels))]

    def evalerror_04(self, preds, dtrain):
        tm._skip_if_no_sklearn()
        from sklearn.metrics import mean_squared_error

        labels = dtrain.get_label()
        return [('error', float(sum(labels != (preds > 0.0))) / len(labels)),
                ('rmse', mean_squared_error(labels, preds))]

    def test_eval_metrics(self):
        tm._skip_if_no_sklearn()
        try:
            from sklearn.model_selection import train_test_split
        except:
            from sklearn.cross_validation import train_test_split
        from sklearn.datasets import load_digits

        digits = load_digits(2)
        X = digits['data']
        y = digits['target']

        Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=0)

        dtrain = xgb.DMatrix(Xt, label=yt)
        dvalid = xgb.DMatrix(Xv, label=yv)

        watchlist = [(dtrain, 'train'), (dvalid, 'val')]

        gbdt_01 = xgb.train(self.xgb_params_01, dtrain, num_boost_round=10)
        gbdt_02 = xgb.train(self.xgb_params_02, dtrain, num_boost_round=10)
        gbdt_03 = xgb.train(self.xgb_params_03, dtrain, num_boost_round=10)
        assert gbdt_01.predict(dvalid)[0] == gbdt_02.predict(dvalid)[0]
        assert gbdt_01.predict(dvalid)[0] == gbdt_03.predict(dvalid)[0]

        gbdt_01 = xgb.train(self.xgb_params_01, dtrain, 10, watchlist,
                            early_stopping_rounds=2)
        gbdt_02 = xgb.train(self.xgb_params_02, dtrain, 10, watchlist,
                            early_stopping_rounds=2)
        gbdt_03 = xgb.train(self.xgb_params_03, dtrain, 10, watchlist,
                            early_stopping_rounds=2)
        gbdt_04 = xgb.train(self.xgb_params_04, dtrain, 10, watchlist,
                            early_stopping_rounds=2)
        assert gbdt_01.predict(dvalid)[0] == gbdt_02.predict(dvalid)[0]
        assert gbdt_01.predict(dvalid)[0] == gbdt_03.predict(dvalid)[0]
        assert gbdt_03.predict(dvalid)[0] != gbdt_04.predict(dvalid)[0]

        gbdt_01 = xgb.train(self.xgb_params_01, dtrain, 10, watchlist,
                            early_stopping_rounds=2, feval=self.evalerror_01)
        gbdt_02 = xgb.train(self.xgb_params_02, dtrain, 10, watchlist,
                            early_stopping_rounds=2, feval=self.evalerror_02)
        gbdt_03 = xgb.train(self.xgb_params_03, dtrain, 10, watchlist,
                            early_stopping_rounds=2, feval=self.evalerror_03)
        gbdt_04 = xgb.train(self.xgb_params_04, dtrain, 10, watchlist,
                            early_stopping_rounds=2, feval=self.evalerror_04)
        assert gbdt_01.predict(dvalid)[0] == gbdt_02.predict(dvalid)[0]
        assert gbdt_01.predict(dvalid)[0] == gbdt_03.predict(dvalid)[0]
        assert gbdt_03.predict(dvalid)[0] != gbdt_04.predict(dvalid)[0]

    def test_cv_multiple_evals(self):
        tm._skip_if_no_sklearn()
        try:
            from sklearn.model_selection import train_test_split
        except:
            from sklearn.cross_validation import train_test_split
        from sklearn.datasets import load_digits
        from sklearn.metrics import roc_auc_score

        digits = load_digits(2)
        X = digits['data']
        y = digits['target']

        Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=0)
        d_matrix = xgb.DMatrix(Xt, yt)

        def _gini(preds, train):
            return 'gini', 2*roc_auc_score(train.get_label(), preds) - 1
        def _auc(preds, train):
            return 'auc', roc_auc_score(train.get_label(), preds)

        _, results = xgb.cv(self.xgb_params_05, d_matrix, stratified=True, feval=[_gini, _auc], feval_apart=True)

        assert len(results.columns) == 4
        assert 'train-gini' in results.columns
        assert 'test-gini' in results.columns
        assert 'train-auc' in results.columns
        assert 'test-auc' in results.columns
