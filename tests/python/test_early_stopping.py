import xgboost as xgb
import testing as tm
import numpy as np
import unittest

rng = np.random.RandomState(1994)


class TestEarlyStopping(unittest.TestCase):

    def test_early_stopping_nonparallel(self):
        tm._skip_if_no_sklearn()
        from sklearn.datasets import load_digits
        try:
            from sklearn.model_selection import train_test_split
        except:
            from sklearn.cross_validation import train_test_split

        digits = load_digits(2)
        X = digits['data']
        y = digits['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        clf1 = xgb.XGBClassifier()
        clf1.fit(X_train, y_train, early_stopping_rounds=5, eval_metric="auc",
                 eval_set=[(X_test, y_test)])
        clf2 = xgb.XGBClassifier()
        clf2.fit(X_train, y_train, early_stopping_rounds=4, eval_metric="auc",
                 eval_set=[(X_test, y_test)])
        # should be the same
        assert clf1.best_score == clf2.best_score
        assert clf1.best_score != 1
        # check overfit
        clf3 = xgb.XGBClassifier()
        clf3.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="auc",
                 eval_set=[(X_test, y_test)])
        assert clf3.best_score == 1

    def evalerror(self, preds, dtrain):
        tm._skip_if_no_sklearn()
        from sklearn.metrics import mean_squared_error

        labels = dtrain.get_label()
        return 'rmse', mean_squared_error(labels, preds)

    def test_cv_early_stopping(self):
        tm._skip_if_no_sklearn()
        from sklearn.datasets import load_digits

        digits = load_digits(2)
        X = digits['data']
        y = digits['target']
        dm = xgb.DMatrix(X, label=y)
        params = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}

        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10, early_stopping_rounds=10)
        assert cv.shape[0] == 10
        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10, early_stopping_rounds=5)
        assert cv.shape[0] == 3
        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10, early_stopping_rounds=1)
        assert cv.shape[0] == 1

        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10, feval=self.evalerror,
                    early_stopping_rounds=10)
        assert cv.shape[0] == 10
        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10, feval=self.evalerror,
                    early_stopping_rounds=1)
        assert cv.shape[0] == 5
        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10, feval=self.evalerror,
                    maximize=True, early_stopping_rounds=1)
        assert cv.shape[0] == 1
