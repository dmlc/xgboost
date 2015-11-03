import xgboost as xgb
import numpy as np
from sklearn.datasets import load_digits
from sklearn.cross_validation import KFold, train_test_split
import unittest

rng = np.random.RandomState(1994)

class TestEarlyStopping(unittest.TestCase):

	def test_early_stopping_nonparallel(self):
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

# TODO: parallel test for early stopping
# TODO: comment out for now. Will re-visit later