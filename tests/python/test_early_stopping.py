import xgboost as xgb
import numpy as np
from sklearn.datasets import load_digits
from sklearn.cross_validation import KFold, train_test_split

rng = np.random.RandomState(1994)

def test_early_stopping_nonparallel():
	# digits = load_digits(2)
	# X = digits['data']
	# y = digits['target']
	# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
	# clf = xgb.XGBClassifier()
	# clf.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="auc",
	#         eval_set=[(X_test, y_test)])
	print("This test will be re-visited later. ")

# TODO: parallel test for early stopping
# TODO: comment out for now. Will re-visit later