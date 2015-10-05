import pickle
import xgboost as xgb

import numpy as np
from sklearn.cross_validation import KFold, train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import load_iris, load_digits, load_boston

rng = np.random.RandomState(1994)

def test_binary_classification():
	digits = load_digits(2)
	y = digits['target']
	X = digits['data']
	kf = KFold(y.shape[0], n_folds=2, shuffle=True, random_state=rng)
	for train_index, test_index in kf:
	    xgb_model = xgb.XGBClassifier().fit(X[train_index],y[train_index])
	    predictions = xgb_model.predict(X[test_index])
	    actuals = y[test_index]
	    print(confusion_matrix(actuals, predictions))

def test_multiclass_classification():
	iris = load_iris()
	y = iris['target']
	X = iris['data']
	kf = KFold(y.shape[0], n_folds=2, shuffle=True, random_state=rng)
	for train_index, test_index in kf:
	    xgb_model = xgb.XGBClassifier().fit(X[train_index],y[train_index])
	    predictions = xgb_model.predict(X[test_index])
	    actuals = y[test_index]
	    print(confusion_matrix(actuals, predictions))






