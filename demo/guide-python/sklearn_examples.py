'''
Created on 1 Apr 2015

@author: Jamie Hall
'''

import sys
sys.path.append('../../wrapper')
import xgboost as xgb

import numpy as np
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris, load_digits, load_boston

rng = np.random.RandomState(31337)


print("Zeros and Ones from the Digits dataset: binary classification")
digits = load_digits(2)
y = digits['target']
X = digits['data']
kf = KFold(y.shape[0], n_folds=2, shuffle=True, random_state=rng)
for train_index, test_index in kf:
    xgb_model = xgb.XGBClassifier().fit(X[train_index],y[train_index])
    predictions = xgb_model.predict(X[test_index])
    actuals = y[test_index]
    print(confusion_matrix(actuals, predictions))

print("Iris: multiclass classification")
iris = load_iris()
y = iris['target']
X = iris['data']
kf = KFold(y.shape[0], n_folds=2, shuffle=True, random_state=rng)
for train_index, test_index in kf:
    xgb_model = xgb.XGBClassifier().fit(X[train_index],y[train_index])
    predictions = xgb_model.predict(X[test_index])
    actuals = y[test_index]
    print(confusion_matrix(actuals, predictions))


