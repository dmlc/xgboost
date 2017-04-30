#pylint: skip-file
import xgboost as xgb
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import time

n = 1000000
num_rounds = 100

X,y = make_classification(n, n_features=50, random_state=7)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test, y_test)

param = {'objective': 'binary:logistic',
         'tree_method': 'exact',
         'updater': 'grow_gpu_hist',
         'max_depth': 8,
         'silent': 1,
         'eval_metric': 'auc'}
res = {}
tmp = time.time()
xgb.train(param, dtrain, num_rounds, [(dtrain, 'train'), (dtest, 'test')],
          evals_result=res)
print ("GPU: %s seconds" % (str(time.time() - tmp)))

tmp = time.time()
param['updater'] = 'grow_fast_histmaker'
xgb.train(param, dtrain, num_rounds, [(dtrain, 'train'), (dtest, 'test')], evals_result=res)
print ("CPU: %s seconds" % (str(time.time() - tmp)))

