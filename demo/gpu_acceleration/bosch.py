import numpy as np
import pandas as pd
import xgboost as xgb
import time
import random
from sklearn.cross_validation import StratifiedKFold

#For sampling rows from input file
random_seed = 9
subset = 0.4

n_rows = 1183747;
train_rows = int(n_rows * subset)
random.seed(random_seed)
skip = sorted(random.sample(xrange(1,n_rows + 1),n_rows-train_rows))
data = pd.read_csv("../data/train_numeric.csv", index_col=0, dtype=np.float32, skiprows=skip)
y = data['Response'].values
del data['Response']
X = data.values

param = {}
param['objective'] = 'binary:logistic'
param['eval_metric'] = 'auc'
param['max_depth'] = 5
param['eta'] = 0.3
param['silent'] = 0
param['updater'] = 'grow_gpu'
#param['updater'] = 'grow_colmaker'

num_round = 20

cv = StratifiedKFold(y, n_folds=5)

for i, (train, test) in enumerate(cv):
    dtrain = xgb.DMatrix(X[train], label=y[train])
    tmp = time.time()
    bst = xgb.train(param, dtrain, num_round)
    boost_time = time.time() - tmp
    res = bst.eval(xgb.DMatrix(X[test], label=y[test]))
    print("Fold {}: {}, Boost Time {}".format(i, res, str(boost_time)))
    del bst

