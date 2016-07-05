#!/usr/bin/python
import xgboost as xgb
import numpy as np

#  this script demonstrate how to fit gamma regression model (with log link function)
#  in xgboost
dtrain = xgb.DMatrix('../data/autoclaims.train')
dtest = xgb.DMatrix('../data/autoclaims.test')

# for gamma regression, we need to set the objective to 'reg:gamma', it also suggests
# to set the base_score to a value between 1 to 5 if the number of iteration is small
param = {'silent':1, 'objective':'reg:gamma', 'booster':'gbtree', 'base_score':3}

# the rest of settings are the same
watchlist  = [(dtest,'eval'), (dtrain,'train')]
num_round = 25

# training and evaluation
bst = xgb.train(param, dtrain, num_round, watchlist)
preds = bst.predict(dtest)
labels = dtest.get_label()
print ('test deviance=%f' % (2 * np.sum((labels - preds) / preds - np.log(labels) + np.log(preds))))
