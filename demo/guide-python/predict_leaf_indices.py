#!/usr/bin/python
import xgboost as xgb

### load data in do training
dtrain = xgb.DMatrix('../data/agaricus.txt.train')
dtest = xgb.DMatrix('../data/agaricus.txt.test')
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'}
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 3
bst = xgb.train(param, dtrain, num_round, watchlist)

print ('start testing predict the leaf indices')
### predict using first 2 tree
leafindex = bst.predict(dtest, ntree_limit=2, pred_leaf=True)
print(leafindex.shape)
print(leafindex)
### predict all trees
leafindex = bst.predict(dtest, pred_leaf=True)
print(leafindex.shape)
