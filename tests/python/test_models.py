import numpy as np
import xgboost as xgb

dpath = 'demo/data/'

def test_glm():
	dtrain = xgb.DMatrix('../data/agaricus.txt.train')
	dtest = xgb.DMatrix('../data/agaricus.txt.test')
	param = {'silent':1, 'objective':'binary:logistic', 'booster':'gblinear',
         'alpha': 0.0001, 'lambda': 1 }
    watchlist  = [(dtest,'eval'), (dtrain,'train')]
	num_round = 4
	bst = xgb.train(param, dtrain, num_round, watchlist)
	preds = bst.predict(dtest)
	labels = dtest.get_label()