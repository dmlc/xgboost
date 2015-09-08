import numpy as np
import xgboost as xgb

dpath = 'demo/data/'
dtrain = xgb.DMatrix(dpath + 'agaricus.txt.train')
dtest = xgb.DMatrix(dpath + 'agaricus.txt.test')

def test_glm():
	param = {'silent':1, 'objective':'binary:logistic', 'booster':'gblinear', 'alpha': 0.0001, 'lambda': 1 }
	watchlist  = [(dtest,'eval'), (dtrain,'train')]
	num_round = 4
	bst = xgb.train(param, dtrain, num_round, watchlist)
	assert isinstance(bst, xgb.core.Booster)
	preds = bst.predict(dtest)
	labels = dtest.get_label()
	err = sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) / float(len(preds))
	assert err < 0.1

def test_custom_objective():
	param = {'max_depth':2, 'eta':1, 'silent':1 }
	watchlist  = [(dtest,'eval'), (dtrain,'train')]
	num_round = 2
	def logregobj(preds, dtrain):
		labels = dtrain.get_label()
		preds = 1.0 / (1.0 + np.exp(-preds))
		grad = preds - labels
		hess = preds * (1.0-preds)
		return grad, hess
	def evalerror(preds, dtrain):
		labels = dtrain.get_label()
		return 'error', float(sum(labels != (preds > 0.0))) / len(labels)
	bst = xgb.train(param, dtrain, num_round, watchlist, logregobj, evalerror)
	assert isinstance(bst, xgb.core.Booster)
	preds = bst.predict(dtest)
	labels = dtest.get_label()
	err = sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) / float(len(preds))
	assert err < 0.1


