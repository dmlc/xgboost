###
# advanced: customized loss function
#
import os
import numpy as np
import xgboost as xgb

print('start running example to used customized objective function')

CURRENT_DIR = os.path.dirname(__file__)
dtrain = xgb.DMatrix(os.path.join(CURRENT_DIR, '../data/agaricus.txt.train'))
dtest = xgb.DMatrix(os.path.join(CURRENT_DIR, '../data/agaricus.txt.test'))

# note: what we are getting is margin value in prediction you must know what
# you are doing
param = {'max_depth': 2, 'eta': 1, 'objective': 'reg:logistic'}
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 10


# user define objective function, given prediction, return gradient and second
# order gradient this is log likelihood loss
def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))  # transform raw leaf weight
    grad = preds - labels
    hess = preds * (1.0 - preds)
    return grad, hess


# user defined evaluation function, return a pair metric_name, result

# NOTE: when you do customized loss function, the default prediction value is
# margin, which means the prediction is score before logistic transformation.
def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))  # transform raw leaf weight
    # return a pair metric_name, result. The metric name must not contain a
    # colon (:) or a space
    return 'my-error', float(sum(labels != (preds > 0.5))) / len(labels)


py_evals_result = {}

# training with customized objective, we can also do step by step training
# simply look at training.py's implementation of train
py_params = param.copy()
py_params.update({'disable_default_eval_metric': True})
py_logreg = xgb.train(py_params, dtrain, num_round, watchlist, obj=logregobj,
                      feval=evalerror, evals_result=py_evals_result)

evals_result = {}
params = param.copy()
params.update({'eval_metric': 'error'})
logreg = xgb.train(params, dtrain, num_boost_round=num_round, evals=watchlist,
                   evals_result=evals_result)


for i in range(len(py_evals_result['train']['my-error'])):
    np.testing.assert_almost_equal(py_evals_result['train']['my-error'],
                                   evals_result['train']['error'])
