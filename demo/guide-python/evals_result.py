##
#  This script demonstrate how to access the eval metrics in xgboost
##

import xgboost as xgb
dtrain = xgb.DMatrix('../data/agaricus.txt.train', silent=True)
dtest = xgb.DMatrix('../data/agaricus.txt.test', silent=True)

param = [('max_depth', 2), ('objective', 'binary:logistic'), ('eval_metric', 'logloss'), ('eval_metric', 'error')]
 
num_round = 2
watchlist = [(dtest,'eval'), (dtrain,'train')]

evals_result = {}
bst = xgb.train(param, dtrain, num_round, watchlist, evals_result=evals_result)

print('Access logloss metric directly from evals_result:')
print(evals_result['eval']['logloss'])

print('')
print('Access metrics through a loop:')
for e_name, e_mtrs in evals_result.items():
    print('- {}'.format(e_name))
    for e_mtr_name, e_mtr_vals in e_mtrs.items():
        print('   - {}'.format(e_mtr_name))
        print('      - {}'.format(e_mtr_vals))

print('')
print('Access complete dictionary:')
print(evals_result)
