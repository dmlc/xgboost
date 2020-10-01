import os
import xgboost as xgb

### simple example for using external memory version

# this is the only difference, add a # followed by a cache prefix name
# several cache file with the prefix will be generated
# currently only support convert from libsvm file
CURRENT_DIR = os.path.dirname(__file__)
dtrain = xgb.DMatrix(os.path.join(CURRENT_DIR, '../data/agaricus.txt.train'))
dtest = xgb.DMatrix(os.path.join(CURRENT_DIR, '../data/agaricus.txt.test'))

# specify validations set to watch performance
param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic'}

# performance notice: set nthread to be the number of your real cpu
# some cpu offer two threads per core, for example, a 4 core cpu with 8 threads, in such case set nthread=4
#param['nthread']=num_real_cpu

watchlist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 2
bst = xgb.train(param, dtrain, num_round, watchlist)
