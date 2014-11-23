import os
import sys
sys.path.append(os.path.dirname(__file__)+'/../wrapper')
import xgboost as xgb
# this is example script of running distributed xgboost using python

# call this additional function to intialize the xgboost sync module
# in distributed mode
xgb.sync_init(sys.argv)
rank = xgb.sync_get_rank()
# read in dataset
dtrain = xgb.DMatrix('train.col%d' % rank)
param = {'max_depth':3, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
param['dsplit'] = 'col'
nround = 3

if rank == 0:
    dtest = xgb.DMatrix('../../demo/data/agaricus.txt.test')
    model = xgb.train(param, dtrain, nround, [(dtrain, 'train') , (dtest, 'test')])
else:
    # if it is a slave node, do not run evaluation
    model = xgb.train(param, dtrain, nround)

if rank == 0:
    model.save_model('%04d.model' % nround)
    # dump model with feature map
    model.dump_model('dump.nice.%d.txt' % xgb.sync_get_world_size(),'../../demo/data/featmap.txt')
# shutdown the synchronization module
xgb.sync_finalize()
