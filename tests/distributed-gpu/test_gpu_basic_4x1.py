#!/usr/bin/python
import xgboost as xgb
from collections import OrderedDict

# Always call this before using distributed module
xgb.rabit.init()
rank = xgb.rabit.get_rank()
world = xgb.rabit.get_world_size()

# Load file, file will be automatically sharded in distributed mode.
dtrain = xgb.DMatrix('../../demo/data/agaricus.txt.train')
dtest = xgb.DMatrix('../../demo/data/agaricus.txt.test')

# Specify parameters via map, definition are same as c++ version
param = {'n_gpus': 4, 'gpu_id': rank, 'tree_method': 'gpu_hist', 'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic' }

# Specify validations set to watch performance
watchlist  = [(dtest,'eval'), (dtrain,'train')]
num_round = 20

# Run training, all the features in training API is available.
# Currently, this script only support calling train once for fault recovery purpose.
bst = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=2)

# Have root save its model
if(rank == 0):
    model_name = "test.model.4x1." + str(rank)
    bst.dump_model(model_name, with_stats=True)
xgb.rabit.tracker_print("Finished training\n")

# Notify the tracker all training has been successful
# This is only needed in distributed training.
xgb.rabit.finalize()
