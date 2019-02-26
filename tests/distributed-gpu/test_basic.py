#!/usr/bin/python
import filecmp
import os

import xgboost as xgb

# Always call this before using distributed module
xgb.rabit.init()

# Set the visible GPU per worker
rank = xgb.rabit.get_rank()
nproc = xgb.rabit.get_world_size()
os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

# Load file, file will be automatically sharded in distributed mode.
dtrain = xgb.DMatrix('../../demo/data/agaricus.txt.train')
dtest = xgb.DMatrix('../../demo/data/agaricus.txt.test')

# Specify parameters via map, definition are same as c++ version
param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic', 'tree_method': 'gpu_hist'}

# Specify validations set to watch performance
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 20

# Run training, all the features in training API is available.
# Currently, this script only support calling train once for fault recovery purpose.
bst = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=2)

# Save the model from every worker.
bst.save_model('test.model.%d' % rank)
xgb.rabit.tracker_print('Finished training from rank %d\n' % rank)

# Notify the tracker all training has been successful
# This is only needed in distributed training.
xgb.rabit.finalize()

for i in range(nproc - 1):
    first = 'test.model.%d' % i
    second = 'test.model.%d' % (i + 1)
    assert filecmp.cmp(first, second), '%s and %s are different' % (first, second)
