#!/usr/bin/python
import xgboost as xgb

# Always call this before using distributed module
xgb.rabit.init()
rank = xgb.rabit.get_rank()
world = xgb.rabit.get_world_size()

# Load file, file will be automatically sharded in distributed mode.
dtrain = xgb.DMatrix('../../demo/data/agaricus.txt.train')
dtest = xgb.DMatrix('../../demo/data/agaricus.txt.test')

# Specify parameters via map, definition are same as c++ version
param = {'n_gpus': 2, 'gpu_id': 2*rank, 'tree_method': 'gpu_hist', 'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic' }

# Specify validations set to watch performance
watchlist  = [(dtest,'eval'), (dtrain,'train')]
num_round = 20

# Run training, all the features in training API is available.
# Currently, this script only support calling train once for fault recovery purpose.
bst = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=2)

# Have each worker save its model
model_name = "test.model.2x2." + str(rank)
bst.dump_model(model_name, with_stats=True)
xgb.rabit.tracker_print("Finished training\n")

if (rank == 0):
    for i in range(0, world):
        model_name_root = "test.model.2x2." + str(i)
        for j in range(0, world):
            if i != j:
                model_name_rank = "test.model.2x2." + str(j)
                with open(model_name_root, 'r') as model_root:
                    with open(model_name_rank, 'r') as model_rank:
                        diff = set(model_root).difference(model_rank)
                if len(diff) != 0:
                    raise Exception('Worker models diverged: test.model.2x2.{} differs from test.model.2x2.{}'.format(i, j))

# Notify the tracker all training has been successful
# This is only needed in distributed training.
xgb.rabit.finalize()
