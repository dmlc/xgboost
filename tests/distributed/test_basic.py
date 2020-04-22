#!/usr/bin/python
import xgboost as xgb

# Always call this before using distributed module
xgb.rabit.init()

# Load file, file will be automatically sharded in distributed mode.
dtrain = xgb.DMatrix('../../demo/data/agaricus.txt.train')
dtest = xgb.DMatrix('../../demo/data/agaricus.txt.test')

# Specify parameters via map, definition are same as c++ version
param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic' }

# Specify validations set to watch performance
watchlist  = [(dtest,'eval'), (dtrain,'train')]
num_round = 20

# Run training, all the features in training API is available.
# Currently, this script only support calling train once for fault recovery purpose.
bst = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=2)

# Save the model, only ask process 0 to save the model.
if xgb.rabit.get_rank() == 0:
    bst.save_model("test.model")
    xgb.rabit.tracker_print("Finished training\n")

# Notify the tracker all training has been successful
# This is only needed in distributed training.
xgb.rabit.finalize()
