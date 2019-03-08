"""Common functions for distributed GPU tests."""
import time
import xgboost as xgb


def run_test(name, params_fun):
    """Runs a distributed GPU test."""
    # Always call this before using distributed module
    xgb.rabit.init()
    rank = xgb.rabit.get_rank()
    world = xgb.rabit.get_world_size()

    # Load file, file will be automatically sharded in distributed mode.
    dtrain = xgb.DMatrix('../../demo/data/agaricus.txt.train')
    dtest = xgb.DMatrix('../../demo/data/agaricus.txt.test')

    params = params_fun(rank)

    # Specify validations set to watch performance
    watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    num_round = 20

    # Run training, all the features in training API is available.
    # Currently, this script only support calling train once for fault recovery purpose.
    bst = xgb.train(params, dtrain, num_round, watchlist, early_stopping_rounds=2)

    # Have each worker save its model
    model_name = "test.model.%s.%d" % (name, rank)
    bst.dump_model(model_name, with_stats=True)
    time.sleep(2)
    xgb.rabit.tracker_print("Finished training\n")

    if (rank == 0):
        for i in range(0, world):
            model_name_root = "test.model.%s.%d" % (name, i)
            for j in range(0, world):
                if i == j:
                    continue
                with open(model_name_root, 'r') as model_root:
                    contents_root = model_root.read()
                    model_name_rank = "test.model.%s.%d" % (name, j)
                    with open(model_name_rank, 'r') as model_rank:
                        contents_rank = model_rank.read()
                        if contents_root != contents_rank:
                            raise Exception(
                                ('Worker models diverged: test.model.%s.%d '
                                 'differs from test.model.%s.%d') % (name, i, name, j))

    xgb.rabit.finalize()
