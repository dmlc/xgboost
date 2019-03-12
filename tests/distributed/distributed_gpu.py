"""Distributed GPU tests."""
import sys
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

    params, n_rounds = params_fun(rank)

    # Specify validations set to watch performance
    watchlist = [(dtest, 'eval'), (dtrain, 'train')]

    # Run training, all the features in training API is available.
    # Currently, this script only support calling train once for fault recovery purpose.
    bst = xgb.train(params, dtrain, n_rounds, watchlist, early_stopping_rounds=2)

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


base_params = {
    'tree_method': 'gpu_hist',
    'max_depth': 2,
    'eta': 1,
    'verbosity': 0,
    'objective': 'binary:logistic'
}


def params_basic_1x4(rank):
    return dict(base_params, **{
        'n_gpus': 1,
        'gpu_id': rank,
    }), 20


def params_basic_2x2(rank):
    return dict(base_params, **{
        'n_gpus': 2,
        'gpu_id': 2*rank,
    }), 20


def params_basic_4x1(rank):
    return dict(base_params, **{
        'n_gpus': 4,
        'gpu_id': rank,
    }), 20


def params_basic_asym(rank):
    return dict(base_params, **{
        'n_gpus': 1 if rank == 0 else 3,
        'gpu_id': rank,
    }), 20


rf_update_params = {
    'subsample': 0.5,
    'colsample_bynode': 0.5
}


def wrap_rf(params_fun):
    def wrapped_params_fun(rank):
        params, n_estimators = params_fun(rank)
        rf_params = dict(rf_update_params, num_parallel_tree=n_estimators)
        return dict(params, **rf_params), 1
    return wrapped_params_fun


params_rf_1x4 = wrap_rf(params_basic_1x4)

params_rf_2x2 = wrap_rf(params_basic_2x2)

params_rf_4x1 = wrap_rf(params_basic_4x1)

params_rf_asym = wrap_rf(params_basic_asym)


test_name = sys.argv[1]
run_test(test_name, globals()['params_%s' % test_name])
