"""Basic distributed tests."""
import argparse
import filecmp
import os

import xgboost as xgb


def run_test(args):
    """Runs the test."""

    # Always call this before using distributed module
    xgb.rabit.init()

    # Set the visible GPU per worker
    rank = xgb.rabit.get_rank()
    if args.device_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids[rank]

    # Load file, file will be automatically sharded in distributed mode.
    dtrain = xgb.DMatrix('../../demo/data/agaricus.txt.train')
    dtest = xgb.DMatrix('../../demo/data/agaricus.txt.test')

    # Specify parameters via map, definition are same as c++ version
    param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic',
             'tree_method': args.tree_method, 'n_gpus': -1}

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


def main():
    """The main function.

    Defines and parses command line arguments and calls the test.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree_method', default='auto')
    parser.add_argument('--device_ids', nargs='+', help='Device IDs for each worker')
    args = parser.parse_args()

    run_test(args)


if __name__ == '__main__':
    main()
