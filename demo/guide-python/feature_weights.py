'''Using feature weight to change column sampling.

    .. versionadded:: 1.3.0
'''

import numpy as np
import xgboost
from matplotlib import pyplot as plt
import argparse


def main(args):
    rng = np.random.RandomState(1994)

    kRows = 1000
    kCols = 10

    X = rng.randn(kRows, kCols)
    y = rng.randn(kRows)
    fw = np.ones(shape=(kCols,))
    for i in range(kCols):
        fw[i] *= float(i)

    dtrain = xgboost.DMatrix(X, y)
    dtrain.feature_weights = fw

    bst = xgboost.train({'tree_method': 'hist',
                         'colsample_bynode': 0.5},
                        dtrain, num_boost_round=10,
                        evals=[(dtrain, 'd')])
    featue_map = bst.get_fscore()
    # feature zero has 0 weight
    assert featue_map.get('f0', None) is None
    assert max(featue_map.values()) == featue_map.get('f9')

    if args.plot:
        xgboost.plot_importance(bst)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--plot',
        type=int,
        default=1,
        help='Set to 0 to disable plotting the evaluation history.')
    args = parser.parse_args()
    main(args)
