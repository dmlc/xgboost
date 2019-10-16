import pickle
import numpy as np
import xgboost as xgb
import os


kRows = 100
kCols = 10


def generate_data():
    X = np.random.randn(kRows, kCols)
    y = np.random.randn(kRows)
    return X, y


def test_model_pickling():
    xgb_params = {
        'verbosity': 0,
        'nthread': 1,
        'tree_method': 'hist'
    }

    X, y = generate_data()
    dtrain = xgb.DMatrix(X, y)
    bst = xgb.train(xgb_params, dtrain)

    dump_0 = bst.get_dump(dump_format='json')
    assert dump_0

    filename = 'model.pkl'

    with open(filename, 'wb') as fd:
        pickle.dump(bst, fd)

    with open(filename, 'rb') as fd:
        bst = pickle.load(fd)

    with open(filename, 'wb') as fd:
        pickle.dump(bst, fd)

    with open(filename, 'rb') as fd:
        bst = pickle.load(fd)

    assert bst.get_dump(dump_format='json') == dump_0

    if os.path.exists(filename):
        os.remove(filename)
