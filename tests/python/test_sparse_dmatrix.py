import numpy as np
import xgboost as xgb
from scipy.sparse import rand

rng = np.random.RandomState(1)

param = {'max_depth': 3, 'objective': 'binary:logistic', 'silent': 1}


def test_sparse_dmatrix_csr():
    nrow = 100
    ncol = 1000
    x = rand(nrow, ncol, density=0.0005, format='csr', random_state=rng)
    assert x.indices.max() < ncol - 1
    x.data[:] = 1
    dtrain = xgb.DMatrix(x, label=np.random.binomial(1, 0.3, nrow))
    assert (dtrain.num_row(), dtrain.num_col()) == (nrow, ncol)
    watchlist = [(dtrain, 'train')]
    bst = xgb.train(param, dtrain, 5, watchlist)
    bst.predict(dtrain)


def test_sparse_dmatrix_csc():
    nrow = 1000
    ncol = 100
    x = rand(nrow, ncol, density=0.0005, format='csc', random_state=rng)
    assert x.indices.max() < nrow - 1
    x.data[:] = 1
    dtrain = xgb.DMatrix(x, label=np.random.binomial(1, 0.3, nrow))
    assert (dtrain.num_row(), dtrain.num_col()) == (nrow, ncol)
    watchlist = [(dtrain, 'train')]
    bst = xgb.train(param, dtrain, 5, watchlist)
    bst.predict(dtrain)
