import numpy as np
import xgboost as xgb

dpath = 'demo/data/'

def test_basic():
    dtrain = xgb.DMatrix(dpath + 'agaricus.txt.train')
    dtest = xgb.DMatrix(dpath + 'agaricus.txt.test')
    param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
    # specify validations set to watch performance
    watchlist  = [(dtest,'eval'), (dtrain,'train')]
    num_round = 2
    bst = xgb.train(param, dtrain, num_round, watchlist)
    # this is prediction
    preds = bst.predict(dtest)
    labels = dtest.get_label()
    err = sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) / float(len(preds))
    # error must be smaller than 10%
    assert err < 0.1

    # save dmatrix into binary buffer
    dtest.save_binary('dtest.buffer')
    # save model
    bst.save_model('xgb.model')
    # load model and data in
    bst2 = xgb.Booster(model_file='xgb.model')
    dtest2 = xgb.DMatrix('dtest.buffer')
    preds2 = bst2.predict(dtest2)
    # assert they are the same
    assert np.sum(np.abs(preds2-preds)) == 0

