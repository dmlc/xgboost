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

def test_plotting():
    bst2 = xgb.Booster(model_file='xgb.model')
    # plotting

    import matplotlib
    matplotlib.use('Agg')

    from matplotlib.axes import Axes
    from graphviz import Digraph

    ax = xgb.plot_importance(bst2)
    assert isinstance(ax, Axes)
    assert ax.get_title() == 'Feature importance'
    assert ax.get_xlabel() == 'F score'
    assert ax.get_ylabel() == 'Features'
    assert len(ax.patches) == 4

    ax = xgb.plot_importance(bst2, color='r',
                             title='t', xlabel='x', ylabel='y')
    assert isinstance(ax, Axes)
    assert ax.get_title() == 't'
    assert ax.get_xlabel() == 'x'
    assert ax.get_ylabel() == 'y'
    assert len(ax.patches) == 4
    for p in ax.patches:
        assert p.get_facecolor() == (1.0, 0, 0, 1.0) # red


    ax = xgb.plot_importance(bst2, color=['r', 'r', 'b', 'b'],
                             title=None, xlabel=None, ylabel=None)
    assert isinstance(ax, Axes)
    assert ax.get_title() == ''
    assert ax.get_xlabel() == ''
    assert ax.get_ylabel() == ''
    assert len(ax.patches) == 4
    assert ax.patches[0].get_facecolor() == (1.0, 0, 0, 1.0) # red
    assert ax.patches[1].get_facecolor() == (1.0, 0, 0, 1.0) # red
    assert ax.patches[2].get_facecolor() == (0, 0, 1.0, 1.0) # blue
    assert ax.patches[3].get_facecolor() == (0, 0, 1.0, 1.0) # blue

    g = xgb.to_graphviz(bst2, num_trees=0)
    assert isinstance(g, Digraph)
    ax = xgb.plot_tree(bst2, num_trees=0)
    assert isinstance(ax, Axes)
