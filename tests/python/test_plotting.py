# -*- coding: utf-8 -*-
import numpy as np
import xgboost as xgb
import testing as tm
import unittest

try:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.axes import Axes
    from graphviz import Digraph
except ImportError:
    pass


tm._skip_if_no_matplotlib()


dpath = 'demo/data/'
rng = np.random.RandomState(1994)


class TestPlotting(unittest.TestCase):

    def test_plotting(self):
        bst2 = xgb.Booster(model_file='xgb.model')

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
            assert p.get_facecolor() == (1.0, 0, 0, 1.0)  # red

        ax = xgb.plot_importance(bst2, color=['r', 'r', 'b', 'b'],
                                 title=None, xlabel=None, ylabel=None)
        assert isinstance(ax, Axes)
        assert ax.get_title() == ''
        assert ax.get_xlabel() == ''
        assert ax.get_ylabel() == ''
        assert len(ax.patches) == 4
        assert ax.patches[0].get_facecolor() == (1.0, 0, 0, 1.0)  # red
        assert ax.patches[1].get_facecolor() == (1.0, 0, 0, 1.0)  # red
        assert ax.patches[2].get_facecolor() == (0, 0, 1.0, 1.0)  # blue
        assert ax.patches[3].get_facecolor() == (0, 0, 1.0, 1.0)  # blue

        g = xgb.to_graphviz(bst2, num_trees=0)
        assert isinstance(g, Digraph)

        ax = xgb.plot_tree(bst2, num_trees=0)
        assert isinstance(ax, Axes)

    def test_importance_plot_lim(self):
        np.random.seed(1)
        dm = xgb.DMatrix(np.random.randn(100, 100), label=[0, 1] * 50)
        bst = xgb.train({}, dm)
        assert len(bst.get_fscore()) == 71
        ax = xgb.plot_importance(bst)
        assert ax.get_xlim() == (0., 11.)
        assert ax.get_ylim() == (-1., 71.)

        ax = xgb.plot_importance(bst, xlim=(0, 5), ylim=(10, 71))
        assert ax.get_xlim() == (0., 5.)
        assert ax.get_ylim() == (10., 71.)
