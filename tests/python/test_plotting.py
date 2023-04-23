import json

import numpy as np
import pytest

import xgboost as xgb
from xgboost import testing as tm

try:
    import matplotlib
    matplotlib.use('Agg')
    from graphviz import Source
    from matplotlib.axes import Axes
except ImportError:
    pass

pytestmark = pytest.mark.skipif(**tm.no_multiple(tm.no_matplotlib(),
                                                 tm.no_graphviz()))


class TestPlotting:
    def test_plotting(self):
        m, _ = tm.load_agaricus(__file__)
        booster = xgb.train({'max_depth': 2, 'eta': 1,
                             'objective': 'binary:logistic'}, m,
                            num_boost_round=2)

        ax = xgb.plot_importance(booster)
        assert isinstance(ax, Axes)
        assert ax.get_title() == 'Feature importance'
        assert ax.get_xlabel() == 'F score'
        assert ax.get_ylabel() == 'Features'
        assert len(ax.patches) == 4

        ax = xgb.plot_importance(booster, color='r',
                                 title='t', xlabel='x', ylabel='y')
        assert isinstance(ax, Axes)
        assert ax.get_title() == 't'
        assert ax.get_xlabel() == 'x'
        assert ax.get_ylabel() == 'y'
        assert len(ax.patches) == 4
        for p in ax.patches:
            assert p.get_facecolor() == (1.0, 0, 0, 1.0)  # red

        ax = xgb.plot_importance(booster, color=['r', 'r', 'b', 'b'],
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

        g = xgb.to_graphviz(booster, num_trees=0)
        assert isinstance(g, Source)

        ax = xgb.plot_tree(booster, num_trees=0)
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

    def run_categorical(self, tree_method: str) -> None:
        X, y = tm.make_categorical(1000, 31, 19, onehot=False)
        reg = xgb.XGBRegressor(
            enable_categorical=True, n_estimators=10, tree_method=tree_method
        )
        reg.fit(X, y)
        trees = reg.get_booster().get_dump(dump_format="json")
        for tree in trees:
            j_tree = json.loads(tree)
            assert "leaf" in j_tree.keys() or isinstance(
                j_tree["split_condition"], list
            )

        graph = xgb.to_graphviz(reg, num_trees=len(j_tree) - 1)
        assert isinstance(graph, Source)
        ax = xgb.plot_tree(reg, num_trees=len(j_tree) - 1)
        assert isinstance(ax, Axes)

    @pytest.mark.skipif(**tm.no_pandas())
    def test_categorical(self) -> None:
        self.run_categorical("approx")
