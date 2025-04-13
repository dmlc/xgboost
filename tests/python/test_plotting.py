import numpy as np
import pytest

import xgboost as xgb
from xgboost import testing as tm
from xgboost.testing.plotting import run_categorical

try:
    import matplotlib

    matplotlib.use("Agg")
    from graphviz import Source
    from matplotlib.axes import Axes
except ImportError:
    pass

pytestmark = pytest.mark.skipif(**tm.no_multiple(tm.no_matplotlib(), tm.no_graphviz()))


class TestPlotting:
    def test_plotting(self):
        m, _ = tm.load_agaricus(__file__)
        booster = xgb.train(
            {"max_depth": 2, "eta": 1, "objective": "binary:logistic"},
            m,
            num_boost_round=2,
        )

        ax = xgb.plot_importance(booster)
        assert isinstance(ax, Axes)
        assert ax.get_title() == "Feature importance"
        assert ax.get_xlabel() == "Importance score"
        assert ax.get_ylabel() == "Features"
        assert len(ax.patches) == 4

        ax = xgb.plot_importance(booster, color="r", title="t", xlabel="x", ylabel="y")
        assert isinstance(ax, Axes)
        assert ax.get_title() == "t"
        assert ax.get_xlabel() == "x"
        assert ax.get_ylabel() == "y"
        assert len(ax.patches) == 4
        for p in ax.patches:
            assert p.get_facecolor() == (1.0, 0, 0, 1.0)  # red

        ax = xgb.plot_importance(
            booster, color=["r", "r", "b", "b"], title=None, xlabel=None, ylabel=None
        )
        assert isinstance(ax, Axes)
        assert ax.get_title() == ""
        assert ax.get_xlabel() == ""
        assert ax.get_ylabel() == ""
        assert len(ax.patches) == 4
        assert ax.patches[0].get_facecolor() == (1.0, 0, 0, 1.0)  # red
        assert ax.patches[1].get_facecolor() == (1.0, 0, 0, 1.0)  # red
        assert ax.patches[2].get_facecolor() == (0, 0, 1.0, 1.0)  # blue
        assert ax.patches[3].get_facecolor() == (0, 0, 1.0, 1.0)  # blue

        g = xgb.to_graphviz(booster, tree_idx=0)
        assert isinstance(g, Source)

        ax = xgb.plot_tree(booster, tree_idx=0)
        assert isinstance(ax, Axes)

    def test_importance_plot_lim(self):
        np.random.seed(1)
        dm = xgb.DMatrix(np.random.randn(100, 100), label=[0, 1] * 50)
        bst = xgb.train({}, dm)
        assert len(bst.get_fscore()) == 71
        ax = xgb.plot_importance(bst)
        assert ax.get_xlim() == (0.0, 11.0)
        assert ax.get_ylim() == (-1.0, 71.0)

        ax = xgb.plot_importance(bst, xlim=(0, 5), ylim=(10, 71))
        assert ax.get_xlim() == (0.0, 5.0)
        assert ax.get_ylim() == (10.0, 71.0)

    @pytest.mark.skipif(**tm.no_pandas())
    def test_categorical(self) -> None:
        run_categorical("approx", "cpu")
