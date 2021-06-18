import sys
import xgboost as xgb
import pytest
import json

sys.path.append("tests/python")
import testing as tm

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.axes import Axes
    from graphviz import Source
except ImportError:
    pass


pytestmark = pytest.mark.skipif(**tm.no_multiple(tm.no_matplotlib(), tm.no_graphviz()))


class TestPlotting:
    @pytest.mark.skipif(**tm.no_pandas())
    def test_categorical(self):
        X, y = tm.make_categorical(1000, 31, 19, onehot=False)
        reg = xgb.XGBRegressor(
            enable_categorical=True, n_estimators=10, tree_method="gpu_hist"
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
