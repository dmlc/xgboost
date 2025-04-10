"""Test plotting functions for XGBoost."""

import json

from graphviz import Source
from matplotlib.axes import Axes

from ..plotting import plot_tree, to_graphviz
from ..sklearn import XGBRegressor
from .data import make_categorical
from .utils import Device


def run_categorical(tree_method: str, device: Device) -> None:
    """Tests plotting functions for categorical features."""
    X, y = make_categorical(1000, 31, 19, onehot=False)
    reg = XGBRegressor(
        enable_categorical=True, n_estimators=10, tree_method=tree_method, device=device
    )
    reg.fit(X, y)
    trees = reg.get_booster().get_dump(dump_format="json")
    for tree in trees:
        j_tree = json.loads(tree)
        assert "leaf" in j_tree.keys() or isinstance(j_tree["split_condition"], list)

    graph = to_graphviz(reg, tree_idx=len(j_tree) - 1)
    assert isinstance(graph, Source)
    ax = plot_tree(reg, tree_idx=len(j_tree) - 1)
    assert isinstance(ax, Axes)
