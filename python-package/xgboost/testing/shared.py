"""Testing code shared by other tests."""

# pylint: disable=invalid-name
import collections
import importlib.util
import json
import os
import tempfile
from typing import Any, Callable, Dict, Type

import numpy as np

import xgboost as xgb
from xgboost._typing import ArrayLike


def validate_leaf_output(leaf: np.ndarray, num_parallel_tree: int) -> None:
    """Validate output for predict leaf tests."""
    for i in range(leaf.shape[0]):  # n_samples
        for j in range(leaf.shape[1]):  # n_rounds
            for k in range(leaf.shape[2]):  # n_classes
                tree_group = leaf[i, j, k, :]
                assert tree_group.shape[0] == num_parallel_tree
                # No sampling, all trees within forest are the same
                assert np.all(tree_group == tree_group[0])


def validate_data_initialization(
    dmatrix: Type, model: Type[xgb.XGBModel], X: ArrayLike, y: ArrayLike
) -> None:
    """Assert that we don't create duplicated DMatrix."""

    old_init = dmatrix.__init__
    count = [0]

    def new_init(self: Any, **kwargs: Any) -> Callable:
        count[0] += 1
        return old_init(self, **kwargs)

    dmatrix.__init__ = new_init
    model(n_estimators=1).fit(X, y, eval_set=[(X, y)])

    assert count[0] == 1
    count[0] = 0  # only 1 DMatrix is created.

    y_copy = y.copy()
    model(n_estimators=1).fit(X, y, eval_set=[(X, y_copy)])
    assert count[0] == 2  # a different Python object is considered different

    dmatrix.__init__ = old_init


# pylint: disable=too-many-arguments,too-many-locals
def get_feature_weights(
    *,
    X: ArrayLike,
    y: ArrayLike,
    fw: np.ndarray,
    parser_path: str,
    tree_method: str,
    model: Type[xgb.XGBModel] = xgb.XGBRegressor,
) -> np.ndarray:
    """Get feature weights using the demo parser."""
    with tempfile.TemporaryDirectory() as tmpdir:
        colsample_bynode = 0.5
        reg = model(
            tree_method=tree_method,
            colsample_bynode=colsample_bynode,
            feature_weights=fw,
        )

        reg.fit(X, y)
        model_path = os.path.join(tmpdir, "model.json")
        reg.save_model(model_path)
        with open(model_path, "r", encoding="utf-8") as fd:
            model = json.load(fd)

        spec = importlib.util.spec_from_file_location("JsonParser", parser_path)
        assert spec is not None
        jsonm = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(jsonm)
        model = jsonm.Model(model)
        splits: Dict[int, int] = {}
        total_nodes = 0
        for tree in model.trees:
            n_nodes = len(tree.nodes)
            total_nodes += n_nodes
            for n in range(n_nodes):
                if tree.is_leaf(n):
                    continue
                if splits.get(tree.split_index(n), None) is None:
                    splits[tree.split_index(n)] = 1
                else:
                    splits[tree.split_index(n)] += 1

        od = collections.OrderedDict(sorted(splits.items()))
        tuples = list(od.items())
        k, v = list(zip(*tuples))
        w = np.polyfit(k, v, deg=1)
        return w
