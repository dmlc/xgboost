"""Tests for parsing trees."""

import json

import numpy as np
import pytest
from sklearn.datasets import make_regression

from ..core import Booster, DMatrix, QuantileDMatrix
from ..sklearn import XGBRegressor
from ..training import train
from .data import make_categorical
from .updater import ResetStrategy
from .utils import Device


def run_tree_to_df_categorical(tree_method: str, device: Device) -> None:
    """Tests tree_to_df with categorical features."""

    import pandas as pd

    X, y = make_categorical(100, 10, 31, onehot=False)
    Xy = DMatrix(X, y, enable_categorical=True)
    booster = train(
        {"tree_method": tree_method, "device": device}, Xy, num_boost_round=10
    )
    df = booster.trees_to_dataframe()

    all_ids = set(df["ID"])
    for _, x in df.iterrows():
        if x["Feature"] == "Leaf":
            # A leaf carries its scalar weight in ``Gain`` and no split info.
            assert pd.isna(x["Split"])
            assert not isinstance(x["Category"], list)
            assert pd.isna(x["Yes"])
            assert pd.isna(x["No"])
            assert pd.isna(x["Missing"])
        else:
            # A categorical split has a missing threshold and a non-empty list of
            # integer category codes rendered as strings.
            assert pd.isna(x["Split"])
            assert isinstance(x["Category"], list) and len(x["Category"]) >= 1
            assert all(isinstance(c, str) and int(c) >= 0 for c in x["Category"])
            # Branch ids must reference existing nodes in the same frame.
            assert x["Yes"] in all_ids
            assert x["No"] in all_ids
            assert x["Missing"] in all_ids


def _expected_leaf_vectors(booster: Booster) -> dict[tuple[int, int], list[float]]:
    """Map ``(tree_id, node_id) -> list[float]`` of leaf outputs."""
    model = json.loads(booster.save_raw(raw_format="json"))
    trees = model["learner"]["gradient_booster"]["model"]["trees"]
    out: dict[tuple[int, int], list[float]] = {}
    for tid, tree in enumerate(trees):
        n_targets = int(tree["tree_param"]["size_leaf_vector"])
        if n_targets <= 1:
            continue
        left = tree["left_children"]
        right = tree["right_children"]
        leaf_weights = tree["leaf_weights"]
        for nid, lc in enumerate(left):
            if lc == -1:  # leaf
                beg = right[nid] * n_targets
                out[(tid, nid)] = leaf_weights[beg : beg + n_targets]
    return out


def run_tree_to_df_vector_leaf_mixed(device: Device) -> None:
    """Tests trees_to_dataframe on a mixed scalar + vector-leaf booster."""
    n_targets = 3
    X, y = make_regression(
        n_samples=512, n_features=10, n_targets=n_targets, random_state=2025
    )
    booster = train(
        {"device": device, "multi_strategy": "multi_output_tree", "max_depth": 3},
        QuantileDMatrix(X, y),
        num_boost_round=6,
        callbacks=[ResetStrategy()],
    )

    df = booster.trees_to_dataframe()
    assert str(df["Gain"].dtype) == "Float64"
    assert "Target" in df.columns

    expected = _expected_leaf_vectors(booster)
    # Tree ids that use vector leaves (each such tree has at least one leaf).
    vector_tids = {tid for tid, _ in expected}
    assert len(vector_tids) > 0

    all_ids = set(df["ID"])
    n_vector_leaf_rows = 0
    for _, x in df.iterrows():
        tid = int(x["Tree"])
        key = (tid, int(x["Node"]))
        if x["Feature"] == "Leaf":
            # Leaf rows don't carry split information
            assert pd.isna(x["Split"])
            assert x["Category"] is None
            assert pd.isna(x["Yes"])
            assert pd.isna(x["No"])
            assert pd.isna(x["Missing"])
            if key in expected:
                # Vector-leaf leaf: expanded into one row per target, with the scalar
                # Gain equal to the target-th entry of the leaf vector.
                n_vector_leaf_rows += 1
                target = int(x["Target"])
                assert target in range(n_targets)
                np.testing.assert_allclose(float(x["Gain"]), expected[key][target])
            else:
                # Scalar-tree leaf: a single row carrying the tree's concrete target.
                assert not pd.isna(x["Target"])
        else:
            assert isinstance(float(x["Gain"]), float)
            assert x["Yes"] in all_ids
            assert x["No"] in all_ids
            assert x["Missing"] in all_ids
            if tid in vector_tids:
                # A split node does not have a concept of target.
                assert pd.isna(x["Target"])
            else:
                # A scalar-tree split belongs to that tree's target.
                assert not pd.isna(x["Target"])
    # Every vector-leaf node expands into exactly `n_targets` rows.
    assert n_vector_leaf_rows == n_targets * len(expected)


def run_split_value_histograms(tree_method: str, device: Device) -> None:
    """Tests split_value_histograms with categorical features."""
    X, y = make_categorical(1000, 10, 13, onehot=False)
    reg = XGBRegressor(tree_method=tree_method, enable_categorical=True, device=device)
    reg.fit(X, y)

    with pytest.raises(ValueError, match="doesn't"):
        reg.get_booster().get_split_value_histogram("3", bins=5)
