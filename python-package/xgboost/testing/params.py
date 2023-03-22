"""Strategies for updater tests."""

from typing import cast

import pytest

strategies = pytest.importorskip("hypothesis.strategies")


exact_parameter_strategy = strategies.fixed_dictionaries(
    {
        "nthread": strategies.integers(1, 4),
        "max_depth": strategies.integers(1, 11),
        "min_child_weight": strategies.floats(0.5, 2.0),
        "alpha": strategies.floats(1e-5, 2.0),
        "lambda": strategies.floats(1e-5, 2.0),
        "eta": strategies.floats(0.01, 0.5),
        "gamma": strategies.floats(1e-5, 2.0),
        "seed": strategies.integers(0, 10),
        # We cannot enable subsampling as the training loss can increase
        # 'subsample': strategies.floats(0.5, 1.0),
        "colsample_bytree": strategies.floats(0.5, 1.0),
        "colsample_bylevel": strategies.floats(0.5, 1.0),
    }
)

hist_parameter_strategy = strategies.fixed_dictionaries(
    {
        "max_depth": strategies.integers(1, 11),
        "max_leaves": strategies.integers(0, 1024),
        "max_bin": strategies.integers(2, 512),
        "grow_policy": strategies.sampled_from(["lossguide", "depthwise"]),
        "min_child_weight": strategies.floats(0.5, 2.0),
        # We cannot enable subsampling as the training loss can increase
        # 'subsample': strategies.floats(0.5, 1.0),
        "colsample_bytree": strategies.floats(0.5, 1.0),
        "colsample_bylevel": strategies.floats(0.5, 1.0),
    }
).filter(
    lambda x: (cast(int, x["max_depth"]) > 0 or cast(int, x["max_leaves"]) > 0)
    and (cast(int, x["max_depth"]) > 0 or x["grow_policy"] == "lossguide")
)

hist_multi_parameter_strategy = strategies.fixed_dictionaries(
    {
        "max_depth": strategies.integers(1, 11),
        "max_leaves": strategies.integers(0, 1024),
        "max_bin": strategies.integers(2, 512),
        "multi_strategy": strategies.sampled_from(
            ["multi_output_tree", "one_output_per_tree"]
        ),
        "grow_policy": strategies.sampled_from(["lossguide", "depthwise"]),
        "min_child_weight": strategies.floats(0.5, 2.0),
        # We cannot enable subsampling as the training loss can increase
        # 'subsample': strategies.floats(0.5, 1.0),
        "colsample_bytree": strategies.floats(0.5, 1.0),
        "colsample_bylevel": strategies.floats(0.5, 1.0),
    }
).filter(
    lambda x: (cast(int, x["max_depth"]) > 0 or cast(int, x["max_leaves"]) > 0)
    and (cast(int, x["max_depth"]) > 0 or x["grow_policy"] == "lossguide")
)

cat_parameter_strategy = strategies.fixed_dictionaries(
    {
        "max_cat_to_onehot": strategies.integers(1, 128),
        "max_cat_threshold": strategies.integers(1, 128),
    }
)
