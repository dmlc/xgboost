"""Strategies for updater tests."""

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import strategies

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
    lambda x: (x["max_depth"] > 0 or x["max_leaves"] > 0)
    and (x["max_depth"] > 0 or x["grow_policy"] == "lossguide")
)


cat_parameter_strategy = strategies.fixed_dictionaries(
    {
        "max_cat_to_onehot": strategies.integers(1, 128),
        "max_cat_threshold": strategies.integers(1, 128),
    }
)
