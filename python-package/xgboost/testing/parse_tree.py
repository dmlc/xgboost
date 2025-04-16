"""Tests for parsing trees."""

import pytest

from ..core import DMatrix
from ..sklearn import XGBRegressor
from ..training import train
from .data import make_categorical
from .utils import Device


def run_tree_to_df_categorical(tree_method: str, device: Device) -> None:
    """Tests tree_to_df with categorical features."""
    X, y = make_categorical(100, 10, 31, onehot=False)
    Xy = DMatrix(X, y, enable_categorical=True)
    booster = train(
        {"tree_method": tree_method, "device": device}, Xy, num_boost_round=10
    )
    df = booster.trees_to_dataframe()
    for _, x in df.iterrows():
        if x["Feature"] != "Leaf":
            assert len(x["Category"]) >= 1


def run_split_value_histograms(tree_method: str, device: Device) -> None:
    """Tests split_value_histograms with categorical features."""
    X, y = make_categorical(1000, 10, 13, onehot=False)
    reg = XGBRegressor(tree_method=tree_method, enable_categorical=True, device=device)
    reg.fit(X, y)

    with pytest.raises(ValueError, match="doesn't"):
        reg.get_booster().get_split_value_histogram("3", bins=5)
