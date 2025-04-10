"""Helpers for testing monotone constraints."""

from typing import Optional

import numpy as np

from .._typing import FeatureNames
from ..core import Booster, DMatrix


def is_increasing(v: np.ndarray) -> bool:
    """Whether is v increasing."""
    return np.count_nonzero(np.diff(v) < 0.0) == 0


def is_decreasing(v: np.ndarray) -> bool:
    """Whether is v decreasing."""
    return np.count_nonzero(np.diff(v) > 0.0) == 0


def is_correctly_constrained(
    learner: Booster, feature_names: Optional[FeatureNames] = None
) -> bool:
    """Whether the monotone constraint is correctly applied."""
    n = 100
    variable_x = np.linspace(0, 1, n).reshape((n, 1))
    fixed_xs_values = np.linspace(0, 1, n)

    for i in range(n):
        fixed_x = fixed_xs_values[i] * np.ones((n, 1))
        monotonically_increasing_x = np.column_stack((variable_x, fixed_x))
        monotonically_increasing_dset = DMatrix(
            monotonically_increasing_x, feature_names=feature_names
        )
        monotonically_increasing_y = learner.predict(monotonically_increasing_dset)

        monotonically_decreasing_x = np.column_stack((fixed_x, variable_x))
        monotonically_decreasing_dset = DMatrix(
            monotonically_decreasing_x, feature_names=feature_names
        )
        monotonically_decreasing_y = learner.predict(monotonically_decreasing_dset)

        if not (
            is_increasing(monotonically_increasing_y)
            and is_decreasing(monotonically_decreasing_y)
        ):
            return False

    return True


NUMBER_OF_DPOINTS = 1000
x1_positively_correlated_with_y = np.random.random(size=NUMBER_OF_DPOINTS)
x2_negatively_correlated_with_y = np.random.random(size=NUMBER_OF_DPOINTS)

x = np.column_stack((x1_positively_correlated_with_y, x2_negatively_correlated_with_y))
zs = np.random.normal(loc=0.0, scale=0.01, size=NUMBER_OF_DPOINTS)
y = (
    5 * x1_positively_correlated_with_y
    + np.sin(10 * np.pi * x1_positively_correlated_with_y)
    - 5 * x2_negatively_correlated_with_y
    - np.cos(10 * np.pi * x2_negatively_correlated_with_y)
    + zs
)
training_dset = DMatrix(x, label=y)
