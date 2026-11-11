"""Regression tests for booster-owned CUDA row sampling."""

import pytest

from xgboost.testing.sampling import (
    check_parallel_samples,
    check_projected_sampling,
    check_row_sampling,
)


@pytest.mark.parametrize("tree_method", ["hist", "approx"])
@pytest.mark.parametrize("sampling_method", ["uniform", "gradient_based"])
@pytest.mark.parametrize("subsample", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("parallel_trees", [1, 3])
def test_row_sampling(
    tree_method: str, sampling_method: str, subsample: float, parallel_trees: int
) -> None:
    check_row_sampling("cuda", tree_method, sampling_method, subsample, parallel_trees)


@pytest.mark.parametrize("tree_method", ["hist", "approx"])
@pytest.mark.parametrize("sampling_method", ["uniform", "gradient_based"])
def test_parallel_samples(tree_method: str, sampling_method: str) -> None:
    check_parallel_samples("cuda", tree_method, sampling_method)


@pytest.mark.parametrize("sampling_method", ["uniform", "gradient_based"])
@pytest.mark.parametrize("ratio", [0.5, 1.0])
@pytest.mark.parametrize("parallel_trees", [1, 3])
def test_projected_sampling(
    sampling_method: str, ratio: float, parallel_trees: int
) -> None:
    check_projected_sampling("cuda", sampling_method, ratio, parallel_trees)
