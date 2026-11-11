"""Regression tests for booster-owned SYCL row sampling."""

import pytest

from xgboost.testing.sampling import check_parallel_samples, check_row_sampling


@pytest.mark.parametrize("sampling_method", ["uniform", "gradient_based"])
@pytest.mark.parametrize("subsample", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("parallel_trees", [1, 3])
def test_row_sampling(
    sampling_method: str, subsample: float, parallel_trees: int
) -> None:
    check_row_sampling("sycl", "hist", sampling_method, subsample, parallel_trees)


@pytest.mark.parametrize("sampling_method", ["uniform", "gradient_based"])
def test_parallel_samples(sampling_method: str) -> None:
    check_parallel_samples("sycl", "hist", sampling_method)
