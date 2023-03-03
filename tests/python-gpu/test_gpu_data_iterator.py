import sys

import pytest
from hypothesis import given, settings, strategies

from xgboost.testing import no_cupy

sys.path.append("tests/python")
from test_data_iterator import run_data_iterator
from test_data_iterator import test_single_batch as cpu_single_batch


def test_gpu_single_batch() -> None:
    cpu_single_batch("gpu_hist")


@pytest.mark.skipif(**no_cupy())
@given(
    strategies.integers(0, 1024),
    strategies.integers(1, 7),
    strategies.integers(0, 8),
    strategies.booleans(),
    strategies.booleans(),
)
@settings(deadline=None, max_examples=10, print_blob=True)
def test_gpu_data_iterator(
    n_samples_per_batch: int,
    n_features: int,
    n_batches: int,
    subsample: bool,
    use_cupy: bool,
) -> None:
    run_data_iterator(
        n_samples_per_batch, n_features, n_batches, "gpu_hist", subsample, use_cupy
    )


def test_cpu_data_iterator() -> None:
    """Make sure CPU algorithm can handle GPU inputs"""
    run_data_iterator(1024, 2, 3, "approx", False, True)
