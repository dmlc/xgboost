import numpy as np
import xgboost as xgb
from hypothesis import given, strategies, settings
import pytest
import sys

sys.path.append("tests/python")
from test_data_iterator import SingleBatch, make_batches
from test_data_iterator import test_single_batch as cpu_single_batch
from test_data_iterator import run_data_iterator
from testing import IteratorForTest, no_cupy


def test_gpu_single_batch() -> None:
    cpu_single_batch("gpu_hist")


@pytest.mark.skipif(**no_cupy())
@given(
    strategies.integers(0, 1024),
    strategies.integers(1, 7),
    strategies.integers(0, 13),
    strategies.booleans(),
)
@settings(deadline=None, print_blob=True)
def test_gpu_data_iterator(
    n_samples_per_batch: int, n_features: int, n_batches: int, subsample: bool
) -> None:
    run_data_iterator(
        n_samples_per_batch, n_features, n_batches, "gpu_hist", subsample, True
    )
    run_data_iterator(
        n_samples_per_batch, n_features, n_batches, "gpu_hist", subsample, False
    )


def test_cpu_data_iterator() -> None:
    """Make sure CPU algorithm can handle GPU inputs"""
    run_data_iterator(1024, 2, 3, "approx", False, True)
