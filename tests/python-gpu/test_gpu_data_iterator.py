import numpy as np
import xgboost as xgb
from hypothesis import given, strategies, settings
import pytest
import sys

sys.path.append("tests/python")
from test_data_iterator import SingleBatch, make_batches
from test_data_iterator import test_single_batch as cpu_single_batch
from test_data_iterator import run_data_iterator
from testing import IteratorForTest


def test_gpu_single_batch() -> None:
    cpu_single_batch("gpu_hist")


@given(
    strategies.integers(0, 1024), strategies.integers(1, 7), strategies.integers(0, 13)
)
@settings(deadline=None)
def test_data_iterator(
    n_samples_per_batch: int, n_features: int, n_batches: int
) -> None:
    run_data_iterator(n_samples_per_batch, n_features, n_batches, "gpu_hist")
