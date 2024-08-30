import sys

import pytest
from hypothesis import given, settings, strategies

from xgboost.testing import no_cupy
from xgboost.testing.updater import check_extmem_qdm, check_quantile_loss_extmem

sys.path.append("tests/python")
from test_data_iterator import run_data_iterator
from test_data_iterator import test_single_batch as cpu_single_batch


def test_gpu_single_batch() -> None:
    cpu_single_batch("hist", "cuda")


@pytest.mark.skipif(**no_cupy())
@given(
    strategies.integers(0, 1024),
    strategies.integers(1, 7),
    strategies.integers(0, 8),
    strategies.booleans(),
    strategies.booleans(),
    strategies.booleans(),
)
@settings(deadline=None, max_examples=16, print_blob=True)
def test_gpu_data_iterator(
    n_samples_per_batch: int,
    n_features: int,
    n_batches: int,
    subsample: bool,
    use_cupy: bool,
    on_host: bool,
) -> None:
    run_data_iterator(
        n_samples_per_batch,
        n_features,
        n_batches,
        "hist",
        subsample=subsample,
        device="cuda",
        use_cupy=use_cupy,
        on_host=on_host,
    )


def test_cpu_data_iterator() -> None:
    """Make sure CPU algorithm can handle GPU inputs"""
    run_data_iterator(
        1024,
        2,
        3,
        "approx",
        device="cuda",
        subsample=False,
        use_cupy=True,
        on_host=False,
    )


@given(
    strategies.integers(1, 2048),
    strategies.integers(1, 8),
    strategies.integers(1, 4),
    strategies.booleans(),
)
@settings(deadline=None, max_examples=10, print_blob=True)
def test_extmem_qdm(
    n_samples_per_batch: int, n_features: int, n_batches: int, on_host: bool
) -> None:
    check_extmem_qdm(n_samples_per_batch, n_features, n_batches, "cuda", on_host)
