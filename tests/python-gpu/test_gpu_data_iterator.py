import sys

import numpy as np
import pytest
from hypothesis import given, settings, strategies

import xgboost as xgb
from xgboost import testing as tm
from xgboost.testing import no_cupy
from xgboost.testing.data_iter import check_invalid_cat_batches, check_uneven_sizes
from xgboost.testing.updater import (
    check_categorical_missing,
    check_categorical_ohe,
    check_extmem_qdm,
    check_quantile_loss_extmem,
)

sys.path.append("tests/python")
from test_data_iterator import run_data_iterator
from test_data_iterator import test_single_batch as cpu_single_batch

# There are lots of warnings if XGBoost is not running on ATS-enabled systems.
pytestmark = pytest.mark.filterwarnings("ignore")


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
    strategies.integers(2, 16),
    strategies.booleans(),
)
@settings(deadline=None, max_examples=10, print_blob=True)
def test_extmem_qdm(
    n_samples_per_batch: int,
    n_features: int,
    n_batches: int,
    n_bins: int,
    on_host: bool,
) -> None:
    check_extmem_qdm(
        n_samples_per_batch,
        n_features,
        n_batches=n_batches,
        n_bins=n_bins,
        device="cuda",
        on_host=on_host,
        is_cat=False,
    )


@given(
    strategies.integers(1, 2048),
    strategies.integers(1, 4),
    strategies.integers(2, 16),
    strategies.booleans(),
)
@settings(deadline=None, max_examples=10, print_blob=True)
@pytest.mark.skipif(**tm.no_cudf())
@pytest.mark.skipif(**tm.no_cupy())
def test_categorical_extmem_qdm(
    n_samples_per_batch: int,
    n_batches: int,
    n_bins: int,
    on_host: bool,
) -> None:
    check_extmem_qdm(
        n_samples_per_batch,
        4,
        n_batches=n_batches,
        n_bins=n_bins,
        device="cuda",
        on_host=on_host,
        is_cat=True,
    )


def test_invalid_device_extmem_qdm() -> None:
    it = tm.IteratorForTest(
        *tm.make_batches(16, 4, 2, use_cupy=False), cache="cache", on_host=True
    )
    Xy = xgb.ExtMemQuantileDMatrix(it)
    with pytest.raises(ValueError, match="cannot be used for GPU"):
        xgb.train({"device": "cuda"}, Xy)

    it = tm.IteratorForTest(
        *tm.make_batches(16, 4, 2, use_cupy=True), cache="cache", on_host=True
    )
    Xy = xgb.ExtMemQuantileDMatrix(it)
    with pytest.raises(ValueError, match="cannot be used for CPU"):
        xgb.train({"device": "cpu"}, Xy)


def test_concat_pages_invalid() -> None:
    it = tm.IteratorForTest(*tm.make_batches(64, 16, 4, use_cupy=True), cache=None)
    Xy = xgb.ExtMemQuantileDMatrix(it)
    with pytest.raises(ValueError, match="can not be used with concatenated pages"):
        xgb.train(
            {
                "device": "cuda",
                "subsample": 0.5,
                "sampling_method": "gradient_based",
                "extmem_single_page": True,
                "objective": "reg:absoluteerror",
            },
            Xy,
        )


def test_concat_pages() -> None:
    boosters = []
    for min_cache_page_bytes in [0, 256, 386, np.iinfo(np.int64).max]:
        it = tm.IteratorForTest(
            *tm.make_batches(64, 16, 4, use_cupy=True),
            cache=None,
            min_cache_page_bytes=min_cache_page_bytes,
            on_host=True,
        )
        Xy = xgb.ExtMemQuantileDMatrix(it)
        booster = xgb.train(
            {
                "device": "cuda",
                "objective": "reg:absoluteerror",
            },
            Xy,
        )
        boosters.append(booster.save_raw(raw_format="json"))

    for model in boosters[1:]:
        assert str(model) == str(boosters[0])


@given(
    strategies.integers(1, 64),
    strategies.integers(1, 8),
    strategies.integers(1, 4),
)
@settings(deadline=None, max_examples=10, print_blob=True)
def test_quantile_objective(
    n_samples_per_batch: int, n_features: int, n_batches: int
) -> None:
    check_quantile_loss_extmem(
        n_samples_per_batch,
        n_features,
        n_batches,
        "hist",
        "cuda",
    )
    check_quantile_loss_extmem(
        n_samples_per_batch,
        n_features,
        n_batches,
        "approx",
        "cuda",
    )


@pytest.mark.parametrize("tree_method", ["hist", "approx"])
@pytest.mark.skipif(**tm.no_cudf())
@pytest.mark.skipif(**tm.no_cupy())
def test_categorical_missing(tree_method: str) -> None:
    check_categorical_missing(
        1024, 4, 5, device="cuda", tree_method=tree_method, extmem=True
    )


@pytest.mark.parametrize("tree_method", ["hist", "approx"])
@pytest.mark.skipif(**tm.no_cudf())
@pytest.mark.skipif(**tm.no_cupy())
def test_categorical_ohe(tree_method: str) -> None:
    check_categorical_ohe(
        rows=1024,
        cols=16,
        rounds=4,
        cats=5,
        device="cuda",
        tree_method=tree_method,
        extmem=True,
    )


@pytest.mark.skipif(**tm.no_cudf())
@pytest.mark.skipif(**tm.no_cupy())
def test_invalid_cat_batches() -> None:
    check_invalid_cat_batches("cuda")


def test_uneven_sizes() -> None:
    check_uneven_sizes("cuda")
