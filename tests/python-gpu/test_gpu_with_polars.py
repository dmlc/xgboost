"""GPU-side polars Categorical regressions for the sparse-codes fix.

Auto-skipped when polars is not installed; the surrounding tests/python-gpu/ suite
already requires CUDA, so the CPU<->GPU training-metric parity check below will
exercise both paths through the same dictionary.
"""

import os
import time
from typing import Any

import numpy as np
import pytest

import xgboost as xgb

pl = pytest.importorskip("polars")


def _make_sparse_categorical_df(
    rng: np.random.Generator, n_rows: int, n_real_cats: int, primer_size: int
) -> Any:
    """Build a polars Categorical with codes that land at sparse positions.

    Primes the StringCache with `primer_size` strings before constructing the real
    feature so the actual categorical codes start at offset >= primer_size, matching
    the sparse-codes reproducer.
    """
    with pl.StringCache():
        # bind the primer Series to a local so its StringCache entries outlive it
        primer = pl.Series(
            "primer", [f"primer_{i}" for i in range(primer_size)], dtype=pl.Categorical
        )
        _unused = primer
        cats = [f"cat_{i:02d}" for i in range(n_real_cats)]
        col = pl.Series("f0", rng.choice(cats, size=n_rows), dtype=pl.Categorical)
    return pl.DataFrame({"f0": col, "f1": rng.normal(size=n_rows)})


def test_categorical_sparse_codes_cpu_gpu_parity() -> None:
    """CPU vs GPU training-metric parity on a polars Categorical with sparse codes.

    A single QuantileDMatrix is consumed by both `device=cpu` and `device=cuda`
    train calls; the per-iteration train RMSE must match within rtol=1e-2 (the
    tolerance band used by tests/python-gpu/test_gpu_updaters.py::test_gpu_hist).
    A regression in the categorical bitfield sizing or split path misroutes some
    fraction of categories and breaks the bound.
    """
    rng = np.random.default_rng(2030)
    n_rows = 1024
    with pl.StringCache():
        df = _make_sparse_categorical_df(
            rng, n_rows=n_rows, n_real_cats=12, primer_size=20_000
        )
        y_floats = rng.normal(size=n_rows).astype(np.float32)

        params = {
            "tree_method": "hist",
            "objective": "reg:squarederror",
            "max_depth": 6,
            "eta": 0.1,
            "seed": 0,
        }
        n_rounds = 16

        dtrain = xgb.QuantileDMatrix(df, y_floats, enable_categorical=True)

        # CPU train must run first; the device=cuda call below migrates cuts to the
        # device and a subsequent CPU train on the same DMatrix would migrate back
        cpu_eval: dict = {}
        xgb.train(
            {**params, "device": "cpu"},
            dtrain,
            num_boost_round=n_rounds,
            evals=[(dtrain, "train")],
            evals_result=cpu_eval,
            verbose_eval=False,
        )

        gpu_eval: dict = {}
        xgb.train(
            {**params, "device": "cuda"},
            dtrain,
            num_boost_round=n_rounds,
            evals=[(dtrain, "train")],
            evals_result=gpu_eval,
            verbose_eval=False,
        )

    cpu_rmse = np.asarray(cpu_eval["train"]["rmse"], dtype=np.float64)
    gpu_rmse = np.asarray(gpu_eval["train"]["rmse"], dtype=np.float64)
    assert cpu_rmse.shape == gpu_rmse.shape == (n_rounds,)
    assert np.isfinite(cpu_rmse).all()
    assert np.isfinite(gpu_rmse).all()
    np.testing.assert_allclose(cpu_rmse, gpu_rmse, rtol=1e-2)


@pytest.mark.parametrize("primer_size", [1_000, 50_000, 500_000])
def test_categorical_sparse_codes_gpu_bench(
    primer_size: int, capsys: pytest.CaptureFixture[str]
) -> None:
    """Times xgb.train(device=cuda) under a sparse-codes dictionary.

    QuantileDMatrix from a host polars df runs CPU-side construction; the GPU hot
    path is engaged only at train time when cut values reach device memory. The
    env-gated assertion catches an O(primer_size) cut-layout regression.
    """
    rng = np.random.default_rng(2032)
    n_real_cats = 16
    n_rows = 2048
    with pl.StringCache():
        primer = pl.Series(
            "primer",
            [f"primer_{i}" for i in range(primer_size)],
            dtype=pl.Categorical,
        )
        _unused = primer
        cats = [f"cat_{i:02d}" for i in range(n_real_cats)]
        df = pl.DataFrame(
            {"f0": pl.Series("f0", rng.choice(cats, size=n_rows), dtype=pl.Categorical)}
        )
        y = rng.integers(0, 2, size=n_rows)
        dm = xgb.QuantileDMatrix(df, y, enable_categorical=True)
        t0 = time.perf_counter()
        booster = xgb.train(
            {"tree_method": "hist", "device": "cuda", "objective": "binary:logistic"},
            dm,
            num_boost_round=1,
        )
        gpu_train_t = time.perf_counter() - t0

    # explicit env gate keeps CI green on contended runners; with the gate set, a
    # regression that re-introduces O(primer_size) cut-value materialisation pushes
    # train well past 10s on this primer range
    if os.environ.get("XGB_PERF_ASSERT") == "1":
        assert gpu_train_t < 10.0, (
            f"gpu_train_t={gpu_train_t:.3f}s for primer={primer_size}"
        )
    pred = booster.inplace_predict(df)
    assert np.isfinite(pred).all()
    with capsys.disabled():
        print(
            f"[sparse_codes_gpu_bench] primer_size={primer_size:>7d} "
            f"n_real_cats={n_real_cats} gpu_train_s={gpu_train_t:.4f}"
        )
