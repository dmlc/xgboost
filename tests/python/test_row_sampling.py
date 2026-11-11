"""Regression tests for booster-owned CPU row sampling."""

from typing import Any

import numpy as np
import pytest

import xgboost as xgb
from xgboost import testing as tm
from xgboost.testing.sampling import (
    check_parallel_samples,
    check_projected_sampling,
    check_row_sampling,
)


@pytest.mark.parametrize("tree_method", ["exact", "hist", "approx"])
@pytest.mark.parametrize("sampling_method", ["uniform", "gradient_based"])
@pytest.mark.parametrize("subsample", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("parallel_trees", [1, 3])
def test_row_sampling(
    tree_method: str, sampling_method: str, subsample: float, parallel_trees: int
) -> None:
    check_row_sampling("cpu", tree_method, sampling_method, subsample, parallel_trees)


@pytest.mark.parametrize("tree_method", ["exact", "hist", "approx"])
@pytest.mark.parametrize("sampling_method", ["uniform", "gradient_based"])
def test_parallel_samples(tree_method: str, sampling_method: str) -> None:
    check_parallel_samples("cpu", tree_method, sampling_method)


@pytest.mark.parametrize("sampling_method", ["uniform", "gradient_based"])
@pytest.mark.parametrize("ratio", [0.5, 1.0])
@pytest.mark.parametrize("parallel_trees", [1, 3])
def test_projected_sampling(
    sampling_method: str, ratio: float, parallel_trees: int
) -> None:
    check_projected_sampling("cpu", sampling_method, ratio, parallel_trees)


@pytest.mark.parametrize("sampling_method", ["uniform", "gradient_based"])
@pytest.mark.parametrize("parallel_trees", [1, 3])
def test_approx_updates_monotone_constraints(
    sampling_method: str, parallel_trees: int
) -> None:
    """Reusing updater storage must not retain the previous tree's constraints."""
    X: np.ndarray = np.arange(128, dtype=np.float32).reshape(-1, 1)
    data = xgb.DMatrix(X, label=-X[:, 0], nthread=2)
    model = xgb.Booster(
        {
            "tree_method": "approx",
            "objective": "reg:squarederror",
            "sampling_method": sampling_method,
            "subsample": 0.5,
            "num_parallel_tree": parallel_trees,
            "monotone_constraints": "(1)",
            "max_depth": 2,
            "learning_rate": 1.0,
            "base_score": 0.0,
            "reg_lambda": 0.0,
            "seed": 41,
            "nthread": 2,
        },
        [data],
    )
    model.update(data, 0)
    # An increasing constraint prevents fitting this decreasing target.
    assert np.ptp(model.predict(data)) == 0
    model.set_param({"monotone_constraints": "(-1)"})
    model.update(data, 1)
    pred = model.predict(data)
    assert np.ptp(pred) > 0
    assert np.all(np.diff(pred) <= 0)
    np.testing.assert_array_equal(
        pred, xgb.Booster(model_file=model.save_raw()).predict(data)
    )


def _distributed_worker(env: dict, params: dict, worker_id: int) -> bytearray:
    with xgb.collective.CommunicatorContext(**env, dmlc_task_id=str(worker_id)):
        rng = np.random.default_rng(worker_id)
        X = rng.normal(size=(96, 4)).astype(np.float32)
        y = rng.normal(size=96).astype(np.float32)
        data = xgb.DMatrix(X, label=y, nthread=2)
        results: dict = {}
        model = xgb.train(
            params,
            data,
            num_boost_round=3,
            evals=[(data, "train")],
            evals_result=results,
            verbose_eval=False,
        )
        cached = model.predict(data)
        raw = model.save_raw()
        np.testing.assert_array_equal(cached, xgb.Booster(model_file=raw).predict(data))
        return raw


@pytest.mark.skipif(**tm.no_loky())
@pytest.mark.parametrize("tree_method", ["hist", "approx"])
@pytest.mark.parametrize("sampling_method", ["uniform", "gradient_based"])
def test_distributed_sampling(tree_method: str, sampling_method: str) -> None:
    from loky import get_reusable_executor

    params: dict[str, Any] = {
        "tree_method": tree_method,
        "sampling_method": sampling_method,
        "subsample": 0.5,
        "num_parallel_tree": 3,
        "max_depth": 2,
        "seed": 11,
        "nthread": 2,
    }
    tracker = xgb.RabitTracker(n_workers=2, host_ip="127.0.0.1", timeout=60)
    tracker.start()
    with get_reusable_executor(max_workers=2) as pool:
        futures = [
            pool.submit(_distributed_worker, tracker.worker_args(), params, i)
            for i in range(2)
        ]
        models = [future.result(timeout=90) for future in futures]
    tracker.wait_for(timeout=10)
    assert models[0] == models[1]
