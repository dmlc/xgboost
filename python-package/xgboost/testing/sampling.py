"""Checks for booster-owned row sampling across tree builders."""

import json

import numpy as np

from .. import Booster, DMatrix, train


def check_row_sampling(
    device: str,
    tree_method: str,
    sampling_method: str,
    subsample: float,
    parallel_trees: int,
) -> None:
    """Check deterministic training and cached predictions, including excluded rows."""
    rng = np.random.default_rng(17)
    X = rng.normal(size=(192, 5)).astype(np.float32)
    X[::7, 1] = np.nan
    y = (2 * X[:, 0] - X[:, 2] + rng.normal(size=192)).astype(np.float32)
    data = DMatrix(X, label=y, nthread=2)
    params = {
        "device": device,
        "tree_method": tree_method,
        "sampling_method": sampling_method,
        "subsample": subsample,
        "num_parallel_tree": parallel_trees,
        "max_depth": 3,
        "seed": 19,
        "nthread": 2,
        "base_score": 0.0,
        "objective": "reg:squarederror",
    }
    results: dict = {}
    model = train(
        params,
        data,
        num_boost_round=3,
        evals=[(data, "train")],
        evals_result=results,
        verbose_eval=False,
    )
    pred = model.predict(data)
    assert np.isfinite(pred).all()
    if device == "cuda":
        assert json.loads(model.save_config())["learner"]["generic_param"][
            "device"
        ].startswith("cuda:")
    assert model.num_boosted_rounds() == 3
    np.testing.assert_allclose(
        results["train"]["rmse"][-1],
        np.sqrt(np.mean((pred - y) ** 2)),
        rtol=1e-6,
        atol=1e-6,
    )
    reloaded = Booster(model_file=model.save_raw())
    # Cached training predictions must include rows excluded from tree statistics.
    np.testing.assert_array_equal(pred, reloaded.predict(data))
    repeated = train(params, data, num_boost_round=3)
    np.testing.assert_array_equal(pred, repeated.predict(data))


def check_parallel_samples(device: str, tree_method: str, sampling_method: str) -> None:
    """Each parallel tree must receive an independent sample of original gradients."""
    rng = np.random.default_rng(23)
    X = rng.normal(size=(256, 4)).astype(np.float32)
    y = rng.normal(size=256).astype(np.float32)
    data = DMatrix(X, label=y, nthread=2)
    model = train(
        {
            "tree_method": tree_method,
            "device": device,
            "sampling_method": sampling_method,
            "subsample": 0.5,
            "num_parallel_tree": 4,
            "seed": 31,
            "nthread": 2,
            "max_depth": 3,
        },
        data,
        num_boost_round=1,
    )
    trees = model.get_dump(dump_format="json")
    assert len(trees) == 4
    assert len(set(trees)) == 4, "Parallel trees reused a row sample"


def check_projected_sampling(
    device: str, sampling_method: str, ratio: float, parallel_trees: int
) -> None:
    """Projection's split and full leaf gradients receive the same sampling weights."""
    rng = np.random.default_rng(43)
    X = rng.normal(size=(128, 5)).astype(np.float32)
    y = (X @ rng.normal(size=(5, 4))).astype(np.float32)
    data = DMatrix(X, label=y, nthread=2)
    params = {
        "device": device,
        "tree_method": "hist",
        "sampling_method": sampling_method,
        "subsample": 0.5,
        "multi_strategy": "multi_output_tree",
        "multi_projection_ratio": ratio,
        "num_parallel_tree": parallel_trees,
        "seed": 7,
        "nthread": 2,
        "max_depth": 3,
    }
    results: dict = {}
    model = train(
        params,
        data,
        num_boost_round=3,
        evals=[(data, "train")],
        evals_result=results,
        verbose_eval=False,
    )
    pred = model.predict(data)
    assert pred.shape == y.shape
    assert np.isfinite(pred).all()
    np.testing.assert_allclose(
        results["train"]["rmse"][-1], np.sqrt(np.mean((pred - y) ** 2)), rtol=1e-6
    )
    np.testing.assert_array_equal(
        pred, Booster(model_file=model.save_raw()).predict(data)
    )
    np.testing.assert_array_equal(
        pred, train(params, data, num_boost_round=3).predict(data)
    )
