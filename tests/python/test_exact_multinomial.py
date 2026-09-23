"""
End-to-end validation of `multi_hessian=exact` through the public Python API.

These tests exercise the real user entry point rather than the C++ internals, so they catch
binding-level problems that the C++ suite cannot. Run with the package from this source tree:

    PYTHONPATH=python-package python -m pytest tests/python/test_exact_multinomial.py

Everything here is deterministic; no test depends on a network fetch except the Covertype
case, which skips cleanly when scikit-learn or its cached dataset is unavailable.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

import xgboost as xgb

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

BASE_PARAMS = {
    "objective": "multi:softprob",
    "tree_method": "hist",
    "device": "cpu",
    "multi_strategy": "multi_output_tree",
    "max_depth": 3,
    "eta": 0.3,
    "lambda": 1.0,
    "base_score": 0.5,
}


def params(n_classes: int, multi_hessian: str | None = None, **overrides):
    out = dict(BASE_PARAMS)
    out["num_class"] = n_classes
    if multi_hessian is not None:
        out["multi_hessian"] = multi_hessian
    out.update(overrides)
    return out


def make_data(n_rows: int, n_cols: int, n_classes: int, seed: int = 11):
    """Deterministic synthetic classification data with class-dependent signal."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, size=n_rows)
    x = rng.normal(size=(n_rows, n_cols))
    # Give each class a distinct mean so the features are informative.
    for k in range(n_classes):
        x[y == k, k % n_cols] += 1.5
    return x.astype(np.float32), y.astype(np.float32)


def train(x, y, prm, rounds: int, weights=None):
    dtrain = xgb.DMatrix(x, label=y, weight=weights)
    booster = xgb.train(prm, dtrain, num_boost_round=rounds)
    return booster, dtrain


def predict(booster, dtrain):
    return booster.predict(dtrain)


# --------------------------------------------------------------------------- #
# Phase 1 - the binding reaches the current build
# --------------------------------------------------------------------------- #


def test_library_is_from_this_source_tree():
    from xgboost.libpath import find_lib_path

    lib = Path(find_lib_path()[0]).resolve()
    repo_lib = (Path(xgb.__file__).resolve().parents[2] / "lib").resolve()
    assert lib.parent == repo_lib, f"loaded {lib}, expected a library under {repo_lib}"


@pytest.mark.parametrize("mode", ["diagonal", "exact"])
def test_parameter_is_accepted_and_reaches_cpp(mode):
    x, y = make_data(128, 4, 3)
    booster, _ = train(x, y, params(3, mode), rounds=1)
    config = json.loads(booster.save_config())
    train_param = config["learner"]["learner_train_param"]
    assert train_param["multi_hessian"] == mode


# --------------------------------------------------------------------------- #
# Phase 2 - the default path is untouched
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_classes", [2, 3, 7])
def test_default_matches_explicit_diagonal(n_classes):
    x, y = make_data(256, 6, n_classes)
    omitted, d1 = train(x, y, params(n_classes), rounds=3)
    explicit, d2 = train(x, y, params(n_classes, "diagonal"), rounds=3)
    np.testing.assert_array_equal(predict(omitted, d1), predict(explicit, d2))


# --------------------------------------------------------------------------- #
# Phase 3 - exact mode trains and produces valid output
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_classes", [2, 3, 7, 10])
def test_exact_mode_trains(n_classes):
    x, y = make_data(256, 6, n_classes)
    booster, dtrain = train(x, y, params(n_classes, "exact"), rounds=4)
    p = predict(booster, dtrain)

    assert p.shape == (x.shape[0], n_classes)
    assert np.all(np.isfinite(p))
    assert np.all(p >= 0.0) and np.all(p <= 1.0)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("n_classes", [3, 7])
def test_exact_differs_from_diagonal(n_classes):
    x, y = make_data(512, 6, n_classes)
    diag, d1 = train(x, y, params(n_classes, "diagonal"), rounds=4)
    exact, d2 = train(x, y, params(n_classes, "exact"), rounds=4)
    assert not np.allclose(predict(diag, d1), predict(exact, d2), atol=1e-4), (
        "exact and diagonal produced the same predictions, so multi_hessian=exact is inert"
    )


@pytest.mark.parametrize("n_classes", [2, 3, 7, 10])
def test_leaf_outputs_are_centered(n_classes):
    """Each vector leaf's K outputs must sum to zero: the gauge the regularizer assumes."""
    x, y = make_data(256, 6, n_classes)
    booster, _ = train(x, y, params(n_classes, "exact"), rounds=2)
    model = json.loads(booster.save_raw("json").decode("utf-8"))
    trees = model["learner"]["gradient_booster"]["model"]["trees"]
    assert trees, "no trees were produced"

    checked = 0
    for tree in trees:
        weights = np.asarray(tree["base_weights"], dtype=np.float64)
        n_targets = int(tree.get("tree_param", {}).get("size_leaf_vector", n_classes))
        assert n_targets == n_classes
        weights = weights.reshape(-1, n_classes)
        # Every node's weight vector is produced by the centered write.
        for row in weights:
            assert abs(row.sum()) < 1e-3, f"leaf outputs not centered: {row}"
            checked += 1
    assert checked > 0


# --------------------------------------------------------------------------- #
# Phase 4/5/6 - intercept-only mathematics
# --------------------------------------------------------------------------- #


def intercept_problem(n_classes: int, n_rows: int = 900):
    """One constant feature, so every tree is a stump and the leaf is the Newton step."""
    x = np.ones((n_rows, 1), dtype=np.float32)
    y = np.empty(n_rows, dtype=np.float32)
    parts = n_classes * (n_classes + 1) / 2.0
    filled = 0
    counts = []
    for k in range(n_classes):
        quota = n_rows - filled if k + 1 == n_classes else int(n_rows * (k + 1) / parts)
        y[filled : filled + quota] = k
        counts.append(quota)
        filled += quota
    assert filled == n_rows
    return x, y, np.asarray(counts, dtype=np.float64) / n_rows


@pytest.mark.parametrize("n_classes", [2, 3, 7])
def test_intercept_only_reaches_empirical_proportions(n_classes):
    x, y, proportions = intercept_problem(n_classes)
    prm = params(n_classes, "exact", eta=1.0, **{"lambda": 0.0})
    prm["max_depth"] = 1
    booster, dtrain = train(x, y, prm, rounds=60)
    p = predict(booster, dtrain)

    # Intercept-only: every row shares one distribution.
    np.testing.assert_allclose(p[0], p[-1], atol=1e-5)
    err = float(np.max(np.abs(p[0] - proportions)))
    assert err < 5e-3, f"K={n_classes}: max probability error {err:.3e}"


@pytest.mark.parametrize("eta", [1.0, 0.5, 0.1])
def test_learning_rate_reaches_same_optimum(eta):
    """Different trajectories, same fixed point. Not an iteration-count claim."""
    n_classes = 3
    x, y, proportions = intercept_problem(n_classes)
    prm = params(n_classes, "exact", eta=eta, **{"lambda": 0.0})
    prm["max_depth"] = 1
    rounds = int(round(60 / eta))
    booster, dtrain = train(x, y, prm, rounds=rounds)
    p = predict(booster, dtrain)
    err = float(np.max(np.abs(p[0] - proportions)))
    assert err < 1e-2, f"eta={eta}: max probability error {err:.3e}"


def test_regularization_shrinks_towards_uniform():
    """A large lambda must pull the intercept solution towards the uniform distribution."""
    n_classes = 3
    x, y, proportions = intercept_problem(n_classes)
    uniform = np.full(n_classes, 1.0 / n_classes)

    def fit(lam):
        prm = params(n_classes, "exact", eta=1.0, **{"lambda": lam})
        prm["max_depth"] = 1
        booster, dtrain = train(x, y, prm, rounds=30)
        return predict(booster, dtrain)[0]

    p_small = fit(0.0)
    p_large = fit(1e5)
    assert np.all(np.isfinite(p_small)) and np.all(np.isfinite(p_large))
    # Unregularized sits near the empirical proportions; heavily regularized sits nearer
    # uniform.
    assert np.max(np.abs(p_small - proportions)) < np.max(np.abs(p_large - proportions))
    assert np.max(np.abs(p_large - uniform)) < np.max(np.abs(p_small - uniform))


# --------------------------------------------------------------------------- #
# Phase 7 - reference-class invariance
# --------------------------------------------------------------------------- #


def test_reference_class_invariance():
    """
    Permuting the class labels must permute the fitted distribution and nothing else.

    This is the production-level check on R = lambda * (I - 11^T / K): with R = lambda * I the
    penalty would depend on which class happens to be the reference, and this would fail.
    """
    n_classes = 4
    x, y, _ = intercept_problem(n_classes)
    perm = np.array([2, 0, 3, 1])

    prm = params(n_classes, "exact", eta=1.0)
    prm["max_depth"] = 1
    base, d_base = train(x, y, prm, rounds=40)
    p_base = predict(base, d_base)[0]

    y_perm = np.asarray([perm[int(v)] for v in y], dtype=np.float32)
    permuted, d_perm = train(x, y_perm, prm, rounds=40)
    p_perm = predict(permuted, d_perm)[0]

    # Undo the permutation: p_perm[perm[k]] should equal p_base[k].
    recovered = np.empty_like(p_base)
    for k in range(n_classes):
        recovered[k] = p_perm[perm[k]]
    np.testing.assert_allclose(recovered, p_base, atol=1e-4)


# --------------------------------------------------------------------------- #
# Phase 9/10/11 - weights, sampling, rounds
# --------------------------------------------------------------------------- #


def test_sample_weights_change_the_fit():
    n_classes = 3
    x, y = make_data(400, 6, n_classes)
    plain, d1 = train(x, y, params(n_classes, "exact"), rounds=3)
    w = np.where(np.arange(x.shape[0]) < x.shape[0] // 2, 0.1, 10.0).astype(np.float32)
    weighted, d2 = train(x, y, params(n_classes, "exact"), rounds=3, weights=w)
    assert not np.allclose(predict(plain, d1), predict(weighted, d2), atol=1e-4)
    assert np.all(np.isfinite(predict(weighted, d2)))


def test_zero_weights_are_tolerated():
    n_classes = 3
    x, y = make_data(300, 5, n_classes)
    w = np.ones(x.shape[0], dtype=np.float32)
    w[::3] = 0.0
    booster, dtrain = train(x, y, params(n_classes, "exact"), rounds=3, weights=w)
    p = predict(booster, dtrain)
    assert np.all(np.isfinite(p))
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("subsample", [1.0, 0.8, 0.5])
def test_subsampling(subsample):
    n_classes = 3
    x, y = make_data(512, 6, n_classes)
    booster, dtrain = train(x, y, params(n_classes, "exact", subsample=subsample), rounds=4)
    p = predict(booster, dtrain)
    assert np.all(np.isfinite(p))
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("rounds", [1, 2, 5, 10])
def test_multiple_rounds(rounds):
    n_classes = 3
    x, y = make_data(256, 6, n_classes)
    booster, dtrain = train(x, y, params(n_classes, "exact"), rounds=rounds)
    p = predict(booster, dtrain)
    assert np.all(np.isfinite(p))
    assert booster.num_boosted_rounds() == rounds


def test_training_loss_decreases():
    n_classes = 3
    x, y = make_data(512, 6, n_classes)
    dtrain = xgb.DMatrix(x, label=y)
    history: dict = {}
    xgb.train(
        params(n_classes, "exact"),
        dtrain,
        num_boost_round=8,
        evals=[(dtrain, "train")],
        evals_result=history,
        verbose_eval=False,
    )
    losses = history["train"]["mlogloss"]
    assert losses[-1] < losses[0], f"loss did not improve: {losses[0]} -> {losses[-1]}"
    assert all(np.isfinite(losses))


# --------------------------------------------------------------------------- #
# Phase 15 - save / load
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_classes", [2, 3, 7])
def test_save_load_round_trip(n_classes):
    x, y = make_data(256, 6, n_classes)
    booster, dtrain = train(x, y, params(n_classes, "exact"), rounds=3)
    before = predict(booster, dtrain)

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "model.json")
        booster.save_model(path)
        restored = xgb.Booster()
        restored.load_model(path)
        after = restored.predict(dtrain)
        raw = Path(path).read_text(encoding="utf-8")

    np.testing.assert_array_equal(before, after)
    # Training-only state must not be in the model.
    assert "multi_hessian" not in raw
    assert "exact_hessian" not in raw


# --------------------------------------------------------------------------- #
# Phase 16 - unsupported configurations must fail
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"multi_strategy": "one_output_per_tree"}, "one_output_per_tree"),
        ({"alpha": 0.5}, "reg_alpha"),
        ({"max_delta_step": 1.0}, "max_delta_step"),
        ({"tree_method": "approx"}, "approx"),
        ({"subsample": 0.5, "sampling_method": "gradient_based"}, "gradient_based"),
    ],
)
def test_unsupported_configurations_are_rejected(overrides, reason):
    x, y = make_data(128, 5, 3)
    with pytest.raises(xgb.core.XGBoostError):
        train(x, y, params(3, "exact", **overrides), rounds=1)


def test_unsupported_objective_is_rejected():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(128, 5)).astype(np.float32)
    y = rng.normal(size=(128, 3)).astype(np.float32)
    prm = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": "cpu",
        "multi_strategy": "multi_output_tree",
        "multi_hessian": "exact",
    }
    with pytest.raises(xgb.core.XGBoostError):
        xgb.train(prm, xgb.DMatrix(x, label=y), num_boost_round=1)


def test_categorical_features_are_rejected():
    """M5.5: categorical features must error, not be silently scanned as ordered."""
    import pandas as pd

    rng = np.random.default_rng(3)
    n = 256
    frame = pd.DataFrame(
        {
            "num": rng.normal(size=n).astype(np.float32),
            "cat": pd.Categorical(rng.integers(0, 4, size=n)),
        }
    )
    y = rng.integers(0, 3, size=n).astype(np.float32)
    dtrain = xgb.DMatrix(frame, label=y, enable_categorical=True)
    with pytest.raises(xgb.core.XGBoostError, match="categorical"):
        xgb.train(params(3, "exact"), dtrain, num_boost_round=1)

    # The same data must still train in the pre-existing diagonal mode.
    booster = xgb.train(params(3, "diagonal"), dtrain, num_boost_round=1)
    assert np.all(np.isfinite(booster.predict(dtrain)))


# --------------------------------------------------------------------------- #
# Phase 18 - numerical stress
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_classes", [2, 3, 7])
@pytest.mark.parametrize("scale", [1e-4, 1.0, 1e4])
def test_extreme_feature_scales(n_classes, scale):
    x, y = make_data(256, 5, n_classes)
    booster, dtrain = train(x * scale, y, params(n_classes, "exact"), rounds=3)
    p = predict(booster, dtrain)
    assert np.all(np.isfinite(p))
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)


def test_highly_imbalanced_classes():
    """One class holds almost all the mass, so the others have near-zero curvature."""
    n, n_classes = 600, 4
    y = np.zeros(n, dtype=np.float32)
    y[:3] = 1.0
    y[3:6] = 2.0
    y[6:9] = 3.0
    rng = np.random.default_rng(5)
    x = rng.normal(size=(n, 4)).astype(np.float32)
    booster, dtrain = train(x, y, params(n_classes, "exact", **{"lambda": 1.0}), rounds=5)
    p = predict(booster, dtrain)
    assert np.all(np.isfinite(p))
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)


def test_single_row_per_class():
    n_classes = 3
    x = np.arange(9, dtype=np.float32).reshape(9, 1)
    y = np.array([0, 1, 2] * 3, dtype=np.float32)
    booster, dtrain = train(x, y, params(n_classes, "exact"), rounds=3)
    p = predict(booster, dtrain)
    assert np.all(np.isfinite(p))
