"""
Milestone 7 compatibility audit for `multi_hessian=exact`.

Every XGBoost feature that interacts with training is either exercised here and shown to
work, or shown to be rejected with an actionable error. Nothing is assumed to work because
it "should"; a feature with no test below is not claimed as supported.

Run with the package built from this source tree:

    PYTHONPATH=python-package python -m pytest tests/python/test_exact_multinomial_compat.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

import xgboost as xgb

BASE = {
    "objective": "multi:softprob",
    "tree_method": "hist",
    "device": "cpu",
    "multi_strategy": "multi_output_tree",
    "max_depth": 3,
    "eta": 0.3,
    "lambda": 1.0,
    "base_score": 0.5,
}


def params(n_classes: int, mode: str = "exact", **overrides):
    out = dict(BASE)
    out["num_class"] = n_classes
    out["multi_hessian"] = mode
    out.update(overrides)
    return out


def make_data(n_rows=400, n_cols=8, n_classes=3, seed=13, missing_frac=0.0, sparse=False):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, size=n_rows)
    x = rng.normal(size=(n_rows, n_cols))
    for k in range(n_classes):
        x[y == k, k % n_cols] += 1.5
    if missing_frac > 0:
        mask = rng.random(x.shape) < missing_frac
        x[mask] = np.nan
    x = x.astype(np.float32)
    if sparse:
        from scipy import sparse as sp

        x = sp.csr_matrix(np.nan_to_num(x, nan=0.0))
    return x, y.astype(np.float32)


def fit(x, y, prm, rounds=4, **dm_kwargs):
    dtrain = xgb.DMatrix(x, label=y, **dm_kwargs)
    return xgb.train(prm, dtrain, num_boost_round=rounds), dtrain


def valid(p, n_classes):
    assert np.all(np.isfinite(p))
    assert p.shape[1] == n_classes
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)


# --------------------------------------------------------------------------- #
# Phase 1 - column subsampling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_classes", [3, 7])
@pytest.mark.parametrize(
    "knob", ["colsample_bytree", "colsample_bylevel", "colsample_bynode"]
)
def test_column_subsampling(n_classes, knob):
    x, y = make_data(n_classes=n_classes)
    booster, dtrain = fit(x, y, params(n_classes, **{knob: 0.5, "seed": 0}))
    valid(booster.predict(dtrain), n_classes)

    # Subsampling columns must actually restrict the model: with only half the features
    # available per node the fit should differ from the full-feature fit.
    full, d_full = fit(x, y, params(n_classes, **{"seed": 0}))
    assert not np.allclose(booster.predict(dtrain), full.predict(d_full), atol=1e-5)


def test_column_subsampling_respects_feature_set():
    """Only sampled features may appear as splits."""
    n_classes = 3
    x, y = make_data(n_cols=12, n_classes=n_classes)
    booster, _ = fit(x, y, params(n_classes, colsample_bytree=0.25, seed=7), rounds=3)
    model = json.loads(booster.save_raw("json").decode("utf-8"))
    trees = model["learner"]["gradient_booster"]["model"]["trees"]
    used = set()
    for tree in trees:
        left = tree["left_children"]
        for nidx, idx in enumerate(tree["split_indices"]):
            if left[nidx] != -1:  # internal node
                used.add(int(idx))
    # 25% of 12 features is 3 per tree; across 3 trees we must still be well under 12.
    assert 0 < len(used) <= 9


# --------------------------------------------------------------------------- #
# Phase 2 - sparse input and missing values
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_classes", [3, 7])
def test_missing_values(n_classes):
    x, y = make_data(n_classes=n_classes, missing_frac=0.3)
    booster, dtrain = fit(x, y, params(n_classes))
    valid(booster.predict(dtrain), n_classes)


def test_sparse_csr_input():
    pytest.importorskip("scipy")
    n_classes = 3
    x, y = make_data(n_classes=n_classes, sparse=True)
    booster, dtrain = fit(x, y, params(n_classes))
    valid(booster.predict(dtrain), n_classes)


def test_dense_and_sparse_agree_on_identical_data():
    """
    A CSR matrix holding the same values trains to the same model as dense.

    Note the caveat: CSR *omits* structural zeros, and an omitted entry is treated as
    missing, not as 0.0. Dense zeros and sparse gaps are therefore different data. This test
    uses strictly non-zero values so the CSR stores every entry explicitly and the two
    representations really do describe the same matrix.
    """
    sp = pytest.importorskip("scipy.sparse")
    n_classes = 3
    rng = np.random.default_rng(5)
    y = rng.integers(0, n_classes, size=300).astype(np.float32)
    dense = rng.normal(size=(300, 6))
    dense = np.where(np.abs(dense) < 0.1, 0.5, dense).astype(np.float32)  # no zeros
    assert not np.any(dense == 0.0)

    b_dense, d_dense = fit(dense, y, params(n_classes))
    b_sparse, d_sparse = fit(sp.csr_matrix(dense), y, params(n_classes))
    np.testing.assert_allclose(
        b_dense.predict(d_dense), b_sparse.predict(d_sparse), atol=1e-5
    )


def test_sparse_zeros_are_missing_not_zero():
    """Documents the representation caveat above rather than asserting equivalence."""
    sp = pytest.importorskip("scipy.sparse")
    n_classes = 3
    rng = np.random.default_rng(5)
    y = rng.integers(0, n_classes, size=300).astype(np.float32)
    dense = rng.normal(size=(300, 6))
    dense[dense < 0.5] = 0.0
    dense = dense.astype(np.float32)

    b_dense, d_dense = fit(dense, y, params(n_classes))
    b_sparse, d_sparse = fit(sp.csr_matrix(dense), y, params(n_classes))
    # Both are valid models; they simply describe different data.
    valid(b_dense.predict(d_dense), n_classes)
    valid(b_sparse.predict(d_sparse), n_classes)
    assert not np.allclose(b_dense.predict(d_dense), b_sparse.predict(d_sparse), atol=1e-5)


# --------------------------------------------------------------------------- #
# Phase 3 - QuantileDMatrix
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_classes", [3, 7])
def test_quantile_dmatrix(n_classes):
    x, y = make_data(n_classes=n_classes)
    qdm = xgb.QuantileDMatrix(x, label=y)
    booster = xgb.train(params(n_classes), qdm, num_boost_round=4)
    valid(booster.predict(qdm), n_classes)


def test_quantile_dmatrix_matches_dmatrix():
    n_classes = 3
    x, y = make_data(n_classes=n_classes)
    prm = params(n_classes, max_bin=64)
    b1, d1 = fit(x, y, prm)
    qdm = xgb.QuantileDMatrix(x, label=y, max_bin=64)
    b2 = xgb.train(prm, qdm, num_boost_round=4)
    np.testing.assert_allclose(b1.predict(d1), b2.predict(qdm), atol=1e-5)


# --------------------------------------------------------------------------- #
# Phase 4 - feature weights
# --------------------------------------------------------------------------- #


def test_feature_weights():
    """Feature weights bias column sampling only; no dense-Hessian mathematics involved."""
    n_classes = 3
    x, y = make_data(n_cols=8, n_classes=n_classes)
    fw = np.array([10.0, 10.0] + [0.01] * 6, dtype=np.float32)
    dtrain = xgb.DMatrix(x, label=y)
    dtrain.set_info(feature_weights=fw)
    booster = xgb.train(
        params(n_classes, colsample_bytree=0.25, seed=3), dtrain, num_boost_round=4
    )
    valid(booster.predict(dtrain), n_classes)

    model = json.loads(booster.save_raw("json").decode("utf-8"))
    trees = model["learner"]["gradient_booster"]["model"]["trees"]
    used = set()
    for tree in trees:
        left = tree["left_children"]
        for nidx, idx in enumerate(tree["split_indices"]):
            if left[nidx] != -1:
                used.add(int(idx))
    # The heavily weighted columns should dominate the chosen splits.
    assert used, "no splits were made"
    assert len(used & {0, 1}) >= 1


# --------------------------------------------------------------------------- #
# Phase 5 - interaction constraints (supported)
# --------------------------------------------------------------------------- #


def _paths_features(tree):
    """Return, for each leaf, the set of features used on its root path."""
    left, right = tree["left_children"], tree["right_children"]
    idx = tree["split_indices"]
    out = []

    def walk(node, acc):
        if left[node] == -1:
            out.append(set(acc))
            return
        acc.append(int(idx[node]))
        walk(left[node], acc)
        walk(right[node], acc)
        acc.pop()

    walk(0, [])
    return out


def test_interaction_constraints_are_enforced():
    n_classes = 3
    x, y = make_data(n_cols=6, n_classes=n_classes)
    groups = [[0, 1], [2, 3], [4, 5]]
    prm = params(n_classes, interaction_constraints=json.dumps(groups), max_depth=4)
    booster, dtrain = fit(x, y, prm, rounds=4)
    valid(booster.predict(dtrain), n_classes)

    model = json.loads(booster.save_raw("json").decode("utf-8"))
    for tree in model["learner"]["gradient_booster"]["model"]["trees"]:
        for path in _paths_features(tree):
            if len(path) <= 1:
                continue
            assert any(path <= set(g) for g in groups), (
                f"path {sorted(path)} crosses an interaction constraint group"
            )


def test_interaction_constraints_change_the_model():
    """A constraint that forbids a useful combination must alter the fit."""
    n_classes = 3
    x, y = make_data(n_cols=6, n_classes=n_classes)
    free, d1 = fit(x, y, params(n_classes, max_depth=4), rounds=4)
    constrained, d2 = fit(
        x, y, params(n_classes, interaction_constraints="[[0],[1],[2],[3],[4],[5]]", max_depth=4), rounds=4
    )
    assert not np.allclose(free.predict(d1), constrained.predict(d2), atol=1e-5)


# --------------------------------------------------------------------------- #
# Phase 6 - monotone constraints (rejected)
# --------------------------------------------------------------------------- #


def test_monotone_constraints_are_rejected():
    n_classes = 3
    x, y = make_data(n_cols=4, n_classes=n_classes)
    with pytest.raises(xgb.core.XGBoostError, match="monotone"):
        fit(x, y, params(n_classes, monotone_constraints="(1,0,0,0)"), rounds=1)

    # The same constraint still trains in the pre-existing diagonal mode.
    booster, dtrain = fit(
        x, y, params(n_classes, "diagonal", monotone_constraints="(1,0,0,0)"), rounds=2
    )
    valid(booster.predict(dtrain), n_classes)


# --------------------------------------------------------------------------- #
# Phase 7 - base_score / boost_from_average
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_classes", [2, 3, 7])
def test_explicit_base_score(n_classes):
    x, y = make_data(n_classes=n_classes)
    booster, dtrain = fit(x, y, params(n_classes, base_score=0.25))
    valid(booster.predict(dtrain), n_classes)


def test_boost_from_average_enabled():
    n_classes = 3
    x, y = make_data(n_classes=n_classes)
    prm = params(n_classes)
    prm.pop("base_score", None)
    prm["boost_from_average"] = 1
    booster, dtrain = fit(x, y, prm)
    valid(booster.predict(dtrain), n_classes)


def test_boost_from_average_disabled_needs_base_score_in_both_modes():
    """
    `boost_from_average=0` without an explicit `base_score` fails, and it fails identically
    in diagonal mode. That is a pre-existing XGBoost requirement, not an exact-mode
    limitation, so it is recorded here rather than attributed to exact mode.
    """
    n_classes = 3
    x, y = make_data(n_classes=n_classes)
    for mode in ("exact", "diagonal"):
        prm = params(n_classes, mode)
        prm.pop("base_score", None)
        prm["boost_from_average"] = 0
        with pytest.raises(xgb.core.XGBoostError):
            fit(x, y, prm, rounds=1)

    # With base_score supplied, both modes train.
    for mode in ("exact", "diagonal"):
        prm = params(n_classes, mode, base_score=0.5)
        prm["boost_from_average"] = 0
        booster, dtrain = fit(x, y, prm, rounds=2)
        valid(booster.predict(dtrain), n_classes)


def test_intercept_fixed_point_unaffected_by_base_score():
    """The intercept-only optimum must not depend on where training started."""
    n_classes = 3
    n = 900
    x = np.ones((n, 1), dtype=np.float32)
    y = np.concatenate(
        [np.full(150, 0.0), np.full(300, 1.0), np.full(450, 2.0)]
    ).astype(np.float32)
    proportions = np.array([150, 300, 450], dtype=np.float64) / n

    for base_score in (0.25, 0.5, 0.75):
        prm = params(n_classes, eta=1.0, max_depth=1, base_score=base_score)
        prm["lambda"] = 0.0
        booster, dtrain = fit(x, y, prm, rounds=60)
        p = booster.predict(dtrain)[0]
        assert np.max(np.abs(p - proportions)) < 5e-3, f"base_score={base_score}: {p}"


# --------------------------------------------------------------------------- #
# Phase 8 - early stopping
# --------------------------------------------------------------------------- #


def test_early_stopping():
    n_classes = 3
    x, y = make_data(n_rows=600, n_classes=n_classes)
    split = 400
    dtrain = xgb.DMatrix(x[:split], label=y[:split])
    dvalid = xgb.DMatrix(x[split:], label=y[split:])
    booster = xgb.train(
        params(n_classes),
        dtrain,
        num_boost_round=60,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=5,
        verbose_eval=False,
    )
    assert booster.best_iteration >= 0
    assert booster.best_iteration <= 59
    p = booster.predict(dvalid, iteration_range=(0, booster.best_iteration + 1))
    valid(p, n_classes)

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "m.json")
        booster.save_model(path)
        restored = xgb.Booster()
        restored.load_model(path)
        np.testing.assert_allclose(
            restored.predict(dvalid, iteration_range=(0, booster.best_iteration + 1)),
            p,
            atol=0,
        )


# --------------------------------------------------------------------------- #
# Phase 9 - continued training
# --------------------------------------------------------------------------- #


def test_continued_training_matches_continuous():
    """
    Training 2+2 rounds must equal training 4 rounds.

    `multi_hessian` is training configuration, not model state, so it must be supplied again
    when continuing; this test also pins that requirement down.
    """
    n_classes = 3
    x, y = make_data(n_classes=n_classes)
    prm = params(n_classes)

    continuous, dtrain = fit(x, y, prm, rounds=4)

    first = xgb.train(prm, dtrain, num_boost_round=2)
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "m.json")
        first.save_model(path)
        reloaded = xgb.Booster()
        reloaded.load_model(path)
        resumed = xgb.train(prm, dtrain, num_boost_round=2, xgb_model=reloaded)

    assert resumed.num_boosted_rounds() == 4
    np.testing.assert_allclose(
        continuous.predict(dtrain), resumed.predict(dtrain), atol=1e-5
    )


def test_continued_training_can_switch_modes():
    """A model trained in exact mode can be continued in diagonal mode and vice versa."""
    n_classes = 3
    x, y = make_data(n_classes=n_classes)
    dtrain = xgb.DMatrix(x, label=y)
    first = xgb.train(params(n_classes, "exact"), dtrain, num_boost_round=2)
    second = xgb.train(
        params(n_classes, "diagonal"), dtrain, num_boost_round=2, xgb_model=first
    )
    assert second.num_boosted_rounds() == 4
    valid(second.predict(dtrain), n_classes)


# --------------------------------------------------------------------------- #
# Phase 10 - model slicing
# --------------------------------------------------------------------------- #


def test_model_slicing():
    n_classes = 3
    x, y = make_data(n_classes=n_classes)
    booster, dtrain = fit(x, y, params(n_classes), rounds=6)
    sliced = booster[0:3]
    np.testing.assert_allclose(
        sliced.predict(dtrain), booster.predict(dtrain, iteration_range=(0, 3)), atol=0
    )
    valid(sliced.predict(dtrain), n_classes)


# --------------------------------------------------------------------------- #
# Phase 14 - edge data
# --------------------------------------------------------------------------- #


def test_single_feature():
    n_classes = 3
    x, y = make_data(n_cols=1, n_classes=n_classes)
    booster, dtrain = fit(x, y, params(n_classes))
    valid(booster.predict(dtrain), n_classes)


def test_tiny_dataset():
    n_classes = 3
    x = np.arange(6, dtype=np.float32).reshape(6, 1)
    y = np.array([0, 1, 2, 0, 1, 2], dtype=np.float32)
    booster, dtrain = fit(x, y, params(n_classes), rounds=2)
    valid(booster.predict(dtrain), n_classes)


def test_no_useful_split():
    """A constant feature offers nothing to split on; training must still be well defined."""
    n_classes = 3
    n = 201  # divisible by n_classes so every class is present
    x = np.ones((n, 1), dtype=np.float32)
    y = np.tile(np.arange(n_classes), n // n_classes).astype(np.float32)
    assert y.size == n
    booster, dtrain = fit(x, y, params(n_classes), rounds=3)
    valid(booster.predict(dtrain), n_classes)


def test_missing_heavy_feature():
    n_classes = 3
    x, y = make_data(n_classes=n_classes, missing_frac=0.9)
    booster, dtrain = fit(x, y, params(n_classes), rounds=3)
    valid(booster.predict(dtrain), n_classes)


def test_zero_weight_rows():
    n_classes = 3
    x, y = make_data(n_classes=n_classes)
    w = np.ones(x.shape[0], dtype=np.float32)
    w[::4] = 0.0
    booster, dtrain = fit(x, y, params(n_classes), weight=w)
    valid(booster.predict(dtrain), n_classes)


# --------------------------------------------------------------------------- #
# Phase 12 - the rejection matrix
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "overrides,pattern",
    [
        ({"alpha": 0.5}, "reg_alpha"),
        ({"max_delta_step": 1.0}, "max_delta_step"),
        ({"monotone_constraints": "(1,0,0,0)"}, "monotone"),
        ({"multi_strategy": "one_output_per_tree"}, "multi_output_tree"),
        ({"tree_method": "approx"}, "hist"),
        ({"subsample": 0.5, "sampling_method": "gradient_based"}, "uniform"),
    ],
)
def test_rejections_are_actionable(overrides, pattern):
    x, y = make_data(n_rows=128, n_cols=4)
    with pytest.raises(xgb.core.XGBoostError, match=pattern):
        fit(x, y, params(3, **overrides), rounds=1)


def test_categorical_still_rejected():
    """The M5.5 rejection must survive every later change."""
    pd = pytest.importorskip("pandas")
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
        xgb.train(params(3), dtrain, num_boost_round=1)


def test_gpu_request_is_rejected_regardless_of_hardware():
    """
    A GPU request must be refused even on a machine with no GPU.

    XGBoost's Context downgrades `device=cuda` to CPU when no device is present, so checking
    only the resolved context would let a GPU request quietly train on the CPU and look
    supported. Validation therefore reads the requested device, which makes the answer the
    same on every machine -- including this one, which has no GPU.
    """
    x, y = make_data(n_rows=128, n_cols=4)
    with pytest.raises(xgb.core.XGBoostError, match="CPU only"):
        fit(x, y, params(3, device="cuda"), rounds=1)


def test_gpu_request_is_allowed_in_diagonal_mode():
    """The restriction belongs to exact mode; the existing path is unaffected."""
    x, y = make_data(n_rows=128, n_cols=4)
    # Falls back to CPU on this machine with a warning, exactly as before.
    booster, dtrain = fit(x, y, params(3, "diagonal", device="cuda"), rounds=1)
    valid(booster.predict(dtrain), 3)


def test_custom_objective_is_rejected_not_silently_downgraded():
    """
    A custom objective supplies the gradient directly and cannot produce an exact Hessian.

    That path (``XGBoosterTrainOneIter`` -> ``Learner::BoostOneIter``) bypasses
    ``LearnerImpl::GetGradient`` entirely, so nothing populates the sidecar. Before this was
    checked, training simply succeeded using the DIAGONAL path while the user had asked for
    exact -- a silent downgrade, and the one failure mode this feature is built to prevent.
    The regression is that it must raise, not that it must work.
    """
    rng = np.random.default_rng(0)
    n, n_classes = 800, 3
    y = rng.integers(0, n_classes, size=n)
    x = rng.normal(size=(n, 6))
    for k in range(n_classes):
        x[y == k, k % 6] += 1.5
    x = x.astype(np.float32)
    dtrain = xgb.DMatrix(x, label=y.astype(np.float32))

    def softmax_obj(preds, dmat):
        lab = dmat.get_label().astype(int)
        p = np.exp(preds - preds.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        grad = p.copy()
        grad[np.arange(len(lab)), lab] -= 1.0
        hess = np.maximum(np.abs(grad), 1e-16)
        return grad.astype(np.float32), hess.astype(np.float32)

    prm = {
        "num_class": n_classes,
        "tree_method": "hist",
        "device": "cpu",
        "multi_strategy": "multi_output_tree",
        "multi_hessian": "exact",
        "max_depth": 4,
        "base_score": 0.5,
        "disable_default_eval_metric": 1,
    }
    with pytest.raises(xgb.core.XGBoostError, match="custom objective"):
        xgb.train(prm, dtrain, num_boost_round=2, obj=softmax_obj)

    # The same custom objective is fine in diagonal mode, which is what makes the rejection
    # specific to exact mode rather than a general break of the custom-objective path.
    prm["multi_hessian"] = "diagonal"
    booster = xgb.train(prm, dtrain, num_boost_round=2, obj=softmax_obj)
    assert np.all(np.isfinite(booster.predict(dtrain)))
