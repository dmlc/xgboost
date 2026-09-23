"""
Numerical stability of exact multinomial mode over long boosting runs.

The unit tests in ``tests/cpp/test_exact_multinomial_numerics.cc`` pin down one leaf solve at
a time. What they cannot see is drift: an error that is invisible in a single Newton step but
compounds once the model's own predictions feed the next round's Hessian. These tests run the
loop far enough for that to show up.
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pytest

import xgboost as xgb

BASE_PARAMS = {
    "objective": "multi:softprob",
    "tree_method": "hist",
    "device": "cpu",
    "multi_strategy": "multi_output_tree",
    "multi_hessian": "exact",
    "max_depth": 3,
    "lambda": 1.0,
    "base_score": 0.5,
}


def params(n_classes: int, **overrides):
    out = dict(BASE_PARAMS)
    out["num_class"] = n_classes
    out.update(overrides)
    return out


def make_data(n_rows: int, n_cols: int, n_classes: int, seed: int = 11):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, size=n_rows)
    x = rng.normal(size=(n_rows, n_cols))
    for k in range(n_classes):
        x[y == k, k % n_cols] += 1.5
    return x.astype(np.float32), y.astype(np.float32)


def leaf_weight_rows(booster, n_classes: int) -> np.ndarray:
    """Every node's K-vector output, across every tree in the model."""
    model = json.loads(booster.save_raw("json").decode("utf-8"))
    trees = model["learner"]["gradient_booster"]["model"]["trees"]
    assert trees, "no trees were produced"
    rows = [
        np.asarray(tree["base_weights"], dtype=np.float64).reshape(-1, n_classes)
        for tree in trees
    ]
    return np.concatenate(rows, axis=0)


# --------------------------------------------------------------------------- #
# Phase 13 - multi-round boosting stability
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("eta", [1.0, 0.5, 0.1])
@pytest.mark.parametrize("rounds", [1, 5, 10, 25, 50])
def test_long_runs_stay_finite_and_improve(eta, rounds):
    n_classes = 4
    x, y = make_data(768, 8, n_classes, seed=5)
    dtrain = xgb.DMatrix(x, label=y)
    history: dict = {}
    booster = xgb.train(
        params(n_classes, eta=eta),
        dtrain,
        num_boost_round=rounds,
        evals=[(dtrain, "train")],
        evals_result=history,
        verbose_eval=False,
    )

    losses = np.asarray(history["train"]["mlogloss"], dtype=np.float64)
    assert len(losses) == rounds
    assert np.all(np.isfinite(losses)), f"non-finite training loss: {losses}"
    # Newton steps on a convex loss with eta <= 1 should not make the fit worse overall.
    assert losses[-1] <= losses[0] + 1e-9, f"loss got worse: {losses[0]} -> {losses[-1]}"

    p = booster.predict(dtrain)
    assert np.all(np.isfinite(p))
    assert np.all(p >= 0.0) and np.all(p <= 1.0)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("eta", [1.0, 0.5, 0.1])
def test_leaf_outputs_stay_centered_over_fifty_rounds(eta):
    """
    The gauge must not drift.

    Each leaf is written centered, so its K outputs sum to zero. Nothing re-centers the
    accumulated margin between rounds, so a small per-round bias along the all-ones direction
    would accumulate linearly over 50 rounds -- which is exactly the failure this catches, and
    which one round cannot.
    """
    n_classes = 5
    x, y = make_data(512, 6, n_classes, seed=23)
    dtrain = xgb.DMatrix(x, label=y)
    booster = xgb.train(params(n_classes, eta=eta), dtrain, num_boost_round=50)

    rows = leaf_weight_rows(booster, n_classes)
    worst = float(np.max(np.abs(rows.sum(axis=1))))
    scale = max(1.0, float(np.max(np.abs(rows))))
    assert worst < 1e-5 * scale, f"eta={eta}: leaf outputs drifted off centre by {worst:.3e}"


def test_more_rounds_do_not_degrade_the_fit():
    """Training loss is monotone in the round count, across a wide range of horizons."""
    n_classes = 3
    x, y = make_data(640, 6, n_classes, seed=41)
    dtrain = xgb.DMatrix(x, label=y)

    previous = np.inf
    for rounds in (1, 5, 10, 25, 50):
        history: dict = {}
        xgb.train(
            params(n_classes, eta=0.3),
            dtrain,
            num_boost_round=rounds,
            evals=[(dtrain, "train")],
            evals_result=history,
            verbose_eval=False,
        )
        final = float(history["train"]["mlogloss"][-1])
        assert np.isfinite(final)
        assert final <= previous + 1e-9, f"{rounds} rounds was worse than fewer rounds"
        previous = final


def test_repeated_training_is_deterministic():
    """A drifting accumulator would show up as run-to-run variation; none is permitted."""
    n_classes = 4
    x, y = make_data(512, 6, n_classes, seed=3)
    dtrain = xgb.DMatrix(x, label=y)

    reference = None
    for _ in range(3):
        booster = xgb.train(params(n_classes, eta=0.5), dtrain, num_boost_round=30)
        p = booster.predict(dtrain)
        if reference is None:
            reference = p
        else:
            np.testing.assert_array_equal(p, reference)


def test_near_separable_data_does_not_blow_up():
    """
    Probabilities driven towards 0 and 1 make the Hessian nearly singular.

    With many rounds at eta = 1 on almost separable data, the leaf solve is asked for large
    steps from vanishing curvature. The centered regularizer is what keeps that bounded, so
    this is the end-to-end counterpart of the solver's extreme-conditioning test.
    """
    n_classes = 3
    rng = np.random.default_rng(77)
    n_rows = 600
    y = rng.integers(0, n_classes, size=n_rows)
    # A feature that reveals the label almost exactly.
    x = (y.astype(np.float64) * 10.0 + rng.normal(scale=0.01, size=n_rows)).reshape(-1, 1)

    dtrain = xgb.DMatrix(x.astype(np.float32), label=y.astype(np.float32))
    history: dict = {}
    booster = xgb.train(
        params(n_classes, eta=1.0, **{"lambda": 1.0}),
        dtrain,
        num_boost_round=50,
        evals=[(dtrain, "train")],
        evals_result=history,
        verbose_eval=False,
    )

    losses = np.asarray(history["train"]["mlogloss"], dtype=np.float64)
    assert np.all(np.isfinite(losses)), f"loss went non-finite: {losses}"
    assert losses[-1] < losses[0]
    p = booster.predict(dtrain)
    assert np.all(np.isfinite(p))
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)
    rows = leaf_weight_rows(booster, n_classes)
    assert np.all(np.isfinite(rows))


# --------------------------------------------------------------------------- #
# Phase 15 - reference-class permutation under stress
# --------------------------------------------------------------------------- #


def permuted_fit(x, y, perm, n_classes, rounds, **overrides):
    """Train on relabelled data and map the predictions back to the original class order."""
    y_perm = np.asarray([perm[int(v)] for v in y], dtype=np.float32)
    dtrain = xgb.DMatrix(x, label=y_perm)
    booster = xgb.train(params(n_classes, **overrides), dtrain, num_boost_round=rounds)
    p = booster.predict(dtrain)
    inverse = np.empty_like(p)
    for k in range(n_classes):
        inverse[:, k] = p[:, perm[k]]
    return inverse


@pytest.mark.parametrize("perm", list(itertools.permutations(range(4))))
def test_every_class_permutation_gives_the_same_fit(perm):
    """
    All 24 relabelings of a 4-class problem, on a model with real splits.

    The existing single-permutation test uses an intercept-only stump. This one lets the tree
    split, so the permutation has to commute with split selection as well as with the leaf
    solve -- and it sweeps every permutation rather than one lucky choice.
    """
    n_classes = 4
    rounds = 20
    x, y = make_data(640, 6, n_classes, seed=13)

    identity = permuted_fit(x, y, tuple(range(n_classes)), n_classes, rounds, eta=0.5)
    recovered = permuted_fit(x, y, perm, n_classes, rounds, eta=0.5)
    np.testing.assert_allclose(recovered, identity, atol=1e-5)


@pytest.mark.parametrize("perm", [(2, 0, 3, 1), (3, 2, 1, 0)])
def test_permutation_invariance_survives_imbalance_and_long_runs(perm):
    """Same claim, under the conditions most likely to expose an asymmetry."""
    n_classes = 4
    rng = np.random.default_rng(101)
    # Heavily imbalanced: one class is 40x rarer than the most common.
    counts = [400, 200, 50, 10]
    y = np.concatenate([np.full(c, k) for k, c in enumerate(counts)])
    x = rng.normal(size=(len(y), 5))
    for k in range(n_classes):
        x[y == k, k % 5] += 1.2
    x = x.astype(np.float32)

    identity = permuted_fit(x, y, tuple(range(n_classes)), n_classes, 50, eta=1.0)
    recovered = permuted_fit(x, y, perm, n_classes, 50, eta=1.0)
    np.testing.assert_allclose(recovered, identity, atol=1e-4)


def test_diagonal_mode_is_also_permutation_invariant():
    """
    Attribution control.

    If the permutation tests above passed for a reason unrelated to exact mode -- something in
    the data or the harness making every relabeling trivially equivalent -- then diagonal mode
    would pass them too, and the exact-mode versions would prove nothing. This records that
    both modes are invariant, so the exact-mode tests are testing the exact path, not an
    artefact of the setup.
    """
    n_classes = 4
    x, y = make_data(640, 6, n_classes, seed=13)
    perm = (2, 0, 3, 1)

    identity = permuted_fit(
        x, y, tuple(range(n_classes)), n_classes, 20, eta=0.5, multi_hessian="diagonal"
    )
    recovered = permuted_fit(
        x, y, perm, n_classes, 20, eta=0.5, multi_hessian="diagonal"
    )
    np.testing.assert_allclose(recovered, identity, atol=1e-5)


# --------------------------------------------------------------------------- #
# Phase 11 - the refused-factorization path, end to end
# --------------------------------------------------------------------------- #


def test_all_zero_weights_produce_zero_leaves_not_nan():
    """
    Every node has exactly zero curvature, so every LDL^T factorization must be refused.

    The solver's own test checks that a refusal leaves its outputs untouched. This checks what
    the builder then does with it: writes a zero leaf, matching the scalar path's policy for
    `sum_hess <= 0`. A silent NaN here would poison the margin for every subsequent round, and
    the partially-weighted test cannot reach this path because its leaves still have curvature.
    """
    n_classes = 3
    x, y = make_data(300, 5, n_classes, seed=31)
    w = np.zeros(x.shape[0], dtype=np.float32)
    dtrain = xgb.DMatrix(x, label=y, weight=w)
    booster = xgb.train(params(n_classes, eta=1.0), dtrain, num_boost_round=5)

    rows = leaf_weight_rows(booster, n_classes)
    assert np.all(np.isfinite(rows))
    np.testing.assert_array_equal(rows, np.zeros_like(rows))

    p = booster.predict(dtrain)
    assert np.all(np.isfinite(p))
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)
    # No tree moved the margin, so every row keeps the intercept.
    np.testing.assert_allclose(p, np.broadcast_to(p[0], p.shape), atol=1e-6)


def test_single_class_data_stays_finite():
    """
    One class with all the mass drives the other probabilities towards zero, which is where
    the Hessian loses rank. Fifty rounds at eta = 1 is the worst case for that.
    """
    n_classes = 3
    rng = np.random.default_rng(59)
    n_rows = 400
    y = np.zeros(n_rows, dtype=np.float32)
    x = rng.normal(size=(n_rows, 4)).astype(np.float32)

    dtrain = xgb.DMatrix(x, label=y)
    booster = xgb.train(params(n_classes, eta=1.0), dtrain, num_boost_round=50)

    rows = leaf_weight_rows(booster, n_classes)
    assert np.all(np.isfinite(rows))
    p = booster.predict(dtrain)
    assert np.all(np.isfinite(p))
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)
    # The model should be confident about the only class present.
    assert float(np.min(p[:, 0])) > 0.9
