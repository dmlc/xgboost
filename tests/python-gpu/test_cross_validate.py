# SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import json
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import pytest
import xgboost as xgb
from pytest import fixture
from xgboost import _cross_validation as xcv
from xgboost import testing as tm

if TYPE_CHECKING:
    import cupy as cp


type XywExtQdm = tuple[
    list[cp.ndarray], list[cp.ndarray], list[cp.ndarray], xgb.ExtMemQuantileDMatrix
]

N_SAMPLES_PER_BATCH, N_FEATURES, N_BATCHES = 16, 4, 2

# Fixed by FoldModels, which pins `boost_from_average` to false and the base score to the
# objective default. It cannot be configured from here.
BASE_SCORE = 0.5

# The CV tree method cannot grow past depth 1 yet. `debug_synchronize` gates the check that
# every training row of a unit, and only those, received a leaf position; the fused-page-pass
# check runs either way.
PARAMS = {"max_depth": 1, "debug_synchronize": True}

pytestmark = pytest.mark.skipif(**tm.no_cupy())


@fixture(autouse=True)
def cuda_async_pool() -> Iterator[None]:
    with xgb.config_context(use_cuda_async_pool=True):
        yield


def make_extqdm() -> XywExtQdm:
    """A fresh external-memory matrix over `N_BATCHES` equally sized batches."""
    X, y, w = tm.make_batches(N_SAMPLES_PER_BATCH, N_FEATURES, N_BATCHES, use_cupy=True)
    it = tm.IteratorForTest(X, y, w, cache=None, min_cache_page_bytes=0, on_host=True)
    return X, y, w, xgb.ExtMemQuantileDMatrix(it)


@fixture(scope="module")
def xyw_extqdm() -> XywExtQdm:
    with xgb.config_context(use_cuda_async_pool=True):
        return make_extqdm()


def get_fold_tree(cv_folds: xcv.FoldModels, k: int, iteration: int = -1) -> dict:
    """The `iteration`-th tree of the k-th fold model, as a JSON object."""
    model = json.loads(cv_folds.save_raw("json"))
    trees = model["cv_folds"][k]["gradient_booster"]["model"]["trees"]
    return trees[iteration]


def get_refit_tree(cv_folds: xcv.FoldModels, iteration: int = -1) -> dict:
    """The `iteration`-th tree of the full-data model, as a JSON object."""
    model = json.loads(cv_folds.save_raw("json"))
    trees = model["refit"]["gradient_booster"]["model"]["trees"]
    return trees[iteration]


def get_booster_tree(booster: xgb.Booster, iteration: int = -1) -> dict:
    """The `iteration`-th tree of a plain booster, as a JSON object."""
    model = json.loads(booster.save_raw("json"))
    trees = model["learner"]["gradient_booster"]["model"]["trees"]
    return trees[iteration]


def assert_same_tree(fused: dict, plain: dict) -> None:
    """Compare a fused-CV tree against the same tree grown by a plain booster.

    The two store the leaf values differently: a fused tree keeps them in `leaf_weights`
    and uses `right_children` as the node-to-leaf mapping, whereas a plain booster with a
    scalar leaf overloads the `split_conditions` slot of the leaf. `base_weights` holds
    the computed weight of every node under both conventions, so comparing it covers the
    leaves as well.

    """
    for key in ("left_children", "split_indices", "default_left", "split_type"):
        assert fused[key] == plain[key], key
    for key in ("base_weights", "sum_hessian", "loss_changes"):
        np.testing.assert_allclose(fused[key], plain[key], rtol=1e-6, atol=1e-6)
    # A leaf has no split condition, and the two disagree on what its slot holds.
    internal = [i for i, c in enumerate(fused["left_children"]) if c != -1]
    np.testing.assert_allclose(
        [fused["split_conditions"][i] for i in internal],
        [plain["split_conditions"][i] for i in internal],
        rtol=1e-6,
        atol=1e-6,
    )


def get_leaf_weight(tree: dict, nidx: int, n_targets: int = 1) -> list[float]:
    """The leaf weight of a node of a vector-leaf tree.

    `leaf_weights` is indexed by leaf index rather than node index, and `SetLeaves`
    repurposes `right_children` as the node-to-leaf mapping, so only `left_children` marks
    a leaf.

    """
    # Guard against -1 reaching here from a child array, which would silently wrap around.
    assert nidx >= 0
    assert tree["left_children"][nidx] == -1
    leaf_idx = tree["right_children"][nidx]
    return tree["leaf_weights"][leaf_idx * n_targets : (leaf_idx + 1) * n_targets]


def stump_leaf_values(tree: dict, features: cp.ndarray) -> cp.ndarray:
    """The leaf value every row of `features` reaches in a depth-1 tree."""
    import cupy as cp

    # The device splits on the binned value, whose cut is the smallest value strictly
    # greater than the raw value, so the raw-feature equivalent of the device comparison is
    # a strict `<`. `features` must already be float32 to match the split condition.
    go_left = features[:, tree["split_indices"][0]] < tree["split_conditions"][0]
    left = get_leaf_weight(tree, tree["left_children"][0])[0]
    right = get_leaf_weight(tree, tree["right_children"][0])[0]
    return cp.where(go_left, left, right).astype(cp.float32)


def fold_rows(k_folds: int, k: int) -> tuple[cp.ndarray, cp.ndarray]:
    """Global row indices of the training and the held-out rows of the k^th fold.

    The folds are split within each batch, hence the per-batch offset.

    """
    import cupy as cp
    from sklearn.model_selection import KFold

    # Every batch has the same number of rows here, hence the same within-batch split.
    train_idx, valid_idx = list(
        KFold(n_splits=k_folds).split(np.arange(N_SAMPLES_PER_BATCH))
    )[k]
    train = [cp.asarray(train_idx) + i * N_SAMPLES_PER_BATCH for i in range(N_BATCHES)]
    valid = [cp.asarray(valid_idx) + i * N_SAMPLES_PER_BATCH for i in range(N_BATCHES)]
    return cp.concatenate(train), cp.concatenate(valid)


def fold_weights(w: list[cp.ndarray], k_folds: int, k: int) -> cp.ndarray:
    """Row weights as the k^th fold sees them: the rows it holds out are zeroed."""
    import cupy as cp

    train_rows, _ = fold_rows(k_folds, k)
    weights = cp.concatenate(w)
    masked = cp.zeros_like(weights)
    masked[train_rows] = weights[train_rows]
    return masked


def expected_gradient(
    y: list[cp.ndarray], weights: cp.ndarray, margin: cp.ndarray | None = None
) -> tuple[cp.ndarray, cp.ndarray]:
    """The squared-error gradient and hessian of every row, as column vectors.

    A zero weight zeroes the gradient of that row, which is how a fold masks out the rows
    it holds out.

    """
    import cupy as cp

    labels = cp.concatenate(y).astype(cp.float32).reshape(-1, 1)
    column_w = weights.astype(cp.float32).reshape(-1, 1)
    predt = BASE_SCORE if margin is None else margin.reshape(-1, 1)
    return (predt - labels) * column_w, column_w


def train_reference(Xy: xgb.DMatrix, n_rounds: int) -> xgb.Booster:
    """A plain booster configured the way `FoldModels` configures every training unit."""
    return xgb.train(
        {
            **PARAMS,
            "base_score": BASE_SCORE,
            "device": "cuda",
            "multi_strategy": "multi_output_tree",
        },
        Xy,
        num_boost_round=n_rounds,
    )


def run_cv(
    Xy: xgb.ExtMemQuantileDMatrix, k_folds: int, n_rounds: int, refit: bool = False
) -> tuple[xcv.FoldModels, xcv.FoldPredictions, xcv.FoldGpairs]:
    """Run `n_rounds` rounds of fused cross-validation to completion.

    The returned gradient is the one of the last round, computed before that round's trees
    were grown.

    """
    cv_folds = xcv.FoldModels(data=Xy, k_folds=k_folds, refit=refit)
    folds = xcv.FoldInfoBatches(Xy, k_folds=k_folds)
    predts = xcv.FoldPredictions()
    cv_folds.init_prediction(Xy, folds, out=predts)
    gpairs = xcv.FoldGpairs()
    tree_method = xcv.FoldTreeMethod(cv_folds, Xy, params=PARAMS)
    for it in range(n_rounds):
        cv_folds.get_gradient(Xy, it, folds, predts, out=gpairs)
        tree_method.update(cv_folds, Xy, folds, gpairs, predts)
    return cv_folds, predts, gpairs


def test_cv_tree_method(xyw_extqdm: XywExtQdm) -> None:
    """The out-parameter protocol, and one round of depth-1 growth."""
    _, _, _, Xy = xyw_extqdm
    k_folds = 3

    cv_folds = xcv.FoldModels(data=Xy, k_folds=k_folds)
    assert cv_folds.num_boosted_rounds() == 0

    predts = xcv.FoldPredictions()
    folds = xcv.FoldInfoBatches(Xy, k_folds=k_folds)
    assert cv_folds.init_prediction(Xy, folds, out=predts) is predts
    gpairs = xcv.FoldGpairs()
    assert cv_folds.get_gradient(Xy, 0, folds, predts, out=gpairs) is gpairs
    tree_method = xcv.FoldTreeMethod(cv_folds, Xy, params=PARAMS)
    tree_method.update(cv_folds, Xy, folds, gpairs, predts)
    assert cv_folds.num_boosted_rounds() == 1

    # The data is continuous and random, so every fold has a positive-gain root split.
    for k in range(k_folds):
        tree = get_fold_tree(cv_folds, k)
        assert tree["left_children"] == [1, -1, -1]
        # 0 and 1 are leaf indices here, not "no child".
        assert tree["right_children"] == [2, 0, 1]
        assert get_leaf_weight(tree, 1) != get_leaf_weight(tree, 2)


def test_cv_fold_info_batches(xyw_extqdm: XywExtQdm) -> None:
    """The handles and the interleaved gradient buffer the Python layer hands back."""
    _, _, _, Xy = xyw_extqdm
    k_folds = 3

    folds = xcv.FoldInfoBatches(Xy, k_folds=k_folds)
    assert isinstance(folds.handle, ctypes.c_void_p)
    assert folds.handle.value is not None
    assert folds.k_folds == k_folds

    cv_folds = xcv.FoldModels(data=Xy, k_folds=k_folds)
    predts = xcv.FoldPredictions()
    assert cv_folds.init_prediction(Xy, folds, out=predts) is predts
    gpairs = xcv.FoldGpairs()
    assert cv_folds.get_gradient(Xy, 0, folds, predts, out=gpairs) is gpairs
    assert isinstance(gpairs.handle, ctypes.c_void_p)
    assert gpairs.handle.value is not None

    float_size = ctypes.sizeof(ctypes.c_float)
    for k in range(k_folds):
        grad, hess = gpairs.get(k, copy=False)
        # Gradient and hessian are two strided views over one interleaved buffer, indexed
        # by the global row index.
        assert grad.shape == (Xy.num_row(), 1)
        assert grad.shape == hess.shape
        assert grad.dtype == hess.dtype
        assert grad.data.ptr + float_size == hess.data.ptr
        assert grad.strides == hess.strides == (2 * float_size, 2 * float_size)

    # The buffer is reusable across rounds.
    assert cv_folds.get_gradient(Xy, 1, folds, predts, out=gpairs) is gpairs


@pytest.mark.skipif(**tm.no_sklearn())
@pytest.mark.parametrize("base_margin", [False, True])
def test_cv_gradient(base_margin: bool) -> None:
    """A fold's gradient covers its training rows and zeroes the rows it holds out."""
    import cupy as cp

    k_folds = 3
    # A local matrix: setting a base margin would leak into the other tests.
    _, y, w, Xy = make_extqdm()
    margin = None
    if base_margin:
        # A distinct margin for every row, the gradient of a row must be calculated from
        # the margin of that same row.
        margin = cp.arange(Xy.num_row(), dtype=cp.float32) / Xy.num_row()
        Xy.set_info(base_margin=margin)

    cv_folds = xcv.FoldModels(data=Xy, k_folds=k_folds)
    folds = xcv.FoldInfoBatches(Xy, k_folds=k_folds)
    predts = xcv.FoldPredictions()
    cv_folds.init_prediction(Xy, folds, out=predts)
    gpairs = xcv.FoldGpairs()
    cv_folds.get_gradient(Xy, 0, folds, predts, out=gpairs)

    for k in range(k_folds):
        grad, hess = gpairs.get(k, copy=False)
        want_grad, want_hess = expected_gradient(y, fold_weights(w, k_folds, k), margin)
        cp.testing.assert_allclose(grad, want_grad)
        cp.testing.assert_allclose(hess, want_hess)


@pytest.mark.skipif(**tm.no_sklearn())
def test_cv_prediction_cache(xyw_extqdm: XywExtQdm) -> None:
    """A fold's training cache accumulates leaf values for its training rows only."""
    import cupy as cp

    X, _, _, Xy = xyw_extqdm
    k_folds = 3
    features = cp.concatenate(X).astype(cp.float32)

    cv_folds = xcv.FoldModels(data=Xy, k_folds=k_folds)
    predts = xcv.FoldPredictions()
    folds = xcv.FoldInfoBatches(Xy, k_folds=k_folds)
    cv_folds.init_prediction(Xy, folds, out=predts)
    gpairs = xcv.FoldGpairs()
    tree_method = xcv.FoldTreeMethod(cv_folds, Xy, params=PARAMS)

    expected = [
        cp.full((Xy.num_row(), 1), BASE_SCORE, dtype=cp.float32) for _ in range(k_folds)
    ]
    for it in range(2):
        cv_folds.get_gradient(Xy, it, folds, predts, out=gpairs)
        tree_method.update(cv_folds, Xy, folds, gpairs, predts)
        assert cv_folds.num_boosted_rounds() == it + 1

        for k in range(k_folds):
            train_rows, valid_rows = fold_rows(k_folds, k)
            leaf_value = stump_leaf_values(get_fold_tree(cv_folds, k), features)
            # Neither leaf may be empty, otherwise the check below is vacuous on one side.
            assert cp.unique(leaf_value[train_rows]).size == 2

            expected[k][train_rows] += leaf_value[train_rows].reshape(-1, 1)
            predt = predts.get(k)
            assert predt.shape == (Xy.num_row(), 1)
            cp.testing.assert_allclose(predt[train_rows], expected[k][train_rows])
            # The rows held out by the fold are padding, nothing may write to them.
            cp.testing.assert_array_equal(
                predt[valid_rows], cp.full((valid_rows.size, 1), BASE_SCORE)
            )


@pytest.mark.skipif(**tm.no_sklearn())
def test_cv_oof_prediction_cache(xyw_extqdm: XywExtQdm) -> None:
    """Each row's OOF prediction comes from the fold that held it out."""
    import cupy as cp

    X, _, _, Xy = xyw_extqdm
    k_folds = 3
    features = cp.concatenate(X).astype(cp.float32)

    cv_folds, predts, _ = run_cv(Xy, k_folds, 1)

    expected = cp.full((Xy.num_row(), 1), BASE_SCORE, dtype=cp.float32)
    for k in range(k_folds):
        _, valid_rows = fold_rows(k_folds, k)
        leaf_value = stump_leaf_values(get_fold_tree(cv_folds, k), features)
        expected[valid_rows] += leaf_value[valid_rows].reshape(-1, 1)

    cp.testing.assert_allclose(predts.get_valid(), expected)


@pytest.mark.skipif(**tm.no_sklearn())
def test_cv_vs_reference(xyw_extqdm: XywExtQdm) -> None:
    """Each fold must train exactly like a booster fitted on that fold's rows alone."""
    import cupy as cp

    X, y, w, Xy = xyw_extqdm
    k_folds, n_rounds = 3, 3

    _, predts, _ = run_cv(Xy, k_folds, n_rounds)

    features, labels, weights = (cp.concatenate(v) for v in (X, y, w))
    for k in range(k_folds):
        train_rows, _ = fold_rows(k_folds, k)
        # `ref` shares the CV cuts, so the reference sees the same bins.
        Xyk = xgb.QuantileDMatrix(
            features[train_rows],
            label=labels[train_rows],
            weight=weights[train_rows],
            ref=Xy,
        )
        booster = train_reference(Xyk, n_rounds)
        margin = cp.asarray(booster.predict(Xyk, output_margin=True)).reshape(-1, 1)
        cp.testing.assert_allclose(
            predts.get(k)[train_rows], margin, rtol=1e-6, atol=1e-6
        )


def test_cv_refit_vs_reference(xyw_extqdm: XywExtQdm) -> None:
    """The refit model must train exactly like a booster fitted on the whole dataset."""
    import cupy as cp

    _, _, _, Xy = xyw_extqdm
    k_folds, n_rounds = 3, 3

    cv_folds, predts, _ = run_cv(Xy, k_folds, n_rounds, refit=True)
    assert cv_folds.refit
    assert cv_folds.num_boosted_rounds() == n_rounds

    booster = train_reference(Xy, n_rounds)
    for it in range(n_rounds):
        assert_same_tree(get_refit_tree(cv_folds, it), get_booster_tree(booster, it))

    margin = cp.asarray(booster.predict(Xy, output_margin=True)).reshape(-1, 1)
    predt = predts.get_refit()
    assert predt.shape == (Xy.num_row(), 1)
    cp.testing.assert_allclose(predt, margin, rtol=1e-6, atol=1e-6)


def test_cv_refit_does_not_disturb_folds(xyw_extqdm: XywExtQdm) -> None:
    """Adding the refit model must leave the fold models and the OOF cache untouched."""
    import cupy as cp

    _, _, _, Xy = xyw_extqdm
    k_folds, n_rounds = 3, 3

    plain, plain_predts, _ = run_cv(Xy, k_folds, n_rounds, refit=False)
    with_refit, refit_predts, _ = run_cv(Xy, k_folds, n_rounds, refit=True)

    for k in range(k_folds):
        for it in range(n_rounds):
            assert get_fold_tree(plain, k, it) == get_fold_tree(with_refit, k, it)
        cp.testing.assert_array_equal(plain_predts.get(k), refit_predts.get(k))
    # The refit model holds no row out, so it must contribute nothing to the OOF cache.
    cp.testing.assert_array_equal(plain_predts.get_valid(), refit_predts.get_valid())


def test_cv_refit_gradient(xyw_extqdm: XywExtQdm) -> None:
    """Unlike a fold, the refit model has a gradient for every row."""
    import cupy as cp

    _, y, w, Xy = xyw_extqdm

    _, _, gpairs = run_cv(Xy, 3, 1, refit=True)
    grad, hess = gpairs.get_refit()
    assert grad.shape == (Xy.num_row(), 1)

    # No row is masked out, so no weight is zeroed.
    want_grad, want_hess = expected_gradient(y, cp.concatenate(w))
    cp.testing.assert_allclose(grad, want_grad)
    cp.testing.assert_allclose(hess, want_hess)


def test_cv_refit_access(xyw_extqdm: XywExtQdm) -> None:
    """The refit model is reachable only through its own getters, and only if asked for."""
    _, _, _, Xy = xyw_extqdm
    k_folds = 3

    cv_folds, predts, gpairs = run_cv(Xy, k_folds, 1, refit=False)
    assert not cv_folds.refit
    assert "refit" not in json.loads(cv_folds.save_raw("json"))
    with pytest.raises(xgb.core.XGBoostError, match="No refit model"):
        predts.get_refit()
    with pytest.raises(xgb.core.XGBoostError, match="No refit model"):
        gpairs.get_refit()

    _, predts, gpairs = run_cv(Xy, k_folds, 1, refit=True)
    # The refit model is a training unit, but it is not a fold, so the fold getters stay
    # bound by the fold count rather than the unit count.
    predts.get(k_folds - 1)
    gpairs.get(k_folds - 1)
    with pytest.raises(xgb.core.XGBoostError):
        predts.get(k_folds)
    with pytest.raises(xgb.core.XGBoostError):
        gpairs.get(k_folds)
