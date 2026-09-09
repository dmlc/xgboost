# SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import json
from collections.abc import Iterator
from dataclasses import dataclass
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
    list[cp.ndarray],
    list[cp.ndarray],
    list[cp.ndarray] | None,
    xgb.ExtMemQuantileDMatrix,
]

N_SAMPLES_PER_BATCH, N_FEATURES, N_BATCHES = 16, 4, 2

# Fixed by FoldModels, which pins `boost_from_average` to false and the base score to the
# objective default. It cannot be configured from here.
BASE_SCORE = 0.5

# A depth that leaves the trees ragged on these fixtures, so that the folds run out of
# candidates at different levels. `debug_synchronize` gates the check that every training row
# of a unit, and only those, received a leaf position; the fused-page-pass check runs either
# way.
PARAMS = {"max_depth": 3, "debug_synchronize": True}

pytestmark = pytest.mark.skipif(**tm.no_cupy())


@fixture(autouse=True)
def cuda_async_pool() -> Iterator[None]:
    with xgb.config_context(use_cuda_async_pool=True):
        yield


def make_extqdm(n_targets: int = 1) -> XywExtQdm:
    """A fresh external-memory matrix over `N_BATCHES` equally sized batches.

    Under `n_targets=2` the weight becomes a second, unrelated target and the matrix carries
    none. Either way the batching is the same, so `fold_rows` describes both.

    """
    import cupy as cp

    X, y, w = tm.make_batches(N_SAMPLES_PER_BATCH, N_FEATURES, N_BATCHES, use_cupy=True)
    multi = n_targets == 2
    labels = [cp.stack([yi, wi], axis=1) for yi, wi in zip(y, w)] if multi else y
    weights = None if multi else w
    it = tm.IteratorForTest(
        X, labels, weights, cache=None, min_cache_page_bytes=0, on_host=True
    )
    return X, labels, weights, xgb.ExtMemQuantileDMatrix(it)


@fixture(scope="module")
def xyw_extqdm() -> XywExtQdm:
    with xgb.config_context(use_cuda_async_pool=True):
        return make_extqdm()


def make_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The same dataset as `make_extqdm`, as single host arrays."""
    return tm.make_regression(
        N_SAMPLES_PER_BATCH * N_BATCHES, N_FEATURES, use_cupy=False
    )


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


def tree_depth(tree: dict) -> int:
    """The number of splits on the longest root-to-leaf path.

    `right_children` is the node-to-leaf mapping for a leaf, so it is a child link only
    where `left_children` says the node is internal.

    """

    def walk(nidx: int) -> int:
        left = tree["left_children"][nidx]
        if left == -1:
            return 0
        return 1 + max(walk(left), walk(tree["right_children"][nidx]))

    return walk(0)


def fold_rows(k_folds: int, k: int) -> tuple[cp.ndarray, cp.ndarray]:
    """Global row indices of the training and the held-out rows of the k^th fold.

    The folds are split within each page, and `make_extqdm` keeps one page per batch, hence
    the per-batch offset.

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


@dataclass
class CvState:
    """The handles of one fused CV run, wired to each other and ready to boost."""

    Xy: xgb.ExtMemQuantileDMatrix
    cv_folds: xcv.FoldModels
    folds: xcv.FoldInfoBatches
    predts: xcv.FoldPredictions
    gpairs: xcv.FoldGpairs
    tree_method: xcv.FoldTreeMethod

    def boost(self, it: int) -> None:
        """One round: the gradient of every unit, then one tree for each."""
        self.cv_folds.get_gradient(
            self.Xy, it, self.folds, self.predts, out=self.gpairs
        )
        self.tree_method.update(
            self.cv_folds, self.Xy, self.folds, self.gpairs, self.predts
        )


def make_cv_state(
    Xy: xgb.ExtMemQuantileDMatrix,
    k_folds: int,
    refit: bool = False,
    params: dict | None = None,
) -> CvState:
    """Set a run up to the point where `boost` can be called, growing nothing yet."""
    cv_folds = xcv.FoldModels(data=Xy, k_folds=k_folds, refit=refit)
    folds = xcv.FoldInfoBatches(Xy, k_folds=k_folds)
    predts = xcv.FoldPredictions()
    cv_folds.init_prediction(Xy, folds, out=predts)
    return CvState(
        Xy=Xy,
        cv_folds=cv_folds,
        folds=folds,
        predts=predts,
        gpairs=xcv.FoldGpairs(),
        tree_method=xcv.FoldTreeMethod(cv_folds, Xy, params=params or PARAMS),
    )


def run_cv(
    Xy: xgb.ExtMemQuantileDMatrix,
    k_folds: int,
    n_rounds: int,
    refit: bool = False,
    params: dict | None = None,
) -> tuple[xcv.FoldModels, xcv.FoldPredictions, xcv.FoldGpairs]:
    """Run `n_rounds` rounds of fused cross-validation to completion.

    The returned gradient is the one of the last round, computed before that round's trees
    were grown.

    """
    state = make_cv_state(Xy, k_folds, refit, params)
    for it in range(n_rounds):
        state.boost(it)
    return state.cv_folds, state.predts, state.gpairs


def run_cv_eval(
    Xy: xgb.ExtMemQuantileDMatrix,
    k_folds: int,
    n_rounds: int,
    *,
    refit: bool = False,
    metrics: str | list[str] | None = None,
    eval_train: bool = True,
) -> tuple[xcv.FoldModels, xcv.FoldPredictions, xcv.CvEvalResult]:
    """`run_cv`, evaluating after every round; the result is the last round's."""
    state = make_cv_state(Xy, k_folds, refit)
    evaluator = xcv.FoldEvaluator(
        state.cv_folds, metrics=metrics, eval_train=eval_train
    )
    result = None
    for it in range(n_rounds):
        state.boost(it)
        # After the update, so the value labelled round `it` describes `it + 1` trees.
        result = evaluator.evaluate(state.cv_folds, Xy, state.folds, state.predts, it)
    assert result is not None
    return state.cv_folds, state.predts, result


def eval_reference(booster: xgb.Booster, Xy: xgb.DMatrix) -> float:
    """`Booster.eval` reports a line, `[0]\\tname-metric:value`; this is the value."""
    return float(booster.eval(Xy).split(":")[-1])


def test_cv_tree_method(xyw_extqdm: XywExtQdm) -> None:
    """The out-parameter protocol, and one round of growth."""
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

    # The data is continuous and random, so every fold splits the root and then keeps
    # splitting until it runs out of gain or of depth. Which of the two it is differs by
    # fold, so only the bounds hold for all of them.
    for k in range(k_folds):
        tree = get_fold_tree(cv_folds, k)
        assert 1 < tree_depth(tree) <= PARAMS["max_depth"]
        leaves = [i for i, c in enumerate(tree["left_children"]) if c == -1]
        assert len({get_leaf_weight(tree, i)[0] for i in leaves}) > 1


def test_cv_tree_method_rejects(xyw_extqdm: XywExtQdm) -> None:
    """Loss-guided growth would make the page passes scale with the node count."""
    _, _, _, Xy = xyw_extqdm
    state = make_cv_state(Xy, 3, params={**PARAMS, "grow_policy": "lossguide"})
    # The parameters are checked from `InitDataOnce`, which the first update runs.
    with pytest.raises(xgb.core.XGBoostError, match="Only the depthwise grow policy"):
        state.boost(0)


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
    assert w is not None  # Single-target, so the matrix carries a weight.
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


@pytest.mark.skipif(**tm.no_sklearn())
def test_cv_prediction_cache(xyw_extqdm: XywExtQdm) -> None:
    """A fold's training cache is written for its training rows only.

    What the accumulated value must be is `test_cv_vs_reference`'s claim; this one owns the
    other half, that the rows a fold holds out are padding it never touches.

    """
    import cupy as cp

    _, _, _, Xy = xyw_extqdm
    k_folds = 3

    cv_folds = xcv.FoldModels(data=Xy, k_folds=k_folds)
    predts = xcv.FoldPredictions()
    folds = xcv.FoldInfoBatches(Xy, k_folds=k_folds)
    cv_folds.init_prediction(Xy, folds, out=predts)
    gpairs = xcv.FoldGpairs()
    tree_method = xcv.FoldTreeMethod(cv_folds, Xy, params=PARAMS)

    for it in range(2):
        cv_folds.get_gradient(Xy, it, folds, predts, out=gpairs)
        tree_method.update(cv_folds, Xy, folds, gpairs, predts)
        assert cv_folds.num_boosted_rounds() == it + 1

        for k in range(k_folds):
            train_rows, valid_rows = fold_rows(k_folds, k)
            predt = predts.get(k)
            assert predt.shape == (Xy.num_row(), 1)
            # The rows held out by the fold are padding, nothing may write to them.
            cp.testing.assert_array_equal(
                predt[valid_rows], cp.full((valid_rows.size, 1), BASE_SCORE)
            )
            # Without this, a run that writes nowhere would satisfy the check above.
            assert cp.all(predt[train_rows] != BASE_SCORE)


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


def test_cv_hist_subtraction(xyw_extqdm: XywExtQdm) -> None:
    """A small histogram cache changes how a sibling is obtained, not what is grown.

    The budget is shared by the units, so one node per unit is the tightest there is. It
    evicts every parent below the root, and the siblings that would have been subtracted are
    built in the same page pass instead. The two runs must agree bit for bit: a histogram
    bin is a fixed-point integer, so a parent bin is the exact sum of its children's.

    """
    import cupy as cp

    _, _, _, Xy = xyw_extqdm
    k_folds, n_rounds = 3, 2

    subtracted, cached_predts, _ = run_cv(Xy, k_folds, n_rounds)
    rebuilt, tight_predts, _ = run_cv(
        Xy, k_folds, n_rounds, params={**PARAMS, "max_cached_hist_node": 1}
    )
    assert json.loads(subtracted.save_raw("json")) == json.loads(
        rebuilt.save_raw("json")
    )
    cp.testing.assert_array_equal(cached_predts.get_valid(), tight_predts.get_valid())


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

    # The refit unit trains inside the same page loop as the folds, but it must not disturb
    # them: the fold trees are the ones the same run without a refit unit grows.
    plain_folds, plain_predts, _ = run_cv(Xy, k_folds, n_rounds)
    for k in range(k_folds):
        for it in range(n_rounds):
            assert get_fold_tree(cv_folds, k, it) == get_fold_tree(plain_folds, k, it)
        cp.testing.assert_array_equal(predts.get(k), plain_predts.get(k))
    cp.testing.assert_array_equal(predts.get_valid(), plain_predts.get_valid())


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


def test_cv_data_iter() -> None:
    """The batches must tile the rows of the input in order."""
    X, y, _ = make_dataset()
    n_samples = X.shape[0]

    assert xcv.CvDataIter(X, y).batch_ptr == [0, n_samples]
    it = xcv.CvDataIter(X, y, batch_size=N_SAMPLES_PER_BATCH)
    assert it.n_batches == N_BATCHES
    # A batch size that does not divide the dataset leaves a shorter last batch.
    short = xcv.CvDataIter(X, y, batch_size=n_samples - 1)
    assert short.batch_ptr == [0, n_samples - 1, n_samples]

    Xy = xgb.ExtMemQuantileDMatrix(it)
    assert (Xy.num_row(), Xy.num_col()) == (n_samples, N_FEATURES)

    with pytest.raises(ValueError, match="they must match"):
        xcv.CvDataIter(X, y[:-1])
    with pytest.raises(ValueError, match="must be positive"):
        xcv.CvDataIter(X, y, batch_size=0)


@pytest.mark.skipif(**tm.no_sklearn())
@pytest.mark.parametrize("n_targets", [1, 2])
def test_cross_val_predict(n_targets: int) -> None:
    """Every estimate must match a booster fitted on the rows its fold kept."""
    import cupy as cp
    from sklearn.model_selection import KFold

    X, y, w = make_dataset()
    # `w` is drawn independently of `y`, so it doubles as a second, unrelated target.
    labels = y if n_targets == 1 else np.stack([y, w], axis=1)
    n_samples, k_folds, n_rounds = X.shape[0], 3, 3

    # The input is on the host, the iterator is what moves it to the device.
    oof = xcv.cross_val_predict(
        X, labels, cv=k_folds, params=PARAMS, num_boost_round=n_rounds
    )
    assert oof.dtype == cp.float32
    assert oof.shape == ((n_samples,) if n_targets == 1 else (n_samples, n_targets))

    # A dataset this small ends up in a single page, so the folds are the unshuffled
    # `KFold` split of the whole dataset.
    Xy = xgb.ExtMemQuantileDMatrix(xcv.CvDataIter(X, labels))
    features, d_labels = cp.asarray(X), cp.asarray(labels)
    expected = cp.empty_like(oof)
    for train_rows, valid_rows in KFold(n_splits=k_folds).split(X):
        train, valid = cp.asarray(train_rows), cp.asarray(valid_rows)
        # `ref` shares the CV cuts, so the reference sees the same bins.
        Xyk = xgb.QuantileDMatrix(features[train], label=d_labels[train], ref=Xy)
        booster = train_reference(Xyk, n_rounds)
        expected[valid] = cp.asarray(
            booster.predict(
                xgb.QuantileDMatrix(features[valid], ref=Xy), output_margin=True
            )
        )

    cp.testing.assert_allclose(oof, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(**tm.no_sklearn())
@pytest.mark.parametrize("n_targets", [1, 2])
def test_cv_evaluate_vs_reference(xyw_extqdm: XywExtQdm, n_targets: int) -> None:
    """Both values of a fold must match a booster fitted on that fold's rows alone.

    The oracle is the metric itself, which also pins the multi-target semantics: `rmse` pools
    every element into one value rather than averaging per-target values.

    """
    import cupy as cp

    k_folds, n_rounds = 3, 3
    # One target reuses the module fixture; two need a matrix of their own.
    X, y, w, Xy = xyw_extqdm if n_targets == 1 else make_extqdm(n_targets)
    weights = None if w is None else cp.concatenate(w)

    _, _, result = run_cv_eval(Xy, k_folds, n_rounds)
    assert result.train is not None
    assert result.train.dtype == result.valid.dtype == np.float64

    features, labels = cp.concatenate(X), cp.concatenate(y)

    def subset(rows: cp.ndarray) -> xgb.QuantileDMatrix:
        # `ref` shares the CV cuts, so the reference sees the same bins.
        wk = None if weights is None else weights[rows]
        return xgb.QuantileDMatrix(
            features[rows], label=labels[rows], weight=wk, ref=Xy
        )

    for k in range(k_folds):
        train_rows, valid_rows = fold_rows(k_folds, k)
        Xyk = subset(train_rows)
        booster = train_reference(Xyk, n_rounds)
        assert result.train[0, k] == pytest.approx(
            eval_reference(booster, Xyk), rel=1e-6
        )
        assert result.valid[0, k] == pytest.approx(
            eval_reference(booster, subset(valid_rows)), rel=1e-6
        )


def test_cv_evaluate_request(xyw_extqdm: XywExtQdm) -> None:
    """The request decides the names and both axes of the buffer, and bad ones are rejected."""
    _, _, _, Xy = xyw_extqdm
    k_folds = 3
    metrics = ["mae", "error@0.5", "error@0.7"]

    # Names come back under `Metric::Name()`, which drops a parameter left at its default and
    # keeps one that is not. `refit=True` adds a unit that trains but is not scored.
    _, _, both = run_cv_eval(Xy, k_folds, 1, metrics=metrics, refit=True)
    assert both.names == ("mae", "error", "error@0.7")
    assert both.train is not None
    assert both.train.shape == both.valid.shape == (3, k_folds)
    # The rows differ, so they are not one metric repeated.
    assert not np.allclose(both.valid[0], both.valid[1])

    # `eval_train=False` drops the training section, which Python decides from the shorter
    # buffer, and leaves the held-out one untouched.
    _, _, valid_only = run_cv_eval(
        Xy, k_folds, 1, metrics=metrics, refit=True, eval_train=False
    )
    assert valid_only.train is None
    np.testing.assert_allclose(valid_only.valid, both.valid, rtol=1e-6)

    _, _, one = run_cv_eval(Xy, k_folds, 1, metrics="mae")
    assert one.names == ("mae",)

    # Not evaluating is a matter of not building an evaluator, so there is no request for it.
    with pytest.raises(xgb.core.XGBoostError, match="must name at least one metric"):
        run_cv_eval(Xy, k_folds, 1, metrics=[])

    # Rejected while the evaluator is built, once per half of the gate: `ndcg` keeps a
    # per-DMatrix cache and fails the `MetricNoCache` cast, `cox-nloglik` fails the deny list.
    cv_folds = xcv.FoldModels(data=Xy, k_folds=k_folds)
    for metric in ("ndcg", "cox-nloglik"):
        with pytest.raises(
            xgb.core.XGBoostError, match=f"`{metric}` metric is not supported"
        ):
            xcv.FoldEvaluator(cv_folds, metrics=[metric])


def test_cv_evaluate_rejects(xyw_extqdm: XywExtQdm) -> None:
    """Evaluating out of step, on another run's caches, or on rows a fold cannot split."""
    _, _, _, Xy = xyw_extqdm
    state = make_cv_state(Xy, 3)
    evaluator = xcv.FoldEvaluator(state.cv_folds)

    stale = "not from the round being evaluated"
    # Before the update, which is the ordering mistake a driver would make.
    with pytest.raises(xgb.core.XGBoostError, match=stale):
        evaluator.evaluate(state.cv_folds, Xy, state.folds, state.predts, 0)

    state.boost(0)
    with pytest.raises(xgb.core.XGBoostError, match=stale):
        evaluator.evaluate(state.cv_folds, Xy, state.folds, state.predts, 1)
    with pytest.raises(xgb.core.XGBoostError, match="needs a committed round"):
        evaluator.evaluate(state.cv_folds, Xy, state.folds, state.predts, -1)

    # Caches of a run with a refit unit describe one unit more than these models do. Read
    # before anything else, so an unboosted run is enough to reach it.
    refit = make_cv_state(Xy, 3, refit=True)
    with pytest.raises(xgb.core.XGBoostError, match="describe 4 training units"):
        evaluator.evaluate(state.cv_folds, Xy, state.folds, refit.predts, 0)

    # A local matrix, since a group would leak into the other tests through the fixture.
    _, _, _, grouped = make_extqdm()
    grouped.set_info(qid=np.full(shape=(N_SAMPLES_PER_BATCH * N_BATCHES), fill_value=0))
    gstate = make_cv_state(grouped, 3)
    with pytest.raises(xgb.core.XGBoostError, match="does not support ranking data"):
        xcv.FoldEvaluator(gstate.cv_folds).evaluate(
            gstate.cv_folds, grouped, gstate.folds, gstate.predts, 0
        )


def test_cv_evaluate_is_inert(xyw_extqdm: XywExtQdm) -> None:
    """Evaluating every round must not change what the run produces."""
    import cupy as cp

    _, _, _, Xy = xyw_extqdm
    k_folds, n_rounds = 3, 3

    quiet_folds, quiet_predts, _ = run_cv(Xy, k_folds, n_rounds, refit=True)
    loud_folds, loud_predts, _ = run_cv_eval(Xy, k_folds, n_rounds, refit=True)

    # Every tree of every unit, byte for byte.
    assert json.loads(quiet_folds.save_raw("json")) == json.loads(
        loud_folds.save_raw("json")
    )
    for k in range(k_folds):
        cp.testing.assert_array_equal(quiet_predts.get(k), loud_predts.get(k))
    cp.testing.assert_array_equal(quiet_predts.get_valid(), loud_predts.get_valid())
    cp.testing.assert_array_equal(quiet_predts.get_refit(), loud_predts.get_refit())
