"""Tests for inference."""

import json
from typing import Any, Optional, Type

import numpy as np
import pytest
from scipy import sparse
from scipy.special import logit  # pylint: disable=no-name-in-module

from ..compat import import_cupy
from ..core import Booster, DMatrix, ExtMemQuantileDMatrix, QuantileDMatrix
from ..training import train
from .data import IteratorForTest
from .shared import validate_leaf_output
from .updater import ResetStrategy, get_basescore
from .utils import Device

# Model parameters, raw output groups, and transformed output width. Use representative
# layouts instead of crossing every objective with every booster and strategy.
_PREDICT_SHAPE_CASES: dict[str, tuple[dict[str, Any], int, int]] = {
    "regression": ({"objective": "reg:squarederror"}, 1, 1),
    "binary": ({"objective": "binary:logistic"}, 1, 1),
    "softprob": ({"objective": "multi:softprob", "num_class": 3}, 3, 3),
    "softmax": ({"objective": "multi:softmax", "num_class": 3}, 3, 1),
    "quantile": (
        {"objective": "reg:quantileerror", "quantile_alpha": [0.25, 0.75]},
        2,
        2,
    ),
    "multi_target": ({"num_target": 2}, 2, 2),
    "vector_leaves": ({"num_target": 2, "multi_strategy": "multi_output_tree"}, 2, 2),
    "distribution": ({"objective": "reg:normal"}, 2, 2),
    "dropout": ({"rate_drop": 0.5, "skip_drop": 0.0}, 1, 1),
    "linear": ({"booster": "gblinear"}, 1, 1),
    "linear_softmax": (
        {"booster": "gblinear", "objective": "multi:softmax", "num_class": 3},
        3,
        1,
    ),
}


def predict_shape_cases(device: Device) -> list[str]:
    """Return shape test case names supported by the prediction device."""
    return [
        name
        for name, (params, _, _) in _PREDICT_SHAPE_CASES.items()
        if device == "cpu" or params.get("booster") != "gblinear"
    ]


# pylint: disable=too-many-locals
def run_predict_shape(device: Device, case: str) -> None:
    """Check dimensions across prediction modes, row counts and strictness.

    Values and margins exercise objective transformation and inplace prediction; SHAP
    retains raw output groups. Leaf layouts are covered by run_predict_leaf.
    """
    params, groups, outputs = _PREDICT_SHAPE_CASES[case]
    rng = np.random.default_rng(2026)
    X = rng.random((24, 4), dtype=np.float32)
    if "num_class" in params or params.get("objective") == "binary:logistic":
        y = rng.integers(0, params.get("num_class", 2), X.shape[0])
    else:
        y = rng.random((X.shape[0], params.get("num_target", 1)))
    linear = params.get("booster") == "gblinear"
    vector_leaf = params.get("multi_strategy") == "multi_output_tree"
    train_params = {"device": device, "nthread": 2, **params}
    if not linear:
        train_params["tree_method"] = "hist"
    model = train(train_params, DMatrix(X, y), num_boost_round=3)
    xp = import_cupy() if device == "cuda" else np

    # Check empty predictions before allocation and after reusing a non-empty output.
    for rows in [0, 4, 0, 1]:
        data = X[:rows]
        # SHAP dimensions come from the model, even if input columns are missing.
        matrices = [DMatrix(data), DMatrix(data[:, :-1])]
        for strict in [False, True]:
            group_axis = (groups,) if strict or groups > 1 else ()
            modes = [
                ({}, (rows, outputs) if strict or outputs > 1 else (rows,)),
                ({"output_margin": True}, (rows,) + group_axis),
                ({"pred_contribs": True}, (rows,) + group_axis + (X.shape[1] + 1,)),
                (
                    {"pred_interactions": True},
                    (rows,) + group_axis + (X.shape[1] + 1,) * 2,
                ),
            ]
            if device == "cpu" and not vector_leaf:
                modes += [
                    ({**kwargs, "approx_contribs": True}, shape)
                    for kwargs, shape in modes[2:]
                ]
            for kwargs, expected in modes:
                for interval in [(0, 0), (0, 1)]:
                    for dmat in matrices:
                        result = model.predict(
                            dmat,
                            strict_shape=strict,
                            iteration_range=interval,
                            **kwargs,
                        )
                        assert result.shape == expected, (
                            case,
                            rows,
                            strict,
                            interval,
                            kwargs,
                            dmat.num_col(),
                        )
                if linear or "pred_contribs" in kwargs or "pred_interactions" in kwargs:
                    continue
                inputs = [xp.asarray(data)]
                if device == "cpu":
                    inputs.append(sparse.csr_matrix(data))
                for value in inputs:
                    result = model.inplace_predict(
                        value,
                        predict_type="margin" if kwargs else "value",
                        strict_shape=strict,
                    )
                    assert result.shape == expected, (case, rows, strict, kwargs)


def _make_leaf_dmatrix(
    device: Device, DMatrixT: Type[DMatrix], X: np.ndarray, y: np.ndarray
) -> tuple[DMatrix, Optional[QuantileDMatrix]]:
    """Make a prediction matrix and an in-memory reference for external memory."""
    if DMatrixT is not ExtMemQuantileDMatrix:
        return DMatrixT(X, y), None

    xp = import_cupy() if device.startswith("cuda") else np
    X, y = xp.asarray(X), xp.asarray(y)
    it = IteratorForTest(
        xp.array_split(X, 4),
        xp.array_split(y, 4),
        None,
        cache="cache",
        on_host=True,
        min_cache_page_bytes=0,
    )
    m = ExtMemQuantileDMatrix(
        it, cache_host_ratio=1.0 if device.startswith("cuda") else None
    )
    return m, QuantileDMatrix(X, y, ref=m)


def _predict_leaf(
    booster: Booster,
    m: DMatrix,
    reference: Optional[DMatrix],
    **kwargs: Any,
) -> np.ndarray:
    """Predict leaves and compare external memory against in-memory data."""
    predt = booster.predict(m, pred_leaf=True, **kwargs)
    if reference is not None:
        ref_predt = booster.predict(reference, pred_leaf=True, **kwargs)
        np.testing.assert_array_equal(predt, ref_predt)
    return predt


def _validate_leaf_indices(leaf: np.ndarray, trees: list[dict]) -> None:
    """Validate that each predicted node is a leaf in the corresponding tree."""
    assert leaf.ndim == 2
    assert leaf.shape[1] == len(trees)
    for tree_idx, tree in enumerate(trees):
        prediction = leaf[:, tree_idx]
        assert np.isfinite(prediction).all()
        node_idx = prediction.astype(np.int64)
        np.testing.assert_array_equal(prediction, node_idx)

        left = np.asarray(tree["left_children"])
        assert ((0 <= node_idx) & (node_idx < left.size)).all()
        assert (left[node_idx] == -1).all()


# pylint: disable=too-many-locals
def run_predict_leaf(device: Device, DMatrixT: Type[DMatrix]) -> np.ndarray:
    """Run tests for leaf index prediction."""
    rows, cols, classes = 100, 4, 5
    n_parallel, n_rounds = 4, 10
    rng = np.random.RandomState(1994)
    X = rng.randn(rows, cols)
    y = rng.randint(low=0, high=classes, size=rows)

    m, reference = _make_leaf_dmatrix(device, DMatrixT, X, y)
    dtrain = DMatrix(X, y) if DMatrixT is ExtMemQuantileDMatrix else m
    booster = train(
        {
            "num_parallel_tree": n_parallel,
            "num_class": classes,
            "tree_method": "hist",
        },
        dtrain,
        num_boost_round=n_rounds,
    )
    booster.set_param({"device": device})
    empty = DMatrix(np.empty((0, cols)))

    leaf = _predict_leaf(booster, m, reference, strict_shape=True)
    assert leaf.shape == (rows, n_rounds, classes, n_parallel)
    validate_leaf_output(leaf, n_parallel)

    n_iters = 2
    sliced = _predict_leaf(
        booster,
        m,
        reference,
        iteration_range=(0, n_iters),
        strict_shape=True,
    )
    assert sliced.shape == (rows, n_iters, classes, n_parallel)

    for end in [0, n_iters]:
        rounds = end or n_rounds
        for strict in [False, True]:
            shape = (
                (0, rounds, classes, n_parallel)
                if strict
                else (0, rounds * classes * n_parallel)
            )
            assert (
                booster.predict(
                    empty, pred_leaf=True, strict_shape=strict, iteration_range=(0, end)
                ).shape
                == shape
            )

    # When there's only 1 tree, the output is a 1 dim vector
    booster = train({"tree_method": "hist"}, num_boost_round=1, dtrain=dtrain)
    booster.set_param({"device": device})
    assert _predict_leaf(booster, m, reference).shape == (rows,)

    assert booster.predict(empty, pred_leaf=True).shape == (0,)
    assert booster.predict(empty, pred_leaf=True, strict_shape=True).shape == (
        0,
        1,
        1,
        1,
    )

    # With no selected trees the forest dimension still exists; avoid inferring it
    # from the zero-length buffer or dividing by the number of boosting rounds.
    booster = train({"num_parallel_tree": n_parallel}, dtrain, num_boost_round=0)
    booster.set_param({"device": device})
    for data, n_rows in [(empty, 0), (m, rows)]:
        assert booster.predict(data, pred_leaf=True).shape == (n_rows, 0)
        assert booster.predict(data, pred_leaf=True, strict_shape=True).shape == (
            n_rows,
            0,
            1,
            n_parallel,
        )

    # The first two rounds have only vector-leaf trees. The full model contains
    # both vector- and scalar-leaf trees.
    mixed_rounds = 4
    booster = train(
        {
            "num_parallel_tree": classes,
            "num_class": classes,
            "tree_method": "hist",
            "multi_strategy": "multi_output_tree",
        },
        dtrain,
        num_boost_round=mixed_rounds,
        callbacks=[ResetStrategy()],
    )
    booster.set_param({"device": device})

    model = json.loads(booster.save_raw(raw_format="json"))
    gbtree_model = model["learner"]["gradient_booster"]["model"]
    trees = gbtree_model["trees"]
    iteration_indptr = gbtree_model["iteration_indptr"]
    leaf_sizes = [int(tree["tree_param"]["size_leaf_vector"]) for tree in trees]
    vector_end = 2
    assert set(leaf_sizes) == {1, classes}
    assert set(leaf_sizes[: iteration_indptr[vector_end]]) == {classes}

    for end, n_trees in [(vector_end, iteration_indptr[vector_end]), (0, len(trees))]:
        kwargs = {"iteration_range": (0, end)}
        predt = _predict_leaf(booster, m, reference, **kwargs)
        selected_trees = trees[:n_trees]
        assert predt.shape == (rows, len(selected_trees))
        assert booster.predict(empty, pred_leaf=True, **kwargs).shape == (
            0,
            len(selected_trees),
        )
        _validate_leaf_indices(predt, selected_trees)
        with pytest.raises(ValueError, match="vector leaf trees"):
            booster.predict(m, pred_leaf=True, strict_shape=True, **kwargs)
        with pytest.raises(ValueError, match="vector leaf trees"):
            booster.predict(empty, pred_leaf=True, strict_shape=True, **kwargs)

    return leaf


def run_base_margin_vs_base_score(device: Device) -> None:
    """Test for the relation between score and margin."""
    from sklearn.datasets import make_classification

    intercept = 0.5

    X, y = make_classification(random_state=2025)
    booster = train(
        {"base_score": intercept, "objective": "binary:logistic", "device": device},
        dtrain=DMatrix(X, y),
        num_boost_round=1,
    )
    np.testing.assert_allclose(get_basescore(booster), intercept)
    predt_0 = booster.predict(DMatrix(X, y))

    margin = np.full(y.shape, fill_value=logit(intercept), dtype=np.float32)
    Xy = DMatrix(X, y, base_margin=margin)
    # 0.2 is a dummy value
    booster = train(
        {"base_score": 0.2, "objective": "binary:logistic", "device": device},
        dtrain=Xy,
        num_boost_round=1,
    )
    np.testing.assert_allclose(get_basescore(booster), 0.2)
    predt_1 = booster.predict(Xy)

    np.testing.assert_allclose(predt_0, predt_1)
