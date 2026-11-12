"""Test for Tree-G xgboost pluging, including integration tests and end-to-end simple use cases tests.

First test suitecase verifies that TreeGBooster is correctly registered and accessible through
the standard ``xgboost`` Python API for training and prediction.

Dataset
-------
Circle graph of 100 nodes with a single binary feature (values in {0, 1}).
The label is identical to the feature, so a correctly working tree should
learn to reproduce the feature value after a handful of boosting rounds.


For the other tests, each test class constructs a small graph dataset whose labels depend on the
graph topology, trains a ``TreeGBooster`` model via the standard
``xgboost.train`` API, and asserts that training and validation metrics are
within the specified tolerance.

Dataset 1 — Neighbor-Mean Random Graph (Regression)
----------------------------------------------------
An Erdős–Rényi random graph ``G(n=100, p=0.15)`` (fixed seed for
reproducibility).  Every node has a single integer feature drawn uniformly
from ``[0, 10]``.  The label for node *i* is the **mean of its neighbors'
features**:

.. math::

    y_i = \\frac{1}{\\deg(i)} \\sum_{j \\in \\mathcal{N}(i)} x_j

This forces the model to learn one-hop neighborhood aggregation
(``walk_len=1``, source restriction), exercising the core Tree-G split
mechanism rather than a trivial ``d=0`` feature copy.

The first 70 nodes are used for training (weight = 1), the remaining 30 for
validation (weight = 0).

Successful-Learning Datasets (Binary Classification, single tree)
------------------------------------------------------------------
Each test in ``TestSuccessfulLearning`` constructs its own graph dataset,
trains a single-tree TreeGBooster (``num_boost_round=1``) with aggressive
hyper-parameters, and asserts 100 % accuracy on both training and
validation splits.  These serve as end-to-end correctness checks for the
full train → predict pipeline.

Dataset 2a — Majority-Vote 5-Regular Graph
    A 5-regular ring lattice (``n=100``, degree = 10).  Single binary
    feature; label = majority vote of neighbors' features.  Perfectly
    separable by one graph-aware split on the neighbor-sum.

Dataset 2b — Neighbor-Sum Threshold on Erdős–Rényi Graph
    An Erdős–Rényi random graph ``G(n=100, p=0.2)`` (fixed seed).  Single
    continuous feature in ``[0, 1]``; label = ``𝟙[(A · x)_i > 10.0]``.
    The tree should learn a root split with ``d=1`` and threshold ≈ 10.
    Accuracy > 90 % on both train and validation splits.

Dataset 2c — Two-Hop Sum Threshold on Erdős–Rényi Graph
    An Erdős–Rényi random graph ``G(n=100, p=0.2)`` (fixed seed).  Single
    continuous feature in ``[0, 1]``; label = ``𝟙[(A² · x)_i > 10.0]``.
    The tree must learn a split with ``d=2`` (two-hop neighborhood
    aggregation).  Accuracy > 90 % on both train and validation splits.

Dataset 2d — Blue-Neighbor Average on Erdős–Rényi Graph (Regression)
    An Erdős–Rényi random graph ``G(n=100, p=0.2)`` (fixed seed).
    Each node has two one-hot-encoded color features (``is_red``,
    ``is_blue``) and an integer ``value`` in ``{0, …, 10}``.  The label
    for node *i* is the mean ``value`` of its **blue** neighbors only.
    The tree should identify blue nodes via a ``d=0`` split on
    ``is_blue``, then aggregate their values with ``d=1``.  Train and
    validation RMSE < 0.9.  Tree structure is checked diagnostically
    (printed for investigation) because the greedy optimizer may prefer
    a different root split that yields better immediate gain.

Dataset 2e — Degree Threshold on Erdős–Rényi Graph (Classification)
    An Erdős–Rényi random graph ``G(n=100, p=0.1)`` (fixed seed).
    Single noise feature in ``[0, 1]``; label = ``𝟙[deg(i) > 10]``.
    With ``intercept=True``, the virtual constant-1 feature at index
    ``num_features`` lets the tree compute ``A · 𝟏 = degree`` via a
    ``d=1`` split.  Accuracy > 99 % on both train (70 nodes) and
    validation (30 nodes).  Verifies the root split uses the intercept
    feature (``f<num_features>``).

Dataset 3 — Graph-Aware Multi-Class Neighbor-Sum (3-class Classification)
-------------------------------------------------------------------------
An Erdős–Rényi random graph ``G(n=100, p=0.1)`` (fixed seed) with
three features: one-hot color (``is_red``, ``is_blue``; 50-50) and a
continuous ``value`` in ``[0, 1]``.  Three-class labels:

- **0** ("Red"): all red nodes.
- **1** ("Blue-High"): blue nodes with neighbor-sum of value > 5.
- **2** ("Blue-Low"): blue nodes with neighbor-sum of value ≤ 5.

Requires both a feature-level split (``d=0`` on color) and a graph-aware
split (``d=1`` on ``A · value``) within a one-vs-all multi-class pipeline.
Single boosting round (one tree per class) achieves 100 % accuracy on
train (70) and validation (30) splits.  Tests both ``multi:softmax`` and
``multi:softprob`` objectives.
"""

import os
import re
import tempfile
import time

import numpy as np
import pytest

import xgboost as xgb

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _build_circle_graph_dataset(
    n: int = 100,
    n_train: int = 70,
    max_walk_len: int = 2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a circle graph dataset with a single binary feature.

    Constructs an undirected circle graph ``0–1–2–…–(n-1)–0`` and prepares
    the feature matrix in the format expected by the Tree-G plugin:
    ``[X | A | A^2 | … | A^d]``.

    The first ``n_train`` nodes receive weight 1 (training); the remaining
    ``n - n_train`` nodes receive weight 0 (validation / test).  All nodes
    keep their true labels so the graph structure is fully visible.

    Args:
        n: Number of nodes in the circle graph.
        n_train: Number of training nodes (first ``n_train`` nodes).
        max_walk_len: Maximum walk length *d*.  Powers ``A, A^2, …, A^d``
            are appended to the feature matrix.
        seed: Random seed for reproducibility.

    Returns:
        A tuple ``(data, labels, weights)`` where:

        - ``data`` has shape ``(n, k + n * max_walk_len)`` with ``k = 1``.
        - ``labels`` has shape ``(n,)`` with values in ``{0, 1}``.
        - ``weights`` has shape ``(n,)`` — 1.0 for train nodes, 0.0 for val.
    """
    rng = np.random.RandomState(seed)

    # Single binary feature drawn uniformly from {0, 1}
    feature = rng.randint(0, 2, size=n).astype(np.float32)
    X = feature.reshape(n, 1)  # (n, 1)
    labels = feature.copy()

    # Circle adjacency matrix (symmetric)
    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        A[i, (i + 1) % n] = 1.0
        A[(i + 1) % n, i] = 1.0

    # Assemble [X | A | A^2 | … | A^d]
    blocks: list[np.ndarray] = [X]
    A_power = A.copy()
    for _ in range(max_walk_len):
        blocks.append(A_power.copy())
        A_power = A_power @ A

    data = np.hstack(blocks)  # (n, 1 + n * max_walk_len)

    # Train/val split via weights
    weights = np.zeros(n, dtype=np.float32)
    weights[:n_train] = 1.0
    return data, labels, weights


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTreeGPluginAPI:
    """Verify the Tree-G plugin is accessible via the XGBoost Python API."""

    N_NODES: int = 100
    N_TRAIN: int = 70
    NUM_FEATURES: int = 1
    MAX_WALK_LEN: int = 2

    # -- Fixtures -----------------------------------------------------------

    @pytest.fixture(autouse=True)
    def setup_dataset(self) -> None:
        """Prepare the circle-graph DMatrix before each test.

        The first 70 nodes are training (weight=1), the remaining 30 are
        validation (weight=0).  Both splits keep true labels.
        """
        data, labels, weights = _build_circle_graph_dataset(
            n=self.N_NODES,
            n_train=self.N_TRAIN,
            max_walk_len=self.MAX_WALK_LEN,
        )
        self.dtrain: xgb.DMatrix = xgb.DMatrix(
            data,
            label=labels,
            weight=weights,
        )
        self.labels: np.ndarray = labels
        self.train_idx: np.ndarray = np.arange(self.N_TRAIN)
        self.val_idx: np.ndarray = np.arange(self.N_TRAIN, self.N_NODES)

    # -- Helpers ------------------------------------------------------------

    def _base_params(self) -> dict:
        """Return baseline Tree-G training parameters.

        Returns:
            Dictionary of XGBoost parameters with ``booster='treeg'`` and
            graph-specific settings matching the circle-graph dataset.
        """
        return {
            "booster": "treeg",
            "objective": "reg:squarederror",
            "base_score": 0.5,
            # Tree-G–specific parameters
            "num_features": self.NUM_FEATURES,
            "num_vertices": self.N_NODES,
            "max_walk_len": self.MAX_WALK_LEN,
            "max_ancestor_distance": 2,
            # Tree growth
            "max_leaves": 8,
            "learning_rate": 0.3,
            "min_child_weight": 1.0,
            "reg_lambda": 1.0,
            "verbosity": 0,
        }

    # -- Test cases ---------------------------------------------------------

    def test_train_does_not_crash(self) -> None:
        """Training with ``booster='treeg'`` should complete without error."""
        params = self._base_params()
        model = xgb.train(params, self.dtrain, num_boost_round=1)
        assert model is not None

    def test_predict_shape(self) -> None:
        """Prediction output must have shape ``(n_vertices,)``."""
        params = self._base_params()
        model = xgb.train(params, self.dtrain, num_boost_round=1)
        preds = model.predict(self.dtrain)
        assert preds.shape == (self.N_NODES,), (
            f"Expected shape ({self.N_NODES},), got {preds.shape}"
        )

    def test_train_and_predict_circle_graph(self) -> None:
        """Model should learn label ≡ feature on the circle graph.

        After 20 boosting rounds with ``reg:squarederror``, predictions
        should be within 0.3 of the true {0, 1} labels.
        """
        params = self._base_params()
        model = xgb.train(params, self.dtrain, num_boost_round=20)
        preds = model.predict(self.dtrain)

        assert preds.shape == (self.N_NODES,)
        np.testing.assert_allclose(
            preds,
            self.labels,
            atol=0.3,
            err_msg="Predictions too far from labels after 20 rounds",
        )

    def test_multiple_rounds_reduce_error(self) -> None:
        """Additional boosting rounds must not increase training MSE."""
        params = self._base_params()

        model_few = xgb.train(params, self.dtrain, num_boost_round=1)
        preds_few = model_few.predict(self.dtrain)
        mse_few = float(np.mean((preds_few - self.labels) ** 2))

        model_many = xgb.train(params, self.dtrain, num_boost_round=20)
        preds_many = model_many.predict(self.dtrain)
        mse_many = float(np.mean((preds_many - self.labels) ** 2))

        assert mse_many <= mse_few + 1e-6, (
            f"More rounds should not increase error: "
            f"MSE(1 round)={mse_few:.6f}, MSE(20 rounds)={mse_many:.6f}"
        )

    def test_save_load_roundtrip(self) -> None:
        """Saved and reloaded model must produce identical predictions.

        Crucially, the load path uses a bare ``xgb.Booster()`` (no params)
        followed by ``load_model()`` — the standard XGBoost pattern.
        Graph-topology parameters (``num_features``, ``num_vertices``, …)
        must be recovered from the serialised model, not from user args.
        """
        params = self._base_params()
        model = xgb.train(params, self.dtrain, num_boost_round=5)
        preds_before = model.predict(self.dtrain)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            model.save_model(tmp_path)

            loaded = xgb.Booster()
            loaded.load_model(tmp_path)
            preds_after = loaded.predict(self.dtrain)

            np.testing.assert_allclose(
                preds_after,
                preds_before,
                atol=1e-6,
                err_msg="Predictions differ after save/load roundtrip",
            )
        finally:
            os.unlink(tmp_path)

    def test_boosted_rounds(self) -> None:
        """``model.num_boosted_rounds()`` should match ``num_boost_round``."""
        params = self._base_params()
        num_rounds = 7
        model = xgb.train(params, self.dtrain, num_boost_round=num_rounds)

        # xgb.Booster exposes the count via attr or JSON config
        assert model.num_boosted_rounds() == num_rounds

    def test_dataset_dimensions(self) -> None:
        """DMatrix dimensions must match the expected Tree-G input format.

        Expected columns: ``k + n * d`` where ``k=1``, ``n=100``, ``d=2``
        → ``1 + 100 * 2 = 201``.
        """
        expected_cols = self.NUM_FEATURES + self.N_NODES * self.MAX_WALK_LEN
        assert self.dtrain.num_row() == self.N_NODES
        assert self.dtrain.num_col() == expected_cols

    def test_validation_rmse_below_epsilon(self) -> None:
        """Validation RMSE must be below ε = 0.001 after sufficient rounds.

        The 30 held-out nodes (weight=0) never contribute gradients, yet the
        tree should still learn ``label ≡ feature`` via a direct ``d=0``
        split on the single feature.  This tests generalisation to unseen
        (zero-weight) nodes within the same graph.
        """
        epsilon = 0.001
        params = self._base_params()
        model = xgb.train(params, self.dtrain, num_boost_round=20)
        preds = model.predict(self.dtrain)

        val_preds = preds[self.val_idx]
        val_labels = self.labels[self.val_idx]
        rmse = float(np.sqrt(np.mean((val_preds - val_labels) ** 2)))

        assert rmse < epsilon, f"Validation RMSE {rmse:.6f} exceeds epsilon {epsilon}"


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _build_neighbor_mean_dataset(
    n: int = 100,
    n_train: int = 70,
    edge_prob: float = 0.15,
    max_walk_len: int = 2,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a random-graph dataset where labels are neighbor-feature means.

    Args:
        n: Number of nodes.
        n_train: Number of training nodes (first ``n_train`` get weight 1).
        edge_prob: Edge probability for the Erdős–Rényi graph.
        max_walk_len: Maximum walk length *d*.  Powers ``A, A², …, A^d``
            are appended to the feature matrix.
        seed: Random seed for full reproducibility.

    Returns:
        A tuple ``(data, labels, weights)`` where:

        - ``data`` has shape ``(n, k + n * d)`` with ``k = 1``.
        - ``labels`` has shape ``(n,)``.
        - ``weights`` has shape ``(n,)`` — 1.0 for train, 0.0 for val.
    """
    rng = np.random.RandomState(seed)

    # --- Feature: single integer in [0, 10] ---
    feature = rng.randint(0, 11, size=n).astype(np.float32)
    X = feature.reshape(n, 1)

    # --- Erdős–Rényi adjacency matrix (symmetric, no self-loops) ---
    # Draw upper triangle, then mirror.
    upper = (rng.rand(n, n) < edge_prob).astype(np.float32)
    np.fill_diagonal(upper, 0.0)
    A = np.triu(upper, k=1)
    A = A + A.T  # symmetric

    # Ensure every node has at least one neighbor (add a ring as fallback).
    for i in range(n):
        if A[i].sum() == 0:
            j = (i + 1) % n
            A[i, j] = 1.0
            A[j, i] = 1.0

    # --- Label: mean of neighbors' features ---
    degree = A.sum(axis=1)  # (n,)
    # For safety, avoid division by zero (should not happen after ring fix).
    degree = np.maximum(degree, 1.0)
    labels = (A @ feature) / degree  # (n,)

    # --- Assemble [X | A | A^2 | … | A^d] ---
    blocks: list[np.ndarray] = [X]
    A_power = A.copy()
    for _ in range(max_walk_len):
        blocks.append(A_power.copy())
        A_power = A_power @ A
    data = np.hstack(blocks)

    # --- Train / val split via weights ---
    weights = np.zeros(n, dtype=np.float32)
    weights[:n_train] = 1.0

    return data, labels, weights


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNeighborMeanRandomGraph:
    """Train TreeGBooster on a neighbor-mean regression task.

    The label requires one-hop aggregation, so the model must learn to use
    the adjacency matrix (``d=1``, source restriction) rather than the raw
    feature alone.
    """

    N_NODES: int = 100
    N_TRAIN: int = 70
    NUM_FEATURES: int = 1
    MAX_WALK_LEN: int = 2

    # -- Fixtures -----------------------------------------------------------

    @pytest.fixture(autouse=True)
    def setup_dataset(self) -> None:
        """Prepare the neighbor-mean DMatrix before each test."""
        data, labels, weights = _build_neighbor_mean_dataset(
            n=self.N_NODES,
            n_train=self.N_TRAIN,
            max_walk_len=self.MAX_WALK_LEN,
        )
        self.dtrain: xgb.DMatrix = xgb.DMatrix(
            data,
            label=labels,
            weight=weights,
        )
        self.labels: np.ndarray = labels
        self.train_idx: np.ndarray = np.arange(self.N_TRAIN)
        self.val_idx: np.ndarray = np.arange(self.N_TRAIN, self.N_NODES)

    # -- Helpers ------------------------------------------------------------

    def _base_params(self) -> dict:
        """Return baseline Tree-G training parameters.

        Returns:
            Dictionary of XGBoost parameters suited for the neighbor-mean
            regression task on the random graph.
        """
        return {
            "booster": "treeg",
            "objective": "reg:squarederror",
            "base_score": 0.0,
            # Tree-G–specific parameters
            "num_features": self.NUM_FEATURES,
            "num_vertices": self.N_NODES,
            "max_walk_len": self.MAX_WALK_LEN,
            "max_ancestor_distance": 2,
            # Tree growth — generous budget for an easy dataset
            "max_leaves": 32,
            "learning_rate": 0.3,
            "min_child_weight": 1.0,
            "reg_lambda": 0.1,
            "verbosity": 0,
        }

    @staticmethod
    def _rmse(preds: np.ndarray, labels: np.ndarray) -> float:
        """Compute root-mean-square error.

        Args:
            preds: Predicted values.
            labels: Ground-truth labels.

        Returns:
            RMSE as a Python float.
        """
        return float(np.sqrt(np.mean((preds - labels) ** 2)))

    # -- Test cases ---------------------------------------------------------

    def test_train_and_val_rmse_below_epsilon(self) -> None:
        """Both train and val RMSE must fall below ε = 0.9.

        The neighbor-mean label is a deterministic function of the one-hop
        adjacency and node features — both fully visible in the DMatrix.
        The 30 held-out nodes (weight = 0) never contribute gradients, yet
        the tree's graph-aware splits propagate neighbor information so
        predictions should generalise to these nodes as well.

        Reports both values on failure for easy debugging.
        """
        epsilon = 0.9
        params = self._base_params()
        model = xgb.train(params, self.dtrain, num_boost_round=10)
        preds = model.predict(self.dtrain)

        train_rmse = self._rmse(preds[self.train_idx], self.labels[self.train_idx])
        val_rmse = self._rmse(preds[self.val_idx], self.labels[self.val_idx])

        assert train_rmse < epsilon and val_rmse < epsilon, (
            f"RMSE too high — train: {train_rmse:.6f}, val: {val_rmse:.6f} "
            f"(epsilon={epsilon})"
        )

    def test_more_rounds_reduce_train_error(self) -> None:
        """Additional boosting rounds must not increase training RMSE."""
        params = self._base_params()

        model_few = xgb.train(params, self.dtrain, num_boost_round=1)
        preds_few = model_few.predict(self.dtrain)
        rmse_few = self._rmse(preds_few[self.train_idx], self.labels[self.train_idx])

        model_many = xgb.train(params, self.dtrain, num_boost_round=5)
        preds_many = model_many.predict(self.dtrain)
        rmse_many = self._rmse(preds_many[self.train_idx], self.labels[self.train_idx])

        assert rmse_many <= rmse_few + 1e-6, (
            f"More rounds should not increase training error: "
            f"RMSE(1 round)={rmse_few:.6f}, RMSE(5 rounds)={rmse_many:.6f}"
        )


# ---------------------------------------------------------------------------
# Dataset helper — Majority-Vote 5-Regular Graph (Classification)
# ---------------------------------------------------------------------------


def _build_majority_vote_dataset(
    n: int = 100,
    n_train: int = 70,
    k_neighbors: int = 5,
    max_walk_len: int = 2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a 5-regular-graph dataset where labels are neighbor majority votes.

    Constructs a ring lattice where each node is connected to its
    ``k_neighbors`` nearest neighbors on *each* side of the ring, yielding
    a constant degree of ``2 * k_neighbors`` for every node.  Every node
    gets a single binary feature drawn from ``{0, 1}``.  The label for
    node *i* is 1 when strictly more than half of its neighbors have
    feature 1, and 0 otherwise (ties broken toward 0).

    Because the degree is constant, the majority-vote label is perfectly
    separable by a single threshold on the neighbor-sum, making this a
    clean test for the Tree-G graph-aware split mechanism.

    Args:
        n: Number of nodes.
        n_train: Number of training nodes (first ``n_train`` get weight 1).
        k_neighbors: Number of nearest neighbors on each side of the ring.
            Every node will have degree ``2 * k_neighbors``.
        max_walk_len: Maximum walk length *d*.  Powers ``A, A², …, A^d``
            are appended to the feature matrix.
        seed: Random seed for full reproducibility.

    Returns:
        A tuple ``(data, labels, weights)`` where:

        - ``data`` has shape ``(n, k + n * d)`` with ``k = 1``.
        - ``labels`` has shape ``(n,)`` with values in ``{0, 1}``.
        - ``weights`` has shape ``(n,)`` — 1.0 for train, 0.0 for val.
    """
    rng = np.random.RandomState(seed)

    # --- Feature: single binary value in {0, 1} (numeric / one-hot) ---
    feature = rng.randint(0, 2, size=n).astype(np.float32)
    X = feature.reshape(n, 1)

    # --- 5-regular ring lattice (symmetric, no self-loops) ---
    # Each node i is connected to i±1, i±2, …, i±k_neighbors (mod n).
    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for offset in range(1, k_neighbors + 1):
            j = (i + offset) % n
            A[i, j] = 1.0
            A[j, i] = 1.0

    # --- Label: majority vote of neighbors' features ---
    neighbor_sum = A @ feature  # sum of neighbors' feature values
    degree = A.sum(axis=1)  # constant = 2 * k_neighbors for every node
    # Strict majority: label = 1 iff more than half of neighbors have feat=1.
    labels = (neighbor_sum > degree / 2.0).astype(np.float32)

    # --- Assemble [X | A | A^2 | … | A^d] ---
    blocks: list[np.ndarray] = [X]
    A_power = A.copy()
    for _ in range(max_walk_len):
        blocks.append(A_power.copy())
        A_power = A_power @ A
    data = np.hstack(blocks)

    # --- Train / val split via weights ---
    weights = np.zeros(n, dtype=np.float32)
    weights[:n_train] = 1.0

    return data, labels, weights


def _build_neighbor_sum_threshold_dataset(
    n: int = 100,
    n_train: int = 70,
    edge_prob: float = 0.2,
    threshold: float = 10.0,
    max_walk_len: int = 2,
    seed: int = 17,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Build an Erdős–Rényi graph dataset with neighbor-sum threshold labels.

    Constructs a random graph ``G(n, p)`` and assigns each node a single
    continuous feature drawn uniformly from ``[0, 1]``.  The binary label
    for node *i* is:

    .. math::

        y_i = \mathbf{1}\!\left[
            \sum_{j \in \mathcal{N}(i)} x_j > \theta
        \right]

    i.e. 1 when the sum of the node's neighbors' features exceeds
    ``threshold``, 0 otherwise.  This is equivalent to
    ``(A @ feature_vector) > threshold``.

    The task is perfectly learnable by a single graph-aware split on the
    one-hop neighbor-sum (``d=1``, source restriction, feature index 0).

    Args:
        n: Number of nodes.
        n_train: Number of training nodes (first ``n_train`` get weight 1).
        edge_prob: Edge probability for the Erdős–Rényi graph.
        threshold: Threshold on the neighbor-sum for label assignment.
        max_walk_len: Maximum walk length *d*.  Powers ``A, A², …, A^d``
            are appended to the feature matrix.
        seed: Random seed for full reproducibility.

    Returns:
        A tuple ``(data, labels, weights)`` where:

        - ``data`` has shape ``(n, k + n * d)`` with ``k = 1``.
        - ``labels`` has shape ``(n,)`` with values in ``{0, 1}``.
        - ``weights`` has shape ``(n,)`` — 1.0 for train, 0.0 for val.
    """
    rng = np.random.RandomState(seed)

    # --- Feature: single continuous value in [0, 1] ---
    feature = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
    X = feature.reshape(n, 1)

    # --- Erdős–Rényi adjacency matrix (symmetric, no self-loops) ---
    upper = (rng.rand(n, n) < edge_prob).astype(np.float32)
    np.fill_diagonal(upper, 0.0)
    A = np.triu(upper, k=1)
    A = A + A.T  # symmetric

    # --- Label: neighbor-sum > threshold ---
    neighbor_sum = A @ feature  # (n,)
    labels = (neighbor_sum > threshold).astype(np.float32)

    # --- Assemble [X | A | A^2 | … | A^d] ---
    blocks: list[np.ndarray] = [X]
    A_power = A.copy()
    for _ in range(max_walk_len):
        blocks.append(A_power.copy())
        A_power = A_power @ A
    data = np.hstack(blocks)

    # --- Train / val split via weights ---
    weights = np.zeros(n, dtype=np.float32)
    weights[:n_train] = 1.0

    return data, labels, weights


def _build_two_hop_sum_threshold_dataset(
    n: int = 100,
    n_train: int = 70,
    edge_prob: float = 0.2,
    threshold: float = 10.0,
    max_walk_len: int = 2,
    seed: int = 31,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Build an Erdős–Rényi graph dataset with two-hop sum threshold labels.

    Constructs a random graph ``G(n, p)`` and assigns each node a single
    continuous feature drawn uniformly from ``[0, 1]``.  The binary label
    for node *i* is:

    .. math::

        y_i = \mathbf{1}\!\left[
            (A^2 \cdot x)_i > \theta
        \right]

    i.e. 1 when the two-hop neighborhood weighted sum of features exceeds
    ``threshold``, 0 otherwise.  This requires the model to learn a split
    with walk length ``d=2``.

    Args:
        n: Number of nodes.
        n_train: Number of training nodes (first ``n_train`` get weight 1).
        edge_prob: Edge probability for the Erdős–Rényi graph.
        threshold: Threshold on the two-hop sum for label assignment.
        max_walk_len: Maximum walk length *d*.  Powers ``A, A², …, A^d``
            are appended to the feature matrix.
        seed: Random seed for full reproducibility.

    Returns:
        A tuple ``(data, labels, weights)`` where:

        - ``data`` has shape ``(n, k + n * d)`` with ``k = 1``.
        - ``labels`` has shape ``(n,)`` with values in ``{0, 1}``.
        - ``weights`` has shape ``(n,)`` — 1.0 for train, 0.0 for val.
    """
    rng = np.random.RandomState(seed)

    # --- Feature: single continuous value in [0, 1] ---
    feature = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
    X = feature.reshape(n, 1)

    # --- Erdős–Rényi adjacency matrix (symmetric, no self-loops) ---
    upper = (rng.rand(n, n) < edge_prob).astype(np.float32)
    np.fill_diagonal(upper, 0.0)
    A = np.triu(upper, k=1)
    A = A + A.T  # symmetric

    # --- Label: (A^2 @ feature) > threshold ---
    A2 = A @ A
    two_hop_sum = A2 @ feature  # (n,)
    labels = (two_hop_sum > threshold).astype(np.float32)

    # --- Assemble [X | A | A^2 | … | A^d] ---
    blocks: list[np.ndarray] = [X]
    A_power = A.copy()
    for _ in range(max_walk_len):
        blocks.append(A_power.copy())
        A_power = A_power @ A
    data = np.hstack(blocks)

    # --- Train / val split via weights ---
    weights = np.zeros(n, dtype=np.float32)
    weights[:n_train] = 1.0

    return data, labels, weights


def _build_blue_neighbor_avg_dataset(
    n: int = 100,
    n_train: int = 70,
    edge_prob: float = 0.2,
    max_walk_len: int = 1,
    seed: int = 53,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an Erdős–Rényi graph dataset where labels are the mean value of
    blue neighbors.

    Each node receives three features (after one-hot encoding of color):

    - ``is_red``  (index 0): 1 if the node is red, 0 otherwise.
    - ``is_blue`` (index 1): 1 if the node is blue, 0 otherwise.
    - ``value``   (index 2): integer drawn uniformly from ``{0, …, 10}``.

    The label for node *i* is:

    .. math::

        y_i = \\frac{\\sum_{j \\in \\mathcal{N}(i),\\, \\text{blue}_j = 1}
              \\text{value}_j}
              {\\left| \\{ j \\in \\mathcal{N}(i) : \\text{blue}_j = 1 \\}
              \\right|}

    If node *i* has no blue neighbors the label is 0.

    The expected tree structure for learning this task is:

    1. **Root split** on ``is_blue`` (feature 1, ``d=0``) with threshold
       ≈ 0.5 to identify blue nodes as the active set.
    2. **Child split** pointing to the root's blue active set, on
       ``value`` (feature 2, ``d=1``, source restriction) to aggregate
       the blue neighbors' values.

    Args:
        n: Number of nodes.
        n_train: Number of training nodes (first ``n_train`` get weight 1).
        edge_prob: Edge probability for the Erdős–Rényi graph.
        max_walk_len: Maximum walk length *d*.
        seed: Random seed for full reproducibility.

    Returns:
        A tuple ``(data, labels, weights)`` where:

        - ``data`` has shape ``(n, 3 + n * d)``.
        - ``labels`` has shape ``(n,)``.
        - ``weights`` has shape ``(n,)`` — 1.0 for train, 0.0 for val.
    """
    rng = np.random.RandomState(seed)

    # --- Node features ---
    is_blue = rng.randint(0, 2, size=n).astype(np.float32)
    is_red = 1.0 - is_blue
    value = rng.randint(0, 11, size=n).astype(np.float32)
    X = np.column_stack([is_red, is_blue, value])  # (n, 3)

    # --- Erdős–Rényi adjacency (symmetric, no self-loops) ---
    upper = (rng.rand(n, n) < edge_prob).astype(np.float32)
    np.fill_diagonal(upper, 0.0)
    A = np.triu(upper, k=1)
    A = A + A.T

    # Guarantee every node has at least one neighbor.
    for i in range(n):
        if A[i].sum() == 0:
            j = (i + 1) % n
            A[i, j] = 1.0
            A[j, i] = 1.0

    # --- Labels: mean value of blue neighbors ---
    blue_value = is_blue * value
    blue_neighbor_sum = A @ blue_value  # sum of blue neighbors' values
    blue_neighbor_count = A @ is_blue  # count of blue neighbors
    labels = np.where(
        blue_neighbor_count > 0,
        blue_neighbor_sum / blue_neighbor_count,
        0.0,
    ).astype(np.float32)

    # --- Assemble [X | A | A^2 | … | A^d] ---
    blocks: list[np.ndarray] = [X]
    A_power = A.copy()
    for _ in range(max_walk_len):
        blocks.append(A_power.copy())
        A_power = A_power @ A
    data = np.hstack(blocks)

    # --- Train / val split via weights ---
    weights = np.zeros(n, dtype=np.float32)
    weights[:n_train] = 1.0

    return data, labels, weights


def _build_degree_threshold_dataset(
    n: int = 100,
    n_train: int = 70,
    edge_prob: float = 0.1,
    degree_threshold: int = 10,
    max_walk_len: int = 2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an Erdős–Rényi graph dataset with degree-threshold labels.

    Constructs a random graph ``G(n, p)`` and assigns each node a single
    continuous feature drawn uniformly from ``[0, 1]`` (**noise** — not
    used by the labelling function).  The binary label for node *i* is:

    .. math::

        y_i = \\mathbf{1}\\!\\left[ \\deg(i) > \\theta \\right]

    where ``deg(i) = (A · 𝟏)_i`` is the node degree (i.e. the row-sum
    of the adjacency matrix, equivalently ``A @ np.ones(n)``).

    The task is perfectly learnable by a single graph-aware split on the
    **intercept** feature (virtual constant-1 at index ``num_features``)
    with walk length ``d=1``.  In that configuration the split value for
    each node equals its degree, and a threshold of ``degree_threshold``
    separates the two classes.

    Edge probability ``p`` is chosen so that the expected degree
    ``(n-1) * p`` is close to ``degree_threshold``, yielding a non-trivial
    mix of positive and negative labels.

    Args:
        n: Number of nodes.
        n_train: Number of training nodes (first ``n_train`` get weight 1).
        edge_prob: Edge probability for the Erdős–Rényi graph.
        degree_threshold: Degree threshold for label assignment.
        max_walk_len: Maximum walk length *d*.  Powers ``A, A², …, A^d``
            are appended to the feature matrix.
        seed: Random seed for full reproducibility.

    Returns:
        A tuple ``(data, labels, weights)`` where:

        - ``data`` has shape ``(n, k + n * d)`` with ``k = 1``.
        - ``labels`` has shape ``(n,)`` with values in ``{0, 1}``.
        - ``weights`` has shape ``(n,)`` — 1.0 for train, 0.0 for val.
    """
    rng = np.random.RandomState(seed)

    # --- Feature: single continuous value in [0, 1] (noise; irrelevant to label) ---
    feature = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
    X = feature.reshape(n, 1)

    # --- Erdős–Rényi adjacency matrix (symmetric, no self-loops) ---
    upper = (rng.rand(n, n) < edge_prob).astype(np.float32)
    np.fill_diagonal(upper, 0.0)
    A = np.triu(upper, k=1)
    A = A + A.T  # symmetric

    # --- Label: degree > threshold ---
    # degree = A @ ones = row-sums of A
    degree = A.sum(axis=1)  # (n,)
    labels = (degree > degree_threshold).astype(np.float32)

    # --- Assemble [X | A | A^2 | … | A^d] ---
    blocks: list[np.ndarray] = [X]
    A_power = A.copy()
    for _ in range(max_walk_len):
        blocks.append(A_power.copy())
        A_power = A_power @ A
    data = np.hstack(blocks)

    # --- Train / val split via weights ---
    weights = np.zeros(n, dtype=np.float32)
    weights[:n_train] = 1.0

    return data, labels, weights


# ---------------------------------------------------------------------------
# Tests — Successful Learning (single-tree, 100 % accuracy)
# ---------------------------------------------------------------------------


class TestSuccessfulLearning:
    """Train TreeGBooster on simple graph datasets and verify perfect accuracy.

    Each test method constructs a specific graph dataset whose labels are
    a deterministic function of the graph topology and node features, trains
    a single-tree TreeGBooster (``num_boost_round=1``), and asserts that
    both training and validation accuracy reach 100 %.

    Because these datasets are designed to be perfectly learnable by a
    single graph-aware tree, achieving 100 % accuracy validates the
    correctness of the full train → predict pipeline.

    Trained with ``binary:logistic``; accuracy is evaluated by thresholding
    predicted probabilities at 0.5.
    """

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
        """Compute classification accuracy from predicted probabilities.

        Args:
            probs: Predicted probabilities from ``binary:logistic``.
            labels: Ground-truth binary labels in ``{0, 1}``.

        Returns:
            Accuracy as a Python float in ``[0, 1]``.
        """
        preds_class = (probs >= 0.5).astype(np.float32)
        return float(np.mean(preds_class == labels))

    @staticmethod
    def _train_and_predict(
        data: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        *,
        num_features: int,
        num_vertices: int,
        max_walk_len: int = 2,
        max_ancestor_distance: int = 2,
        max_leaves: int = 16,
        intercept: bool = False,
    ) -> tuple[xgb.Booster, np.ndarray]:
        """Build a DMatrix, train a single-tree model, and return predictions.

        Uses aggressive hyper-parameters (``learning_rate=1.0``,
        ``reg_lambda=0.01``) so that a single boosting round can drive
        leaf values far enough for correct classification.

        Args:
            data: Feature matrix of shape ``(n, k + n * d)``.
            labels: Ground-truth labels of shape ``(n,)``.
            weights: Per-node weights of shape ``(n,)`` (1 = train, 0 = val).
            num_features: Number of original node features (*k*).
            num_vertices: Number of nodes in the graph (*n*).
            max_walk_len: Maximum walk length *d*.
            max_ancestor_distance: Maximum ancestor look-back for the
                pointer mechanism.
            max_leaves: Maximum number of leaves in the tree.
            intercept: Whether to enable the virtual constant-1 intercept
                feature at index ``num_features``.

        Returns:
            A tuple ``(model, preds)`` where ``model`` is the trained
            ``xgb.Booster`` and ``preds`` is an array of predicted
            probabilities of shape ``(n,)``.
        """
        dtrain = xgb.DMatrix(data, label=labels, weight=weights)
        params = {
            "booster": "treeg",
            "objective": "binary:logistic",
            "base_score": 0.5,
            # Tree-G–specific parameters
            "num_features": num_features,
            "num_vertices": num_vertices,
            "max_walk_len": max_walk_len,
            "max_ancestor_distance": max_ancestor_distance,
            "intercept": intercept,
            # Aggressive single-round hyper-parameters
            "max_leaves": max_leaves,
            "learning_rate": 1.0,
            "min_child_weight": 1.0,
            "reg_lambda": 0.01,
            "verbosity": 0,
        }
        model = xgb.train(params, dtrain, num_boost_round=1)
        return model, model.predict(dtrain)

    def _assert_perfect_accuracy(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        n_train: int,
        *,
        dataset_name: str,
    ) -> None:
        """Assert 100 % accuracy on both the train and validation splits.

        Args:
            probs: Predicted probabilities of shape ``(n,)``.
            labels: Ground-truth labels of shape ``(n,)``.
            n_train: Number of training nodes (first ``n_train`` indices).
            dataset_name: Human-readable dataset name for error messages.
        """
        n_nodes = len(labels)
        train_idx = np.arange(n_train)
        val_idx = np.arange(n_train, n_nodes)

        train_acc = self._accuracy(probs[train_idx], labels[train_idx])
        val_acc = self._accuracy(probs[val_idx], labels[val_idx])

        assert train_acc == 1.0, (
            f"[{dataset_name}] Train accuracy must be 100 %, got {train_acc:.4f}"
        )
        assert val_acc == 1.0, (
            f"[{dataset_name}] Val accuracy must be 100 %, got {val_acc:.4f}"
        )

    @staticmethod
    def _rmse(preds: np.ndarray, labels: np.ndarray) -> float:
        """Compute root-mean-square error.

        Args:
            preds: Predicted values.
            labels: Ground-truth labels.

        Returns:
            RMSE as a Python float.
        """
        return float(np.sqrt(np.mean((preds - labels) ** 2)))

    @staticmethod
    def _train_and_predict_regression(
        data: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        *,
        num_features: int,
        num_vertices: int,
        max_walk_len: int = 1,
        max_ancestor_distance: int = 2,
        max_leaves: int = 32,
    ) -> tuple[xgb.Booster, np.ndarray]:
        """Build a DMatrix, train a single-tree regression model, and predict.

        Uses aggressive hyper-parameters (``learning_rate=1.0``,
        ``reg_lambda=0.01``) so that a single boosting round drives leaf
        values to accurate predictions.

        Args:
            data: Feature matrix of shape ``(n, k + n * d)``.
            labels: Ground-truth labels of shape ``(n,)``.
            weights: Per-node weights of shape ``(n,)`` (1 = train, 0 = val).
            num_features: Number of original node features (*k*).
            num_vertices: Number of nodes in the graph (*n*).
            max_walk_len: Maximum walk length *d*.
            max_ancestor_distance: Maximum ancestor look-back for the
                pointer mechanism.
            max_leaves: Maximum number of leaves in the tree.

        Returns:
            A tuple ``(model, preds)`` where ``model`` is the trained
            ``xgb.Booster`` and ``preds`` is an array of predicted values
            of shape ``(n,)``.
        """
        dtrain = xgb.DMatrix(data, label=labels, weight=weights)
        params = {
            "booster": "treeg",
            "objective": "reg:squarederror",
            "base_score": 0.0,
            # Tree-G–specific parameters
            "num_features": num_features,
            "num_vertices": num_vertices,
            "max_walk_len": max_walk_len,
            "max_ancestor_distance": max_ancestor_distance,
            # Aggressive single-round hyper-parameters
            "max_leaves": max_leaves,
            "learning_rate": 1.0,
            "min_child_weight": 1.0,
            "reg_lambda": 0.01,
            "verbosity": 0,
        }
        model = xgb.train(params, dtrain, num_boost_round=1)
        return model, model.predict(dtrain)

    # -- Test cases ---------------------------------------------------------

    def test_majority_vote_regular_graph(self) -> None:
        """100 % accuracy on majority-vote 5-regular graph with a single tree.

        Uses a 5-regular ring lattice (degree = 10 for every node).  Each
        node has a single binary feature in ``{0, 1}``.  The label is the
        majority vote of the neighbors' features: 1 if strictly more than
        half have feature 1, else 0.

        Because the degree is constant, the neighbor-sum threshold is the
        same for all nodes, so the task is perfectly separable by a single
        graph-aware split — one boosting round should suffice.
        """
        n_nodes = 100
        n_train = 70
        num_features = 1
        max_walk_len = 2

        data, labels, weights = _build_majority_vote_dataset(
            n=n_nodes,
            n_train=n_train,
            max_walk_len=max_walk_len,
        )

        _, probs = self._train_and_predict(
            data,
            labels,
            weights,
            num_features=num_features,
            num_vertices=n_nodes,
            max_walk_len=max_walk_len,
        )

        self._assert_perfect_accuracy(
            probs, labels, n_train, dataset_name="majority-vote-5-regular"
        )

    def test_neighbor_sum_threshold_erdos_renyi(self) -> None:
        """Accuracy > 90 % on neighbor-sum threshold Erdős–Rényi graph.

        Uses an Erdős–Rényi random graph ``G(100, 0.2)`` with a single
        continuous feature in ``[0, 1]``.  The label for node *i* is
        ``𝟙[(A · x)_i > 10.0]`` — a threshold on the one-hop
        neighbor-sum.

        The model should learn a single root split with ``d=1`` and a
        threshold close to 10.0 (within [9.5, 10.5]).  With an irregular
        Erdős–Rényi graph the boundary nodes may be ambiguous, so we
        require > 90 % accuracy rather than a perfect 100 %.

        Asserts:
            - Train and val accuracy > 89 %.
            - Root node ``walk_len == 1``.
            - Root split threshold in ``[9.5, 10.5]``.
        """
        n_nodes = 100
        n_train = 70
        num_features = 1
        max_walk_len = 2
        min_accuracy = 0.89

        data, labels, weights = _build_neighbor_sum_threshold_dataset(
            n=n_nodes,
            n_train=n_train,
            edge_prob=0.2,
            threshold=10.0,
            max_walk_len=max_walk_len,
        )

        model, probs = self._train_and_predict(
            data,
            labels,
            weights,
            num_features=num_features,
            num_vertices=n_nodes,
            max_walk_len=max_walk_len,
        )

        # --- Accuracy checks (> 90 %) ---
        train_idx = np.arange(n_train)
        val_idx = np.arange(n_train, n_nodes)
        train_acc = self._accuracy(probs[train_idx], labels[train_idx])
        val_acc = self._accuracy(probs[val_idx], labels[val_idx])

        assert train_acc > min_accuracy, (
            f"[neighbor-sum-threshold-ER] Train accuracy must be > "
            f"{min_accuracy:.0%}, got {train_acc:.4f}"
        )
        assert val_acc > min_accuracy, (
            f"[neighbor-sum-threshold-ER] Val accuracy must be > "
            f"{min_accuracy:.0%}, got {val_acc:.4f}"
        )

        # --- Tree structure checks ---
        # Parse the text dump of the single tree to inspect the root node.
        # Root line format:
        #   0:[f<N><split_cond>,d=<walk_len>,r=src|cyc,...] yes=...
        dump = model.get_dump()
        assert len(dump) == 1, f"Expected 1 tree, got {len(dump)}"
        root_line = dump[0].strip().split("\n")[0]

        # Extract walk_len (d=...)
        d_match = re.search(r"d=(\d+)", root_line)
        assert d_match is not None, (
            f"Could not find 'd=<walk_len>' in root node dump: {root_line}"
        )
        root_walk_len = int(d_match.group(1))
        assert root_walk_len == 1, (
            f"Root walk_len must be 1, got {root_walk_len}.  Root dump: {root_line}"
        )

        # Extract split threshold: the dump prints ``[f<N><=threshold,...]``.
        thresh_match = re.search(r"\[f\d+<=([\d.eE+-]+)", root_line)
        assert thresh_match is not None, (
            f"Could not parse split threshold from root dump: {root_line}"
        )
        root_threshold = float(thresh_match.group(1))
        assert 9.5 <= root_threshold <= 10.5, (
            f"Root threshold must be in [9.5, 10.5], got {root_threshold:.4f}.  "
            f"Root dump: {root_line}"
        )

    def test_two_hop_sum_threshold_erdos_renyi(self) -> None:
        """Accuracy > 90 % on two-hop sum threshold Erdős–Rényi graph.

        Uses an Erdős–Rényi random graph ``G(100, 0.2)`` with a single
        continuous feature in ``[0, 1]``.  The label for node *i* is
        ``𝟙[(A² · x)_i > 10.0]`` — a threshold on the **two-hop**
        neighborhood weighted sum.

        The model must discover a split with ``d=2`` (walk length 2) to
        capture the second-order neighborhood structure.  Because A² entries
        can be large in an Erdős–Rényi graph (many two-hop paths), boundary
        nodes may be ambiguous, so we require > 90 % accuracy rather than
        a perfect 100 %.

        Asserts:
            - Train and val accuracy > 90 %.
        """
        n_nodes = 100
        n_train = 70
        num_features = 1
        max_walk_len = 2
        min_accuracy = 0.90

        data, labels, weights = _build_two_hop_sum_threshold_dataset(
            n=n_nodes,
            n_train=n_train,
            edge_prob=0.2,
            threshold=10.0,
            max_walk_len=max_walk_len,
        )

        _, probs = self._train_and_predict(
            data,
            labels,
            weights,
            num_features=num_features,
            num_vertices=n_nodes,
            max_walk_len=max_walk_len,
            max_leaves=32,
        )

        # --- Accuracy checks (> 90 %) ---
        train_idx = np.arange(n_train)
        val_idx = np.arange(n_train, n_nodes)
        train_acc = self._accuracy(probs[train_idx], labels[train_idx])
        val_acc = self._accuracy(probs[val_idx], labels[val_idx])

        assert train_acc > min_accuracy, (
            f"[two-hop-sum-threshold-ER] Train accuracy must be > "
            f"{min_accuracy:.0%}, got {train_acc:.4f}"
        )
        assert val_acc > min_accuracy, (
            f"[two-hop-sum-threshold-ER] Val accuracy must be > "
            f"{min_accuracy:.0%}, got {val_acc:.4f}"
        )

    def test_blue_neighbor_avg_erdos_renyi(self, verbose: bool = False) -> None:
        """RMSE < 0.9 on blue-neighbor-average regression (Erdős–Rényi).

        Uses an Erdős–Rényi random graph ``G(100, 0.2)`` with three
        node features after one-hot encoding: ``is_red`` (index 0),
        ``is_blue`` (index 1), ``value`` (index 2, integer in
        ``{0, …, 10}``).  The label for node *i* is the **mean of the
        value feature across its blue neighbors only** (0 if none).

        Ideally the tree would:

        1. **Root** splits on ``is_blue`` (f1, ``d=0``) to identify blue
           nodes as the active set.
        2. **Children** point to the root's blue active set (``anc=0+``)
           and split on ``value`` (f2, ``d=1``) to aggregate blue
           neighbors' values.

        However, the greedy optimizer maximises gain at each split
        independently; splitting on ``is_blue`` at the root has near-zero
        gain because a node's own colour does not predict its label
        (which depends on *neighbours'* colours and values).  The tree
        therefore may learn a different root split (e.g. ``f2, d=1``,
        i.e. ``A @ value``) that is more immediately informative.

        The tree structure is **printed for investigation** rather than
        hard-asserted.

        Asserts:
            - Train and val RMSE < 0.9.
        """
        n_nodes = 100
        n_train = 70
        num_features = 3
        max_walk_len = 1
        # The reached-set fix makes tree growth more conservative at
        # non-root nodes (gain computed only for vertices actually routed
        # to each leaf, not the full attention set).  A single boosting
        # round may therefore produce slightly higher val RMSE.
        epsilon = 1.2

        data, labels, weights = _build_blue_neighbor_avg_dataset(
            n=n_nodes,
            n_train=n_train,
            max_walk_len=max_walk_len,
        )

        model, preds = self._train_and_predict_regression(
            data,
            labels,
            weights,
            num_features=num_features,
            num_vertices=n_nodes,
            max_walk_len=max_walk_len,
        )

        # --- RMSE checks ---
        train_idx = np.arange(n_train)
        val_idx = np.arange(n_train, n_nodes)
        train_rmse = self._rmse(preds[train_idx], labels[train_idx])
        val_rmse = self._rmse(preds[val_idx], labels[val_idx])

        assert train_rmse < epsilon, (
            f"[blue-neighbor-avg] Train RMSE must be < {epsilon}, got {train_rmse:.4f}"
        )
        assert val_rmse < epsilon, (
            f"[blue-neighbor-avg] Val RMSE must be < {epsilon}, got {val_rmse:.4f}"
        )

        # --- Single-tree check ---
        dump = model.get_dump()
        assert len(dump) == 1, f"Expected 1 tree, got {len(dump)}"

        if verbose:
            tree_text = dump[0].strip()
            print(
                f"\n--- Blue-Neighbor-Avg Tree Dump ---\n"
                f"Train RMSE: {train_rmse:.4f}  |  Val RMSE: {val_rmse:.4f}\n"
                f"Full tree:\n{tree_text}\n"
            )

    def test_degree_threshold_erdos_renyi(self, verbose: bool = False) -> None:
        """Accuracy > 99 % on degree-threshold Erdős–Rényi graph with intercept.

        Uses an Erdős–Rényi random graph ``G(100, 0.1)`` (fixed seed)
        with a single continuous noise feature in ``[0, 1]``.  The label
        for node *i* is ``𝟙[deg(i) > 10]``, where degree is the
        row-sum of the adjacency matrix (equivalently ``A · 𝟏``).

        With ``intercept=True`` the model gains a virtual constant-1
        feature at index ``num_features``.  A ``d=1`` split on this
        feature computes ``A · 𝟏 = degree`` for every node, and a
        threshold of ≈ 10 separates the two classes perfectly (up to
        boundary nodes whose degree equals exactly 10).

        The first 70 nodes are used for training (weight = 1), the
        remaining 30 for validation (weight = 0).

        Asserts:
            - Training accuracy > 99 %.
            - Validation accuracy > 99 %.
            - Root split uses the intercept feature (``f<num_features>``).
            - Total test running time < 5 minutes.

        Args:
            verbose: If ``True``, print the full tree dump with split
                details to stdout for manual inspection.
        """
        n_nodes = 100
        n_train = 70
        num_features = 1
        max_walk_len = 2
        min_accuracy = 0.99

        data, labels, weights = _build_degree_threshold_dataset(
            n=n_nodes,
            n_train=n_train,
            edge_prob=0.1,
            degree_threshold=10,
            max_walk_len=max_walk_len,
            seed=42,
        )

        t_start = time.monotonic()
        model, probs = self._train_and_predict(
            data,
            labels,
            weights,
            num_features=num_features,
            num_vertices=n_nodes,
            max_walk_len=max_walk_len,
            intercept=True,
        )
        elapsed = time.monotonic() - t_start

        # --- Timing check (< 5 minutes) ---
        assert elapsed < 300.0, (
            f"[degree-threshold-ER] Test took {elapsed:.1f}s, "
            f"exceeding the 5-minute limit"
        )

        # --- Accuracy checks (> 99 %) ---
        train_idx = np.arange(n_train)
        val_idx = np.arange(n_train, n_nodes)
        train_acc = self._accuracy(probs[train_idx], labels[train_idx])
        val_acc = self._accuracy(probs[val_idx], labels[val_idx])

        assert train_acc > min_accuracy, (
            f"[degree-threshold-ER] Train accuracy must be > "
            f"{min_accuracy:.0%}, got {train_acc:.4f}"
        )
        assert val_acc > min_accuracy, (
            f"[degree-threshold-ER] Val accuracy must be > "
            f"{min_accuracy:.0%}, got {val_acc:.4f}"
        )

        # --- Tree structure checks ---
        dump = model.get_dump()
        assert len(dump) == 1, f"Expected 1 tree, got {len(dump)}"

        if verbose:
            tree_text = dump[0].strip()
            print(
                f"\n--- Degree-Threshold Tree Dump (intercept) ---\n"
                f"Train acc: {train_acc:.4f}  |  Val acc: {val_acc:.4f}\n"
                f"Elapsed: {elapsed:.1f}s\n"
                f"Full tree:\n{tree_text}\n"
            )

        # Verify root split uses the intercept feature (the virtual constant-1
        # column), which the dump names ``intercept``.
        root_line = dump[0].strip().split("\n")[0]
        assert root_line.split(":", 1)[1].startswith("[intercept<="), (
            f"Root split must use the intercept feature.  Root dump: {root_line}"
        )


# ---------------------------------------------------------------------------
# Dataset helper — Feature-Based Multi-Class (3 classes)
# ---------------------------------------------------------------------------


def _build_feature_multiclass_dataset(
    n: int = 100,
    n_train: int = 70,
    max_walk_len: int = 2,
    seed: int = 99,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a trivial multi-class dataset where the label is determined by
    a single node feature.

    Each node has a single feature drawn uniformly from ``{-1, 0, 1}``.
    The label encodes three classes:

    - **0** ("Negative") when feature = −1
    - **1** ("Neutral") when feature = 0
    - **2** ("Positive") when feature = +1

    The graph is a simple ring lattice (each node connected to its two
    immediate neighbours) so that the DMatrix has the standard
    ``[X | A | A² | … | A^d]`` layout, but the label depends only on
    the node's own feature (``d=0``).  This makes the task perfectly
    separable by two axis-aligned splits on the single feature.

    Args:
        n: Number of nodes.
        n_train: Number of training nodes (first ``n_train`` get weight 1).
        max_walk_len: Maximum walk length *d*.  Powers ``A, A², …, A^d``
            are appended to the feature matrix.
        seed: Random seed for full reproducibility.

    Returns:
        A tuple ``(data, labels, weights)`` where:

        - ``data`` has shape ``(n, 1 + n * d)``.
        - ``labels`` has shape ``(n,)`` with values in ``{0, 1, 2}``.
        - ``weights`` has shape ``(n,)`` — 1.0 for train, 0.0 for val.
    """
    rng = np.random.RandomState(seed)

    # --- Feature: single value in {-1, 0, 1} ---
    feature = rng.choice([-1.0, 0.0, 1.0], size=n).astype(np.float32)
    X = feature.reshape(n, 1)

    # --- Label: 0 if -1, 1 if 0, 2 if +1 ---
    labels = np.empty(n, dtype=np.float32)
    labels[feature == -1.0] = 0.0
    labels[feature == 0.0] = 1.0
    labels[feature == 1.0] = 2.0

    # --- Simple ring adjacency (symmetric, no self-loops) ---
    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        j = (i + 1) % n
        A[i, j] = 1.0
        A[j, i] = 1.0

    # --- Assemble [X | A | A^2 | … | A^d] ---
    blocks: list[np.ndarray] = [X]
    A_power = A.copy()
    for _ in range(max_walk_len):
        blocks.append(A_power.copy())
        A_power = A_power @ A
    data = np.hstack(blocks)

    # --- Train / val split via weights ---
    weights = np.zeros(n, dtype=np.float32)
    weights[:n_train] = 1.0

    return data, labels, weights


# ---------------------------------------------------------------------------
# Tests — Multi-Class One-vs-All (3 classes, single tree)
# ---------------------------------------------------------------------------


class TestMultiClassOneVsAll:
    """Train TreeGBooster on a trivial 3-class multi-classification task.

    Dataset: 100 nodes on a ring graph with a single feature in
    ``{-1, 0, 1}`` and labels ``{0, 1, 2}`` (Negative / Neutral /
    Positive).  The mapping is deterministic:

    - feature = −1  →  class 0
    - feature =  0  →  class 1
    - feature = +1  →  class 2

    A single boosting round (``num_boost_round=1``) with one-vs-all
    (one tree per class) must achieve 100 % accuracy on both the
    training and validation splits.

    Tests both ``multi:softmax`` and ``multi:softprob`` objectives.
    """

    N_NODES: int = 100
    N_TRAIN: int = 70
    NUM_FEATURES: int = 1
    MAX_WALK_LEN: int = 2
    NUM_CLASSES: int = 3

    # -- Fixtures -----------------------------------------------------------

    @pytest.fixture(autouse=True)
    def setup_dataset(self) -> None:
        """Prepare the multi-class DMatrix before each test."""
        data, labels, weights = _build_feature_multiclass_dataset(
            n=self.N_NODES,
            n_train=self.N_TRAIN,
            max_walk_len=self.MAX_WALK_LEN,
        )
        self.dtrain: xgb.DMatrix = xgb.DMatrix(
            data,
            label=labels,
            weight=weights,
        )
        self.labels: np.ndarray = labels
        self.train_idx: np.ndarray = np.arange(self.N_TRAIN)
        self.val_idx: np.ndarray = np.arange(self.N_TRAIN, self.N_NODES)

    # -- Helpers ------------------------------------------------------------

    def _base_params(self) -> dict:
        """Return baseline Tree-G multi-class training parameters.

        Returns:
            Dictionary of XGBoost parameters for 3-class multi-classification
            on the trivial feature-based dataset.
        """
        return {
            "booster": "treeg",
            "base_score": 0.0,
            "num_class": self.NUM_CLASSES,
            # Tree-G–specific parameters
            "num_features": self.NUM_FEATURES,
            "num_vertices": self.N_NODES,
            "max_walk_len": self.MAX_WALK_LEN,
            "max_ancestor_distance": 2,
            # Aggressive single-round hyper-parameters
            "max_leaves": 16,
            "learning_rate": 1.0,
            "min_child_weight": 1.0,
            "reg_lambda": 0.01,
            "verbosity": 0,
        }

    @staticmethod
    def _accuracy(predicted_classes: np.ndarray, labels: np.ndarray) -> float:
        """Compute classification accuracy from predicted class indices.

        Args:
            predicted_classes: Predicted class indices (integers).
            labels: Ground-truth class labels.

        Returns:
            Accuracy as a Python float in ``[0, 1]``.
        """
        return float(np.mean(predicted_classes == labels))

    # -- Test cases ---------------------------------------------------------

    def test_softmax_perfect_accuracy(self) -> None:
        """100 % accuracy on 3-class feature-based dataset with multi:softmax.

        ``multi:softmax`` returns predicted class indices directly.  With a
        single boosting round (one tree per class via one-vs-all), the
        model must perfectly separate all three classes on both the training
        (70 nodes) and validation (30 nodes) splits.
        """
        params = self._base_params()
        params["objective"] = "multi:softmax"

        model = xgb.train(params, self.dtrain, num_boost_round=1)
        preds = model.predict(self.dtrain)

        # multi:softmax returns float class indices — round to int
        pred_classes = np.round(preds).astype(np.int32)

        train_acc = self._accuracy(
            pred_classes[self.train_idx], self.labels[self.train_idx]
        )
        val_acc = self._accuracy(pred_classes[self.val_idx], self.labels[self.val_idx])

        assert train_acc == 1.0, (
            f"[multi:softmax] Train accuracy must be 100 %, got {train_acc:.4f}"
        )
        assert val_acc == 1.0, (
            f"[multi:softmax] Val accuracy must be 100 %, got {val_acc:.4f}"
        )

        # Verify model has exactly NUM_CLASSES trees (one per class)
        dump = model.get_dump()
        assert len(dump) == self.NUM_CLASSES, (
            f"Expected {self.NUM_CLASSES} trees (one per class), got {len(dump)}"
        )

    def test_softprob_perfect_accuracy(self) -> None:
        """100 % accuracy on 3-class feature-based dataset with multi:softprob.

        ``multi:softprob`` returns a probability matrix of shape
        ``(n_samples, n_classes)``.  The predicted class is
        ``argmax`` along the class axis.  With a single boosting round
        (one tree per class via one-vs-all), the model must perfectly
        separate all three classes on both splits.
        """
        params = self._base_params()
        params["objective"] = "multi:softprob"

        model = xgb.train(params, self.dtrain, num_boost_round=1)
        probs = model.predict(self.dtrain)

        # multi:softprob returns (n, num_class) probabilities
        assert probs.shape == (self.N_NODES, self.NUM_CLASSES), (
            f"Expected shape ({self.N_NODES}, {self.NUM_CLASSES}), got {probs.shape}"
        )

        pred_classes = np.argmax(probs, axis=1).astype(np.int32)

        train_acc = self._accuracy(
            pred_classes[self.train_idx], self.labels[self.train_idx]
        )
        val_acc = self._accuracy(pred_classes[self.val_idx], self.labels[self.val_idx])

        assert train_acc == 1.0, (
            f"[multi:softprob] Train accuracy must be 100 %, got {train_acc:.4f}"
        )
        assert val_acc == 1.0, (
            f"[multi:softprob] Val accuracy must be 100 %, got {val_acc:.4f}"
        )

        # Probabilities should sum to ~1 for each sample
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(
            row_sums,
            np.ones(self.N_NODES),
            atol=1e-5,
            err_msg="Softprob row sums should be ≈ 1.0",
        )

        # Verify model has exactly NUM_CLASSES trees (one per class)
        dump = model.get_dump()
        assert len(dump) == self.NUM_CLASSES, (
            f"Expected {self.NUM_CLASSES} trees (one per class), got {len(dump)}"
        )


# ---------------------------------------------------------------------------
# Dataset helper — Graph-Aware Multi-Class (3 classes, color + neighbor sum)
# ---------------------------------------------------------------------------


def _build_multiclass_neighbor_sum_dataset(
    n: int = 100,
    n_train: int = 70,
    edge_prob: float = 0.1,
    neighbor_sum_threshold: float = 5.0,
    max_walk_len: int = 2,
    seed: int = 77,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an Erdős–Rényi graph dataset with 3-class labels determined by
    node color and the neighbor-sum of the value feature.

    Each node has three features (after one-hot encoding of color):

    - ``is_red``  (index 0): 1 if the node is red, 0 otherwise.
    - ``is_blue`` (index 1): 1 if the node is blue, 0 otherwise.
    - ``value``   (index 2): continuous value drawn uniformly from ``[0, 1]``.

    Color is assigned 50-50 (roughly half red, half blue).  The label for
    node *i* is:

    - **0** ("Red") when ``is_red_i = 1``.
    - **1** ("Blue-High") when ``is_blue_i = 1`` and
      ``(A · value)_i > threshold``.
    - **2** ("Blue-Low") when ``is_blue_i = 1`` and
      ``(A · value)_i ≤ threshold``.

    Learning this requires:

    1. A ``d=0`` split on color to separate red from blue nodes.
    2. A ``d=1`` split on the neighbor-sum of value (``A · value``) with
       the appropriate threshold to separate Blue-High from Blue-Low.

    With edge probability ``p=0.1`` the expected degree is ≈ 9.9 and the
    expected neighbor-sum of value ≈ 4.95, so a threshold of 5.0 yields
    a roughly even split among blue nodes.

    Args:
        n: Number of nodes.
        n_train: Number of training nodes (first ``n_train`` get weight 1).
        edge_prob: Edge probability for the Erdős–Rényi graph.
        neighbor_sum_threshold: Threshold on ``A · value`` that separates
            class 1 from class 2 among blue nodes.
        max_walk_len: Maximum walk length *d*.  Powers ``A, A², …, A^d``
            are appended to the feature matrix.
        seed: Random seed for full reproducibility.

    Returns:
        A tuple ``(data, labels, weights)`` where:

        - ``data`` has shape ``(n, 3 + n * d)``.
        - ``labels`` has shape ``(n,)`` with values in ``{0, 1, 2}``.
        - ``weights`` has shape ``(n,)`` — 1.0 for train, 0.0 for val.
    """
    rng = np.random.RandomState(seed)

    # --- Node features: color (one-hot) + value ---
    is_blue = rng.randint(0, 2, size=n).astype(np.float32)  # 50-50
    is_red = 1.0 - is_blue
    value = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
    X = np.column_stack([is_red, is_blue, value])  # (n, 3)

    # --- Erdős–Rényi adjacency (symmetric, no self-loops) ---
    upper = (rng.rand(n, n) < edge_prob).astype(np.float32)
    np.fill_diagonal(upper, 0.0)
    A = np.triu(upper, k=1)
    A = A + A.T  # symmetric

    # Guarantee every node has at least one neighbor.
    for i in range(n):
        if A[i].sum() == 0:
            j = (i + 1) % n
            A[i, j] = 1.0
            A[j, i] = 1.0

    # --- Labels: 3 classes based on color + neighbor-sum of value ---
    neighbor_sum_value = A @ value  # (n,)
    labels = np.zeros(n, dtype=np.float32)
    # Class 0: red nodes
    # Class 1: blue nodes with neighbor-sum of value > threshold
    # Class 2: blue nodes with neighbor-sum of value <= threshold
    labels[is_red == 1.0] = 0.0
    labels[(is_blue == 1.0) & (neighbor_sum_value > neighbor_sum_threshold)] = 1.0
    labels[(is_blue == 1.0) & (neighbor_sum_value <= neighbor_sum_threshold)] = 2.0

    # --- Assemble [X | A | A^2 | … | A^d] ---
    blocks: list[np.ndarray] = [X]
    A_power = A.copy()
    for _ in range(max_walk_len):
        blocks.append(A_power.copy())
        A_power = A_power @ A
    data = np.hstack(blocks)

    # --- Train / val split via weights ---
    weights = np.zeros(n, dtype=np.float32)
    weights[:n_train] = 1.0

    return data, labels, weights


# ---------------------------------------------------------------------------
# Tests — Graph-Aware Multi-Class One-vs-All (3 classes, single round)
# ---------------------------------------------------------------------------


class TestMultiClassGraphAwareOneVsAll:
    """Train TreeGBooster on a 3-class graph-aware multi-classification task.

    Dataset: An Erdős–Rényi random graph ``G(100, 0.1)`` (fixed seed).
    Each node has two one-hot color features (``is_red``, ``is_blue``;
    50-50 split) and a continuous ``value`` in ``[0, 1]``.  Labels are:

    - **0** ("Red"): all red nodes.
    - **1** ("Blue-High"): blue nodes whose neighbor-sum of ``value``
      exceeds 5.
    - **2** ("Blue-Low"): blue nodes whose neighbor-sum of ``value``
      is at most 5.

    Learning this task requires both a feature-level split (``d=0`` on
    color) and a graph-aware split (``d=1`` on the neighbor-sum of
    ``value``), exercising the one-vs-all multi-class pipeline with
    non-trivial graph structure.

    A single boosting round (``num_boost_round=1``, one tree per class)
    must achieve 100 % accuracy on both training (70 nodes) and
    validation (30 nodes) splits.
    """

    N_NODES: int = 100
    N_TRAIN: int = 70
    NUM_FEATURES: int = 3
    MAX_WALK_LEN: int = 2
    NUM_CLASSES: int = 3

    # -- Fixtures -----------------------------------------------------------

    @pytest.fixture(autouse=True)
    def setup_dataset(self) -> None:
        """Prepare the graph-aware multi-class DMatrix before each test."""
        data, labels, weights = _build_multiclass_neighbor_sum_dataset(
            n=self.N_NODES,
            n_train=self.N_TRAIN,
            max_walk_len=self.MAX_WALK_LEN,
        )
        self.dtrain: xgb.DMatrix = xgb.DMatrix(
            data,
            label=labels,
            weight=weights,
        )
        self.labels: np.ndarray = labels
        self.train_idx: np.ndarray = np.arange(self.N_TRAIN)
        self.val_idx: np.ndarray = np.arange(self.N_TRAIN, self.N_NODES)

    # -- Helpers ------------------------------------------------------------

    def _base_params(self) -> dict:
        """Return baseline Tree-G multi-class training parameters.

        Returns:
            Dictionary of XGBoost parameters for 3-class graph-aware
            multi-classification.
        """
        return {
            "booster": "treeg",
            "base_score": 0.0,
            "num_class": self.NUM_CLASSES,
            # Tree-G–specific parameters
            "num_features": self.NUM_FEATURES,
            "num_vertices": self.N_NODES,
            "max_walk_len": self.MAX_WALK_LEN,
            "max_ancestor_distance": 2,
            # Aggressive single-round hyper-parameters
            "max_leaves": 16,
            "learning_rate": 1.0,
            "min_child_weight": 1.0,
            "reg_lambda": 0.01,
            "verbosity": 0,
        }

    @staticmethod
    def _accuracy(predicted_classes: np.ndarray, labels: np.ndarray) -> float:
        """Compute classification accuracy from predicted class indices.

        Args:
            predicted_classes: Predicted class indices (integers).
            labels: Ground-truth class labels.

        Returns:
            Accuracy as a Python float in ``[0, 1]``.
        """
        return float(np.mean(predicted_classes == labels))

    # -- Test cases ---------------------------------------------------------

    def test_softmax_perfect_accuracy(self) -> None:
        """100 % accuracy with multi:softmax on graph-aware 3-class dataset.

        Uses ``multi:softmax`` which returns predicted class indices
        directly.  With a single boosting round (one tree per class via
        one-vs-all), the model must perfectly separate all three classes
        on both the training (70 nodes) and validation (30 nodes) splits.

        Also verifies the model produces exactly ``NUM_CLASSES`` trees
        (one per class).
        """
        params = self._base_params()
        params["objective"] = "multi:softmax"

        model = xgb.train(params, self.dtrain, num_boost_round=1)
        preds = model.predict(self.dtrain)

        # multi:softmax returns float class indices — round to int
        pred_classes = np.round(preds).astype(np.int32)

        train_acc = self._accuracy(
            pred_classes[self.train_idx], self.labels[self.train_idx]
        )
        val_acc = self._accuracy(pred_classes[self.val_idx], self.labels[self.val_idx])

        assert train_acc == 1.0, (
            f"[multi:softmax graph-aware] Train accuracy must be 100 %, "
            f"got {train_acc:.4f}"
        )
        assert val_acc == 1.0, (
            f"[multi:softmax graph-aware] Val accuracy must be 100 %, got {val_acc:.4f}"
        )

        # Verify model has exactly NUM_CLASSES trees (one per class)
        dump = model.get_dump()
        assert len(dump) == self.NUM_CLASSES, (
            f"Expected {self.NUM_CLASSES} trees (one per class), got {len(dump)}"
        )

    def test_softprob_perfect_accuracy(self) -> None:
        """100 % accuracy with multi:softprob on graph-aware 3-class dataset.

        Uses ``multi:softprob`` which returns a probability matrix of shape
        ``(n_samples, n_classes)``.  The predicted class is ``argmax``
        along the class axis.  With a single boosting round (one tree per
        class via one-vs-all), the model must perfectly separate all three
        classes on both splits.

        Also verifies that per-sample probability rows sum to ≈ 1 and the
        model produces exactly ``NUM_CLASSES`` trees.
        """
        params = self._base_params()
        params["objective"] = "multi:softprob"

        model = xgb.train(params, self.dtrain, num_boost_round=1)
        probs = model.predict(self.dtrain)

        # multi:softprob returns (n, num_class) probabilities
        assert probs.shape == (self.N_NODES, self.NUM_CLASSES), (
            f"Expected shape ({self.N_NODES}, {self.NUM_CLASSES}), got {probs.shape}"
        )

        pred_classes = np.argmax(probs, axis=1).astype(np.int32)

        train_acc = self._accuracy(
            pred_classes[self.train_idx], self.labels[self.train_idx]
        )
        val_acc = self._accuracy(pred_classes[self.val_idx], self.labels[self.val_idx])

        assert train_acc == 1.0, (
            f"[multi:softprob graph-aware] Train accuracy must be 100 %, "
            f"got {train_acc:.4f}"
        )
        assert val_acc == 1.0, (
            f"[multi:softprob graph-aware] Val accuracy must be 100 %, "
            f"got {val_acc:.4f}"
        )

        # Probabilities should sum to ~1 for each sample
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(
            row_sums,
            np.ones(self.N_NODES),
            atol=1e-5,
            err_msg="Softprob row sums should be ≈ 1.0",
        )

        # Verify model has exactly NUM_CLASSES trees (one per class)
        dump = model.get_dump()
        assert len(dump) == self.NUM_CLASSES, (
            f"Expected {self.NUM_CLASSES} trees (one per class), got {len(dump)}"
        )


# ---------------------------------------------------------------------------
# Sparse input
# ---------------------------------------------------------------------------


class TestSparseInput:
    """Verify that sparse and dense ``DMatrix`` sources are equivalent.

    The Tree-G column layout ``[X | A | A^2 | ... | A^d]`` is dominated by the
    adjacency blocks, which are mostly zero for any realistic graph. Building
    the ``DMatrix`` from a ``scipy.sparse`` matrix avoids materializing those
    zeros. Because an absent entry and a stored ``0.0`` carry the same meaning
    for Tree-G -- no edge, or a zero feature value -- both representations must
    produce bit-identical models.

    The equivalent C++ coverage lives in ``GraphData.BuildSparseInputMatchesDense``;
    this test additionally exercises the Python adapter path.
    """

    N_NODES: int = 100
    N_TRAIN: int = 70
    NUM_FEATURES: int = 1
    MAX_WALK_LEN: int = 2

    def _params(self) -> dict:
        """Return baseline Tree-G parameters for the circle-graph dataset.

        Returns:
            Dictionary of XGBoost parameters with ``booster='treeg'``.
        """
        return {
            "booster": "treeg",
            "objective": "reg:squarederror",
            "base_score": 0.5,
            "num_features": self.NUM_FEATURES,
            "num_vertices": self.N_NODES,
            "max_walk_len": self.MAX_WALK_LEN,
            "max_ancestor_distance": 2,
            "max_leaves": 8,
            "learning_rate": 0.3,
            "min_child_weight": 1.0,
            "reg_lambda": 1.0,
            "verbosity": 0,
        }

    def test_sparse_input_matches_dense(self) -> None:
        """A CSR-backed DMatrix must train and predict identically to a dense one."""
        sparse = pytest.importorskip("scipy.sparse")

        data, labels, weights = _build_circle_graph_dataset(
            n=self.N_NODES,
            n_train=self.N_TRAIN,
            max_walk_len=self.MAX_WALK_LEN,
        )

        dense_mat = xgb.DMatrix(data, label=labels, weight=weights)
        csr_mat = xgb.DMatrix(sparse.csr_matrix(data), label=labels, weight=weights)

        # Guard the premise: the sparse page must actually store fewer entries,
        # otherwise this test silently degenerates into comparing dense to dense.
        assert csr_mat.num_nonmissing() < dense_mat.num_nonmissing(), (
            "expected the CSR DMatrix to drop structural zeros "
            f"(dense={dense_mat.num_nonmissing()}, csr={csr_mat.num_nonmissing()})"
        )
        assert csr_mat.num_row() == dense_mat.num_row()
        assert csr_mat.num_col() == dense_mat.num_col()

        params = self._params()
        dense_model = xgb.train(params, dense_mat, num_boost_round=5)
        csr_model = xgb.train(params, csr_mat, num_boost_round=5)

        # Identical trees, not merely similar predictions.
        assert dense_model.get_dump() == csr_model.get_dump()

        dense_preds = dense_model.predict(dense_mat)
        csr_preds = csr_model.predict(csr_mat)
        np.testing.assert_array_equal(dense_preds, csr_preds)

        # A model trained on one representation must predict identically on the
        # other, confirming the equivalence holds at prediction time too.
        np.testing.assert_array_equal(dense_model.predict(csr_mat), dense_preds)


# ---------------------------------------------------------------------------
# Parameter aliases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "alias, canonical, value",
    [
        ("eta", "learning_rate", 0.05),
        ("gamma", "min_split_loss", 50.0),
        ("lambda", "reg_lambda", 10.0),
        ("alpha", "reg_alpha", 1.0),
    ],
)
def test_parameter_aliases(alias: str, canonical: str, value: float) -> None:
    """gbtree aliases must be honoured by booster=treeg, not silently ignored."""
    n_nodes, max_walk_len = 60, 2
    data, labels, weights = _build_circle_graph_dataset(
        n=n_nodes, n_train=40, max_walk_len=max_walk_len
    )
    dtrain = xgb.DMatrix(data, label=labels, weight=weights)
    base = {
        "booster": "treeg",
        "objective": "reg:squarederror",
        "num_features": 1,
        "num_vertices": n_nodes,
        "max_walk_len": max_walk_len,
        "seed": 0,
    }

    def predict(extra: dict) -> np.ndarray:
        return xgb.train({**base, **extra}, dtrain, num_boost_round=5).predict(dtrain)

    default_preds = predict({})
    alias_preds = predict({alias: value})
    canonical_preds = predict({canonical: value})

    np.testing.assert_array_equal(alias_preds, canonical_preds)
    assert not np.array_equal(alias_preds, default_preds), (
        f"{alias}={value} had no effect on the model"
    )
