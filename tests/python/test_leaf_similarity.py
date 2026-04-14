"""Tests for leaf similarity computation."""

import ctypes

import numpy as np
import pytest
import xgboost as xgb
from sklearn.datasets import load_diabetes, load_iris
from xgboost import testing as tm
from xgboost.core import (
    _LIB,
    _check_call,
    c_bst_ulong,
    ctypes2numpy,
    from_pystr_to_cstr,
)


class TestLeafSimilarity:
    """Tests for Booster.compute_leaf_similarity()"""

    @pytest.mark.parametrize(
        ("param", "num_boost_round"),
        [
            ({"max_depth": 4, "eta": 0.3, "objective": "binary:logistic"}, 8),
            (
                {
                    "max_depth": 3,
                    "eta": 0.3,
                    "objective": "multi:softprob",
                    "num_class": 3,
                },
                6,
            ),
            (
                {
                    "max_depth": 4,
                    "eta": 1.0,
                    "objective": "binary:logistic",
                    "num_parallel_tree": 3,
                },
                5,
            ),
        ],
    )
    @pytest.mark.parametrize(
        ("weight_type", "column"), [("gain", "Gain"), ("cover", "Cover")]
    )
    @pytest.mark.skipif(**tm.no_pandas())
    def test_leaf_similarity_weight_api(
        self, param: dict, num_boost_round: int, weight_type: str, column: str
    ) -> None:
        """Test the low-level tree weight API shape and order contract."""
        dtrain, _ = tm.load_agaricus(__file__)
        bst = xgb.train(param, dtrain, num_boost_round=num_boost_round)

        leaves = bst.predict(dtrain, pred_leaf=True, strict_shape=True)
        expected_len = int(np.prod(leaves.shape[1:]))
        expected_weights = np.zeros(expected_len, dtype=np.float32)
        trees_df = bst.trees_to_dataframe()
        split_nodes = trees_df[trees_df["Feature"] != "Leaf"]
        tree_weights = split_nodes.groupby("Tree")[column].sum()
        for tree_id, weight in tree_weights.items():
            expected_weights[int(tree_id)] = weight

        config = from_pystr_to_cstr(
            (f'{{"weight_type":"{weight_type}","iteration_begin":0,"iteration_end":0}}')
        )
        out_len = c_bst_ulong()
        out_weights = ctypes.POINTER(ctypes.c_float)()

        _check_call(
            _LIB.XGBoosterGetLeafSimilarityWeights(
                bst.handle, config, ctypes.byref(out_len), ctypes.byref(out_weights)
            )
        )

        assert out_len.value == expected_len
        weights = ctypes2numpy(out_weights, out_len.value, np.float32)
        np.testing.assert_allclose(weights, expected_weights, rtol=1e-6, atol=1e-3)

    def test_leaf_similarity(self) -> None:
        """Test basic leaf similarity computation."""
        dtrain, _ = tm.load_agaricus(__file__)
        param = {"max_depth": 4, "eta": 0.3, "objective": "binary:logistic"}
        bst = xgb.train(param, dtrain, num_boost_round=10)

        X = dtrain.get_data()
        dm_query = xgb.DMatrix(X[:10])
        dm_ref = xgb.DMatrix(X[100:150])

        # Test shape and range
        similarity = bst.compute_leaf_similarity(dm_query, dm_ref)
        assert similarity.shape == (10, 50)
        assert similarity.min() >= 0.0
        assert similarity.max() <= 1.0 + 1e-6

        # Self-similarity diagonal should be 1.0
        dm_self = xgb.DMatrix(X[:20])
        self_sim = bst.compute_leaf_similarity(dm_self, dm_self)
        np.testing.assert_allclose(np.diag(self_sim), 1.0, rtol=1e-5)

        # Test weight types
        sim_gain = bst.compute_leaf_similarity(dm_query, dm_ref, weight_type="gain")
        sim_cover = bst.compute_leaf_similarity(dm_query, dm_ref, weight_type="cover")
        assert sim_gain.shape == sim_cover.shape

        # Default should be uniform
        sim_uniform = bst.compute_leaf_similarity(
            dm_query, dm_ref, weight_type="uniform"
        )
        sim_default = bst.compute_leaf_similarity(dm_query, dm_ref)
        np.testing.assert_array_equal(sim_default, sim_uniform)

        # Invalid weight_type
        with pytest.raises(ValueError, match="weight_type must be"):
            bst.compute_leaf_similarity(dm_query, dm_ref, weight_type="invalid")

    @pytest.mark.parametrize(
        "param",
        [
            {
                "max_depth": 3,
                "eta": 0.3,
                "objective": "multi:softprob",
                "num_class": 3,
            },
            {
                "booster": "dart",
                "max_depth": 4,
                "eta": 0.3,
                "objective": "binary:logistic",
            },
            {
                "max_depth": 4,
                "eta": 1.0,
                "objective": "binary:logistic",
                "num_parallel_tree": 3,
            },
        ],
    )
    @pytest.mark.parametrize("weight_type", ["uniform", "gain", "cover"])
    def test_leaf_similarity_supported_tree_modes(
        self, param: dict, weight_type: str
    ) -> None:
        """Test supported tree model modes."""
        if param.get("objective") == "multi:softprob":
            X, y = load_iris(return_X_y=True)
            dtrain = xgb.DMatrix(X, label=y)
            dm_query = xgb.DMatrix(X[:5])
            dm_ref = xgb.DMatrix(X[10:20])
            rounds = 8
        else:
            dtrain, _ = tm.load_agaricus(__file__)
            X = dtrain.get_data()
            dm_query = xgb.DMatrix(X[:10])
            dm_ref = xgb.DMatrix(X[100:130])
            rounds = 8 if param.get("booster") == "dart" else 5

        bst = xgb.train(param, dtrain, num_boost_round=rounds)
        similarity = bst.compute_leaf_similarity(
            dm_query, dm_ref, weight_type=weight_type
        )
        assert similarity.shape == (dm_query.num_row(), dm_ref.num_row())
        assert similarity.min() >= 0.0
        assert similarity.max() <= 1.0 + 1e-6

    @pytest.mark.parametrize("weight_type", ["uniform", "gain", "cover"])
    def test_leaf_similarity_one_output_per_tree_multi_target(
        self, weight_type: str
    ) -> None:
        """Test multi-target model with one output per tree."""
        X, y = load_diabetes(return_X_y=True)
        y = np.column_stack([y, y * 0.5])
        dtrain = xgb.DMatrix(X, label=y)
        bst = xgb.train(
            {
                "max_depth": 3,
                "eta": 0.3,
                "tree_method": "hist",
                "objective": "reg:squarederror",
                "multi_strategy": "one_output_per_tree",
                "num_target": 2,
            },
            dtrain,
            num_boost_round=6,
        )

        similarity = bst.compute_leaf_similarity(
            xgb.DMatrix(X[:5]), xgb.DMatrix(X[10:20]), weight_type=weight_type
        )
        assert similarity.shape == (5, 10)
        assert similarity.min() >= 0.0
        assert similarity.max() <= 1.0 + 1e-6

    @pytest.mark.parametrize("weight_type", ["uniform", "gain", "cover"])
    def test_leaf_similarity_gblinear_error(self, weight_type: str) -> None:
        """Test unsupported gblinear booster with stable error."""
        dtrain, _ = tm.load_agaricus(__file__)
        bst = xgb.train({"booster": "gblinear", "objective": "binary:logistic"}, dtrain)
        X = dtrain.get_data()

        with pytest.raises(
            xgb.core.XGBoostError, match="Leaf similarity is only defined"
        ):
            bst.compute_leaf_similarity(
                xgb.DMatrix(X[:5]), xgb.DMatrix(X[10:20]), weight_type=weight_type
            )

    @pytest.mark.parametrize("weight_type", ["uniform", "gain", "cover"])
    def test_leaf_similarity_multi_output_tree_error(self, weight_type: str) -> None:
        """Test unsupported multi-output tree with stable error."""
        X, y = load_diabetes(return_X_y=True)
        y = np.column_stack([y, y * 0.5])
        dtrain = xgb.DMatrix(X, label=y)
        bst = xgb.train(
            {
                "max_depth": 3,
                "eta": 0.3,
                "tree_method": "hist",
                "objective": "reg:squarederror",
                "multi_strategy": "multi_output_tree",
                "num_target": 2,
            },
            dtrain,
            num_boost_round=6,
        )

        with pytest.raises(
            xgb.core.XGBoostError,
            match="Leaf similarity does not support multi_output_tree",
        ):
            bst.compute_leaf_similarity(
                xgb.DMatrix(X[:5]), xgb.DMatrix(X[10:20]), weight_type=weight_type
            )
