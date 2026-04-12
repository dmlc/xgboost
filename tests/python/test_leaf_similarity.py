"""Tests for leaf similarity computation."""

import ctypes

import numpy as np
import pytest

import xgboost as xgb
from xgboost.core import (
    _LIB,
    _check_call,
    c_bst_ulong,
    ctypes2numpy,
    from_pystr_to_cstr,
)
from xgboost import testing as tm

rng = np.random.RandomState(1994)


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
    @pytest.mark.parametrize(("weight_type", "column"), [("gain", "Gain"), ("cover", "Cover")])
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
            (
                "{"
                f'"weight_type":"{weight_type}",'
                '"iteration_begin":0,'
                '"iteration_end":0'
                "}"
            )
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
        sim_uniform = bst.compute_leaf_similarity(dm_query, dm_ref, weight_type="uniform")
        sim_default = bst.compute_leaf_similarity(dm_query, dm_ref)
        np.testing.assert_array_equal(sim_default, sim_uniform)

        # Invalid weight_type
        with pytest.raises(ValueError, match="weight_type must be"):
            bst.compute_leaf_similarity(dm_query, dm_ref, weight_type="invalid")
