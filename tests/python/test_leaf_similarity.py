"""Tests for leaf similarity computation."""

import numpy as np
import pytest

import xgboost as xgb
from xgboost import testing as tm

rng = np.random.RandomState(1994)


class TestLeafSimilarity:
    """Tests for Booster.compute_leaf_similarity()"""

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
        assert similarity.max() <= 1.0

        # Self-similarity diagonal should be 1.0
        dm_self = xgb.DMatrix(X[:20])
        self_sim = bst.compute_leaf_similarity(dm_self, dm_self)
        np.testing.assert_allclose(np.diag(self_sim), 1.0, rtol=1e-5)

        # Test weight types
        sim_gain = bst.compute_leaf_similarity(dm_query, dm_ref, weight_type="gain")
        sim_cover = bst.compute_leaf_similarity(dm_query, dm_ref, weight_type="cover")
        assert sim_gain.shape == sim_cover.shape

        # Default should be gain
        sim_default = bst.compute_leaf_similarity(dm_query, dm_ref)
        np.testing.assert_array_equal(sim_default, sim_gain)

        # Invalid weight_type
        with pytest.raises(ValueError, match="weight_type must be"):
            bst.compute_leaf_similarity(dm_query, dm_ref, weight_type="invalid")
