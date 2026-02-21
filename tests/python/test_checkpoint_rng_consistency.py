"""Test for issue #11982: Checkpoint RNG consistency in distributed training."""

import tempfile
import numpy as np
import pytest
import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

import xgboost.testing as tm


class TestCheckpointRNGConsistency:
    """Test RNG state preservation during checkpoint save/load operations."""

    def test_checkpoint_rng_consistency_single_thread(self):
        """Test that checkpoint save/load preserves RNG state in single-threaded mode."""
        # Create reproducible test data
        X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
        dtrain = xgb.DMatrix(X, label=y)
        
        # Parameters that rely heavily on RNG (low subsample)
        params = {
            'objective': 'reg:squarederror',
            'subsample': 0.2,  # Low subsample makes RNG state critical
            'colsample_bytree': 0.8,
            'seed': 1994,
            'nthread': 1,
            'max_depth': 6,
            'eta': 0.3,
        }
        
        # Test 1: Full training without checkpoint
        model_full = xgb.train(params, dtrain, num_boost_round=50)
        pred_full = model_full.predict(dtrain)
        
        # Test 2: Training with checkpoint at round 25
        model_checkpoint = xgb.train(params, dtrain, num_boost_round=25)
        
        with tempfile.NamedTemporaryFile(suffix='.ubj', delete=False) as f:
            checkpoint_path = f.name
            model_checkpoint.save_model(checkpoint_path)
        
        # Resume training from checkpoint
        model_resumed = xgb.train(params, dtrain, num_boost_round=25, xgb_model=checkpoint_path)
        pred_resumed = model_resumed.predict(dtrain)
        
        # The predictions should be identical (or extremely close)
        mse_diff = mean_squared_error(pred_full, pred_resumed)
        
        # With proper RNG state preservation, this should be near zero
        assert mse_diff < 1e-10, f"Checkpoint resume changed predictions: MSE diff = {mse_diff}"
        
        # Additional check: predictions should be element-wise nearly identical
        np.testing.assert_array_almost_equal(pred_full, pred_resumed, decimal=10,
            err_msg="Predictions differ after checkpoint resume - RNG state not preserved")

    def test_rng_state_serialization_deserialization(self):
        """Test that RNG state can be properly serialized and deserialized."""
        # Create a simple model to trigger RNG state changes
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        dtrain = xgb.DMatrix(X, label=y)
        
        params = {
            'objective': 'reg:squarederror',
            'subsample': 0.5,
            'seed': 123,
            'nthread': 1,
        }
        
        # Train a small model to evolve RNG state
        model = xgb.train(params, dtrain, num_boost_round=5)
        
        # Save and reload the model
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            model_path = f.name
            model.save_model(model_path)
        
        # Load model back
        model_loaded = xgb.Booster()
        model_loaded.load_model(model_path)
        
        # Continue training from both models
        model_continued = xgb.train(params, dtrain, num_boost_round=5, xgb_model=model)
        model_loaded_continued = xgb.train(params, dtrain, num_boost_round=5, xgb_model=model_loaded)
        
        # Predictions should be identical
        pred_original = model_continued.predict(dtrain)
        pred_loaded = model_loaded_continued.predict(dtrain)
        
        np.testing.assert_array_almost_equal(pred_original, pred_loaded, decimal=10,
            err_msg="RNG state not properly preserved across save/load cycle")

    def test_subsample_reproducibility_with_checkpoint(self):
        """Test that subsample behavior is identical before and after checkpoint."""
        X, y = make_regression(n_samples=2000, n_features=50, noise=0.05, random_state=2024)
        dtrain = xgb.DMatrix(X, label=y)
        
        params = {
            'objective': 'reg:squarederror',
            'subsample': 0.1,  # Very low subsample to make RNG critical
            'colsample_bytree': 0.3,
            'seed': 2024,
            'nthread': 1,
            'max_depth': 8,
            'eta': 0.1,
        }
        
        # Scenario A: Full training
        model_a = xgb.train(params, dtrain, num_boost_round=100)
        pred_a = model_a.predict(dtrain)
        
        # Scenario B: Checkpoint at multiple points
        for checkpoint_round in [20, 40, 60, 80]:
            model_b1 = xgb.train(params, dtrain, num_boost_round=checkpoint_round)
            
            with tempfile.NamedTemporaryFile(suffix='.ubj', delete=False) as f:
                checkpoint_path = f.name
                model_b1.save_model(checkpoint_path)
            
            remaining_rounds = 100 - checkpoint_round
            model_b2 = xgb.train(params, dtrain, num_boost_round=remaining_rounds, 
                               xgb_model=checkpoint_path)
            pred_b = model_b2.predict(dtrain)
            
            # Should be identical regardless of checkpoint timing
            mse_diff = mean_squared_error(pred_a, pred_b)
            assert mse_diff < 1e-12, \
                f"Checkpoint at round {checkpoint_round} caused drift: MSE diff = {mse_diff}"

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_distributed_checkpoint_consistency_simulation(self):
        """
        Simulate the distributed training scenario from issue #11982.
        This test validates that the fix prevents accuracy drops.
        """
        # Simulate the user's exact scenario
        X, y = make_regression(n_samples=5000, n_features=100, noise=0.1, random_state=1994)
        dtrain = xgb.DMatrix(X, label=y)
        
        # Parameters matching the issue report
        params = {
            'objective': 'reg:squarederror',
            'subsample': 0.2,  # Exact value from issue
            'max_delta_step': 5,  # Exact value from issue
            'seed': 1994,  # Seed from issue
            'nthread': 1,  # Single thread for reproducibility
            'max_depth': 6,
            'eta': 0.3,
        }
        
        # Multiple checkpoint/resume cycles
        num_rounds_total = 100
        checkpoint_intervals = [10, 25, 50]
        
        reference_model = xgb.train(params, dtrain, num_boost_round=num_rounds_total)
        reference_pred = reference_model.predict(dtrain)
        reference_mse = mean_squared_error(y, reference_pred)
        
        for checkpoint_at in checkpoint_intervals:
            # Train to checkpoint
            model_part1 = xgb.train(params, dtrain, num_boost_round=checkpoint_at)
            
            # Save checkpoint
            with tempfile.NamedTemporaryFile(suffix='.ubj', delete=False) as f:
                checkpoint_path = f.name
                model_part1.save_model(checkpoint_path)
            
            # Resume training
            remaining_rounds = num_rounds_total - checkpoint_at
            model_part2 = xgb.train(params, dtrain, num_boost_round=remaining_rounds,
                                  xgb_model=checkpoint_path)
            
            resumed_pred = model_part2.predict(dtrain)
            resumed_mse = mean_squared_error(y, resumed_pred)
            
            # Check that MSE is consistent (no accuracy drop)
            mse_diff = abs(reference_mse - resumed_mse)
            
            # This should be very small with proper RNG state preservation
            assert mse_diff < 1e-10, \
                f"Accuracy drop detected with checkpoint at round {checkpoint_at}: " \
                f"Reference MSE: {reference_mse:.10f}, Resumed MSE: {resumed_mse:.10f}, " \
                f"Difference: {mse_diff:.10f}"
            
            # Also check prediction-level consistency
            pred_diff = np.mean(np.abs(reference_pred - resumed_pred))
            assert pred_diff < 1e-10, \
                f"Prediction divergence with checkpoint at round {checkpoint_at}: {pred_diff}"

if __name__ == "__main__":
    test = TestCheckpointRNGConsistency()
    test.test_checkpoint_rng_consistency_single_thread()
    test.test_rng_state_serialization_deserialization()
    test.test_subsample_reproducibility_with_checkpoint()
    test.test_distributed_checkpoint_consistency_simulation()
    print("All tests passed! ✅")