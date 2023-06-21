"""Loading a pickled model generated by test_pickling.py, only used by
`test_gpu_with_dask.py`"""
import json
import os

import numpy as np
import pytest
from test_gpu_pickling import build_dataset, load_pickle, model_path

import xgboost as xgb
from xgboost import testing as tm


class TestLoadPickle:
    def test_load_pkl(self) -> None:
        """Test whether prediction is correct."""
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
        bst = load_pickle(model_path)
        x, y = build_dataset()
        if isinstance(bst, xgb.Booster):
            test_x = xgb.DMatrix(x)
            res = bst.predict(test_x)
        else:
            res = bst.predict(x)
            assert len(res) == 10
            bst.set_params(n_jobs=1)  # triggers a re-configuration
            res = bst.predict(x)

        assert len(res) == 10

    def test_context_is_removed(self) -> None:
        """Under invalid CUDA_VISIBLE_DEVICES, context should reset"""
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
        bst = load_pickle(model_path)
        config = bst.save_config()
        config = json.loads(config)
        assert config["learner"]["generic_param"]["gpu_id"] == "-1"

    def test_context_is_preserved(self) -> None:
        """Test the device context is preserved after pickling."""
        assert "CUDA_VISIBLE_DEVICES" not in os.environ.keys()
        bst = load_pickle(model_path)
        config = bst.save_config()
        config = json.loads(config)
        assert config["learner"]["generic_param"]["gpu_id"] == "0"

    def test_wrap_gpu_id(self) -> None:
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
        bst = load_pickle(model_path)
        config = bst.save_config()
        config = json.loads(config)
        assert config["learner"]["generic_param"]["device"] == "cuda:0"

        x, y = build_dataset()
        test_x = xgb.DMatrix(x)
        res = bst.predict(test_x)
        assert len(res) == 10

    def test_training_on_cpu_only_env(self) -> None:
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
        rng = np.random.RandomState(1994)
        X = rng.randn(10, 10)
        y = rng.randn(10)
        with tm.captured_output() as (out, err):
            # Test no thrust exception is thrown
            with pytest.raises(xgb.core.XGBoostError):
                xgb.train({"tree_method": "gpu_hist"}, xgb.DMatrix(X, y))

            assert out.getvalue().find("No visible GPU is found") != -1
