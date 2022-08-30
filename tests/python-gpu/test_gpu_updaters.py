from typing import Dict, Any
import numpy as np
import sys
import gc
import pytest
import xgboost as xgb
from hypothesis import given, strategies, assume, settings, note

sys.path.append("tests/python")
import testing as tm
import test_updaters as test_up


parameter_strategy = strategies.fixed_dictionaries({
    'max_depth': strategies.integers(0, 11),
    'max_leaves': strategies.integers(0, 256),
    'max_bin': strategies.integers(2, 1024),
    'grow_policy': strategies.sampled_from(['lossguide', 'depthwise']),
    'min_child_weight': strategies.floats(0.5, 2.0),
    'seed': strategies.integers(0, 10),
    # We cannot enable subsampling as the training loss can increase
    # 'subsample': strategies.floats(0.5, 1.0),
    'colsample_bytree': strategies.floats(0.5, 1.0),
    'colsample_bylevel': strategies.floats(0.5, 1.0),
}).filter(lambda x: (x['max_depth'] > 0 or x['max_leaves'] > 0) and (
    x['max_depth'] > 0 or x['grow_policy'] == 'lossguide'))


def train_result(param, dmat: xgb.DMatrix, num_rounds: int) -> dict:
    result: xgb.callback.TrainingCallback.EvalsLog = {}
    booster = xgb.train(
        param,
        dmat,
        num_rounds,
        [(dmat, "train")],
        verbose_eval=False,
        evals_result=result,
    )
    assert booster.num_features() == dmat.num_col()
    assert booster.num_boosted_rounds() == num_rounds

    return result


class TestGPUUpdaters:
    cputest = test_up.TestTreeMethod()

    @given(parameter_strategy, strategies.integers(1, 20), tm.dataset_strategy)
    @settings(deadline=None, print_blob=True)
    def test_gpu_hist(self, param, num_rounds, dataset):
        param["tree_method"] = "gpu_hist"
        param = dataset.set_params(param)
        result = train_result(param, dataset.get_dmat(), num_rounds)
        note(result)
        assert tm.non_increasing(result["train"][dataset.metric])

    @given(tm.sparse_datasets_strategy)
    @settings(deadline=None, print_blob=True)
    def test_sparse(self, dataset):
        param = {"tree_method": "hist", "max_bin": 64}
        hist_result = train_result(param, dataset.get_dmat(), 16)
        note(hist_result)
        assert tm.non_increasing(hist_result['train'][dataset.metric])

        param = {"tree_method": "gpu_hist", "max_bin": 64}
        gpu_hist_result = train_result(param, dataset.get_dmat(), 16)
        note(gpu_hist_result)
        assert tm.non_increasing(gpu_hist_result['train'][dataset.metric])

        np.testing.assert_allclose(
            hist_result["train"]["rmse"], gpu_hist_result["train"]["rmse"], rtol=1e-2
        )

    @given(strategies.integers(10, 400), strategies.integers(3, 8),
           strategies.integers(1, 2), strategies.integers(4, 7))
    @settings(deadline=None, print_blob=True)
    @pytest.mark.skipif(**tm.no_pandas())
    def test_categorical_ohe(self, rows, cols, rounds, cats):
        self.cputest.run_categorical_ohe(rows, cols, rounds, cats, "gpu_hist")

    @given(
        tm.categorical_dataset_strategy,
        test_up.exact_parameter_strategy,
        test_up.hist_parameter_strategy,
        test_up.cat_parameter_strategy,
        strategies.integers(4, 32),
    )
    @settings(deadline=None, print_blob=True)
    @pytest.mark.skipif(**tm.no_pandas())
    def test_categorical(
        self,
        dataset: tm.TestDataset,
        exact_parameters: Dict[str, Any],
        hist_parameters: Dict[str, Any],
        cat_parameters: Dict[str, Any],
        n_rounds: int,
    ) -> None:
        cat_parameters.update(exact_parameters)
        cat_parameters.update(hist_parameters)
        cat_parameters["tree_method"] = "gpu_hist"

        results = train_result(cat_parameters, dataset.get_dmat(), n_rounds)
        tm.non_increasing(results["train"]["rmse"])

    @given(
        test_up.hist_parameter_strategy,
        test_up.cat_parameter_strategy,
    )
    @settings(deadline=None, print_blob=True)
    def test_categorical_ames_housing(
        self,
        hist_parameters: Dict[str, Any],
        cat_parameters: Dict[str, Any],
    ) -> None:
        cat_parameters.update(hist_parameters)
        dataset = tm.TestDataset(
            "ames_housing", tm.get_ames_housing, "reg:squarederror", "rmse"
        )
        cat_parameters["tree_method"] = "gpu_hist"
        results = train_result(cat_parameters, dataset.get_dmat(), 16)
        tm.non_increasing(results["train"]["rmse"])

    @given(
        strategies.integers(10, 400),
        strategies.integers(3, 8),
        strategies.integers(4, 7)
    )
    @settings(deadline=None, print_blob=True)
    @pytest.mark.skipif(**tm.no_pandas())
    def test_categorical_missing(self, rows, cols, cats):
        self.cputest.run_categorical_missing(rows, cols, cats, "gpu_hist")

    @pytest.mark.skipif(**tm.no_pandas())
    def test_max_cat(self) -> None:
        self.cputest.run_max_cat("gpu_hist")

    def test_categorical_32_cat(self):
        '''32 hits the bound of integer bitset, so special test'''
        rows = 1000
        cols = 10
        cats = 32
        rounds = 4
        self.cputest.run_categorical_ohe(rows, cols, rounds, cats, "gpu_hist")

    @pytest.mark.skipif(**tm.no_cupy())
    def test_invalid_category(self):
        self.cputest.run_invalid_category("gpu_hist")

    @pytest.mark.skipif(**tm.no_cupy())
    @given(parameter_strategy, strategies.integers(1, 20),
           tm.dataset_strategy)
    @settings(deadline=None, print_blob=True)
    def test_gpu_hist_device_dmatrix(self, param, num_rounds, dataset):
        # We cannot handle empty dataset yet
        assume(len(dataset.y) > 0)
        param['tree_method'] = 'gpu_hist'
        param = dataset.set_params(param)
        result = train_result(param, dataset.get_device_dmat(), num_rounds)
        note(result)
        assert tm.non_increasing(result['train'][dataset.metric], tolerance=1e-3)

    @given(parameter_strategy, strategies.integers(1, 20),
           tm.dataset_strategy)
    @settings(deadline=None, print_blob=True)
    def test_external_memory(self, param, num_rounds, dataset):
        if dataset.name.endswith("-l1"):
            return
        # We cannot handle empty dataset yet
        assume(len(dataset.y) > 0)
        param['tree_method'] = 'gpu_hist'
        param = dataset.set_params(param)
        m = dataset.get_external_dmat()
        external_result = train_result(param, m, num_rounds)
        del m
        gc.collect()
        assert tm.non_increasing(external_result['train'][dataset.metric])

    def test_empty_dmatrix_prediction(self):
        # FIXME(trivialfis): This should be done with all updaters
        kRows = 0
        kCols = 100

        X = np.empty((kRows, kCols))
        y = np.empty((kRows))

        dtrain = xgb.DMatrix(X, y)

        bst = xgb.train({'verbosity': 2,
                         'tree_method': 'gpu_hist',
                         'gpu_id': 0},
                        dtrain,
                        verbose_eval=True,
                        num_boost_round=6,
                        evals=[(dtrain, 'Train')])

        kRows = 100
        X = np.random.randn(kRows, kCols)

        dtest = xgb.DMatrix(X)
        predictions = bst.predict(dtest)
        np.testing.assert_allclose(predictions, 0.5, 1e-6)

    @pytest.mark.mgpu
    @given(tm.dataset_strategy, strategies.integers(0, 10))
    @settings(deadline=None, max_examples=10, print_blob=True)
    def test_specified_gpu_id_gpu_update(self, dataset, gpu_id):
        param = {'tree_method': 'gpu_hist', 'gpu_id': gpu_id}
        param = dataset.set_params(param)
        result = train_result(param, dataset.get_dmat(), 10)
        assert tm.non_increasing(result['train'][dataset.metric])
