import numpy as np
import sys
import gc
import pytest
import xgboost as xgb
from hypothesis import given, strategies, assume, settings, note

sys.path.append("tests/python")
import testing as tm

parameter_strategy = strategies.fixed_dictionaries({
    'max_depth': strategies.integers(0, 11),
    'max_leaves': strategies.integers(0, 256),
    'max_bin': strategies.integers(2, 1024),
    'grow_policy': strategies.sampled_from(['lossguide', 'depthwise']),
    'single_precision_histogram': strategies.booleans(),
    'min_child_weight': strategies.floats(0.5, 2.0),
    'seed': strategies.integers(0, 10),
    # We cannot enable subsampling as the training loss can increase
    # 'subsample': strategies.floats(0.5, 1.0),
    'colsample_bytree': strategies.floats(0.5, 1.0),
    'colsample_bylevel': strategies.floats(0.5, 1.0),
}).filter(lambda x: (x['max_depth'] > 0 or x['max_leaves'] > 0) and (
    x['max_depth'] > 0 or x['grow_policy'] == 'lossguide'))


def train_result(param, dmat, num_rounds):
    result = {}
    xgb.train(param, dmat, num_rounds, [(dmat, 'train')], verbose_eval=False,
              evals_result=result)
    return result


class TestGPUUpdaters:
    @given(parameter_strategy, strategies.integers(1, 20), tm.dataset_strategy)
    @settings(deadline=None)
    def test_gpu_hist(self, param, num_rounds, dataset):
        param["tree_method"] = "gpu_hist"
        param = dataset.set_params(param)
        result = train_result(param, dataset.get_dmat(), num_rounds)
        note(result)
        assert tm.non_increasing(result["train"][dataset.metric])

    def run_categorical_basic(self, rows, cols, rounds, cats):
        onehot, label = tm.make_categorical(rows, cols, cats, True)
        cat, _ = tm.make_categorical(rows, cols, cats, False)

        by_etl_results = {}
        by_builtin_results = {}

        parameters = {"tree_method": "gpu_hist", "predictor": "gpu_predictor"}

        m = xgb.DMatrix(onehot, label, enable_categorical=False)
        xgb.train(
            parameters,
            m,
            num_boost_round=rounds,
            evals=[(m, "Train")],
            evals_result=by_etl_results,
        )

        m = xgb.DMatrix(cat, label, enable_categorical=True)
        xgb.train(
            parameters,
            m,
            num_boost_round=rounds,
            evals=[(m, "Train")],
            evals_result=by_builtin_results,
        )

        # There are guidelines on how to specify tolerance based on considering output as
        # random variables. But in here the tree construction is extremely sensitive to
        # floating point errors. An 1e-5 error in a histogram bin can lead to an entirely
        # different tree.  So even though the test is quite lenient, hypothesis can still
        # pick up falsifying examples from time to time.
        np.testing.assert_allclose(
            np.array(by_etl_results["Train"]["rmse"]),
            np.array(by_builtin_results["Train"]["rmse"]),
            rtol=1e-3,
        )
        assert tm.non_increasing(by_builtin_results["Train"]["rmse"])

    @given(strategies.integers(10, 400), strategies.integers(3, 8),
           strategies.integers(1, 2), strategies.integers(4, 7))
    @settings(deadline=None)
    @pytest.mark.skipif(**tm.no_pandas())
    def test_categorical(self, rows, cols, rounds, cats):
        self.run_categorical_basic(rows, cols, rounds, cats)

    def test_categorical_32_cat(self):
        '''32 hits the bound of integer bitset, so special test'''
        rows = 1000
        cols = 10
        cats = 32
        rounds = 4
        self.run_categorical_basic(rows, cols, rounds, cats)

    @pytest.mark.skipif(**tm.no_cupy())
    @given(parameter_strategy, strategies.integers(1, 20),
           tm.dataset_strategy)
    @settings(deadline=None)
    def test_gpu_hist_device_dmatrix(self, param, num_rounds, dataset):
        # We cannot handle empty dataset yet
        assume(len(dataset.y) > 0)
        param['tree_method'] = 'gpu_hist'
        param = dataset.set_params(param)
        result = train_result(param, dataset.get_device_dmat(), num_rounds)
        note(result)
        assert tm.non_increasing(result['train'][dataset.metric])

    @given(parameter_strategy, strategies.integers(1, 20),
           tm.dataset_strategy)
    @settings(deadline=None)
    def test_external_memory(self, param, num_rounds, dataset):
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
    @settings(deadline=None, max_examples=10)
    def test_specified_gpu_id_gpu_update(self, dataset, gpu_id):
        param = {'tree_method': 'gpu_hist', 'gpu_id': gpu_id}
        param = dataset.set_params(param)
        result = train_result(param, dataset.get_dmat(), 10)
        assert tm.non_increasing(result['train'][dataset.metric])
