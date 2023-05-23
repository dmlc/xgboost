import sys

import numpy as np
import pytest
from hypothesis import assume, given, settings, strategies

import xgboost as xgb
from xgboost import testing as tm
from xgboost.compat import PANDAS_INSTALLED

if PANDAS_INSTALLED:
    from hypothesis.extra.pandas import column, data_frames, range_indexes
else:
    def noop(*args, **kwargs):
        pass
    column, data_frames, range_indexes = noop, noop, noop

sys.path.append("tests/python")
from test_predict import run_predict_leaf  # noqa
from test_predict import run_threaded_predict  # noqa

rng = np.random.RandomState(1994)

shap_parameter_strategy = strategies.fixed_dictionaries({
    'max_depth': strategies.integers(1, 11),
    'max_leaves': strategies.integers(0, 256),
    'num_parallel_tree': strategies.sampled_from([1, 10]),
}).filter(lambda x: x['max_depth'] > 0 or x['max_leaves'] > 0)

predict_parameter_strategy = strategies.fixed_dictionaries({
    'max_depth': strategies.integers(1, 8),
    'num_parallel_tree': strategies.sampled_from([1, 4]),
})

pytestmark = tm.timeout(20)


class TestGPUPredict:
    def test_predict(self):
        iterations = 10
        np.random.seed(1)
        test_num_rows = [10, 1000, 5000]
        test_num_cols = [10, 50, 500]
        # This test passes for tree_method=gpu_hist and tree_method=exact. but
        # for `hist` and `approx` the floating point error accumulates faster
        # and fails even tol is set to 1e-4.  For `hist`, the mismatching rate
        # with 5000 rows is 0.04.
        for num_rows in test_num_rows:
            for num_cols in test_num_cols:
                dtrain = xgb.DMatrix(np.random.randn(num_rows, num_cols),
                                     label=[0, 1] * int(num_rows / 2))
                dval = xgb.DMatrix(np.random.randn(num_rows, num_cols),
                                   label=[0, 1] * int(num_rows / 2))
                dtest = xgb.DMatrix(np.random.randn(num_rows, num_cols),
                                    label=[0, 1] * int(num_rows / 2))
                watchlist = [(dtrain, 'train'), (dval, 'validation')]
                res = {}
                param = {
                    "objective": "binary:logistic",
                    "predictor": "gpu_predictor",
                    'eval_metric': 'logloss',
                    'tree_method': 'gpu_hist',
                    'max_depth': 1
                }
                bst = xgb.train(param, dtrain, iterations, evals=watchlist,
                                evals_result=res)
                assert self.non_increasing(res["train"]["logloss"])
                gpu_pred_train = bst.predict(dtrain, output_margin=True)
                gpu_pred_test = bst.predict(dtest, output_margin=True)
                gpu_pred_val = bst.predict(dval, output_margin=True)

                param["predictor"] = "cpu_predictor"
                bst_cpu = xgb.train(param, dtrain, iterations, evals=watchlist)
                cpu_pred_train = bst_cpu.predict(dtrain, output_margin=True)
                cpu_pred_test = bst_cpu.predict(dtest, output_margin=True)
                cpu_pred_val = bst_cpu.predict(dval, output_margin=True)

                np.testing.assert_allclose(cpu_pred_train, gpu_pred_train,
                                           rtol=1e-6)
                np.testing.assert_allclose(cpu_pred_val, gpu_pred_val,
                                           rtol=1e-6)
                np.testing.assert_allclose(cpu_pred_test, gpu_pred_test,
                                           rtol=1e-6)

    def non_increasing(self, L):
        return all((y - x) < 0.001 for x, y in zip(L, L[1:]))

    # Test case for a bug where multiple batch predictions made on a
    # test set produce incorrect results
    @pytest.mark.skipif(**tm.no_sklearn())
    def test_multi_predict(self):
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split

        n = 1000
        X, y = make_regression(n, random_state=rng)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=123)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)

        params = {}
        params["tree_method"] = "gpu_hist"

        params['predictor'] = "gpu_predictor"
        bst_gpu_predict = xgb.train(params, dtrain)

        params['predictor'] = "cpu_predictor"
        bst_cpu_predict = xgb.train(params, dtrain)

        predict0 = bst_gpu_predict.predict(dtest)
        predict1 = bst_gpu_predict.predict(dtest)
        cpu_predict = bst_cpu_predict.predict(dtest)

        assert np.allclose(predict0, predict1)
        assert np.allclose(predict0, cpu_predict)

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_sklearn(self):
        m, n = 15000, 14
        tr_size = 2500
        X = np.random.rand(m, n)
        y = 200 * np.matmul(X, np.arange(-3, -3 + n))
        X_train, y_train = X[:tr_size, :], y[:tr_size]
        X_test, y_test = X[tr_size:, :], y[tr_size:]

        # First with cpu_predictor
        params = {'tree_method': 'gpu_hist',
                  'predictor': 'cpu_predictor',
                  'n_jobs': -1,
                  'seed': 123}
        m = xgb.XGBRegressor(**params).fit(X_train, y_train)
        cpu_train_score = m.score(X_train, y_train)
        cpu_test_score = m.score(X_test, y_test)

        # Now with gpu_predictor
        params['predictor'] = 'gpu_predictor'

        m = xgb.XGBRegressor(**params).fit(X_train, y_train)
        gpu_train_score = m.score(X_train, y_train)
        gpu_test_score = m.score(X_test, y_test)

        assert np.allclose(cpu_train_score, gpu_train_score)
        assert np.allclose(cpu_test_score, gpu_test_score)

    def run_inplace_base_margin(self, booster, dtrain, X, base_margin):
        import cupy as cp
        dtrain.set_info(base_margin=base_margin)
        from_inplace = booster.inplace_predict(data=X, base_margin=base_margin)
        from_dmatrix = booster.predict(dtrain)
        cp.testing.assert_allclose(from_inplace, from_dmatrix)

    def run_inplace_predict_cupy(self, device: int) -> None:
        import cupy as cp
        cp.cuda.runtime.setDevice(device)
        rows = 1000
        cols = 10
        missing = 11            # set to integer for testing

        cp_rng = cp.random.RandomState(1994)
        cp.random.set_random_state(cp_rng)

        X = cp.random.randn(rows, cols)
        missing_idx = [i for i in range(0, cols, 4)]
        X[:, missing_idx] = missing  # set to be missing
        y = cp.random.randn(rows)

        dtrain = xgb.DMatrix(X, y)

        booster = xgb.train(
            {'tree_method': 'gpu_hist', "gpu_id": device}, dtrain, num_boost_round=10
        )

        test = xgb.DMatrix(X[:10, ...], missing=missing)
        predt_from_array = booster.inplace_predict(X[:10, ...], missing=missing)
        predt_from_dmatrix = booster.predict(test)
        cp.testing.assert_allclose(predt_from_array, predt_from_dmatrix)

        def predict_dense(x):
            cp.cuda.runtime.setDevice(device)
            inplace_predt = booster.inplace_predict(x)
            d = xgb.DMatrix(x)
            copied_predt = cp.array(booster.predict(d))
            return cp.all(copied_predt == inplace_predt)

        # Don't do this on Windows, see issue #5793
        if sys.platform.startswith("win"):
            pytest.skip(
                'Multi-threaded in-place prediction with cuPy is not working on Windows'
            )
        for i in range(10):
            run_threaded_predict(X, rows, predict_dense)

        base_margin = cp_rng.randn(rows)
        self.run_inplace_base_margin(booster, dtrain, X, base_margin)

        # Create a wide dataset
        X = cp_rng.randn(100, 10000)
        y = cp_rng.randn(100)

        missing_idx = [i for i in range(0, X.shape[1], 16)]
        X[:, missing_idx] = missing
        reg = xgb.XGBRegressor(
            tree_method="gpu_hist", n_estimators=8, missing=missing, gpu_id=device
        )
        reg.fit(X, y)

        gpu_predt = reg.predict(X)
        reg.set_params(predictor="cpu_predictor")
        cpu_predt = reg.predict(X)
        np.testing.assert_allclose(gpu_predt, cpu_predt, atol=1e-6)
        cp.cuda.runtime.setDevice(0)

    @pytest.mark.skipif(**tm.no_cupy())
    def test_inplace_predict_cupy(self):
        self.run_inplace_predict_cupy(0)

    @pytest.mark.xfail
    @pytest.mark.skipif(**tm.no_cupy())
    @pytest.mark.mgpu
    def test_inplace_predict_cupy_specified_device(self):
        import cupy as cp
        n_devices = cp.cuda.runtime.getDeviceCount()
        for d in range(n_devices):
            self.run_inplace_predict_cupy(d)

    @pytest.mark.skipif(**tm.no_cupy())
    @pytest.mark.skipif(**tm.no_cudf())
    def test_inplace_predict_cudf(self):
        import cudf
        import cupy as cp
        import pandas as pd
        rows = 1000
        cols = 10
        rng = np.random.RandomState(1994)
        cp.cuda.runtime.setDevice(0)
        X = rng.randn(rows, cols)
        X = pd.DataFrame(X)
        y = rng.randn(rows)
        X = cudf.from_pandas(X)

        dtrain = xgb.DMatrix(X, y)

        booster = xgb.train({'tree_method': 'gpu_hist'},
                            dtrain, num_boost_round=10)
        test = xgb.DMatrix(X)
        predt_from_array = booster.inplace_predict(X)
        predt_from_dmatrix = booster.predict(test)

        cp.testing.assert_allclose(predt_from_array, predt_from_dmatrix)

        def predict_df(x):
            # column major array
            inplace_predt = booster.inplace_predict(x.values)
            d = xgb.DMatrix(x)
            copied_predt = cp.array(booster.predict(d))
            assert cp.all(copied_predt == inplace_predt)

            inplace_predt = booster.inplace_predict(x)
            return cp.all(copied_predt == inplace_predt)

        for i in range(10):
            run_threaded_predict(X, rows, predict_df)

        base_margin = cudf.Series(rng.randn(rows))
        self.run_inplace_base_margin(booster, dtrain, X, base_margin)

    @given(
        strategies.integers(1, 10), tm.make_dataset_strategy(), shap_parameter_strategy
    )
    @settings(deadline=None, max_examples=20, print_blob=True)
    def test_shap(self, num_rounds, dataset, param):
        if dataset.name.endswith("-l1"):  # not supported by the exact tree method
            return
        param.update({"predictor": "gpu_predictor", "gpu_id": 0})
        param = dataset.set_params(param)
        dmat = dataset.get_dmat()
        bst = xgb.train(param, dmat, num_rounds)
        test_dmat = xgb.DMatrix(dataset.X, dataset.y, dataset.w, dataset.margin)
        shap = bst.predict(test_dmat, pred_contribs=True)
        margin = bst.predict(test_dmat, output_margin=True)
        assume(len(dataset.y) > 0)
        assert np.allclose(np.sum(shap, axis=len(shap.shape) - 1), margin, 1e-3, 1e-3)

    @given(
        strategies.integers(1, 10), tm.make_dataset_strategy(), shap_parameter_strategy
    )
    @settings(deadline=None, max_examples=10, print_blob=True)
    def test_shap_interactions(self, num_rounds, dataset, param):
        if dataset.name.endswith("-l1"):  # not supported by the exact tree method
            return
        param.update({"predictor": "gpu_predictor", "gpu_id": 0})
        param = dataset.set_params(param)
        dmat = dataset.get_dmat()
        bst = xgb.train(param, dmat, num_rounds)
        test_dmat = xgb.DMatrix(dataset.X, dataset.y, dataset.w, dataset.margin)
        shap = bst.predict(test_dmat, pred_interactions=True)
        margin = bst.predict(test_dmat, output_margin=True)
        assume(len(dataset.y) > 0)
        assert np.allclose(np.sum(shap, axis=(len(shap.shape) - 1, len(shap.shape) - 2)),
                           margin,
                           1e-3, 1e-3)

    def test_shap_categorical(self):
        X, y = tm.make_categorical(100, 20, 7, False)
        Xy = xgb.DMatrix(X, y, enable_categorical=True)
        booster = xgb.train({"tree_method": "gpu_hist"}, Xy, num_boost_round=10)

        booster.set_param({"predictor": "gpu_predictor"})
        shap = booster.predict(Xy, pred_contribs=True)
        margin = booster.predict(Xy, output_margin=True)
        np.testing.assert_allclose(
            np.sum(shap, axis=len(shap.shape) - 1), margin, rtol=1e-3
        )

        booster.set_param({"predictor": "cpu_predictor"})
        shap = booster.predict(Xy, pred_contribs=True)
        margin = booster.predict(Xy, output_margin=True)
        np.testing.assert_allclose(
            np.sum(shap, axis=len(shap.shape) - 1), margin, rtol=1e-3
        )

    def test_predict_leaf_basic(self):
        gpu_leaf = run_predict_leaf('gpu_predictor')
        cpu_leaf = run_predict_leaf('cpu_predictor')
        np.testing.assert_equal(gpu_leaf, cpu_leaf)

    def run_predict_leaf_booster(self, param, num_rounds, dataset):
        param = dataset.set_params(param)
        m = dataset.get_dmat()
        booster = xgb.train(param, dtrain=dataset.get_dmat(), num_boost_round=num_rounds)
        booster.set_param({'predictor': 'cpu_predictor'})
        cpu_leaf = booster.predict(m, pred_leaf=True)

        booster.set_param({'predictor': 'gpu_predictor'})
        gpu_leaf = booster.predict(m, pred_leaf=True)

        np.testing.assert_equal(cpu_leaf, gpu_leaf)

    @given(predict_parameter_strategy, tm.make_dataset_strategy())
    @settings(deadline=None, max_examples=20, print_blob=True)
    def test_predict_leaf_gbtree(self, param, dataset):
        # Unsupported for random forest
        if param.get("num_parallel_tree", 1) > 1 and dataset.name.endswith("-l1"):
            return

        param['booster'] = 'gbtree'
        param['tree_method'] = 'gpu_hist'
        self.run_predict_leaf_booster(param, 10, dataset)

    @given(predict_parameter_strategy, tm.make_dataset_strategy())
    @settings(deadline=None, max_examples=20, print_blob=True)
    def test_predict_leaf_dart(self, param: dict, dataset: tm.TestDataset) -> None:
        # Unsupported for random forest
        if param.get("num_parallel_tree", 1) > 1 and dataset.name.endswith("-l1"):
            return

        param['booster'] = 'dart'
        param['tree_method'] = 'gpu_hist'
        self.run_predict_leaf_booster(param, 10, dataset)

    @pytest.mark.skipif(**tm.no_sklearn())
    @pytest.mark.skipif(**tm.no_pandas())
    @given(df=data_frames([column('x0', elements=strategies.integers(min_value=0, max_value=3)),
                           column('x1', elements=strategies.integers(min_value=0, max_value=5))],
                          index=range_indexes(min_size=20, max_size=50)))
    @settings(deadline=None, max_examples=20, print_blob=True)
    def test_predict_categorical_split(self, df):
        from sklearn.metrics import mean_squared_error

        df = df.astype('category')
        x0, x1 = df['x0'].to_numpy(), df['x1'].to_numpy()
        y = (x0 * 10 - 20) + (x1 - 2)
        dtrain = xgb.DMatrix(df, label=y, enable_categorical=True)

        params = {
            'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor',
            'max_depth': 3, 'learning_rate': 1.0, 'base_score': 0.0, 'eval_metric': 'rmse'
        }

        eval_history = {}
        bst = xgb.train(params, dtrain, num_boost_round=5, evals=[(dtrain, 'train')],
                        verbose_eval=False, evals_result=eval_history)

        pred = bst.predict(dtrain)
        rmse = mean_squared_error(y_true=y, y_pred=pred, squared=False)
        np.testing.assert_almost_equal(rmse, eval_history['train']['rmse'][-1], decimal=5)

    @pytest.mark.skipif(**tm.no_cupy())
    @pytest.mark.parametrize("n_classes", [2, 3])
    def test_predict_dart(self, n_classes):
        import cupy as cp
        from sklearn.datasets import make_classification
        n_samples = 1000
        X_, y_ = make_classification(
            n_samples=n_samples, n_informative=5, n_classes=n_classes
        )
        X, y = cp.array(X_), cp.array(y_)

        Xy = xgb.DMatrix(X, y)
        if n_classes == 2:
            params = {
                "tree_method": "gpu_hist",
                "booster": "dart",
                "rate_drop": 0.5,
                "objective": "binary:logistic"
            }
        else:
            params = {
                "tree_method": "gpu_hist",
                "booster": "dart",
                "rate_drop": 0.5,
                "objective": "multi:softprob",
                "num_class": n_classes
            }

        booster = xgb.train(params, Xy, num_boost_round=32)
        # predictor=auto
        inplace = booster.inplace_predict(X)
        copied = booster.predict(Xy)
        cpu_inplace = booster.inplace_predict(X_)
        booster.set_param({"predictor": "cpu_predictor"})
        cpu_copied = booster.predict(Xy)

        copied = cp.array(copied)
        cp.testing.assert_allclose(cpu_inplace, copied, atol=1e-6)
        cp.testing.assert_allclose(cpu_copied, copied, atol=1e-6)
        cp.testing.assert_allclose(inplace, copied, atol=1e-6)

        booster.set_param({"predictor": "gpu_predictor"})
        inplace = booster.inplace_predict(X)
        copied = booster.predict(Xy)

        copied = cp.array(copied)
        cp.testing.assert_allclose(inplace, copied, atol=1e-6)

    @pytest.mark.skipif(**tm.no_cupy())
    def test_dtypes(self):
        import cupy as cp
        rows = 1000
        cols = 10
        rng = cp.random.RandomState(1994)
        orig = rng.randint(low=0, high=127, size=rows * cols).reshape(
            rows, cols
        )
        y = rng.randint(low=0, high=127, size=rows)
        dtrain = xgb.DMatrix(orig, label=y)
        booster = xgb.train({"tree_method": "gpu_hist"}, dtrain)

        predt_orig = booster.inplace_predict(orig)
        # all primitive types in numpy
        for dtype in [
            cp.signedinteger,
            cp.byte,
            cp.short,
            cp.intc,
            cp.int_,
            cp.longlong,
            cp.unsignedinteger,
            cp.ubyte,
            cp.ushort,
            cp.uintc,
            cp.uint,
            cp.ulonglong,
            cp.floating,
            cp.half,
            cp.single,
            cp.double,
        ]:
            X = cp.array(orig, dtype=dtype)
            predt = booster.inplace_predict(X)
            cp.testing.assert_allclose(predt, predt_orig)

        # boolean
        orig = cp.random.binomial(1, 0.5, size=rows * cols).reshape(
            rows, cols
        )
        predt_orig = booster.inplace_predict(orig)
        for dtype in [cp.bool8, cp.bool_]:
            X = cp.array(orig, dtype=dtype)
            predt = booster.inplace_predict(X)
            cp.testing.assert_allclose(predt, predt_orig)

        # unsupported types
        for dtype in [
            cp.complex64,
            cp.complex128,
        ]:
            X = cp.array(orig, dtype=dtype)
            with pytest.raises(ValueError):
                booster.inplace_predict(X)
