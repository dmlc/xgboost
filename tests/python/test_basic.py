import json
import os
import pathlib
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

import xgboost as xgb
from xgboost import testing as tm
from xgboost.core import Booster

dpath = 'demo/data/'
rng = np.random.RandomState(1994)


class TestBasic:
    def test_compat(self):
        from xgboost.compat import lazy_isinstance
        a: np.ndarray = np.array([1, 2, 3])
        assert lazy_isinstance(a, 'numpy', 'ndarray')
        assert not lazy_isinstance(a, 'numpy', 'dataframe')

    def test_basic(self):
        dtrain, dtest = tm.load_agaricus(__file__)
        param: Dict = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
        # specify validations set to watch performance
        watchlist: List[Tuple] = [(dtrain, 'train')]
        num_round: int = 2
        bst: Booster = xgb.train(param, dtrain, num_round, watchlist, verbose_eval=True)

        train_preds: np.ndarray = bst.predict(dtrain)
        train_labels: np.ndarray = dtrain.get_label()
        train_err: float = sum(1 for i in range(len(train_preds))
                  if int(train_preds[i] > 0.5) != train_labels[i]) / float(len(train_preds))
        # error must be smaller than 10%
        assert train_err < 0.1

        test_preds: np.ndarray = bst.predict(dtest)
        test_labels: np.ndarray = dtest.get_label()
        test_err: float = sum(1 for i in range(len(test_preds))
                  if int(test_preds[i] > 0.5) != test_labels[i]) / float(len(test_preds))
        # error must be smaller than 10%
        assert test_err < 0.1

        with tempfile.TemporaryDirectory() as tmpdir:
            dtest_path = os.path.join(tmpdir, 'dtest.dmatrix')
            # save dmatrix into binary buffer
            dtest.save_binary(dtest_path)
            # save model
            model_path = os.path.join(tmpdir, 'model.booster')
            bst.save_model(model_path)
            # load model and data in
            bst2: Booster = xgb.Booster(model_file=model_path)
            dtest2: xgb.DMatrix = xgb.DMatrix(dtest_path)
            preds2: np.ndarray = bst2.predict(dtest2)
            # assert they are the same
            assert np.sum(np.abs(preds2 - test_preds)) == 0

    def test_metric_config(self):
        # Make sure that the metric configuration happens in booster so the
        # string `['error', 'auc']` doesn't get passed down to core.
        dtrain, dtest = tm.load_agaricus(__file__)
        param: Dict = {'max_depth': 2, 'eta': 1, 'verbosity': 0,
                 'objective': 'binary:logistic', 'eval_metric': ['error', 'auc']}
        watchlist: List[Tuple] = [(dtest, 'eval'), (dtrain, 'train')]
        num_round: int = 2
        booster_0: Booster = xgb.train(param, dtrain, num_round, watchlist)
        predt_0: np.ndarray = booster_0.predict(dtrain)
        with tempfile.TemporaryDirectory() as tmpdir:
            path: str = os.path.join(tmpdir, 'model.json')
            booster_0.save_model(path)

            booster_1: Booster = xgb.Booster(params=param, model_file=path)
            predt_1: np.ndarray = booster_1.predict(dtrain)
            np.testing.assert_allclose(predt_0, predt_1)

    def test_multiclass(self):
        dtrain, dtest = tm.load_agaricus(__file__)
        param: Dict = {'max_depth': 2, 'eta': 1, 'verbosity': 0, 'num_class': 2}
        # specify validations set to watch performance
        watchlist: List[Tuple] = [(dtest, 'eval'), (dtrain, 'train')]
        num_round: int = 2
        bst: Booster = xgb.train(param, dtrain, num_round, watchlist)
        # this is prediction
        preds: np.ndarray = bst.predict(dtest)
        labels: np.ndarray = dtest.get_label()
        err: float = sum(1 for i in range(len(preds))
                  if preds[i] != labels[i]) / float(len(preds))
        # error must be smaller than 10%
        assert err < 0.1

        with tempfile.TemporaryDirectory() as tmpdir:
            dtest_path: str = os.path.join(tmpdir, 'dtest.buffer')
            model_path: str = os.path.join(tmpdir, 'xgb.model')
            # save dmatrix into binary buffer
            dtest.save_binary(dtest_path)
            # save model
            bst.save_model(model_path)
            # load model and data in
            bst2: Booster = xgb.Booster(model_file=model_path)
            dtest2: xgb.DMatrix = xgb.DMatrix(dtest_path)
            preds2: np.ndarray = bst2.predict(dtest2)
            # assert they are the same
            assert np.sum(np.abs(preds2 - preds)) == 0

    def test_dump(self):
        data: np.ndarray = np.random.randn(100, 2)
        target: np.ndarray = np.array([0, 1] * 50)
        features: List[str] = ['Feature1', 'Feature2']

        dm: xgb.DMatrix = xgb.DMatrix(data, label=target, feature_names=features)
        params: Dict = {'objective': 'binary:logistic',
                  'eval_metric': 'logloss',
                  'eta': 0.3,
                  'max_depth': 1}

        bst: Booster = xgb.train(params, dm, num_boost_round=1)

        # number of feature importances should == number of features
        dump1: List[str] = bst.get_dump()
        assert len(dump1) == 1, 'Expected only 1 tree to be dumped.'
        len(dump1[0].splitlines()) == 3, 'Expected 1 root and 2 leaves - 3 lines in dump.'

        dump2: List[str] = bst.get_dump(with_stats=True)
        assert dump2[0].count('\n') == 3, 'Expected 1 root and 2 leaves - 3 lines in dump.'
        msg = 'Expected more info when with_stats=True is given.'
        assert dump2[0].find('\n') > dump1[0].find('\n'), msg

        dump3: List[str] = bst.get_dump(dump_format="json")
        dump3j: Any = json.loads(dump3[0])
        assert dump3j['nodeid'] == 0, 'Expected the root node on top.'

        dump4: List[str] = bst.get_dump(dump_format="json", with_stats=True)
        dump4j: Any = json.loads(dump4[0])
        assert 'gain' in dump4j, "Expected 'gain' to be dumped in JSON."

        with pytest.raises(ValueError):
            bst.get_dump(fmap="foo")

    def test_feature_score(self):
        rng: np.random.RandomState = np.random.RandomState(0)
        data: np.ndarray = rng.randn(100, 2)
        target: np.ndarray = np.array([0, 1] * 50)
        features: List[str] = ["F0"]
        with pytest.raises(ValueError):
            xgb.DMatrix(data, label=target, feature_names=features)

        params: Dict = {"objective": "binary:logistic"}
        dm: xgb.DMatrix = xgb.DMatrix(data, label=target, feature_names=["F0", "F1"])
        bst: Booster = xgb.train(params, dm, num_boost_round=1)
        # no error since feature names might be assigned before the booster seeing data
        # and booster doesn't known about the actual number of features.
        bst.feature_names = ["F0"]
        with pytest.raises(ValueError):
            bst.get_fscore()

        bst.feature_names = None
        # Use JSON to make sure the output has native Python type
        scores: Any = json.loads(json.dumps(bst.get_fscore()))
        np.testing.assert_allclose(scores["f0"], 6.0)

    def test_load_file_invalid(self):
        with pytest.raises(xgb.core.XGBoostError):
            xgb.Booster(model_file='incorrect_path')

        with pytest.raises(xgb.core.XGBoostError):
            xgb.Booster(model_file=u'不正なパス')

    @pytest.mark.parametrize("path", ["모델.ubj", "がうる・ぐら.json"], ids=["path-0", "path-1"])
    def test_unicode_path(self, tmpdir, path):
        model_path = pathlib.Path(tmpdir) / path
        dtrain, _ = tm.load_agaricus(__file__)
        param = {"max_depth": 2, "eta": 1, "objective": "binary:logistic"}
        bst = xgb.train(param, dtrain, num_boost_round=2)
        bst.save_model(model_path)

        bst2 = xgb.Booster(model_file=model_path)
        assert bst.get_dump(dump_format="text") == bst2.get_dump(dump_format="text")

    def test_dmatrix_numpy_init_omp(self):

        rows: List[int] = [1000, 11326, 15000]
        cols: int = 50
        for row in rows:
            X: np.ndarray = np.random.randn(row, cols)
            y: np.ndarray = np.random.randn(row).astype('f')
            dm: xgb.DMatrix = xgb.DMatrix(X, y, nthread=0)
            np.testing.assert_array_equal(dm.get_label(), y)
            assert dm.num_row() == row
            assert dm.num_col() == cols

            dm = xgb.DMatrix(X, y, nthread=10)
            np.testing.assert_array_equal(dm.get_label(), y)
            assert dm.num_row() == row
            assert dm.num_col() == cols

    def test_cv(self):
        dm, _ = tm.load_agaricus(__file__)
        params: Dict = {'max_depth': 2, 'eta': 1, 'verbosity': 0,
                  'objective': 'binary:logistic'}

        # return np.ndarray
        cv: List[str] = xgb.cv(params, dm, num_boost_round=10, nfold=10, as_pandas=False)
        assert isinstance(cv, dict)
        assert len(cv) == (4)

    def test_cv_no_shuffle(self):
        dm, _ = tm.load_agaricus(__file__)
        params: Dict = {'max_depth': 2, 'eta': 1, 'verbosity': 0,
                  'objective': 'binary:logistic'}

        # return np.ndarray
        cv: List[str] = xgb.cv(params, dm, num_boost_round=10, shuffle=False, nfold=10,
                    as_pandas=False)
        assert isinstance(cv, dict)
        assert len(cv) == (4)

    def test_cv_explicit_fold_indices(self):
        dm, _ = tm.load_agaricus(__file__)
        params: Dict = {'max_depth': 2, 'eta': 1, 'verbosity': 0, 'objective':
                  'binary:logistic'}
        folds: List[Tuple] = [
            # Train        Test
            ([1, 3], [5, 8]),
            ([7, 9], [23, 43]),
        ]

        # return np.ndarray
        cv: List[str] = xgb.cv(params, dm, num_boost_round=10, folds=folds,
                    as_pandas=False)
        assert isinstance(cv, dict)
        assert len(cv) == (4)

    @pytest.mark.skipif(**tm.skip_s390x())
    def test_cv_explicit_fold_indices_labels(self):
        params: Dict = {'max_depth': 2, 'eta': 1, 'verbosity': 0, 'objective':
                  'reg:squarederror'}
        N: int = 100
        F: int = 3
        dm: xgb.DMatrix = xgb.DMatrix(data=np.random.randn(N, F), label=np.arange(N))
        folds: List[Tuple] = [
            # Train        Test
            ([1, 3], [5, 8]),
            ([7, 9], [23, 43, 11]),
        ]

        # Use callback to log the test labels in each fold
        class Callback(xgb.callback.TrainingCallback):
            def __init__(self) -> None:
                super().__init__()

            def after_iteration(
                self, model,
                epoch: int,
                evals_log: xgb.callback.TrainingCallback.EvalsLog
            ):
                print([fold.dtest.get_label() for fold in model.cvfolds])

        cb: Callback = Callback()

        # Run cross validation and capture standard out to test callback result
        with tm.captured_output() as (out, err):
            xgb.cv(
                params, dm, num_boost_round=1, folds=folds, callbacks=[cb],
                as_pandas=False
            )
            output: str = out.getvalue().strip()
        solution: str = ('[array([5., 8.], dtype=float32), array([23., 43., 11.],' +
                    ' dtype=float32)]')
        assert output == solution


class TestBasicPathLike:
    """Unit tests using pathlib.Path for file interaction."""

    def test_DMatrix_init_from_path(self):
        """Initialization from the data path."""
        dtrain, _ = tm.load_agaricus(__file__)
        assert dtrain.num_row() == 6513
        assert dtrain.num_col() == 127

    def test_DMatrix_save_to_path(self):
        """Saving to a binary file using pathlib from a DMatrix."""
        data: np.ndarray = np.random.randn(100, 2)
        target: np.ndarray = np.array([0, 1] * 50)
        features: List[str] = ['Feature1', 'Feature2']

        dm: xgb.DMatrix = xgb.DMatrix(data, label=target, feature_names=features)

        # save, assert exists, remove file
        binary_path: Path = Path("dtrain.bin")
        dm.save_binary(binary_path)
        assert binary_path.exists()
        Path.unlink(binary_path)

    def test_Booster_init_invalid_path(self):
        """An invalid model_file path should raise XGBoostError."""
        with pytest.raises(xgb.core.XGBoostError):
            xgb.Booster(model_file=Path("invalidpath"))

    def test_Booster_save_and_load(self):
        """Saving and loading model files from paths."""
        save_path: Path = Path("saveload.model")

        data: np.ndarray = np.random.randn(100, 2)
        target: np.ndarray = np.array([0, 1] * 50)
        features: List[str] = ['Feature1', 'Feature2']

        dm: xgb.DMatrix = xgb.DMatrix(data, label=target, feature_names=features)
        params: Dict = {'objective': 'binary:logistic',
                  'eval_metric': 'logloss',
                  'eta': 0.3,
                  'max_depth': 1}

        bst: Booster = xgb.train(params, dm, num_boost_round=1)

        # save, assert exists
        bst.save_model(save_path)
        assert save_path.exists()

        def dump_assertions(dump):
            """Assertions for the expected dump from Booster"""
            assert len(dump) == 1, 'Exepcted only 1 tree to be dumped.'
            assert len(dump[0].splitlines()) == 3, 'Expected 1 root and 2 leaves - 3 lines.'

        # load the model again using Path
        bst2: Booster = xgb.Booster(model_file=save_path)
        dump2: List[str] = bst2.get_dump()
        dump_assertions(dump2)

        # load again using load_model
        bst3: Booster = xgb.Booster()
        bst3.load_model(save_path)
        dump3: List[str] = bst3.get_dump()
        dump_assertions(dump3)

        # remove file
        Path.unlink(save_path)
