'''Test model IO with pickle.'''
import json
import os
import pickle
import subprocess

import numpy as np
import pytest

import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import testing as tm

model_path = './model.pkl'

pytestmark = tm.timeout(30)


def build_dataset():
    N = 10
    x = np.linspace(0, N*N, N*N)
    x = x.reshape((N, N))
    y = np.linspace(0, N, N)
    return x, y


def save_pickle(bst, path):
    with open(path, 'wb') as fd:
        pickle.dump(bst, fd)


def load_pickle(path):
    with open(path, 'rb') as fd:
        bst = pickle.load(fd)
    return bst


class TestPickling:
    args_template = [
        "pytest",
        "--verbose",
        "-s",
        "--fulltrace"]

    def run_pickling(self, bst) -> None:
        save_pickle(bst, model_path)
        args = [
            "pytest", "--verbose", "-s", "--fulltrace",
            "./tests/python-gpu/load_pickle.py::TestLoadPickle::test_load_pkl"
        ]
        command = ''
        for arg in args:
            command += arg
            command += ' '

        cuda_environment = {'CUDA_VISIBLE_DEVICES': '-1'}
        env = os.environ.copy()
        # Passing new_environment directly to `env' argument results
        # in failure on Windows:
        #    Fatal Python error: _Py_HashRandomization_Init: failed to
        #    get random numbers to initialize Python
        env.update(cuda_environment)

        # Load model in a CPU only environment.
        status = subprocess.call(command, env=env, shell=True)
        assert status == 0
        os.remove(model_path)

    # TODO: This test is too slow
    @pytest.mark.skipif(**tm.no_sklearn())
    def test_pickling(self):
        x, y = build_dataset()
        train_x = xgb.DMatrix(x, label=y)

        param = {'tree_method': 'gpu_hist', "gpu_id": 0}
        bst = xgb.train(param, train_x)
        self.run_pickling(bst)

        bst = xgb.XGBRegressor(**param).fit(x, y)
        self.run_pickling(bst)

        param = {"booster": "gblinear", "updater": "gpu_coord_descent", "gpu_id": 0}
        bst = xgb.train(param, train_x)
        self.run_pickling(bst)

        bst = xgb.XGBRegressor(**param).fit(x, y)
        self.run_pickling(bst)

    @pytest.mark.mgpu
    def test_wrap_gpu_id(self):
        X, y = build_dataset()
        dtrain = xgb.DMatrix(X, y)

        bst = xgb.train({'tree_method': 'gpu_hist',
                         'gpu_id': 1},
                        dtrain, num_boost_round=6)

        model_path = 'model.pkl'
        save_pickle(bst, model_path)
        cuda_environment = {'CUDA_VISIBLE_DEVICES': '0'}
        env = os.environ.copy()
        env.update(cuda_environment)
        args = self.args_template.copy()
        args.append(
            "./tests/python-gpu/"
            "load_pickle.py::TestLoadPickle::test_wrap_gpu_id"
        )
        status = subprocess.call(args, env=env)
        assert status == 0
        os.remove(model_path)

    def test_pickled_predictor(self):
        x, y = build_dataset()
        train_x = xgb.DMatrix(x, label=y)

        param = {'tree_method': 'gpu_hist',
                 'verbosity': 1, 'predictor': 'gpu_predictor'}
        bst = xgb.train(param, train_x)
        config = json.loads(bst.save_config())
        assert config['learner']['gradient_booster']['gbtree_train_param'][
            'predictor'] == 'gpu_predictor'

        save_pickle(bst, model_path)

        args = self.args_template.copy()
        args.append(
            "./tests/python-gpu/"
            "load_pickle.py::TestLoadPickle::test_predictor_type_is_auto")

        cuda_environment = {'CUDA_VISIBLE_DEVICES': '-1'}
        env = os.environ.copy()
        env.update(cuda_environment)

        # Load model in a CPU only environment.
        status = subprocess.call(args, env=env)
        assert status == 0

        args = self.args_template.copy()
        args.append(
            "./tests/python-gpu/"
            "load_pickle.py::TestLoadPickle::test_predictor_type_is_gpu")

        # Load in environment that has GPU.
        env = os.environ.copy()
        assert 'CUDA_VISIBLE_DEVICES' not in env.keys()
        status = subprocess.call(args, env=env)
        assert status == 0

        os.remove(model_path)

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_predict_sklearn_pickle(self):
        from sklearn.datasets import load_digits
        x, y = load_digits(return_X_y=True)

        kwargs = {'tree_method': 'gpu_hist',
                  'predictor': 'gpu_predictor',
                  'objective': 'binary:logistic',
                  'n_estimators': 10}

        model = XGBClassifier(**kwargs)
        model.fit(x, y)

        save_pickle(model, "model.pkl")
        del model

        # load model
        model: xgb.XGBClassifier = load_pickle("model.pkl")
        os.remove("model.pkl")

        gpu_pred = model.predict(x, output_margin=True)

        # Switch to CPU predictor
        bst = model.get_booster()
        bst.set_param({'predictor': 'cpu_predictor'})
        cpu_pred = model.predict(x, output_margin=True)
        np.testing.assert_allclose(cpu_pred, gpu_pred, rtol=1e-5)

    def test_training_on_cpu_only_env(self):
        cuda_environment = {'CUDA_VISIBLE_DEVICES': '-1'}
        env = os.environ.copy()
        env.update(cuda_environment)
        args = self.args_template.copy()
        args.append(
            "./tests/python-gpu/"
            "load_pickle.py::TestLoadPickle::test_training_on_cpu_only_env")
        status = subprocess.call(args, env=env)
        assert status == 0
