'''Test model IO with pickle.'''
import pickle
import unittest
import numpy as np
import subprocess
import os
import json
import xgboost as xgb
from xgboost import XGBClassifier

model_path = './model.pkl'


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


class TestPickling(unittest.TestCase):
    def test_pickling(self):
        x, y = build_dataset()
        train_x = xgb.DMatrix(x, label=y)
        param = {'tree_method': 'gpu_hist',
                 'verbosity': 1}
        bst = xgb.train(param, train_x)

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

    def test_pickled_predictor(self):
        args_templae = [
            "pytest",
            "--verbose",
            "-s",
            "--fulltrace"]

        x, y = build_dataset()
        train_x = xgb.DMatrix(x, label=y)

        param = {'tree_method': 'gpu_hist',
                 'verbosity': 1, 'predictor': 'gpu_predictor'}
        bst = xgb.train(param, train_x)
        config = json.loads(bst.save_config())
        assert config['learner']['gradient_booster']['gbtree_train_param'][
            'predictor'] == 'gpu_predictor'

        save_pickle(bst, model_path)

        args = args_templae.copy()
        args.append(
            "./tests/python-gpu/"
            "load_pickle.py::TestLoadPickle::test_predictor_type_is_auto")

        cuda_environment = {'CUDA_VISIBLE_DEVICES': '-1'}
        env = os.environ.copy()
        env.update(cuda_environment)

        # Load model in a CPU only environment.
        status = subprocess.call(args, env=env)
        assert status == 0

        args = args_templae.copy()
        args.append(
            "./tests/python-gpu/"
            "load_pickle.py::TestLoadPickle::test_predictor_type_is_gpu")

        # Load in environment that has GPU.
        env = os.environ.copy()
        assert 'CUDA_VISIBLE_DEVICES' not in env.keys()
        status = subprocess.call(args, env=env)
        assert status == 0

    def test_predict_sklearn_pickle(self):
        x, y = build_dataset()

        kwargs = {'tree_method': 'gpu_hist',
                  'predictor': 'gpu_predictor',
                  'verbosity': 1,
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
