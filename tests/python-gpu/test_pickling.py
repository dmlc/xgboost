'''Test model IO with pickle.'''
import pickle
import unittest
import numpy as np
import subprocess
import os
import xgboost as xgb

model_path = './model.pkl'


def build_dataset():
    N = 10
    x = np.linspace(0, N*N, N*N)
    x = x.reshape((N, N))
    y = np.linspace(0, N, N)
    return x, y


class TestPickling(unittest.TestCase):
    def test_pickling(self):
        x, y = build_dataset()
        train_x = xgb.DMatrix(x, label=y)
        param = {'tree_method': 'gpu_hist',
                 'gpu_id': 0,
                 'n_gpus': -1,
                 'verbosity': 1}
        bst = xgb.train(param, train_x)

        with open(model_path, 'wb') as fd:
            pickle.dump(bst, fd)
        args = ["pytest",
                "--verbose",
                "-s",
                "--fulltrace",
                "./tests/python-gpu/load_pickle.py"]
        command = ''
        for arg in args:
            command += arg
            command += ' '

        cuda_environment = {'CUDA_VISIBLE_DEVICES': ''}
        env = os.environ
        # Passing new_environment directly to `env' argument results
        # in failure on Windows:
        #    Fatal Python error: _Py_HashRandomization_Init: failed to
        #    get random numbers to initialize Python
        env.update(cuda_environment)

        # Load model in a CPU only environment.
        status = subprocess.call(command, env=env, shell=True)
        assert status == 0
        os.remove(model_path)
