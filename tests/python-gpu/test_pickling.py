'''Test model IO with pickle.'''
import pickle
import unittest
import numpy as np
import subprocess

model_path = './model.pkl'


def build_dataset():
    N = 10
    x = np.linspace(0, N*N, N*N)
    x = x.reshape((N, N))
    y = np.linspace(0, N, N)
    return x, y


class TestPickling(unittest.TestCase):
    def test_pickling(self):
        import os
        import xgboost as xgb
        x, y = build_dataset()
        train_x = xgb.DMatrix(x, label=y)
        param = {'tree_method': 'gpu_hist',
                 'gpu_id': 0,
                 'n_gpus': -1,
                 'verbosity': 1}
        bst = xgb.train(param, train_x)

        with open(model_path, 'wb') as fd:
            pickle.dump(bst, fd)
        sys_path = os.environ['PATH']
        args = ["pytest",
                "--verbose",
                "-s",
                "--fulltrace",
                "./tests/python-gpu/load_pickle.py"]
        command = ''
        for arg in args:
            command += arg
            command += ' '
        # Load model in a CPU only environment.
        status = subprocess.call(command,
                                 env={'CUDA_VISIBLE_DEVICES': '',
                                      'PATH': sys_path}, shell=True)
        assert status == 0
        os.remove(model_path)
