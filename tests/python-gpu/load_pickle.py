'''Loading a pickled model generated by test_pickling.py'''
import pickle
import unittest
import os
import xgboost as xgb
import sys

sys.path.append("tests/python")
from test_pickling import build_dataset, model_path


class TestLoadPickle(unittest.TestCase):
    def test_load_pkl(self):
        assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
        with open(model_path, 'rb') as fd:
            bst = pickle.load(fd)
        x, y = build_dataset()
        test_x = xgb.DMatrix(x)
        res = bst.predict(test_x)
        assert len(res) == 10
