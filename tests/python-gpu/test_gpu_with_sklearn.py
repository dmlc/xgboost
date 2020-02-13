import xgboost as xgb
import pytest
import sys
import numpy as np
import unittest

sys.path.append("tests/python")
import testing as tm               # noqa
import test_with_sklearn as twskl  # noqa

pytestmark = pytest.mark.skipif(**tm.no_sklearn())

rng = np.random.RandomState(1994)


def test_gpu_binary_classification():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import KFold

    digits = load_digits(2)
    y = digits['target']
    X = digits['data']
    kf = KFold(n_splits=2, shuffle=True, random_state=rng)
    for cls in (xgb.XGBClassifier, xgb.XGBRFClassifier):
        for train_index, test_index in kf.split(X, y):
            xgb_model = cls(
                random_state=42, tree_method='gpu_hist',
                n_estimators=4, gpu_id='0').fit(X[train_index], y[train_index])
            preds = xgb_model.predict(X[test_index])
            labels = y[test_index]
            err = sum(1 for i in range(len(preds))
                      if int(preds[i] > 0.5) != labels[i]) / float(len(preds))
            assert err < 0.1


class TestGPUBoostFromPrediction(unittest.TestCase):
    cpu_test = twskl.TestBoostFromPrediction()

    def test_boost_from_prediction_gpu_hist(self):
        self.cpu_test.run_boost_from_prediction('gpu_hist')
