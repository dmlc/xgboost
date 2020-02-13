import sys
import unittest
import numpy as np
sys.path.append("tests/python")
# Don't import the test class, otherwise they will run twice.
import test_basic_models as test_bm  # noqa
rng = np.random.RandomState(1994)


class TestGPUBasicModels(unittest.TestCase):
    cputest = test_bm.TestModels()

    def test_eta_decay_gpu_hist(self):
        self.cputest.run_eta_decay('gpu_hist')
