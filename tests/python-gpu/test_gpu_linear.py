import sys

sys.path.append('tests/python/')
import test_linear
import testing as tm
import unittest


class TestGPULinear(unittest.TestCase):
    def test_gpu_coordinate(self):
        tm._skip_if_no_sklearn()
        variable_param = {'alpha': [.005, .1], 'lambda': [0.005],
                          'coordinate_selection': ['cyclic', 'random', 'greedy'], 'n_gpus': [-1, 1]}
        test_linear.assert_updater_accuracy('gpu_coord_descent', variable_param)
