import numpy as np
import unittest
import sys
sys.path.append("tests/python")
# Don't import the test class, otherwise they will run twice.
import test_interaction_constraints as test_ic  # noqa
rng = np.random.RandomState(1994)


class TestGPUInteractionConstraints(unittest.TestCase):
    cputest = test_ic.TestInteractionConstraints()

    def test_interaction_constraints(self):
        self.cputest.run_interaction_constraints(tree_method='gpu_hist')

    def test_training_accuracy(self):
        self.cputest.training_accuracy(tree_method='gpu_hist')
