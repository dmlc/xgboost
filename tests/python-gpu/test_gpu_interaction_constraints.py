import numpy as np
import unittest
import pytest
import sys
sys.path.append("tests/python")
# Don't import the test class, otherwise they will run twice.
import test_interaction_constraints as test_ic
rng = np.random.RandomState(1994)


@pytest.mark.gpu
class TestGPUInteractionConstraints(unittest.TestCase):
    cputest = test_ic.TestInteractionConstraints()

    def test_interaction_constraints(self):
        self.cputest.test_interaction_constraints(tree_method='gpu_hist')

    def test_training_accuracy(self):
        self.cputest.test_training_accuracy(tree_method='gpu_hist')
