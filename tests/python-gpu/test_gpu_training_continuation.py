import numpy as np
import pytest

from xgboost.testing.continuation import run_training_continuation_model_output

rng = np.random.RandomState(1994)


class TestGPUTrainingContinuation:
    @pytest.mark.parametrize("tree_method", ["hist", "approx"])
    def test_model_output(self, tree_method: str) -> None:
        run_training_continuation_model_output("cuda", tree_method)
