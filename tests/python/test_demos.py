import os
import subprocess
import sys
import pytest
import testing as tm


CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
DEMO_DIR = os.path.join(ROOT_DIR, 'demo')
PYTHON_DEMO_DIR = os.path.join(DEMO_DIR, 'guide-python')


def test_basic_walkthrough():
    script = os.path.join(PYTHON_DEMO_DIR, 'basic_walkthrough.py')
    cmd = ['python', script]
    subprocess.check_call(cmd)
    os.remove('dump.nice.txt')
    os.remove('dump.raw.txt')


def test_custom_multiclass_objective():
    script = os.path.join(PYTHON_DEMO_DIR, 'custom_softmax.py')
    cmd = ['python', script, '--plot=0']
    subprocess.check_call(cmd)


def test_custom_rmsle_objective():
    major, minor = sys.version_info[:2]
    if minor < 6:
        pytest.skip('Skipping RMLSE test due to Python version being too low.')
    script = os.path.join(PYTHON_DEMO_DIR, 'custom_rmsle.py')
    cmd = ['python', script, '--plot=0']
    subprocess.check_call(cmd)


@pytest.mark.skipif(**tm.no_sklearn())
def test_sklearn_demo():
    script = os.path.join(PYTHON_DEMO_DIR, 'sklearn_examples.py')
    cmd = ['python', script]
    subprocess.check_call(cmd)
    assert os.path.exists('best_boston.pkl')
    os.remove('best_boston.pkl')


@pytest.mark.skipif(**tm.no_sklearn())
def test_sklearn_parallel_demo():
    script = os.path.join(PYTHON_DEMO_DIR, 'sklearn_parallel.py')
    cmd = ['python', script]
    subprocess.check_call(cmd)


@pytest.mark.skipif(**tm.no_sklearn())
def test_sklearn_evals_result_demo():
    script = os.path.join(PYTHON_DEMO_DIR, 'sklearn_evals_result.py')
    cmd = ['python', script]
    subprocess.check_call(cmd)


def test_boost_from_prediction_demo():
    script = os.path.join(PYTHON_DEMO_DIR, 'boost_from_prediction.py')
    cmd = ['python', script]
    subprocess.check_call(cmd)


def test_predict_first_ntree_demo():
    script = os.path.join(PYTHON_DEMO_DIR, 'predict_first_ntree.py')
    cmd = ['python', script]
    subprocess.check_call(cmd)


def test_predict_leaf_indices_demo():
    script = os.path.join(PYTHON_DEMO_DIR, 'predict_leaf_indices.py')
    cmd = ['python', script]
    subprocess.check_call(cmd)


def test_generalized_linear_model_demo():
    script = os.path.join(PYTHON_DEMO_DIR, 'generalized_linear_model.py')
    cmd = ['python', script]
    subprocess.check_call(cmd)


def test_custom_objective_demo():
    script = os.path.join(PYTHON_DEMO_DIR, 'custom_objective.py')
    cmd = ['python', script]
    subprocess.check_call(cmd)


def test_cross_validation_demo():
    script = os.path.join(PYTHON_DEMO_DIR, 'cross_validation.py')
    cmd = ['python', script]
    subprocess.check_call(cmd)


def test_external_memory_demo():
    script = os.path.join(PYTHON_DEMO_DIR, 'external_memory.py')
    cmd = ['python', script]
    subprocess.check_call(cmd)


def test_evals_result_demo():
    script = os.path.join(PYTHON_DEMO_DIR, 'evals_result.py')
    cmd = ['python', script]
    subprocess.check_call(cmd)


def test_aft_demo():
    script = os.path.join(DEMO_DIR, 'aft_survival', 'aft_survival_demo.py')
    cmd = ['python', script]
    subprocess.check_call(cmd)
    assert os.path.exists('aft_model.json')
    os.remove('aft_model.json')


# gpu_acceleration is not tested due to covertype dataset is being too huge.
# gamma regression is not tested as it requires running a R script first.
# aft viz is not tested due to ploting is not controled
# aft tunning is not tested due to extra dependency.
