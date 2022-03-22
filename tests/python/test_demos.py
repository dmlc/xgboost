import os
import subprocess
import pytest
import testing as tm
import sys


ROOT_DIR = tm.PROJECT_ROOT
DEMO_DIR = os.path.join(ROOT_DIR, 'demo')
PYTHON_DEMO_DIR = os.path.join(DEMO_DIR, 'guide-python')
CLI_DEMO_DIR = os.path.join(DEMO_DIR, 'CLI')


def test_basic_walkthrough():
    script = os.path.join(PYTHON_DEMO_DIR, 'basic_walkthrough.py')
    cmd = ['python', script]
    subprocess.check_call(cmd)
    os.remove('dump.nice.txt')
    os.remove('dump.raw.txt')


@pytest.mark.skipif(**tm.no_matplotlib())
def test_custom_multiclass_objective():
    script = os.path.join(PYTHON_DEMO_DIR, 'custom_softmax.py')
    cmd = ['python', script, '--plot=0']
    subprocess.check_call(cmd)


@pytest.mark.skipif(**tm.no_matplotlib())
def test_custom_rmsle_objective():
    script = os.path.join(PYTHON_DEMO_DIR, 'custom_rmsle.py')
    cmd = ['python', script, '--plot=0']
    subprocess.check_call(cmd)


@pytest.mark.skipif(**tm.no_matplotlib())
def test_feature_weights_demo():
    script = os.path.join(PYTHON_DEMO_DIR, 'feature_weights.py')
    cmd = ['python', script, '--plot=0']
    subprocess.check_call(cmd)


@pytest.mark.skipif(**tm.no_sklearn())
def test_sklearn_demo():
    script = os.path.join(PYTHON_DEMO_DIR, 'sklearn_examples.py')
    cmd = ['python', script]
    subprocess.check_call(cmd)
    assert os.path.exists('best_calif.pkl')
    os.remove('best_calif.pkl')


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


@pytest.mark.skipif(**tm.no_sklearn())
@pytest.mark.skipif(**tm.no_pandas())
def test_aft_demo():
    script = os.path.join(DEMO_DIR, 'aft_survival', 'aft_survival_demo.py')
    cmd = ['python', script]
    subprocess.check_call(cmd)
    assert os.path.exists('aft_model.json')
    os.remove('aft_model.json')


@pytest.mark.skipif(**tm.no_matplotlib())
def test_callbacks_demo():
    script = os.path.join(PYTHON_DEMO_DIR, 'callbacks.py')
    cmd = ['python', script, '--plot=0']
    subprocess.check_call(cmd)


def test_continuation_demo():
    script = os.path.join(PYTHON_DEMO_DIR, 'continuation.py')
    cmd = ['python', script]
    subprocess.check_call(cmd)


@pytest.mark.skipif(**tm.no_sklearn())
@pytest.mark.skipif(**tm.no_matplotlib())
def test_multioutput_reg() -> None:
    script = os.path.join(PYTHON_DEMO_DIR, "multioutput_regression.py")
    cmd = ['python', script, "--plot=0"]
    subprocess.check_call(cmd)


# gpu_acceleration is not tested due to covertype dataset is being too huge.
# gamma regression is not tested as it requires running a R script first.
# aft viz is not tested due to ploting is not controled
# aft tunning is not tested due to extra dependency.


def test_cli_regression_demo():
    reg_dir = os.path.join(CLI_DEMO_DIR, 'regression')
    script = os.path.join(reg_dir, 'mapfeat.py')
    cmd = ['python', script]
    subprocess.check_call(cmd, cwd=reg_dir)

    script = os.path.join(reg_dir, 'mknfold.py')
    cmd = ['python', script, 'machine.txt', '1']
    subprocess.check_call(cmd, cwd=reg_dir)

    exe = os.path.join(tm.PROJECT_ROOT, 'xgboost')
    conf = os.path.join(reg_dir, 'machine.conf')
    subprocess.check_call([exe, conf], cwd=reg_dir)


@pytest.mark.skipif(condition=sys.platform.startswith("win"),
                    reason='Test requires sh execution.')
def test_cli_binary_classification():
    cls_dir = os.path.join(CLI_DEMO_DIR, 'binary_classification')
    with tm.DirectoryExcursion(cls_dir, cleanup=True):
        subprocess.check_call(['./runexp.sh'])
        os.remove('0002.model')

# year prediction is not tested due to data size being too large.
# rank is not tested as it requires unrar command.
