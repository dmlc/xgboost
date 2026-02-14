import os
import subprocess
import sys
import tempfile

import pytest

import xgboost
from xgboost import testing as tm

pytestmark = tm.timeout(30)

DEMO_DIR = tm.demo_dir(__file__)
PYTHON_DEMO_DIR = os.path.join(DEMO_DIR, "guide-python")
CLI_DEMO_DIR = os.path.join(DEMO_DIR, "CLI")


PYTHON = sys.executable


def test_basic_walkthrough() -> None:
    script = os.path.join(PYTHON_DEMO_DIR, "basic_walkthrough.py")
    cmd = [PYTHON, script]
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.check_call(cmd, cwd=tmpdir)


@pytest.mark.skipif(**tm.no_pandas())
def test_categorical() -> None:
    script = os.path.join(PYTHON_DEMO_DIR, "categorical.py")
    cmd = [PYTHON, script]
    subprocess.check_call(cmd)


@pytest.mark.skipif(**tm.no_pandas())
def test_cat_pipeline() -> None:
    script = os.path.join(PYTHON_DEMO_DIR, "cat_pipeline.py")
    cmd = [PYTHON, script]
    subprocess.check_call(cmd)


@pytest.mark.skipif(**tm.no_matplotlib())
def test_custom_multiclass_objective() -> None:
    script = os.path.join(PYTHON_DEMO_DIR, "custom_softmax.py")
    cmd = [PYTHON, script, "--plot=0"]
    subprocess.check_call(cmd)


@pytest.mark.skipif(**tm.no_matplotlib())
def test_custom_rmsle_objective() -> None:
    script = os.path.join(PYTHON_DEMO_DIR, "custom_rmsle.py")
    cmd = [PYTHON, script, "--plot=0"]
    subprocess.check_call(cmd)


@pytest.mark.skipif(**tm.no_matplotlib())
def test_feature_weights_demo() -> None:
    script = os.path.join(PYTHON_DEMO_DIR, "feature_weights.py")
    cmd = [PYTHON, script, "--plot=0"]
    subprocess.check_call(cmd)


@pytest.mark.skipif(**tm.no_sklearn())
def test_sklearn_demo() -> None:
    script = os.path.join(PYTHON_DEMO_DIR, "sklearn_examples.py")
    cmd = [PYTHON, script]
    subprocess.check_call(cmd)
    assert os.path.exists("best_calif.pkl")
    os.remove("best_calif.pkl")


@pytest.mark.skipif(**tm.no_sklearn())
@pytest.mark.timeout(60)
def test_sklearn_parallel_demo() -> None:
    script = os.path.join(PYTHON_DEMO_DIR, "sklearn_parallel.py")
    cmd = [PYTHON, script]
    subprocess.check_call(cmd)


@pytest.mark.skipif(**tm.no_sklearn())
def test_sklearn_evals_result_demo() -> None:
    script = os.path.join(PYTHON_DEMO_DIR, "sklearn_evals_result.py")
    cmd = [PYTHON, script]
    subprocess.check_call(cmd)


def test_boost_from_prediction_demo() -> None:
    script = os.path.join(PYTHON_DEMO_DIR, "boost_from_prediction.py")
    cmd = [PYTHON, script]
    subprocess.check_call(cmd)


def test_predict_first_ntree_demo() -> None:
    script = os.path.join(PYTHON_DEMO_DIR, "predict_first_ntree.py")
    cmd = [PYTHON, script]
    subprocess.check_call(cmd)


def test_individual_trees() -> None:
    script = os.path.join(PYTHON_DEMO_DIR, "individual_trees.py")
    cmd = [PYTHON, script]
    subprocess.check_call(cmd)


def test_predict_leaf_indices_demo() -> None:
    script = os.path.join(PYTHON_DEMO_DIR, "predict_leaf_indices.py")
    cmd = [PYTHON, script]
    subprocess.check_call(cmd)


def test_generalized_linear_model_demo() -> None:
    script = os.path.join(PYTHON_DEMO_DIR, "generalized_linear_model.py")
    cmd = [PYTHON, script]
    subprocess.check_call(cmd)


def test_cross_validation_demo() -> None:
    script = os.path.join(PYTHON_DEMO_DIR, "cross_validation.py")
    cmd = [PYTHON, script]
    subprocess.check_call(cmd)


def test_external_memory_demo() -> None:
    script = os.path.join(PYTHON_DEMO_DIR, "external_memory.py")
    cmd = [PYTHON, script, "--device=cpu"]
    subprocess.check_call(cmd)


def test_distributed_extmem_basic_demo() -> None:
    script = os.path.join(PYTHON_DEMO_DIR, "distributed_extmem_basic.py")
    cmd = [PYTHON, script, "--device=cpu"]
    subprocess.check_call(cmd)


def test_evals_result_demo() -> None:
    script = os.path.join(PYTHON_DEMO_DIR, "evals_result.py")
    cmd = [PYTHON, script]
    subprocess.check_call(cmd)


@pytest.mark.skipif(**tm.no_sklearn())
@pytest.mark.skipif(**tm.no_pandas())
def test_aft_demo() -> None:
    script = os.path.join(DEMO_DIR, "aft_survival", "aft_survival_demo.py")
    cmd = [PYTHON, script]
    subprocess.check_call(cmd)
    assert os.path.exists("aft_model.json")
    os.remove("aft_model.json")


@pytest.mark.skipif(**tm.no_matplotlib())
def test_callbacks_demo() -> None:
    script = os.path.join(PYTHON_DEMO_DIR, "callbacks.py")
    cmd = [PYTHON, script, "--plot=0"]
    subprocess.check_call(cmd)


def test_continuation_demo() -> None:
    script = os.path.join(PYTHON_DEMO_DIR, "continuation.py")
    cmd = [PYTHON, script]
    subprocess.check_call(cmd)


@pytest.mark.skipif(**tm.no_sklearn())
@pytest.mark.skipif(**tm.no_matplotlib())
def test_multioutput_reg() -> None:
    script = os.path.join(PYTHON_DEMO_DIR, "multioutput_regression.py")
    cmd = [PYTHON, script, "--plot=0"]
    subprocess.check_call(cmd)


@pytest.mark.skipif(**tm.no_sklearn())
def test_quantile_reg() -> None:
    script = os.path.join(PYTHON_DEMO_DIR, "quantile_regression.py")
    cmd = [PYTHON, script]
    subprocess.check_call(cmd)


@pytest.mark.skipif(**tm.no_ubjson())
def test_json_model() -> None:
    script = os.path.join(PYTHON_DEMO_DIR, "model_parser.py")

    def run_test(reg: xgboost.XGBRegressor) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "reg.json")
            reg.save_model(path)
            cmd = [PYTHON, script, f"--model={path}"]
            subprocess.check_call(cmd)

            path = os.path.join(tmpdir, "reg.ubj")
            reg.save_model(path)
            cmd = [PYTHON, script, f"--model={path}"]
            subprocess.check_call(cmd)

    # numerical
    X, y = tm.make_sparse_regression(100, 10, 0.5, False)
    reg = xgboost.XGBRegressor(n_estimators=2, tree_method="hist")
    reg.fit(X, y)
    run_test(reg)

    # categorical
    X, y = tm.make_categorical(
        n_samples=1000,
        n_features=10,
        n_categories=6,
        onehot=False,
        sparsity=0.5,
        cat_ratio=0.5,
        shuffle=True,
    )
    reg = xgboost.XGBRegressor(n_estimators=2, tree_method="hist")
    reg.fit(X, y)
    run_test(reg)


# - gpu_acceleration is not tested due to covertype dataset is being too huge.
# - gamma regression is not tested as it requires running a R script first.
# - aft viz is not tested due to ploting is not controlled
# - aft tunning is not tested due to extra dependency.
