import os
import subprocess

import pytest

from xgboost import testing as tm


@pytest.mark.skipif(**tm.no_dask())
def test_dask_cpu_training_demo():
    script = os.path.join(tm.demo_dir(__file__), "dask", "cpu_training.py")
    cmd = ["python", script]
    subprocess.check_call(cmd)


@pytest.mark.skipif(**tm.no_dask())
def test_dask_cpu_survival_demo():
    script = os.path.join(tm.demo_dir(__file__), "dask", "cpu_survival.py")
    cmd = ["python", script]
    subprocess.check_call(cmd)


# Not actually run on CI due to missing dask_ml.
@pytest.mark.skipif(**tm.no_dask())
@pytest.mark.skipif(**tm.no_dask_ml())
def test_dask_callbacks_demo():
    script = os.path.join(tm.demo_dir(__file__), "dask", "dask_callbacks.py")
    cmd = ["python", script]
    subprocess.check_call(cmd)


@pytest.mark.skipif(**tm.no_dask())
def test_dask_sklearn_demo():
    script = os.path.join(tm.demo_dir(__file__), "dask", "sklearn_cpu_training.py")
    cmd = ["python", script]
    subprocess.check_call(cmd)
