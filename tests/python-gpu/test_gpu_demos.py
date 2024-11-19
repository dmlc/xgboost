import os
import subprocess
import sys

import pytest

from xgboost import testing as tm

DEMO_DIR = tm.demo_dir(__file__)
PYTHON_DEMO_DIR = os.path.join(DEMO_DIR, "guide-python")


@pytest.mark.skipif(**tm.no_cupy())
def test_data_iterator():
    script = os.path.join(PYTHON_DEMO_DIR, "quantile_data_iterator.py")
    cmd = ["python", script]
    subprocess.check_call(cmd)


def test_update_process_demo():
    script = os.path.join(PYTHON_DEMO_DIR, "update_process.py")
    cmd = ["python", script]
    subprocess.check_call(cmd)


def test_categorical_demo():
    script = os.path.join(PYTHON_DEMO_DIR, "categorical.py")
    cmd = ["python", script]
    subprocess.check_call(cmd)


@pytest.mark.skipif(**tm.no_rmm())
@pytest.mark.skipif(**tm.no_cupy())
def test_external_memory_demo():
    script = os.path.join(PYTHON_DEMO_DIR, "external_memory.py")
    cmd = ["python", script, "--device=cuda"]
    subprocess.check_call(cmd)


@pytest.mark.skipif(**tm.no_rmm())
@pytest.mark.skipif(**tm.no_cupy())
def test_distributed_extmem_basic_demo():
    script = os.path.join(PYTHON_DEMO_DIR, "distributed_extmem_basic.py")
    cmd = ["python", script, "--device=cuda"]
    subprocess.check_call(cmd)
