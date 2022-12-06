import os
import subprocess
import sys

import pytest

from xgboost import testing as tm

sys.path.append("tests/python")
import test_demos as td  # noqa


@pytest.mark.skipif(**tm.no_cupy())
def test_data_iterator():
    script = os.path.join(td.PYTHON_DEMO_DIR, 'quantile_data_iterator.py')
    cmd = ['python', script]
    subprocess.check_call(cmd)


def test_update_process_demo():
    script = os.path.join(td.PYTHON_DEMO_DIR, 'update_process.py')
    cmd = ['python', script]
    subprocess.check_call(cmd)


def test_categorical_demo():
    script = os.path.join(td.PYTHON_DEMO_DIR, 'categorical.py')
    cmd = ['python', script]
    subprocess.check_call(cmd)
