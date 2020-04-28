import os
import subprocess
import pytest
import sys
sys.path.append("tests/python")
import testing as tm            # NOQA


CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
DEMO_DIR = os.path.join(ROOT_DIR, 'demo', 'guide-python')


@pytest.mark.skipif(**tm.no_cupy())
def test_data_iter():
    script = os.path.join(DEMO_DIR, 'data_iterator.py')
    cmd = ['python', script]
    subprocess.check_call(cmd)
