import os
import subprocess
import sys
import pytest
sys.path.append("tests/python")
import testing as tm
import test_demos as td         # noqa

pytestmark = pytest.mark.no_rmm_pool_setup

@pytest.mark.skipif(**tm.no_cupy())
def test_data_iterator():
    script = os.path.join(td.PYTHON_DEMO_DIR, 'data_iterator.py')
    cmd = ['python', script]
    subprocess.check_call(cmd)
