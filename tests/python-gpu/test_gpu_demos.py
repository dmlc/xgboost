import os
import subprocess
import sys
sys.path.append("tests/python")
import test_demos as td         # noqa


def test_data_iterator():
    script = os.path.join(td.PYTHON_DEMO_DIR, 'data_iterator.py')
    cmd = ['python', script]
    subprocess.check_call(cmd)
