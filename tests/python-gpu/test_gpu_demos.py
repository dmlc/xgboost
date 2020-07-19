import errno
import os
import subprocess
import sys
import pytest
sys.path.append("tests/python")
import testing as tm
import test_demos as td         # noqa


def maybe_makedirs(path):
    path = os.path.normpath(path)
    print("mkdir -p " + path)
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


@pytest.mark.skipif(**tm.no_cupy())
def test_data_iterator():
    script = os.path.join(td.PYTHON_DEMO_DIR, 'data_iterator.py')
    cmd = ['python', script]
    subprocess.check_call(cmd)


@pytest.mark.skipif(**tm.no_rmm())
def test_rmm_integration_demo():
    demo_dir = os.path.join(td.DEMO_DIR, 'rmm-integration')
    build_dir = os.path.join(demo_dir, 'build')
    maybe_makedirs(build_dir)
    subprocess.check_call(['cmake', '..'], cwd=build_dir)
    subprocess.check_call(['make'], cwd=build_dir)
    script = os.path.join(demo_dir, 'rmm_integration_demo.py')
    subprocess.check_call(['python', script])
