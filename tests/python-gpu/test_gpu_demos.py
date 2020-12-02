import os
import subprocess
import sys
import pytest
sys.path.append("tests/python")
import testing as tm
import test_demos as td         # noqa


@pytest.mark.skipif(**tm.no_cupy())
def test_data_iterator():
    script = os.path.join(td.PYTHON_DEMO_DIR, 'data_iterator.py')
    cmd = ['python', script]
    subprocess.check_call(cmd)


@pytest.mark.skipif(**tm.no_dask())
@pytest.mark.skipif(**tm.no_dask_cuda())
@pytest.mark.skipif(**tm.no_cupy())
@pytest.mark.mgpu
def test_dask_training():
    script = os.path.join(tm.PROJECT_ROOT, 'demo', 'dask', 'gpu_training.py')
    cmd = ['python', script, '--ddqdm=1']
    subprocess.check_call(cmd)

    cmd = ['python', script, '--ddqdm=0']
    subprocess.check_call(cmd)
