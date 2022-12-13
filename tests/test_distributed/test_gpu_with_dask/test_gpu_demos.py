import os
import subprocess

import pytest

from xgboost import testing as tm


@pytest.mark.skipif(**tm.no_dask())
@pytest.mark.skipif(**tm.no_dask_cuda())
@pytest.mark.skipif(**tm.no_cupy())
@pytest.mark.mgpu
def test_dask_training():
    script = os.path.join(tm.demo_dir(__file__), "dask", "gpu_training.py")
    cmd = ["python", script]
    subprocess.check_call(cmd)


@pytest.mark.skipif(**tm.no_dask_cuda())
@pytest.mark.skipif(**tm.no_dask())
@pytest.mark.mgpu
def test_dask_sklearn_demo():
    script = os.path.join(tm.demo_dir(__file__), "dask", "sklearn_gpu_training.py")
    cmd = ["python", script]
    subprocess.check_call(cmd)
