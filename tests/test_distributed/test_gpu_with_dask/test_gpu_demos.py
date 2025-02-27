import os
import subprocess

import pytest

from xgboost import testing as tm

pytestmark = [
    pytest.mark.skipif(**tm.no_dask()),
    pytest.mark.skipif(**tm.no_dask_cuda()),
    tm.timeout(60),
]


@pytest.mark.skipif(**tm.no_cupy())
@pytest.mark.mgpu
def test_dask_training() -> None:
    script = os.path.join(tm.demo_dir(__file__), "dask", "gpu_training.py")
    cmd = ["python", script]
    subprocess.check_call(cmd)


@pytest.mark.mgpu
def test_dask_sklearn_demo() -> None:
    script = os.path.join(tm.demo_dir(__file__), "dask", "sklearn_gpu_training.py")
    cmd = ["python", script]
    subprocess.check_call(cmd)


@pytest.mark.mgpu
@pytest.mark.skipif(**tm.no_cupy())
def test_forward_logging_demo() -> None:
    script = os.path.join(tm.demo_dir(__file__), "dask", "forward_logging.py")
    cmd = ["python", script]
    subprocess.check_call(cmd)
