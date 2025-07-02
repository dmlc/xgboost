import pytest

import xgboost.testing as tm
from xgboost.utils import get_device_cpu_affinity


@pytest.mark.skipif(**tm.no_multiple(tm.no_cuda(), tm.no_pynvml()))
def test_get_cpu_affinity() -> None:
    cpus = get_device_cpu_affinity("cuda:0")
    assert cpus
    cpus = get_device_cpu_affinity("cuda")
    assert cpus
    cpus = get_device_cpu_affinity(None)
    assert cpus

    with pytest.raises(ValueError, match="Invalid device"):
        get_device_cpu_affinity("cpu")
