# pylint: disable=c-extension-no-member
"""Various helper functions."""

import math
import os
import warnings
from typing import List, Optional, Sequence, Tuple

_MASK_SIZE = 64


class _BitField64:
    """A simplified version of the bit field in XGBoost C++."""

    def __init__(self, mask: Sequence) -> None:
        self.mask: List[int] = []
        for m in mask:
            self.mask.append(m)

    @staticmethod
    def to_bit(i: int) -> Tuple[int, int]:
        """Split the index into bit position and value position."""
        int_pos, bit_pos = 0, 0
        if i == 0:
            return int_pos, bit_pos

        int_pos = i // _MASK_SIZE
        bit_pos = i % _MASK_SIZE
        return int_pos, bit_pos

    def check(self, i: int) -> bool:
        """Check whether the i bit is set."""
        ip, bp = self.to_bit(i)
        value = self.mask[ip]
        test_bit = 1 << bp
        res = value & test_bit
        return bool(res)


def _get_uuid(ordinal: int) -> str:
    """Construct a string representation of UUID."""
    from cuda.bindings import runtime as cudart

    status, prop = cudart.cudaGetDeviceProperties(ordinal)
    if status != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(cudart.cudaGetErrorString(status))

    dash_pos = {0, 4, 6, 8, 10}
    uuid = "GPU"

    for i in range(16):
        if i in dash_pos:
            uuid += "-"
        h = hex(0xFF & int(prop.uuid.bytes[i]))
        assert h[:2] == "0x"
        h = h[2:]

        while len(h) < 2:
            h = "0" + h
        uuid += h
    return uuid


def _get_ordinal(device: Optional[str]) -> int:
    if device is None:
        device = "cuda"

    def current_device() -> int:
        """Get the current GPU ordinal."""
        from cuda.bindings import runtime as cudart

        status, cur = cudart.cudaGetDevice()
        if status != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(cudart.cudaGetErrorString(status))
        return cur

    split = device.split(":")
    if len(split) == 1:
        ordinal = current_device()
    elif len(split) == 2:
        ordinal = int(split[1])
    else:
        raise ValueError(f"Invalid device: {device}")
    if split[0] not in {"gpu", "cuda"}:
        raise ValueError(f"Invalid device: {device}")
    return ordinal


def get_device_cpu_affinity(device: Optional[str]) -> List[int]:
    """Get optimal affinity using `nvidia-ml-py
    <https://pypi.org/project/nvidia-ml-py/>`__ and `cuda-python
    <https://nvidia.github.io/cuda-python/latest/>`__. This is CUDA-only.

    Parameters
    ----------
    device :
        CUDA device. Same as the `device` parameter for the :py:class:`xgboost.Booster`
        and the :py:class:`XGBRegressor`.

    Returns
    -------
    A list of CPU index.

    """
    try:
        ordinal = _get_ordinal(device)
        uuid = _get_uuid(ordinal)
    except ImportError:
        warnings.warn("Failed to import `cuda`. Please install `cuda-python`.")
        return []

    try:
        import pynvml as nm

        cnt = os.cpu_count()
        assert cnt is not None

        nm.nvmlInit()

        hdl = nm.nvmlDeviceGetHandleByUUID(uuid)
        affinity = nm.nvmlDeviceGetCpuAffinity(
            hdl,
            math.ceil(cnt / _MASK_SIZE),
        )
        cpumask = _BitField64(affinity)
        cpus = list(filter(cpumask.check, range(cnt)))

        nm.nvmlShutdown()
        return cpus
    except ImportError:
        warnings.warn(
            "Failed to import pynvml. Please install `nvidia-ml-py`", UserWarning
        )

        return []


def set_device_cpu_affinity(device: Optional[str] = None) -> None:
    """Set optimal affinity using `nvidia-ml-py
    <https://pypi.org/project/nvidia-ml-py/>`__ and `cuda-python
    <https://nvidia.github.io/cuda-python/latest/>`__. This is CUDA-only.

    Parameters
    ----------
    device :
        CUDA device. Same as the `device` parameter for the :py:class:`xgboost.Booster`
        and the :py:class:`XGBRegressor`.

    """
    cpus = get_device_cpu_affinity(device)
    os.sched_setaffinity(0, cpus)
