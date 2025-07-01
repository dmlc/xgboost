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


def _get_ordinal(device: Optional[str]) -> int:
    if device is None:
        device = "cuda"

    def current_device() -> int:
        """Get the current GPU ordinal."""
        try:
            from cuda.bindings import runtime as cudart

            status, cur = cudart.cudaGetDevice()
            if status != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(cudart.cudaGetErrorString(status))
            return cur
        except ImportError:
            warnings.warn("Failed to import `cuda`. Use the first device.")
            return 0

    split = device.split(":")
    if len(split) == 1:
        ordinal = current_device()
    elif len(split) == 2:
        ordinal = int(split[1])
    else:
        raise ValueError(f"Invalid device: {device}")
    if split[0] != "gpu" and split[0] != "cuda":
        raise ValueError(f"Invalid device: {device}")
    return ordinal


def get_device_cpu_affinity(device: Optional[str]) -> List[int]:
    """Get optimal affinity using nvml. CUDA-only and requires `nvidia-ml-py`.

    Parameters
    ----------
    device :
        CUDA device. Same as the `device` parameter for the :py:class:`xgboost.Booster`
        and the :py:class:`XGBRegressor`.

    Returns
    -------
    A list of CPU index.

    """
    ordinal = _get_ordinal(device)

    try:
        import pynvml as nm

        cnt = os.cpu_count()
        assert cnt is not None

        nm.nvmlInit()

        hdl = nm.nvmlDeviceGetHandleByIndex(ordinal)
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
    """Set affinity according to nvml. CUDA-only and requires `nvidia-ml-py`.

    Parameters
    ----------
    device :
        CUDA device. Same as the `device` parameter for the :py:class:`xgboost.Booster`
        and the :py:class:`XGBRegressor`.

    """
    cpus = get_device_cpu_affinity(device)
    os.sched_setaffinity(0, cpus)
