# pylint: disable=c-extension-no-member
"""Various helper functions."""

import math
import os
import warnings
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from cuda.bindings import runtime as cudart

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


def _checkcu(status: "cudart.cudaError_t") -> None:
    from cuda.bindings import runtime as cudart

    if status != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(cudart.cudaGetErrorString(status))


def _get_ordinal(device: Optional[str]) -> int:
    if device is None:
        device = "cuda"

    split = device.split(":")
    if len(split) == 1:
        ordinal = 0
    elif len(split) == 2:
        ordinal = int(split[1])
    else:
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
    ordinal = _get_ordinal(device)

    try:
        import pynvml as nm

        def current_device() -> int:
            """Get the current GPU ordinal."""
            from cuda.bindings import runtime as cudart

            status, cur = cudart.cudaGetDevice()
            _checkcu(status)
            return cur

        nm.nvmlInit()

        ordinal = ordinal if ordinal is not None else current_device()
        cpus = get_device_cpu_affinity(device)
        os.sched_setaffinity(0, cpus)

        nm.nvmlShutdown()
    except ImportError:
        msg = (
            "Failed to import pynvml. CPU affinity is not set. "
            "Please install `nvidia-ml-py`"
        )
        warnings.warn(msg, UserWarning)
