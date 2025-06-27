"""Various helper functions."""

import math
import os
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from cuda.bindings import runtime as cudart

_mask_size = 64


class _BitField64:
    """A simplified version of the bit field in XGBoost."""

    def __init__(self, mask: Sequence) -> None:
        self.mask: list[int] = []
        for m in mask:
            self.mask.append(m)

    @staticmethod
    def to_bit(i: int) -> tuple[int, int]:
        int_pos, bit_pos = 0, 0
        if i == 0:
            return int_pos, bit_pos

        int_pos = i // _mask_size
        bit_pos = i % _mask_size
        return int_pos, bit_pos

    def check(self, i: int) -> bool:
        ip, bp = self.to_bit(i)
        value = self.mask[ip]
        test_bit = 1 << bp
        res = value & test_bit
        return bool(res)


def _checkcu(status: "cudart.cudaError_t") -> None:
    from cuda.bindings import runtime as cudart

    if status != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(cudart.cudaGetErrorString(status))


def _get_uuid(ordinal: int) -> str:
    """Construct a string representation of UUID."""
    from cuda.bindings import runtime as cudart

    status, prop = cudart.cudaGetDeviceProperties(ordinal)
    _checkcu(status)

    dash_pos = {0, 4, 6, 8, 10}
    uuid = "GPU"

    for i in range(16):
        if i in dash_pos:
            uuid += "-"
        h = hex(0xFF & np.int32(prop.uuid.bytes[i]))
        assert h[:2] == "0x"
        h = h[2:]

        while len(h) < 2:
            h = "0" + h
        uuid += h
    return uuid


def get_cpu_affinity(ordinal: int) -> list[int]:
    """Get optimal affinity using nvml.

    Parameters
    ----------
    ordinal :
        CUDA device ordinal.

    Returns
    -------
    A list of CPU index.

    """
    import pynvml as nm

    cnt = os.cpu_count()
    assert cnt is not None

    uuid = _get_uuid(ordinal)
    hdl = nm.nvmlDeviceGetHandleByUUID(uuid)

    affinity = nm.nvmlDeviceGetCpuAffinity(
        hdl,
        math.ceil(cnt / _mask_size),
    )
    cpumask = _BitField64(affinity)

    ints = []
    for i, mask in enumerate(affinity):
        ints.append(mask)

    cpus = []
    for i in range(cnt):
        if cpumask.check(i):
            cpus.append(i)

    return cpus


def set_cpu_affinity() -> None:
    """Set affinity according to nvml."""
    import pynvml as nm

    def current_device() -> int:
        """Get the current GPU ordinal."""
        from cuda.bindings import runtime as cudart

        status, ordinal = cudart.cudaGetDevice()
        _checkcu(status)
        return ordinal

    nm.nvmlInit()

    cpus = get_cpu_affinity(current_device())
    os.sched_setaffinity(0, cpus)

    nm.nvmlShutdown()
