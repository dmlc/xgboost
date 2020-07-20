# coding: utf-8
"""Callback interface to specify external GPU memory allocator"""
from .core import _LIB, _check_call, c_str


def set_gpu_alloc_callback(libpath):
    """
    Registery custom methods for allocating and deallocating GPU memory

    Parameters
    ----------
    libpath : str
        Path to shared library containing two functions allocate() and deallocate()
    """
    _check_call(_LIB.XGBRegisterGPUDeviceAllocator(c_str(libpath)))
