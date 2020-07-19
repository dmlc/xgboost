# coding: utf-8
"""Callback interface to specify external GPU memory allocator"""
from .core import _LIB, _check_call


def set_gpu_alloc_callback(allocate_func, deallocate_func):
    """
    Registery custom methods for allocating and deallocating GPU memory

    Parameters
    ----------
    allocate_func : a callback function
        Callback function for allocating GPU memory
    deallocate_func : a callback function
        Callback function for deallocating GPU memory
    """
    _check_call(_LIB.XGBRegisterGPUDeviceAllocator(allocate_func, deallocate_func))
