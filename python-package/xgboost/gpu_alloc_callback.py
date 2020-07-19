# coding: utf-8
"""Callback interface to specify external GPU memory allocator"""
import ctypes
from .core import _LIB, _check_call

def set_gpu_alloc_callback(allocate_func, deallocate_func):
    _check_call(_LIB.XGBRegisterGPUDeviceAllocator(allocate_func, deallocate_func))

