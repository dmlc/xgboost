# -*- coding: utf-8 -*-
"""Pytest fixtures to initialize tests"""
import ctypes
import os
import xgboost


def pytest_sessionstart(session):
    libpath = os.environ.get('XGBOOST_RMM_TEST_LIBPATH', None)
    if not libpath:
        return
    print('Initializing RMM pool')
    bridge = ctypes.cdll.LoadLibrary(libpath)
    xgboost.set_gpu_alloc_callback(bridge.allocate, bridge.deallocate)
