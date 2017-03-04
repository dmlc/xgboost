# -*- coding: utf-8 -*-
import unittest
import ctypes
import xgboost as xgb

class TestAPIInternals(unittest.TestCase):
    """
    For usage with Travis. Allows unit integrity tests to be performed that validate
        xbgoost internals continue to work as expected. Tests that cannot be performed
        at the client should be implemented in lib.XGPerformIntegrityTests().

        On failure, this will throw an exception.
    """

    def test_api_internals(self):
        lib_path = xgb.find_lib_path()
        if len(lib_path) == 0:
            return None
        lib = ctypes.cdll.LoadLibrary(lib_path[0])
        lib.XGBGetLastError.restype = ctypes.c_char_p

        # this either works or it doesn't (throws exception)
        ret = lib.XGPerformIntegrityTests()

        if ret != 0:
            raise NameError(lib.XGBGetLastError())

# unittest.main()
