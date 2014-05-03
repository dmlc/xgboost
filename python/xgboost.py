# module for xgboost
import ctypes 

# load in xgboost library
#xglib = ctypes.cdll.LoadLibrary('./libxgboostpy.so')

# entry type of sparse matrix
class REntry(ctypes.Structure):
    _fields_ = [("findex", ctypes.c_uint), ("fvalue", ctypes.c_float) ]


class DMatrix:
    def __init__(fname = None):
        self.__handle = xglib.
    
