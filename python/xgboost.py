# module for xgboost
import ctypes 

# load in xgboost library
xglib = ctypes.cdll.LoadLibrary('./libxgboostpy.so')

# entry type of sparse matrix
class REntry(ctypes.Structure):
    _fields_ = [("findex", ctypes.c_uint), ("fvalue", ctypes.c_float) ]


class DMatrix:
    def __init__(self,fname = None):
        self.__handle = xglib.XGDMatrixCreate();
        if fname != None:
            xglib.XGDMatrixLoad(self.__handle, ctypes.c_char_p(fname), 0)
    def __del__(self):
        xglib.XGDMatrixFree(self.__handle)

dmata = DMatrix('xx.buffer')


