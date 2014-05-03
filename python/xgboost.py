# module for xgboost
import ctypes 
import numpy
import scipy.sparse as scp

# load in xgboost library
xglib = ctypes.cdll.LoadLibrary('./libxgboostpy.so')

# entry type of sparse matrix
class REntry(ctypes.Structure):
    _fields_ = [("findex", ctypes.c_uint), ("fvalue", ctypes.c_float) ]

# data matrix used in xgboost
class DMatrix:
    # constructor
    def __init__(self, data=None, label=None):
        self.handle = xglib.XGDMatrixCreate();
        if data == None:
            return        
        if type(data) is str:
            xglib.XGDMatrixLoad(self.handle, ctypes.c_char_p(data), 1) 
        elif type(data) is scp.csr_matrix:
            self.__init_from_csr(data)
        else:
            try:
                csr = scp.csr_matrix(data)
                self.__init_from_csr(data)
            except:
                raise "DMatrix", "can not intialize DMatrix from"+type(data)                
        if label != None:
            self.set_label(label)

    # convert data from csr matrix
    def __init_from_csr(self,csr):
        assert len(csr.indices) == len(csr.data)
        xglib.XGDMatrixParseCSR( self.handle, 
                                 ( ctypes.c_ulong  * len(csr.indptr) )(*csr.indptr),
                                 ( ctypes.c_uint  * len(csr.indices) )(*csr.indices),
                                 ( ctypes.c_float * len(csr.data) )(*csr.data),
                                 len(csr.indptr), len(csr.data) )
    # destructor
    def __del__(self):
        xglib.XGDMatrixFree(self.handle)
    # load data from file 
    def load(self, fname):
        xglib.XGDMatrixLoad(self.handle, ctypes.c_char_p(fname), 1)
    # set label of dmatrix
    def set_label(self, label):
        xglib.XGDMatrixSetLabel(self.handle, (ctypes.c_float*len(label))(*label), len(label) );
    # get label from dmatrix
    def get_label(self):
        GetLabel = xglib.XGDMatrixGetLabel
        GetLabel.restype = ctypes.POINTER( ctypes.c_float )
        length = ctypes.c_ulong()
        labels = GetLabel(self.handle, ctypes.byref(length));
        return [ labels[i] for i in xrange(length.value) ]
    # append a row to DMatrix
    def add_row(self, row):
        xglib.XGDMatrixAddRow(self.handle, (REntry*len(row))(*row), len(row) );
    

mat = DMatrix('xx.buffer')
lb = mat.get_label()
print len(lb)
mat.set_label(lb)
mat.add_row( [(1,2), (3,4)] )
