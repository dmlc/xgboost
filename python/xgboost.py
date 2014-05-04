# module for xgboost
import ctypes 
# optinally have scipy sparse, though not necessary
import scipy.sparse as scp

# entry type of sparse matrix
class REntry(ctypes.Structure):
    _fields_ = [("findex", ctypes.c_uint), ("fvalue", ctypes.c_float) ]

# load in xgboost library
xglib = ctypes.cdll.LoadLibrary('./libxgboostpy.so')

xglib.XGDMatrixCreate.restype = ctypes.c_void_p
xglib.XGDMatrixNumRow.restype = ctypes.c_ulong
xglib.XGDMatrixGetLabel.restype = ctypes.POINTER( ctypes.c_float )
xglib.XGDMatrixGetRow.restype = ctypes.POINTER( REntry )

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
    def load(self, fname, silent=True):
        xglib.XGDMatrixLoad(self.handle, ctypes.c_char_p(fname), int(silent))
    # load data from file 
    def save_binary(self, fname, silent=True):
        xglib.XGDMatrixSaveBinary(self.handle, ctypes.c_char_p(fname), int(silent))
    # set label of dmatrix
    def set_label(self, label):
        xglib.XGDMatrixSetLabel(self.handle, (ctypes.c_float*len(label))(*label), len(label) );
    # get label from dmatrix
    def get_label(self):
        length = ctypes.c_ulong()
        labels = xglib.XGDMatrixGetLabel(self.handle, ctypes.byref(length));
        return [ labels[i] for i in xrange(length.value) ]
    # clear everything
    def clear(self):
        xglib.XGDMatrixClear(self.handle)
    def num_row(self):
        return xglib.XGDMatrixNumRow(self.handle)
    # append a row to DMatrix
    def add_row(self, row, label):
        xglib.XGDMatrixAddRow(self.handle, (REntry*len(row))(*row), len(row), label )
    # get n-throw from DMatrix
    def __getitem__(self, ridx):
        length = ctypes.c_ulong()
        row = xglib.XGDMatrixGetRow(self.handle, ridx, ctypes.byref(length) );
        return [ (int(row[i].findex),row[i].fvalue) for i in xrange(length.value) ]



mat = DMatrix('xx.buffer')
print mat.num_row()
mat.clear()
