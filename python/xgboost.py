# module for xgboost
import ctypes 
import os
# optinally have scipy sparse, though not necessary
import numpy as np
import scipy.sparse as scp

# set this line correctly
XGBOOST_PATH = os.path.dirname(__file__)+'/libxgboostpy.so'

# entry type of sparse matrix
class REntry(ctypes.Structure):
    _fields_ = [("findex", ctypes.c_uint), ("fvalue", ctypes.c_float) ]

# load in xgboost library
xglib = ctypes.cdll.LoadLibrary(XGBOOST_PATH)

xglib.XGDMatrixCreate.restype = ctypes.c_void_p
xglib.XGDMatrixNumRow.restype = ctypes.c_ulong
xglib.XGDMatrixGetLabel.restype = ctypes.POINTER( ctypes.c_float )
xglib.XGDMatrixGetRow.restype = ctypes.POINTER( REntry )
xglib.XGBoosterPredict.restype = ctypes.POINTER( ctypes.c_float ) 

# data matrix used in xgboost
class DMatrix:
    # constructor
    def __init__(self, data=None, label=None):
        self.handle = xglib.XGDMatrixCreate()
        if data == None:
            return
        if isinstance(data,str):
            xglib.XGDMatrixLoad(self.handle, ctypes.c_char_p(data), 1) 
            
        elif isinstance(data,scp.csr_matrix):
            self.__init_from_csr(data)
        else:
            try:
                csr = scp.csr_matrix(data)
                self.__init_from_csr(csr)
            except:
                raise Exception, "can not intialize DMatrix from"+str(type(data))
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
        xglib.XGDMatrixSetLabel(self.handle, (ctypes.c_float*len(label))(*label), len(label) )
    # set group size of dmatrix, used for rank
    def set_group(self, group):
        xglib.XGDMatrixSetGroup(self.handle, (ctypes.c_uint*len(group))(*group), len(group) )
    # set weight of each instances
    def set_weight(self, weight):
        xglib.XGDMatrixSetWeight(self.handle, (ctypes.c_uint*len(weight))(*weight), len(weight) )
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
    def add_row(self, row):
        xglib.XGDMatrixAddRow(self.handle, (REntry*len(row))(*row), len(row) )
    # get n-throw from DMatrix
    def __getitem__(self, ridx):
        length = ctypes.c_ulong()
        row = xglib.XGDMatrixGetRow(self.handle, ridx, ctypes.byref(length) );
        return [ (int(row[i].findex),row[i].fvalue) for i in xrange(length.value) ]

class Booster:
    """learner class """
    def __init__(self, params, cache=[]):
        """ constructor, param: """    
        for d in cache:
            assert isinstance(d,DMatrix)
        dmats = ( ctypes.c_void_p  * len(cache) )(*[ ctypes.c_void_p(d.handle) for d in cache])
        self.handle = xglib.XGBoosterCreate( dmats, len(cache) )
        self.set_param( params )
    def __del__(self):
        xglib.XGBoosterFree(self.handle) 
    def set_param(self, params,pv=None):
        if isinstance(params,dict):
            for k, v in params.iteritems():
                xglib.XGBoosterSetParam( self.handle, ctypes.c_char_p(k), ctypes.c_char_p(str(v)) )        
        elif isinstance(params,str) and pv != None:
            xglib.XGBoosterSetParam( self.handle, ctypes.c_char_p(params), ctypes.c_char_p(str(pv)) )
        else:
            for k, v in params:
                xglib.XGBoosterSetParam( self.handle, ctypes.c_char_p(k), ctypes.c_char_p(str(v)) )             
    def update(self, dtrain):
        """ update """
        assert isinstance(dtrain, DMatrix)
        xglib.XGBoosterUpdateOneIter( self.handle, dtrain.handle )
    def update_interact(self, dtrain, action, booster_index=None):
        """ beta: update with specified action"""
        assert isinstance(dtrain, DMatrix)
        if booster_index != None:
            self.set_param('interact:booster_index', str(booster_index))
        xglib.XGBoosterUpdateInteract( self.handle, dtrain.handle, ctypes.c_char_p(str(action)) )
    def eval_set(self, evals, it = 0):
        for d in evals:
            assert isinstance(d[0], DMatrix)
            assert isinstance(d[1], str)
        dmats = ( ctypes.c_void_p * len(evals) )(*[ ctypes.c_void_p(d[0].handle) for d in evals])
        evnames = ( ctypes.c_char_p * len(evals) )(*[ ctypes.c_char_p(d[1]) for d in evals])
        xglib.XGBoosterEvalOneIter( self.handle, it, dmats, evnames, len(evals) )
    def eval(self, mat, name = 'eval', it = 0 ):
        self.eval_set( [(mat,name)], it)
    def predict(self, data):
        length = ctypes.c_ulong()
        preds = xglib.XGBoosterPredict( self.handle, data.handle, ctypes.byref(length))
        return [ preds[i] for i in xrange(length.value) ]        
    def save_model(self, fname):
        """ save model to file """
        xglib.XGBoosterSaveModel( self.handle, ctypes.c_char_p(fname) )
    def load_model(self, fname):
        """load model from file"""
        xglib.XGBoosterLoadModel( self.handle, ctypes.c_char_p(fname) )
    def dump_model(self, fname, fmap=''):
        """dump model into text file"""
        xglib.XGBoosterDumpModel( self.handle, ctypes.c_char_p(fname), ctypes.c_char_p(fmap) )

def train(params, dtrain, num_boost_round = 10, evals = []):
    """ train a booster with given paramaters """
    bst = Booster(params, [dtrain] )
    for i in xrange(num_boost_round):
        bst.update( dtrain )
        if len(evals) != 0:
            bst.eval_set( evals, i )
    return bst
    
