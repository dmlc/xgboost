# Author: Tianqi Chen, Bing Xu
# module for xgboost
import ctypes 
import os
# optinally have scipy sparse, though not necessary
import numpy
import numpy.ctypeslib 
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
xglib.XGDMatrixGetLabel.restype =  ctypes.POINTER( ctypes.c_float )
xglib.XGDMatrixGetWeight.restype =  ctypes.POINTER( ctypes.c_float )
xglib.XGDMatrixGetRow.restype = ctypes.POINTER( REntry )
xglib.XGBoosterCreate.restype = ctypes.c_void_p
xglib.XGBoosterPredict.restype = ctypes.POINTER( ctypes.c_float ) 

def ctypes2numpy( cptr, length ):
    # convert a ctypes pointer array to numpy
    assert isinstance( cptr, ctypes.POINTER( ctypes.c_float ) )
    res = numpy.zeros( length, dtype='float32' )
    assert ctypes.memmove( res.ctypes.data, cptr, length * res.strides[0] )
    return res

# data matrix used in xgboost
class DMatrix:
    # constructor
    def __init__(self, data=None, label=None, missing=0.0, weight = None):
        # force into void_p, mac need to pass things in as void_p
        self.handle = ctypes.c_void_p( xglib.XGDMatrixCreate() )
        if data == None:
            return
        if isinstance(data,str):
            xglib.XGDMatrixLoad(self.handle, ctypes.c_char_p(data.encode('utf-8')), 1)             
        elif isinstance(data,scp.csr_matrix):
            self.__init_from_csr(data)
        elif isinstance(data, numpy.ndarray) and len(data.shape) == 2:
            self.__init_from_npy2d(data, missing)
        else:
            try:
                csr = scp.csr_matrix(data)
                self.__init_from_csr(csr)
            except:
                raise Exception("can not intialize DMatrix from"+str(type(data)))
        if label != None:
            self.set_label(label)
        if weight !=None:
            self.set_weight(weight)

    # convert data from csr matrix
    def __init_from_csr(self,csr):
        assert len(csr.indices) == len(csr.data)
        xglib.XGDMatrixParseCSR( self.handle, 
                                 ( ctypes.c_ulong  * len(csr.indptr) )(*csr.indptr),
                                 ( ctypes.c_uint  * len(csr.indices) )(*csr.indices),
                                 ( ctypes.c_float * len(csr.data) )(*csr.data),
                                 len(csr.indptr), len(csr.data) )
    # convert data from numpy matrix
    def __init_from_npy2d(self,mat,missing):
        data = numpy.array( mat.reshape(mat.size), dtype='float32' )
        xglib.XGDMatrixParseMat( self.handle, 
                                 data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 
                                 mat.shape[0], mat.shape[1], ctypes.c_float(missing) )
    # destructor
    def __del__(self):
        xglib.XGDMatrixFree(self.handle)    
    # load data from file 
    def load(self, fname, silent=True):
        xglib.XGDMatrixLoad(self.handle, ctypes.c_char_p(fname.encode('utf-8')), int(silent))
    # load data from file 
    def save_binary(self, fname, silent=True):
        xglib.XGDMatrixSaveBinary(self.handle, ctypes.c_char_p(fname.encode('utf-8')), int(silent))
    # set label of dmatrix
    def set_label(self, label):
        xglib.XGDMatrixSetLabel(self.handle, (ctypes.c_float*len(label))(*label), len(label) )
    # set group size of dmatrix, used for rank
    def set_group(self, group):
        xglib.XGDMatrixSetGroup(self.handle, (ctypes.c_uint*len(group))(*group), len(group) )
    # set weight of each instances
    def set_weight(self, weight):
        xglib.XGDMatrixSetWeight(self.handle, (ctypes.c_float*len(weight))(*weight), len(weight) )
    # get label from dmatrix
    def get_label(self):
        length = ctypes.c_ulong()
        labels = xglib.XGDMatrixGetLabel(self.handle, ctypes.byref(length))
        return ctypes2numpy( labels, length.value );
    # get weight from dmatrix
    def get_weight(self):
        length = ctypes.c_ulong()
        weights = xglib.XGDMatrixGetWeight(self.handle, ctypes.byref(length))
        return ctypes2numpy( weights, length.value );
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
        return [ (int(row[i].findex),row[i].fvalue) for i in range(length.value) ]

class Booster:
    """learner class """
    def __init__(self, params={}, cache=[]):
        """ constructor, param: """    
        for d in cache:
            assert isinstance(d,DMatrix)
        dmats = ( ctypes.c_void_p  * len(cache) )(*[ d.handle for d in cache])
        self.handle = ctypes.c_void_p( xglib.XGBoosterCreate( dmats, len(cache) ) )
        self.set_param( params )
    def __del__(self):
        xglib.XGBoosterFree(self.handle) 
    def set_param(self, params, pv=None):
        if isinstance(params,dict):
            for k, v in params.items():
                xglib.XGBoosterSetParam(
                    self.handle, ctypes.c_char_p(k.encode('utf-8')), 
                    ctypes.c_char_p(str(v).encode('utf-8')))        
        elif isinstance(params,str) and pv != None:
            xglib.XGBoosterSetParam(
                self.handle, ctypes.c_char_p(params.encode('utf-8')),
                ctypes.c_char_p(str(pv).encode('utf-8')) )
        else:
            for k, v in params:
                xglib.XGBoosterSetParam(
                    self.handle, ctypes.c_char_p(k.encode('utf-8')),
                    ctypes.c_char_p(str(v).encode('utf-8')) )             
    def update(self, dtrain):
        """ update """
        assert isinstance(dtrain, DMatrix)
        xglib.XGBoosterUpdateOneIter( self.handle, dtrain.handle )
    def boost(self, dtrain, grad, hess, bst_group = -1):
        """ update """
        assert len(grad) == len(hess)
        assert isinstance(dtrain, DMatrix)
        xglib.XGBoosterBoostOneIter( self.handle, dtrain.handle,
                                     (ctypes.c_float*len(grad))(*grad),
                                     (ctypes.c_float*len(hess))(*hess),
                                     len(grad), bst_group )
    def update_interact(self, dtrain, action, booster_index=None):
        """ beta: update with specified action"""
        assert isinstance(dtrain, DMatrix)
        if booster_index != None:
            self.set_param('interact:booster_index', str(booster_index))
        xglib.XGBoosterUpdateInteract(
            self.handle, dtrain.handle, ctypes.c_char_p(str(action)) )
    def eval_set(self, evals, it = 0):
        for d in evals:
            assert isinstance(d[0], DMatrix)
            assert isinstance(d[1], str)
        dmats = ( ctypes.c_void_p * len(evals) )(*[ d[0].handle for d in evals])
        evnames = ( ctypes.c_char_p * len(evals) )(
            *[ctypes.c_char_p(d[1].encode('utf-8')) for d in evals])
        xglib.XGBoosterEvalOneIter( self.handle, it, dmats, evnames, len(evals) )
    def eval(self, mat, name = 'eval', it = 0 ):
        self.eval_set( [(mat,name)], it)
    def predict(self, data, bst_group = -1):
        length = ctypes.c_ulong()
        preds = xglib.XGBoosterPredict( self.handle, data.handle, ctypes.byref(length), bst_group)
        return ctypes2numpy( preds, length.value )
    def save_model(self, fname):
        """ save model to file """
        xglib.XGBoosterSaveModel(self.handle, ctypes.c_char_p(fname.encode('utf-8')))
    def load_model(self, fname):
        """load model from file"""
        xglib.XGBoosterLoadModel( self.handle, ctypes.c_char_p(fname.encode('utf-8')) )
    def dump_model(self, fname, fmap=''):
        """dump model into text file"""
        xglib.XGBoosterDumpModel(
            self.handle, ctypes.c_char_p(fname.encode('utf-8')), 
            ctypes.c_char_p(fmap.encode('utf-8')))

def train(params, dtrain, num_boost_round = 10, evals = [], obj=None):
    """ train a booster with given paramaters """
    bst = Booster(params, [dtrain] )
    if obj == None:
        for i in range(num_boost_round):
            bst.update( dtrain )
            if len(evals) != 0:
                bst.eval_set( evals, i )
    else:
        # try customized objective function
        for i in range(num_boost_round):
            pred = bst.predict( dtrain )
            grad, hess = obj( pred, dtrain )
            bst.boost( dtrain, grad, hess )
            if len(evals) != 0:
                bst.eval_set( evals, i )        
    return bst

