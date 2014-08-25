# Author: Tianqi Chen, Bing Xu
# module for xgboost
import ctypes
import os
# optinally have scipy sparse, though not necessary
import numpy
import sys
import numpy.ctypeslib
import scipy.sparse as scp

# set this line correctly
XGBOOST_PATH = os.path.dirname(__file__)+'/libxgboostwrapper.so'

# load in xgboost library
xglib = ctypes.cdll.LoadLibrary(XGBOOST_PATH)

xglib.XGDMatrixCreateFromFile.restype = ctypes.c_void_p
xglib.XGDMatrixCreateFromCSR.restype = ctypes.c_void_p
xglib.XGDMatrixCreateFromMat.restype = ctypes.c_void_p
xglib.XGDMatrixSliceDMatrix.restype = ctypes.c_void_p
xglib.XGDMatrixGetFloatInfo.restype = ctypes.POINTER(ctypes.c_float)
xglib.XGDMatrixGetUIntInfo.restype = ctypes.POINTER(ctypes.c_uint)
xglib.XGDMatrixNumRow.restype = ctypes.c_ulong

xglib.XGBoosterCreate.restype = ctypes.c_void_p
xglib.XGBoosterPredict.restype = ctypes.POINTER(ctypes.c_float)
xglib.XGBoosterEvalOneIter.restype = ctypes.c_char_p
xglib.XGBoosterDumpModel.restype = ctypes.POINTER(ctypes.c_char_p)


def ctypes2numpy(cptr, length, dtype):
    # convert a ctypes pointer array to numpy
    assert isinstance(cptr, ctypes.POINTER(ctypes.c_float))
    res = numpy.zeros(length, dtype=dtype)
    assert ctypes.memmove(res.ctypes.data, cptr, length * res.strides[0])
    return res

# data matrix used in xgboost
class DMatrix:
    # constructor
    def __init__(self, data, label=None, missing=0.0, weight = None):
        # force into void_p, mac need to pass things in as void_p
        if data == None:
            self.handle = None
            return
        if isinstance(data, str):
            self.handle = ctypes.c_void_p(
                xglib.XGDMatrixCreateFromFile(ctypes.c_char_p(data.encode('utf-8')), 1))
        elif isinstance(data, scp.csr_matrix):
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
    def __init_from_csr(self, csr):
        assert len(csr.indices) == len(csr.data)
        self.handle = ctypes.c_void_p(xglib.XGDMatrixCreateFromCSR(
            (ctypes.c_ulong  * len(csr.indptr))(*csr.indptr),
            (ctypes.c_uint  * len(csr.indices))(*csr.indices),
            (ctypes.c_float * len(csr.data))(*csr.data),
            len(csr.indptr), len(csr.data)))
    # convert data from numpy matrix
    def __init_from_npy2d(self,mat,missing):
        data = numpy.array(mat.reshape(mat.size), dtype='float32')
        self.handle = ctypes.c_void_p(xglib.XGDMatrixCreateFromMat(
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            mat.shape[0], mat.shape[1], ctypes.c_float(missing)))
    # destructor
    def __del__(self):
        xglib.XGDMatrixFree(self.handle)
    def get_float_info(self, field):
        length = ctypes.c_ulong()
        ret = xglib.XGDMatrixGetFloatInfo(self.handle, ctypes.c_char_p(field.encode('utf-8')),
                                          ctypes.byref(length))
        return ctypes2numpy(ret, length.value, 'float32')
    def get_uint_info(self, field):
        length = ctypes.c_ulong()
        ret = xglib.XGDMatrixGetUIntInfo(self.handle, ctypes.c_char_p(field.encode('utf-8')),
                                         ctypes.byref(length))
        return ctypes2numpy(ret, length.value, 'uint32')
    def set_float_info(self, field, data):
        xglib.XGDMatrixSetFloatInfo(self.handle, ctypes.c_char_p(field.encode('utf-8')),
                                    (ctypes.c_float*len(data))(*data), len(data))
    def set_uint_info(self, field, data):
        xglib.XGDMatrixSetUIntInfo(self.handle, ctypes.c_char_p(field.encode('utf-8')),
                                   (ctypes.c_uint*len(data))(*data), len(data))
    # load data from file
    def save_binary(self, fname, silent=True):
        xglib.XGDMatrixSaveBinary(self.handle, ctypes.c_char_p(fname.encode('utf-8')), int(silent))
    # set label of dmatrix
    def set_label(self, label):
        self.set_float_info('label', label)
    # set weight of each instances
    def set_weight(self, weight):
        self.set_float_info('weight', weight)
    # set initialized margin prediction
    def set_base_margin(self, margin):
        """
        set base margin of booster to start from
        this can be used to specify a prediction value of
        existing model to be base_margin
        However, remember margin is needed, instead of transformed prediction
        e.g. for logistic regression: need to put in value before logistic transformation
        see also example/demo.py
        """
        self.set_float_info('base_margin', margin)
    # set group size of dmatrix, used for rank
    def set_group(self, group):
        xglib.XGDMatrixSetGroup(self.handle, (ctypes.c_uint*len(group))(*group), len(group))
    # get label from dmatrix
    def get_label(self):
        return self.get_float_info('label')
    # get weight from dmatrix
    def get_weight(self):
        return self.get_float_info('weight')
    # get base_margin from dmatrix
    def get_base_margin(self):
        return self.get_float_info('base_margin')
    def num_row(self):
        return xglib.XGDMatrixNumRow(self.handle)
    # slice the DMatrix to return a new DMatrix that only contains rindex
    def slice(self, rindex):
        res = DMatrix(None)
        res.handle = ctypes.c_void_p(xglib.XGDMatrixSliceDMatrix(
            self.handle, (ctypes.c_int*len(rindex))(*rindex), len(rindex)))
        return res

class Booster:
    """learner class """
    def __init__(self, params={}, cache=[], model_file = None):
        """ constructor, param: """
        for d in cache:
            assert isinstance(d, DMatrix)
        dmats = (ctypes.c_void_p  * len(cache))(*[ d.handle for d in cache])
        self.handle = ctypes.c_void_p(xglib.XGBoosterCreate(dmats, len(cache)))
        self.set_param({'seed':0})
        self.set_param(params)
        if model_file != None:
            self.load_model(model_file)
    def __del__(self):
        xglib.XGBoosterFree(self.handle)
    def set_param(self, params, pv=None):
        if isinstance(params, dict):
            for k, v in params.items():
                xglib.XGBoosterSetParam(
                    self.handle, ctypes.c_char_p(k.encode('utf-8')),
                    ctypes.c_char_p(str(v).encode('utf-8')))
        elif isinstance(params,str) and pv != None:
            xglib.XGBoosterSetParam(
                self.handle, ctypes.c_char_p(params.encode('utf-8')),
                ctypes.c_char_p(str(pv).encode('utf-8')))
        else:
            for k, v in params:
                xglib.XGBoosterSetParam(
                    self.handle, ctypes.c_char_p(k.encode('utf-8')),
                    ctypes.c_char_p(str(v).encode('utf-8')))
    def update(self, dtrain, it):
        """
        update
          dtrain: the training DMatrix
          it: current iteration number
        """
        assert isinstance(dtrain, DMatrix)
        xglib.XGBoosterUpdateOneIter(self.handle, it, dtrain.handle)
    def boost(self, dtrain, grad, hess):
        """ update """
        assert len(grad) == len(hess)
        assert isinstance(dtrain, DMatrix)
        xglib.XGBoosterBoostOneIter(self.handle, dtrain.handle,
                                    (ctypes.c_float*len(grad))(*grad),
                                    (ctypes.c_float*len(hess))(*hess),
                                    len(grad))
    def eval_set(self, evals, it = 0):
        for d in evals:
            assert isinstance(d[0], DMatrix)
            assert isinstance(d[1], str)
        dmats = (ctypes.c_void_p * len(evals) )(*[ d[0].handle for d in evals])
        evnames = (ctypes.c_char_p * len(evals))(
            * [ctypes.c_char_p(d[1].encode('utf-8')) for d in evals])
        return xglib.XGBoosterEvalOneIter(self.handle, it, dmats, evnames, len(evals))
    def eval(self, mat, name = 'eval', it = 0):
        return self.eval_set( [(mat,name)], it)
    def predict(self, data, output_margin=False):
        """
        predict with data
            data: the dmatrix storing the input
            output_margin: whether output raw margin value that is untransformed
        """
        length = ctypes.c_ulong()
        preds = xglib.XGBoosterPredict(self.handle, data.handle,
                                       int(output_margin), ctypes.byref(length))
        return ctypes2numpy(preds, length.value, 'float32')
    def save_model(self, fname):
        """ save model to file """
        xglib.XGBoosterSaveModel(self.handle, ctypes.c_char_p(fname.encode('utf-8')))
    def load_model(self, fname):
        """load model from file"""
        xglib.XGBoosterLoadModel( self.handle, ctypes.c_char_p(fname.encode('utf-8')) )
    def dump_model(self, fo, fmap=''):
        """dump model into text file"""
        if isinstance(fo,str):
            fo = open(fo,'w')
            need_close = True
        else:
            need_close = False
        ret = self.get_dump(fmap)
        for i in range(len(ret)):
            fo.write('booster[%d]:\n' %i)
            fo.write( ret[i] )
        if need_close:
            fo.close()
    def get_dump(self, fmap=''):
        """get dump of model as list of strings """
        length = ctypes.c_ulong()
        sarr = xglib.XGBoosterDumpModel(self.handle, ctypes.c_char_p(fmap.encode('utf-8')), ctypes.byref(length))
        res = []
        for i in range(length.value):
            res.append( str(sarr[i]) )
        return res
    def get_fscore(self, fmap=''):
        """ get feature importance of each feature """
        trees = self.get_dump(fmap)
        fmap = {}
        for tree in trees:
            print tree
            for l in tree.split('\n'):
                arr = l.split('[')
                if len(arr) == 1:
                    continue
                fid = arr[1].split(']')[0]
                fid = fid.split('<')[0]
                if fid not in fmap:
                    fmap[fid] = 1
                else:
                    fmap[fid]+= 1
        return fmap

def evaluate(bst, evals, it, feval = None):
    """evaluation on eval set"""
    if feval != None:
        res = '[%d]' % it
        for dm, evname in evals:
            name, val = feval(bst.predict(dm), dm)
            res += '\t%s-%s:%f' % (evname, name, val)
    else:
        res = bst.eval_set(evals, it)

    return res

def train(params, dtrain, num_boost_round = 10, evals = [], obj=None, feval=None):
    """ train a booster with given paramaters """
    bst = Booster(params, [dtrain]+[ d[0] for d in evals ] )
    if obj == None:
        for i in range(num_boost_round):
            bst.update( dtrain, i )
            if len(evals) != 0:
                sys.stderr.write(evaluate(bst, evals, i, feval)+'\n')
    else:
        # try customized objective function
        for i in range(num_boost_round):
            pred = bst.predict( dtrain )
            grad, hess = obj( pred, dtrain )
            bst.boost( dtrain, grad, hess )
            if len(evals) != 0:
                sys.stderr.write(evaluate(bst, evals, i, feval)+'\n')
    return bst
