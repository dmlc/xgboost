"""
xgboost: eXtreme Gradient Boosting library
Author: Tianqi Chen, Bing Xu

"""
import ctypes
import os
# optinally have scipy sparse, though not necessary
import numpy as np
import sys
import numpy.ctypeslib
import scipy.sparse as scp

# set this line correctly
if os.name == 'nt':
    XGBOOST_PATH = os.path.dirname(__file__)+'/../windows/x64/Release/xgboost_wrapper.dll'
else:
    XGBOOST_PATH = os.path.dirname(__file__)+'/libxgboostwrapper.so'

# load in xgboost library
xglib = ctypes.cdll.LoadLibrary(XGBOOST_PATH)
# DMatrix functions
xglib.XGDMatrixCreateFromFile.restype = ctypes.c_void_p
xglib.XGDMatrixCreateFromCSR.restype = ctypes.c_void_p
xglib.XGDMatrixCreateFromCSC.restype = ctypes.c_void_p
xglib.XGDMatrixCreateFromMat.restype = ctypes.c_void_p
xglib.XGDMatrixSliceDMatrix.restype = ctypes.c_void_p
xglib.XGDMatrixGetFloatInfo.restype = ctypes.POINTER(ctypes.c_float)
xglib.XGDMatrixGetUIntInfo.restype = ctypes.POINTER(ctypes.c_uint)
xglib.XGDMatrixNumRow.restype = ctypes.c_ulong
# booster functions
xglib.XGBoosterCreate.restype = ctypes.c_void_p
xglib.XGBoosterPredict.restype = ctypes.POINTER(ctypes.c_float)
xglib.XGBoosterEvalOneIter.restype = ctypes.c_char_p
xglib.XGBoosterDumpModel.restype = ctypes.POINTER(ctypes.c_char_p)


def ctypes2numpy(cptr, length, dtype):
    """convert a ctypes pointer array to numpy array """
    assert isinstance(cptr, ctypes.POINTER(ctypes.c_float))
    res = numpy.zeros(length, dtype=dtype)
    assert ctypes.memmove(res.ctypes.data, cptr, length * res.strides[0])
    return res

class DMatrix:
    """data matrix used in xgboost"""
    # constructor
    def __init__(self, data, label=None, missing=0.0, weight = None):
        """ constructor of DMatrix

            Args:
                data: string/numpy array/scipy.sparse
                      data source, string type is the path of svmlight format txt file or xgb buffer
                label: list or numpy 1d array, optional
                       label of training data
                missing: float
                         value in data which need to be present as missing value
                weight: list or numpy 1d array, optional
                        weight for each instances
        """
        # force into void_p, mac need to pass things in as void_p
        if data is None:
            self.handle = None
            return
        if isinstance(data, str):
            self.handle = ctypes.c_void_p(
                xglib.XGDMatrixCreateFromFile(ctypes.c_char_p(data.encode('utf-8')), 0))
        elif isinstance(data, scp.csr_matrix):
            self.__init_from_csr(data)
        elif isinstance(data, scp.csc_matrix):
            self.__init_from_csc(data)            
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

    def __init_from_csr(self, csr):
        """convert data from csr matrix"""
        assert len(csr.indices) == len(csr.data)
        self.handle = ctypes.c_void_p(xglib.XGDMatrixCreateFromCSR(
            (ctypes.c_ulong  * len(csr.indptr))(*csr.indptr),
            (ctypes.c_uint  * len(csr.indices))(*csr.indices),
            (ctypes.c_float * len(csr.data))(*csr.data),
            len(csr.indptr), len(csr.data)))

    def __init_from_csc(self, csc):
        """convert data from csr matrix"""
        assert len(csc.indices) == len(csc.data)
        self.handle = ctypes.c_void_p(xglib.XGDMatrixCreateFromCSC(
            (ctypes.c_ulong  * len(csc.indptr))(*csc.indptr),
            (ctypes.c_uint * len(csc.indices))(*csc.indices),
            (ctypes.c_float * len(csc.data))(*csc.data),
            len(csc.indptr), len(csc.data)))

    def __init_from_npy2d(self,mat,missing):
        """convert data from numpy matrix"""
        data = numpy.array(mat.reshape(mat.size), dtype='float32')
        self.handle = ctypes.c_void_p(xglib.XGDMatrixCreateFromMat(
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            mat.shape[0], mat.shape[1], ctypes.c_float(missing)))

    def __del__(self):
        """destructor"""
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

    def save_binary(self, fname, silent=True):
        """save DMatrix to XGBoost buffer
            Args:
                fname: string
                       name of buffer file
                slient: bool, option
                       whether print info
           Returns:
                None
        """
        xglib.XGDMatrixSaveBinary(self.handle, ctypes.c_char_p(fname.encode('utf-8')), int(silent))

    def set_label(self, label):
        """set label of dmatrix
            Args:
                label: list
                       label for DMatrix
            Returns:
                None
        """
        self.set_float_info('label', label)

    def set_weight(self, weight):
        """set weight of each instances
            Args:
                weight: float
                        weight for positive instance
            Returns:
                None
        """
        self.set_float_info('weight', weight)

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

    def set_group(self, group):
        """set group size of dmatrix, used for rank
            Args:
                group:

            Returns:
                None
        """
        xglib.XGDMatrixSetGroup(self.handle, (ctypes.c_uint*len(group))(*group), len(group))

    def get_label(self):
        """get label from dmatrix
            Args:
                None
            Returns:
                list, label of data
        """
        return self.get_float_info('label')

    def get_weight(self):
        """get weight from dmatrix
            Args:
                None
            Returns:
                float, weight
        """
        return self.get_float_info('weight')
    def get_base_margin(self):
        """get base_margin from dmatrix
            Args:
                None
            Returns:
                float, base margin
        """
        return self.get_float_info('base_margin')
    def num_row(self):
        """get number of rows
            Args:
                None
            Returns:
                int, num rows
        """
        return xglib.XGDMatrixNumRow(self.handle)
    def slice(self, rindex):
        """slice the DMatrix to return a new DMatrix that only contains rindex
            Args:
                rindex: list
                        list of index to be chosen
            Returns:
                res: DMatrix
                     new DMatrix with chosen index
        """
        res = DMatrix(None)
        res.handle = ctypes.c_void_p(xglib.XGDMatrixSliceDMatrix(
            self.handle, (ctypes.c_int*len(rindex))(*rindex), len(rindex)))
        return res

class Booster:
    """learner class """
    def __init__(self, params={}, cache=[], model_file = None):
        """ constructor
            Args:
                params: dict
                        params for boosters
                cache: list
                        list of cache item
                model_file: string
                        path of model file
            Returns:
                None
        """
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

    def update(self, dtrain, it, fobj=None):
        """
        update
            Args:
                dtrain: DMatrix
                        the training DMatrix
                it: int
                    current iteration number
                fobj: function
                    cutomzied objective function
            Returns:
                None
        """
        assert isinstance(dtrain, DMatrix)
        if fobj is None:
            xglib.XGBoosterUpdateOneIter(self.handle, it, dtrain.handle)
        else:
            pred = self.predict( dtrain )
            grad, hess = fobj( pred, dtrain )
            self.boost( dtrain, grad, hess )

    def boost(self, dtrain, grad, hess):
        """ update
            Args:
                dtrain: DMatrix
                        the training DMatrix
                grad: list
                        the first order of gradient
                hess: list
                        the second order of gradient
        """
        assert len(grad) == len(hess)
        assert isinstance(dtrain, DMatrix)
        xglib.XGBoosterBoostOneIter(self.handle, dtrain.handle,
                                    (ctypes.c_float*len(grad))(*grad),
                                    (ctypes.c_float*len(hess))(*hess),
                                    len(grad))
    def eval_set(self, evals, it = 0, feval = None):
        """evaluates by metric
            Args:
                evals: list of tuple (DMatrix, string)
                       lists of items to be evaluated
                it: int
                    current iteration
                feval: function
                       custom evaluation function
            Returns:
                evals result
        """
        if feval is None:
            for d in evals:
                assert isinstance(d[0], DMatrix)
                assert isinstance(d[1], str)
            dmats = (ctypes.c_void_p * len(evals) )(*[ d[0].handle for d in evals])
            evnames = (ctypes.c_char_p * len(evals))(
                * [ctypes.c_char_p(d[1].encode('utf-8')) for d in evals])
            return xglib.XGBoosterEvalOneIter(self.handle, it, dmats, evnames, len(evals))
        else:
            res = '[%d]' % it
            for dm, evname in evals:
                name, val = feval(self.predict(dm), dm)
                res += '\t%s-%s:%f' % (evname, name, val)
            return res
    def eval(self, mat, name = 'eval', it = 0):
        return self.eval_set( [(mat,name)], it)
    def predict(self, data, output_margin=False, ntree_limit=0):
        """
        predict with data
            Args:
                data: DMatrix
                      the dmatrix storing the input
                output_margin: bool
                               whether output raw margin value that is untransformed

                ntree_limit: int
                             limit number of trees in prediction, default to 0, 0 means using all the trees
            Returns:
                numpy array of prediction
        """
        length = ctypes.c_ulong()
        preds = xglib.XGBoosterPredict(self.handle, data.handle,
                                       int(output_margin), ntree_limit, ctypes.byref(length))
        return ctypes2numpy(preds, length.value, 'float32')
    def save_model(self, fname):
        """ save model to file
            Args:
                fname: string
                       file name of saving model
            Returns:
                None
        """
        xglib.XGBoosterSaveModel(self.handle, ctypes.c_char_p(fname.encode('utf-8')))
    def load_model(self, fname):
        """load model from file
            Args:
                fname: string
                       file name of saving model
            Returns:
                None
        """
        xglib.XGBoosterLoadModel( self.handle, ctypes.c_char_p(fname.encode('utf-8')) )
    def dump_model(self, fo, fmap=''):
        """dump model into text file
            Args:
                fo: string
                    file name to be dumped
                fmap: string, optional
                      file name of feature map names
            Returns:
                None
        """
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
            print (tree)
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

def train(params, dtrain, num_boost_round = 10, evals = [], obj=None, feval=None):
    """ train a booster with given paramaters
        Args:
            params: dict
                    params of booster
            dtrain: DMatrix
                    data to be trained
            num_boost_round: int 
                             num of round to be boosted
            watchlist: list of pairs (DMatrix, string) 
                       list of items to be evaluated during training, this allows user to watch performance on validation set
            obj:  function
                   cutomized objective function
            feval: function
                   cutomized evaluation function
        Returns: Booster model trained
    """
    bst = Booster(params, [dtrain]+[ d[0] for d in evals ] )
    for i in range(num_boost_round):
        bst.update( dtrain, i, obj )
        if len(evals) != 0:
            bst_eval_set=bst.eval_set(evals, i, feval)
            if isinstance(bst_eval_set,str):
                sys.stderr.write(bst_eval_set+'\n')
            else:
                sys.stderr.write(bst_eval_set.decode()+'\n')
    return bst

class CVPack:
    def __init__(self, dtrain, dtest, param):
        self.dtrain = dtrain
        self.dtest = dtest
        self.watchlist = watchlist = [ (dtrain,'train'), (dtest, 'test') ]
        self.bst = Booster(param, [dtrain,dtest])
    def update(self, r, fobj):
        self.bst.update(self.dtrain, r, fobj)
    def eval(self, r, feval):
        return self.bst.eval_set(self.watchlist, r, feval)

def mknfold(dall, nfold, param, seed, evals=[], fpreproc = None):
    """
    mk nfold list of cvpack from randidx
    """
    np.random.seed(seed)
    randidx = np.random.permutation(dall.num_row())
    kstep = len(randidx) / nfold
    idset = [randidx[ (i*kstep) : min(len(randidx),(i+1)*kstep) ] for i in range(nfold)]
    ret = []
    for k in range(nfold):
        dtrain = dall.slice(np.concatenate([idset[i] for i in range(nfold) if k != i]))
        dtest = dall.slice(idset[k])
        # run preprocessing on the data set if needed
        if fpreproc is not None:
            dtrain, dtest, tparam = fpreproc(dtrain, dtest, param.copy())
        else:
            tparam = param
        plst = list(tparam.items()) + [('eval_metric', itm) for itm in evals]
        ret.append(CVPack(dtrain, dtest, plst))
    return ret

def aggcv(rlist, show_stdv=True):
    """
    aggregate cross validation results
    """
    cvmap = {}
    ret = rlist[0].split()[0]
    for line in rlist:
        arr = line.split()
        assert ret == arr[0]
        for it in arr[1:]:
            if not isinstance(it,str):
                it=it.decode()
            k, v  = it.split(':')
            if k not in cvmap:
                cvmap[k] = []
            cvmap[k].append(float(v))
    for k, v in sorted(cvmap.items(), key = lambda x:x[0]):
        v = np.array(v)
        if not isinstance(ret,str):
            ret = ret.decode()
        if show_stdv:
            ret += '\tcv-%s:%f+%f' % (k, np.mean(v), np.std(v))
        else:
            ret += '\tcv-%s:%f' % (k, np.mean(v))
    return ret

def cv(params, dtrain, num_boost_round = 10, nfold=3, metrics=[], \
        obj = None, feval = None, fpreproc = None, show_stdv = True, seed = 0):
    """ cross validation  with given paramaters
        Args:
            params: dict
                    params of booster
            dtrain: DMatrix
                    data to be trained
            num_boost_round: int
                             num of round to be boosted
            nfold: int
                   number of folds to do cv
            metrics: list of strings
                     evaluation metrics to be watched in cv
            obj: function 
                 custom objective function
            feval: function
                   custom evaluation function
            fpreproc: function
                      preprocessing function that takes dtrain, dtest,
                      param and return transformed version of dtrain, dtest, param
            show_stdv: bool
                       whether display standard deviation
            seed: int 
                  seed used to generate the folds, this is passed to numpy.random.seed

        Returns: list(string) of evaluation history
    """
    results = []
    cvfolds = mknfold(dtrain, nfold, params, seed, metrics, fpreproc)
    for i in range(num_boost_round):
        for f in cvfolds:
            f.update(i, obj)
        res = aggcv([f.eval(i, feval) for f in cvfolds], show_stdv)
        sys.stderr.write(res+'\n')
        results.append(res)
    return results
