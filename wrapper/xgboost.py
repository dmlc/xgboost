# coding: utf-8
"""
xgboost: eXtreme Gradient Boosting library

Version: 0.40
Authors: Tianqi Chen, Bing Xu
Early stopping by Zygmunt Zając
"""

from __future__ import absolute_import

import os
import sys
import re
import ctypes
import platform
import collections

import numpy as np
import scipy.sparse

try:
    from sklearn.base import BaseEstimator
    from sklearn.base import RegressorMixin, ClassifierMixin
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_INSTALLED = True
except ImportError:
    SKLEARN_INSTALLED = False

class XGBoostLibraryNotFound(Exception):
    pass

class XGBoostError(Exception):
    pass

__all__ = ['DMatrix', 'CVPack', 'Booster', 'aggcv', 'cv', 'mknfold', 'train']

if sys.version_info[0] == 3:
    string_types = str,
else:
    string_types = basestring,


def load_xglib():
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    dll_path = [curr_path]
    if os.name == 'nt':
        if platform.architecture()[0] == '64bit':
            dll_path.append(os.path.join(curr_path, '../windows/x64/Release/'))
        else:
            dll_path.append(os.path.join(curr_path, '../windows/Release/'))
    if os.name == 'nt':
        dll_path = [os.path.join(p, 'xgboost_wrapper.dll') for p in dll_path]
    else:
        dll_path = [os.path.join(p, 'libxgboostwrapper.so') for p in dll_path]
    lib_path = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]
    if len(dll_path) == 0:
        raise XGBoostLibraryNotFound('cannot find find the files in the candicate path ' + str(dll_path))
    lib = ctypes.cdll.LoadLibrary(lib_path[0])

    # DMatrix functions
    lib.XGDMatrixCreateFromFile.restype = ctypes.c_void_p
    lib.XGDMatrixCreateFromCSR.restype = ctypes.c_void_p
    lib.XGDMatrixCreateFromCSC.restype = ctypes.c_void_p
    lib.XGDMatrixCreateFromMat.restype = ctypes.c_void_p
    lib.XGDMatrixSliceDMatrix.restype = ctypes.c_void_p
    lib.XGDMatrixGetFloatInfo.restype = ctypes.POINTER(ctypes.c_float)
    lib.XGDMatrixGetUIntInfo.restype = ctypes.POINTER(ctypes.c_uint)
    lib.XGDMatrixNumRow.restype = ctypes.c_ulong

    # Booster functions
    lib.XGBoosterCreate.restype = ctypes.c_void_p
    lib.XGBoosterPredict.restype = ctypes.POINTER(ctypes.c_float)
    lib.XGBoosterEvalOneIter.restype = ctypes.c_char_p
    lib.XGBoosterDumpModel.restype = ctypes.POINTER(ctypes.c_char_p)
    lib.XGBoosterGetModelRaw.restype = ctypes.POINTER(ctypes.c_char)
    lib.XGBoosterLoadModelFromBuffer.restype = ctypes.c_void_p

    return lib

# load the XGBoost library globally
xglib = load_xglib()


def ctypes2numpy(cptr, length, dtype):
    """
    Convert a ctypes pointer array to a numpy array.
    """
    if not isinstance(cptr, ctypes.POINTER(ctypes.c_float)):
        raise RuntimeError('expected float pointer')
    res = np.zeros(length, dtype=dtype)
    if not ctypes.memmove(res.ctypes.data, cptr, length * res.strides[0]):
        raise RuntimeError('memmove failed')
    return res


def ctypes2buffer(cptr, length):
    if not isinstance(cptr, ctypes.POINTER(ctypes.c_char)):
        raise RuntimeError('expected char pointer')
    res = bytearray(length)
    rptr = (ctypes.c_char * length).from_buffer(res)
    if not ctypes.memmove(rptr, cptr, length):
        raise RuntimeError('memmove failed')
    return res


def c_str(string):
    return ctypes.c_char_p(string.encode('utf-8'))


def c_array(ctype, values):
    return (ctype * len(values))(*values)


class DMatrix(object):
    def __init__(self, data, label=None, missing=0.0, weight=None, silent=False):
        """
        Data matrix used in XGBoost.

        Parameters
        ----------
        data : string/numpy array/scipy.sparse
            Data source, string type is the path of svmlight format txt file,
            xgb buffer or path to cache_file
        label : list or numpy 1-D array (optional)
            Label of the training data.
        missing : float
            Value in the data which needs to be present as a missing value.
        weight : list or numpy 1-D array (optional)
            Weight for each instance.
        silent: boolean
            Whether print messages during construction
        """
        # force into void_p, mac need to pass things in as void_p
        if data is None:
            self.handle = None
            return
        if isinstance(data, string_types):
            self.handle = ctypes.c_void_p(xglib.XGDMatrixCreateFromFile(c_str(data), int(silent)))
        elif isinstance(data, scipy.sparse.csr_matrix):
            self._init_from_csr(data)
        elif isinstance(data, scipy.sparse.csc_matrix):
            self._init_from_csc(data)
        elif isinstance(data, np.ndarray) and len(data.shape) == 2:
            self._init_from_npy2d(data, missing)
        else:
            try:
                csr = scipy.sparse.csr_matrix(data)
                self._init_from_csr(csr)
            except:
                raise TypeError('can not intialize DMatrix from {}'.format(type(data).__name__))
        if label is not None:
            self.set_label(label)
        if weight is not None:
            self.set_weight(weight)

    def _init_from_csr(self, csr):
        """
        Initialize data from a CSR matrix.
        """
        if len(csr.indices) != len(csr.data):
            raise ValueError('length mismatch: {} vs {}'.format(len(csr.indices), len(csr.data)))
        self.handle = ctypes.c_void_p(xglib.XGDMatrixCreateFromCSR(
            c_array(ctypes.c_ulong, csr.indptr),
            c_array(ctypes.c_uint, csr.indices),
            c_array(ctypes.c_float, csr.data),
            len(csr.indptr), len(csr.data)))

    def _init_from_csc(self, csc):
        """
        Initialize data from a CSC matrix.
        """
        if len(csc.indices) != len(csc.data):
            raise ValueError('length mismatch: {} vs {}'.format(len(csc.indices), len(csc.data)))
        self.handle = ctypes.c_void_p(xglib.XGDMatrixCreateFromCSC(
            c_array(ctypes.c_ulong, csc.indptr),
            c_array(ctypes.c_uint, csc.indices),
            c_array(ctypes.c_float, csc.data),
            len(csc.indptr), len(csc.data)))

    def _init_from_npy2d(self, mat, missing):
        """
        Initialize data from a 2-D numpy matrix.
        """
        data = np.array(mat.reshape(mat.size), dtype=np.float32)
        self.handle = ctypes.c_void_p(xglib.XGDMatrixCreateFromMat(
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            mat.shape[0], mat.shape[1], ctypes.c_float(missing)))

    def __del__(self):
        xglib.XGDMatrixFree(self.handle)

    def get_float_info(self, field):
        length = ctypes.c_ulong()
        ret = xglib.XGDMatrixGetFloatInfo(self.handle, c_str(field), ctypes.byref(length))
        return ctypes2numpy(ret, length.value, np.float32)

    def get_uint_info(self, field):
        length = ctypes.c_ulong()
        ret = xglib.XGDMatrixGetUIntInfo(self.handle, c_str(field), ctypes.byref(length))
        return ctypes2numpy(ret, length.value, np.uint32)

    def set_float_info(self, field, data):
        xglib.XGDMatrixSetFloatInfo(self.handle, c_str(field),
                                    c_array(ctypes.c_float, data), len(data))

    def set_uint_info(self, field, data):
        xglib.XGDMatrixSetUIntInfo(self.handle, c_str(field),
                                   c_array(ctypes.c_uint, data), len(data))

    def save_binary(self, fname, silent=True):
        """
        Save DMatrix to an XGBoost buffer.

        Parameters
        ----------
        fname : string
            Name of the output buffer file.
        silent : bool (optional; default: True)
            If set, the output is suppressed.
        """
        xglib.XGDMatrixSaveBinary(self.handle, c_str(fname), int(silent))

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
        """
        Set weight of each instance.

        Parameters
        ----------
        weight : float
            Weight for positive instance.
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
        """
        Set group size of DMatrix (used for ranking).

        Parameters
        ----------
        group : int
            Group size.
        """
        xglib.XGDMatrixSetGroup(self.handle, c_array(ctypes.c_uint, group), len(group))

    def get_label(self):
        """
        Get the label of the DMatrix.

        Returns
        -------
        label : list
        """
        return self.get_float_info('label')

    def get_weight(self):
        """
        Get the weight of the DMatrix.

        Returns
        -------
        weight : float
        """
        return self.get_float_info('weight')

    def get_base_margin(self):
        """
        Get the base margin of the DMatrix.

        Returns
        -------
        base_margin : float
        """
        return self.get_float_info('base_margin')

    def num_row(self):
        """
        Get the number of rows in the DMatrix.

        Returns
        -------
        number of rows : int
        """
        return xglib.XGDMatrixNumRow(self.handle)

    def slice(self, rindex):
        """
        Slice the DMatrix and return a new DMatrix that only contains `rindex`.

        Parameters
        ----------
        rindex : list
            List of indices to be selected.

        Returns
        -------
        res : DMatrix
            A new DMatrix containing only selected indices.
        """
        res = DMatrix(None)
        res.handle = ctypes.c_void_p(xglib.XGDMatrixSliceDMatrix(
            self.handle, c_array(ctypes.c_int, rindex), len(rindex)))
        return res


class Booster(object):
    def __init__(self, params=None, cache=(), model_file=None):
        """
        Learner class.

        Parameters
        ----------
        params : dict
            Parameters for boosters.
        cache : list
            List of cache items.
        model_file : string
            Path to the model file.
        """
        for d in cache:
            if not isinstance(d, DMatrix):
                raise TypeError('invalid cache item: {}'.format(type(d).__name__))
        dmats = c_array(ctypes.c_void_p, [d.handle for d in cache])
        self.handle = ctypes.c_void_p(xglib.XGBoosterCreate(dmats, len(cache)))
        self.set_param({'seed': 0})
        self.set_param(params or {})
        if model_file is not None:
            self.load_model(model_file)

    def __del__(self):
        xglib.XGBoosterFree(self.handle)

    def __getstate__(self):
        # can't pickle ctypes pointers
        # put model content in bytearray
        this = self.__dict__.copy()
        handle = this['handle']
        if handle is not None:
            raw = self.save_raw()
            this["handle"] = raw
        return this

    def __setstate__(self, state):
        # reconstruct handle from raw data
        handle = state['handle']
        if handle is not None:
            buf = handle
            dmats = c_array(ctypes.c_void_p, [])
            handle = ctypes.c_void_p(xglib.XGBoosterCreate(dmats, 0))
            length = ctypes.c_ulong(len(buf))
            ptr = (ctypes.c_char * len(buf)).from_buffer(buf)
            xglib.XGBoosterLoadModelFromBuffer(handle, ptr, length)
            state['handle'] = handle
        self.__dict__.update(state)
        self.set_param({'seed': 0})

    def __copy__(self):
        return self.__deepcopy__()

    def __deepcopy__(self):
        return Booster(model_file = self.save_raw())

    def copy(self):
        """
        Copy the booster object

        Returns
        --------
        a copied booster model
        """
        return self.__copy__()

    def set_param(self, params, pv=None):
        if isinstance(params, collections.Mapping):
            params = params.items()
        elif isinstance(params, string_types) and pv is not None:
            params = [(params, pv)]
        for k, v in params:
            xglib.XGBoosterSetParam(self.handle, c_str(k), c_str(str(v)))

    def update(self, dtrain, it, fobj=None):
        """
        Update (one iteration).

        Parameters
        ----------
        dtrain : DMatrix
            Training data.
        it : int
            Current iteration number.
        fobj : function
            Customized objective function.
        """
        if not isinstance(dtrain, DMatrix):
            raise TypeError('invalid training matrix: {}'.format(type(dtrain).__name__))
        if fobj is None:
            xglib.XGBoosterUpdateOneIter(self.handle, it, dtrain.handle)
        else:
            pred = self.predict(dtrain)
            grad, hess = fobj(pred, dtrain)
            self.boost(dtrain, grad, hess)

    def boost(self, dtrain, grad, hess):
        """
        Update.

        Parameters
        ----------
        dtrain : DMatrix
            The training DMatrix.
        grad : list
            The first order of gradient.
        hess : list
            The second order of gradient.
        """
        if len(grad) != len(hess):
            raise ValueError('grad / hess length mismatch: {} / {}'.format(len(grad), len(hess)))
        if not isinstance(dtrain, DMatrix):
            raise TypeError('invalid training matrix: {}'.format(type(dtrain).__name__))
        xglib.XGBoosterBoostOneIter(self.handle, dtrain.handle,
                                    c_array(ctypes.c_float, grad),
                                    c_array(ctypes.c_float, hess),
                                    len(grad))

    def eval_set(self, evals, it=0, feval=None):
        """
        Evaluate by a metric.

        Parameters
        ----------
        evals : list of tuples (DMatrix, string)
            List of items to be evaluated.
        it : int
            Current iteration.
        feval : function
            Custom evaluation function.

        Returns
        -------
        evaluation result
        """
        if feval is None:
            for d in evals:
                if not isinstance(d[0], DMatrix):
                    raise TypeError('expected DMatrix, got {}'.format(type(d[0]).__name__))
                if not isinstance(d[1], string_types):
                    raise TypeError('expected string, got {}'.format(type(d[1]).__name__))
            dmats = c_array(ctypes.c_void_p, [d[0].handle for d in evals])
            evnames = c_array(ctypes.c_char_p, [c_str(d[1]) for d in evals])
            return xglib.XGBoosterEvalOneIter(self.handle, it, dmats, evnames, len(evals))
        else:
            res = '[%d]' % it
            for dm, evname in evals:
                name, val = feval(self.predict(dm), dm)
                res += '\t%s-%s:%f' % (evname, name, val)
            return res

    def eval(self, mat, name='eval', it=0):
        return self.eval_set([(mat, name)], it)

    def predict(self, data, output_margin=False, ntree_limit=0, pred_leaf=False):
        """
        Predict with data.

        NOTE: This function is not thread safe.
              For each booster object, predict can only be called from one thread.
              If you want to run prediction using multiple thread, call bst.copy() to make copies
              of model object and then call predict

        Parameters
        ----------
        data : DMatrix
            The dmatrix storing the input.
        output_margin : bool
            Whether to output the raw untransformed margin value.
        ntree_limit : int
            Limit number of trees in the prediction; defaults to 0 (use all trees).
        pred_leaf : bool
            When this option is on, the output will be a matrix of (nsample, ntrees)
            with each record indicating the predicted leaf index of each sample in each tree.
            Note that the leaf index of a tree is unique per tree, so you may find leaf 1
            in both tree 1 and tree 0.

        Returns
        -------
        prediction : numpy array
        """
        option_mask = 0x00
        if output_margin:
            option_mask |= 0x01
        if pred_leaf:
            option_mask |= 0x02
        length = ctypes.c_ulong()
        preds = xglib.XGBoosterPredict(self.handle, data.handle,
                                       option_mask, ntree_limit, ctypes.byref(length))
        preds = ctypes2numpy(preds, length.value, np.float32)
        if pred_leaf:
            preds = preds.astype(np.int32)
        nrow = data.num_row()
        if preds.size != nrow and preds.size % nrow == 0:
            preds = preds.reshape(nrow, preds.size / nrow)
        return preds

    def save_model(self, fname):
        """
        Save the model to a file.

        Parameters
        ----------
        fname : string
            Output file name
        """
        if isinstance(fname, string_types):  # assume file name
            xglib.XGBoosterSaveModel(self.handle, c_str(fname))
        else:
            raise TypeError("fname must be a string")

    def save_raw(self):
        """
        Save the model to a in memory buffer represetation

        Returns
        -------
        a in memory buffer represetation of the model
        """
        length = ctypes.c_ulong()
        cptr = xglib.XGBoosterGetModelRaw(self.handle,
                                          ctypes.byref(length))
        return ctypes2buffer(cptr, length.value)

    def load_model(self, fname):
        """
        Load the model from a file.

        Parameters
        ----------
        fname : string or a memory buffer
            Input file name or memory buffer(see also save_raw)
        """
        if isinstance(fname, str):  # assume file name
            xglib.XGBoosterLoadModel(self.handle, c_str(fname))
        else:
            buf = fname
            length = ctypes.c_ulong(len(buf))
            ptr = (ctypes.c_char * len(buf)).from_buffer(buf)
            xglib.XGBoosterLoadModelFromBuffer(self.handle, ptr, length)

    def dump_model(self, fo, fmap='', with_stats=False):
        """
        Dump model into a text file.

        Parameters
        ----------
        fo : string
            Output file name.
        fmap : string, optional
            Name of the file containing feature map names.
        with_stats : bool (optional)
            Controls whether the split statistics are output.
        """
        if isinstance(fo, string_types):
            fo = open(fo, 'w')
            need_close = True
        else:
            need_close = False
        ret = self.get_dump(fmap, with_stats)
        for i in range(len(ret)):
            fo.write('booster[{}]:\n'.format(i))
            fo.write(ret[i])
        if need_close:
            fo.close()

    def get_dump(self, fmap='', with_stats=False):
        """
        Returns the dump the model as a list of strings.
        """
        length = ctypes.c_ulong()
        sarr = xglib.XGBoosterDumpModel(self.handle, c_str(fmap),
                                        int(with_stats), ctypes.byref(length))
        res = []
        for i in range(length.value):
            res.append(str(sarr[i].decode('ascii')))
        return res

    def get_fscore(self, fmap=''):
        """
        Get feature importance of each feature.
        """
        trees = self.get_dump(fmap)
        fmap = {}
        for tree in trees:
            for l in tree.split('\n'):
                arr = l.split('[')
                if len(arr) == 1:
                    continue
                fid = arr[1].split(']')[0]
                fid = fid.split('<')[0]
                if fid not in fmap:
                    fmap[fid] = 1
                else:
                    fmap[fid] += 1
        return fmap


def train(params, dtrain, num_boost_round=10, evals=(), obj=None, feval=None,
    early_stopping_rounds=None,evals_result=None):
    """
    Train a booster with given parameters.

    Parameters
    ----------
    params : dict
        Booster params.
    dtrain : DMatrix
        Data to be trained.
    num_boost_round: int
        Number of boosting iterations.
    watchlist (evals): list of pairs (DMatrix, string)
        List of items to be evaluated during training, this allows user to watch
        performance on the validation set.
    obj : function
        Customized objective function.
    feval : function
        Customized evaluation function.
    early_stopping_rounds: int
        Activates early stopping. Validation error needs to decrease at least
        every <early_stopping_rounds> round(s) to continue training.
        Requires at least one item in evals.
        If there's more than one, will use the last.
        Returns the model from the last iteration (not the best one).
        If early stopping occurs, the model will have two additional fields:
        bst.best_score and bst.best_iteration.
    evals_result: dict
        This dictionary stores the evaluation results of all the items in watchlist

    Returns
    -------
    booster : a trained booster model
    """

    evals = list(evals)
    bst = Booster(params, [dtrain] + [d[0] for d in evals])

    if evals_result is not None:
        if type(evals_result) is not dict:
            raise TypeError('evals_result has to be a dictionary')
        else:
            evals_name = [d[1] for d in evals]
            evals_result.clear()
            evals_result.update({key:[] for key in evals_name})

    if not early_stopping_rounds:
        for i in range(num_boost_round):
            bst.update(dtrain, i, obj)
            if len(evals) != 0:
                bst_eval_set = bst.eval_set(evals, i, feval)
                if isinstance(bst_eval_set, string_types):
                    msg = bst_eval_set
                else:
                    msg = bst_eval_set.decode()

                sys.stderr.write(msg + '\n')
                if evals_result is not None:
                    res = re.findall(":([0-9.]+).",msg)
                    for key,val in zip(evals_name,res):
                        evals_result[key].append(val)
        return bst

    else:
        # early stopping

        if len(evals) < 1:
            raise ValueError('For early stopping you need at least one set in evals.')

        sys.stderr.write("Will train until {} error hasn't decreased in {} rounds.\n".format(evals[-1][1], early_stopping_rounds))

        # is params a list of tuples? are we using multiple eval metrics?
        if type(params) == list:
            if len(params) != len(dict(params).items()):
                raise ValueError('Check your params. Early stopping works with single eval metric only.')
            params = dict(params)

        # either minimize loss or maximize AUC/MAP/NDCG
        maximize_score = False
        if 'eval_metric' in params:
            maximize_metrics = ('auc', 'map', 'ndcg')
            if list(filter(lambda x: params['eval_metric'].startswith(x), maximize_metrics)):
                maximize_score = True

        if maximize_score:
            best_score = 0.0
        else:
            best_score = float('inf')

        best_msg = ''
        best_score_i = 0

        for i in range(num_boost_round):
            bst.update(dtrain, i, obj)
            bst_eval_set = bst.eval_set(evals, i, feval)

            if isinstance(bst_eval_set, string_types):
                msg = bst_eval_set
            else:
                msg = bst_eval_set.decode()

            sys.stderr.write(msg + '\n')

            if evals_result is not None:
                res = re.findall(":([0-9.]+).",msg)
                for key,val in zip(evals_name,res):
                    evals_result[key].append(val)

            score = float(msg.rsplit(':', 1)[1])
            if (maximize_score and score > best_score) or \
                    (not maximize_score and score < best_score):
                best_score = score
                best_score_i = i
                best_msg = msg
            elif i - best_score_i >= early_stopping_rounds:
                sys.stderr.write("Stopping. Best iteration:\n{}\n\n".format(best_msg))
                bst.best_score = best_score
                bst.best_iteration = best_score_i
                break
        bst.best_score = best_score
        bst.best_iteration = best_score_i
        return bst

class CVPack(object):
    def __init__(self, dtrain, dtest, param):
        self.dtrain = dtrain
        self.dtest = dtest
        self.watchlist = [(dtrain, 'train'), (dtest, 'test')]
        self.bst = Booster(param, [dtrain, dtest])

    def update(self, r, fobj):
        self.bst.update(self.dtrain, r, fobj)

    def eval(self, r, feval):
        return self.bst.eval_set(self.watchlist, r, feval)


def mknfold(dall, nfold, param, seed, evals=(), fpreproc=None):
    """
    Make an n-fold list of CVPack from random indices.
    """
    evals = list(evals)
    np.random.seed(seed)
    randidx = np.random.permutation(dall.num_row())
    kstep = len(randidx) / nfold
    idset = [randidx[(i * kstep): min(len(randidx), (i + 1) * kstep)] for i in range(nfold)]
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
    Aggregate cross-validation results.
    """
    cvmap = {}
    ret = rlist[0].split()[0]
    for line in rlist:
        arr = line.split()
        assert ret == arr[0]
        for it in arr[1:]:
            if not isinstance(it, string_types):
                it = it.decode()
            k, v = it.split(':')
            if k not in cvmap:
                cvmap[k] = []
            cvmap[k].append(float(v))
    for k, v in sorted(cvmap.items(), key=lambda x: x[0]):
        v = np.array(v)
        if not isinstance(ret, string_types):
            ret = ret.decode()
        if show_stdv:
            ret += '\tcv-%s:%f+%f' % (k, np.mean(v), np.std(v))
        else:
            ret += '\tcv-%s:%f' % (k, np.mean(v))
    return ret


def cv(params, dtrain, num_boost_round=10, nfold=3, metrics=(),
       obj=None, feval=None, fpreproc=None, show_stdv=True, seed=0):
    """
    Cross-validation with given paramaters.

    Parameters
    ----------
    params : dict
        Booster params.
    dtrain : DMatrix
        Data to be trained.
    num_boost_round : int
        Number of boosting iterations.
    nfold : int
        Number of folds in CV.
    metrics : list of strings
        Evaluation metrics to be watched in CV.
    obj : function
        Custom objective function.
    feval : function
        Custom evaluation function.
    fpreproc : function
        Preprocessing function that takes (dtrain, dtest, param) and returns
        transformed versions of those.
    show_stdv : bool
        Whether to display the standard deviation.
    seed : int
        Seed used to generate the folds (passed to numpy.random.seed).

    Returns
    -------
    evaluation history : list(string)
    """
    results = []
    cvfolds = mknfold(dtrain, nfold, params, seed, metrics, fpreproc)
    for i in range(num_boost_round):
        for f in cvfolds:
            f.update(i, obj)
        res = aggcv([f.eval(i, feval) for f in cvfolds], show_stdv)
        sys.stderr.write(res + '\n')
        results.append(res)
    return results


# used for compatiblity without sklearn
XGBModelBase = object
XGBClassifier = object
XGBRegressor = object
if SKLEARN_INSTALLED:
    XGBModelBase = BaseEstimator
    XGBRegressor = RegressorMixin
    XGBClassifier = ClassifierMixin

class XGBModel(XGBModelBase):
    """
    Implementation of the Scikit-Learn API for XGBoost.

    Parameters
    ----------
    max_depth : int
        Maximum tree depth for base learners.
    learning_rate : float
        Boosting learning rate (xgb's "eta")
    n_estimators : int
        Number of boosted trees to fit.
    silent : boolean
        Whether to print messages while running boosting.
    objective : string
        Specify the learning task and the corresponding learning objective.

    nthread : int
        Number of parallel threads used to run xgboost.
    gamma : float
        Minimum loss reduction required to make a further partition on a leaf node of the tree.
    min_child_weight : int
        Minimum sum of instance weight(hessian) needed in a child.
    max_delta_step : int
        Maximum delta step we allow each tree's weight estimation to be.
    subsample : float
        Subsample ratio of the training instance.
    colsample_bytree : float
        Subsample ratio of columns when constructing each tree.

    base_score:
        The initial prediction score of all instances, global bias.
    seed : int
        Random number seed.
    missing : float, optional
        Value in the data which needs to be present as a missing value. If
        None, defaults to np.nan.
    """
    def __init__(self, max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective="reg:linear",
                 nthread=-1, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1,
                 base_score=0.5, seed=0, missing=None):
        if not SKLEARN_INSTALLED:
            raise XGBoostError('sklearn needs to be installed in order to use this module')
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.silent = silent
        self.objective = objective

        self.nthread = nthread
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree

        self.base_score = base_score
        self.seed = seed
        self.missing = missing if missing is not None else np.nan

        self._Booster = None

    def __setstate__(self, state):
        # backward compatiblity code
        # load booster from raw if it is raw
        # the booster now support pickle
        bst = state["_Booster"]
        if bst is not None and not isinstance(bst, Booster):
            state["_Booster"] = Booster(model_file=bst)
        self.__dict__.update(state)

    def booster(self):
        """
        get the underlying xgboost Booster of this model
        will raise an exception when fit was not called

        Returns
        -------
        booster : a xgboost booster of underlying model
        """
        if self._Booster is None:
            raise XGBoostError('need to call fit beforehand')
        return self._Booster

    def get_params(self, deep=False):
        params = super(XGBModel, self).get_params(deep=deep)
        if params['missing'] is np.nan:
            params['missing'] = None  # sklearn doesn't handle nan. see #4725
        return params

    def get_xgb_params(self):
        xgb_params = self.get_params()

        xgb_params['silent'] = 1 if self.silent else 0

        if self.nthread <= 0:
            xgb_params.pop('nthread', None)
        return xgb_params

    def fit(self, X, y):
        trainDmatrix = DMatrix(X, label=y, missing=self.missing)
        self._Booster = train(self.get_xgb_params(), trainDmatrix, self.n_estimators)
        return self

    def predict(self, X):
        testDmatrix = DMatrix(X, missing=self.missing)
        return self.booster().predict(testDmatrix)


class XGBClassifier(XGBModel, XGBClassifier):
    __doc__ = """
    Implementation of the scikit-learn API for XGBoost classification
    """ + "\n".join(XGBModel.__doc__.split('\n')[2:])

    def __init__(self, max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective="binary:logistic",
                 nthread=-1, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1,
                 base_score=0.5, seed=0, missing=None):
        super(XGBClassifier, self).__init__(max_depth, learning_rate, n_estimators, silent, objective,
                                            nthread, gamma, min_child_weight, max_delta_step, subsample,
                                            colsample_bytree,
                                            base_score, seed, missing)

    def fit(self, X, y, sample_weight=None):
        self.classes_ = list(np.unique(y))
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ > 2:
            # Switch to using a multiclass objective in the underlying XGB instance
            self.objective = "multi:softprob"
            xgb_options = self.get_xgb_params()
            xgb_options['num_class'] = self.n_classes_
        else:
            xgb_options = self.get_xgb_params()

        self._le = LabelEncoder().fit(y)
        training_labels = self._le.transform(y)

        if sample_weight is not None:
            trainDmatrix = DMatrix(X, label=training_labels, weight=sample_weight,
                                   missing=self.missing)
        else:
            trainDmatrix = DMatrix(X, label=training_labels,
                                   missing=self.missing)

        self._Booster = train(xgb_options, trainDmatrix, self.n_estimators)

        return self

    def predict(self, X):
        testDmatrix = DMatrix(X, missing=self.missing)
        class_probs = self.booster().predict(testDmatrix)
        if len(class_probs.shape) > 1:
            column_indexes = np.argmax(class_probs, axis=1)
        else:
            column_indexes = np.repeat(0, X.shape[0])
            column_indexes[class_probs > 0.5] = 1
        return self._le.inverse_transform(column_indexes)

    def predict_proba(self, X):
        testDmatrix = DMatrix(X, missing=self.missing)
        class_probs = self.booster().predict(testDmatrix)
        if self.objective == "multi:softprob":
            return class_probs
        else:
            classone_probs = class_probs
            classzero_probs = 1.0 - classone_probs
            return np.vstack((classzero_probs, classone_probs)).transpose()

class XGBRegressor(XGBModel, XGBRegressor):
    __doc__ = """
    Implementation of the scikit-learn API for XGBoost regression
    """ + "\n".join(XGBModel.__doc__.split('\n')[2:])

    pass
