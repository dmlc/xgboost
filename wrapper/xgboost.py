# coding: utf-8
"""
xgboost: eXtreme Gradient Boosting library

Version: 0.40
Authors: Tianqi Chen, Bing Xu
Early stopping by Zygmunt Zając
"""
# pylint: disable=too-many-arguments, too-many-locals, too-many-lines, invalid-name
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
    """Error throwed by when xgboost is not found"""
    pass

class XGBoostError(Exception):
    """Error throwed by xgboost trainer."""
    pass

__all__ = ['DMatrix', 'CVPack', 'Booster', 'aggcv', 'cv', 'mknfold', 'train']

if sys.version_info[0] == 3:
    # pylint: disable=invalid-name
    STRING_TYPES = str,
else:
    # pylint: disable=invalid-name
    STRING_TYPES = basestring,

def load_xglib():
    """Load the xgboost library."""
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
        raise XGBoostLibraryNotFound(
            'cannot find find the files in the candicate path ' + str(dll_path))
    lib = ctypes.cdll.LoadLibrary(lib_path[0])
    lib.XGBGetLastError.restype = ctypes.c_char_p

    return lib

# load the XGBoost library globally
_LIB = load_xglib()

def _check_call(ret):
    """Check the return value of C API call

    This function will raise exception when error occurs.
    Wrap every API call with this function

    Parameters
    ----------
    ret : int
        return value from API calls
    """
    if ret != 0:
        raise XGBoostError(_LIB.XGBGetLastError())


def ctypes2numpy(cptr, length, dtype):
    """Convert a ctypes pointer array to a numpy array.
    """
    if not isinstance(cptr, ctypes.POINTER(ctypes.c_float)):
        raise RuntimeError('expected float pointer')
    res = np.zeros(length, dtype=dtype)
    if not ctypes.memmove(res.ctypes.data, cptr, length * res.strides[0]):
        raise RuntimeError('memmove failed')
    return res


def ctypes2buffer(cptr, length):
    """Convert ctypes pointer to buffer type."""
    if not isinstance(cptr, ctypes.POINTER(ctypes.c_char)):
        raise RuntimeError('expected char pointer')
    res = bytearray(length)
    rptr = (ctypes.c_char * length).from_buffer(res)
    if not ctypes.memmove(rptr, cptr, length):
        raise RuntimeError('memmove failed')
    return res


def c_str(string):
    """Convert a python string to cstring."""
    return ctypes.c_char_p(string.encode('utf-8'))


def c_array(ctype, values):
    """Convert a python string to c array."""
    return (ctype * len(values))(*values)


class DMatrix(object):
    """Data Matrix used in XGBoost."""
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
        if isinstance(data, STRING_TYPES):
            self.handle = ctypes.c_void_p()
            _check_call(_LIB.XGDMatrixCreateFromFile(c_str(data),
                                                     int(silent),
                                                     ctypes.byref(self.handle)))
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
        self.handle = ctypes.c_void_p()
        _check_call(_LIB.XGDMatrixCreateFromCSR(c_array(ctypes.c_ulong, csr.indptr),
                                                c_array(ctypes.c_uint, csr.indices),
                                                c_array(ctypes.c_float, csr.data),
                                                len(csr.indptr), len(csr.data),
                                                ctypes.byref(self.handle)))

    def _init_from_csc(self, csc):
        """
        Initialize data from a CSC matrix.
        """
        if len(csc.indices) != len(csc.data):
            raise ValueError('length mismatch: {} vs {}'.format(len(csc.indices), len(csc.data)))
        self.handle = ctypes.c_void_p()
        _check_call(_LIB.XGDMatrixCreateFromCSC(c_array(ctypes.c_ulong, csc.indptr),
                                                c_array(ctypes.c_uint, csc.indices),
                                                c_array(ctypes.c_float, csc.data),
                                                len(csc.indptr), len(csc.data),
                                                ctypes.byref(self.handle)))

    def _init_from_npy2d(self, mat, missing):
        """
        Initialize data from a 2-D numpy matrix.
        """
        data = np.array(mat.reshape(mat.size), dtype=np.float32)
        self.handle = ctypes.c_void_p()
        _check_call(_LIB.XGDMatrixCreateFromMat(data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                                mat.shape[0], mat.shape[1],
                                                ctypes.c_float(missing),
                                                ctypes.byref(self.handle)))

    def __del__(self):
        _check_call(_LIB.XGDMatrixFree(self.handle))

    def get_float_info(self, field):
        """Get float property from the DMatrix.

        Parameters
        ----------
        field: str
            The field name of the information

        Returns
        -------
        info : array
            a numpy array of float information of the data
        """
        length = ctypes.c_ulong()
        ret = ctypes.POINTER(ctypes.c_float)()
        _check_call(_LIB.XGDMatrixGetFloatInfo(self.handle,
                                               c_str(field),
                                               ctypes.byref(length),
                                               ctypes.byref(ret)))
        return ctypes2numpy(ret, length.value, np.float32)

    def get_uint_info(self, field):
        """Get unsigned integer property from the DMatrix.

        Parameters
        ----------
        field: str
            The field name of the information

        Returns
        -------
        info : array
            a numpy array of float information of the data
        """
        length = ctypes.c_ulong()
        ret = ctypes.POINTER(ctypes.c_uint)()
        _check_call(_LIB.XGDMatrixGetUIntInfo(self.handle,
                                              c_str(field),
                                              ctypes.byref(length),
                                              ctypes.byref(ret)))
        return ctypes2numpy(ret, length.value, np.uint32)

    def set_float_info(self, field, data):
        """Set float type property into the DMatrix.

        Parameters
        ----------
        field: str
            The field name of the information

        data: numpy array
            The array ofdata to be set
        """
        _check_call(_LIB.XGDMatrixSetFloatInfo(self.handle,
                                               c_str(field),
                                               c_array(ctypes.c_float, data),
                                               len(data)))

    def set_uint_info(self, field, data):
        """Set uint type property into the DMatrix.

        Parameters
        ----------
        field: str
            The field name of the information

        data: numpy array
            The array ofdata to be set
        """
        _check_call(_LIB.XGDMatrixSetUIntInfo(self.handle,
                                              c_str(field),
                                              c_array(ctypes.c_uint, data),
                                              len(data)))

    def save_binary(self, fname, silent=True):
        """Save DMatrix to an XGBoost buffer.

        Parameters
        ----------
        fname : string
            Name of the output buffer file.
        silent : bool (optional; default: True)
            If set, the output is suppressed.
        """
        _check_call(_LIB.XGDMatrixSaveBinary(self.handle,
                                             c_str(fname),
                                             int(silent)))

    def set_label(self, label):
        """Set label of dmatrix

        Parameters
        ----------
        label: array like
            The label information to be set into DMatrix
        """
        self.set_float_info('label', label)

    def set_weight(self, weight):
        """ Set weight of each instance.

        Parameters
        ----------
        weight : array like
            Weight for each data point
        """
        self.set_float_info('weight', weight)

    def set_base_margin(self, margin):
        """ Set base margin of booster to start from.

        This can be used to specify a prediction value of
        existing model to be base_margin
        However, remember margin is needed, instead of transformed prediction
        e.g. for logistic regression: need to put in value before logistic transformation
        see also example/demo.py

        Parameters
        ----------
        margin: array like
            Prediction margin of each datapoint
        """
        self.set_float_info('base_margin', margin)

    def set_group(self, group):
        """Set group size of DMatrix (used for ranking).

        Parameters
        ----------
        group : array like
            Group size of each group
        """
        _check_call(_LIB.XGDMatrixSetGroup(self.handle,
                                           c_array(ctypes.c_uint, group),
                                           len(group)))

    def get_label(self):
        """Get the label of the DMatrix.

        Returns
        -------
        label : array
        """
        return self.get_float_info('label')

    def get_weight(self):
        """Get the weight of the DMatrix.

        Returns
        -------
        weight : array
        """
        return self.get_float_info('weight')

    def get_base_margin(self):
        """Get the base margin of the DMatrix.

        Returns
        -------
        base_margin : float
        """
        return self.get_float_info('base_margin')

    def num_row(self):
        """Get the number of rows in the DMatrix.

        Returns
        -------
        number of rows : int
        """
        ret = ctypes.c_ulong()
        _check_call(_LIB.XGDMatrixNumRow(self.handle,
                                         ctypes.byref(ret)))
        return ret.value

    def slice(self, rindex):
        """Slice the DMatrix and return a new DMatrix that only contains `rindex`.

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
        res.handle = ctypes.c_void_p()
        _check_call(_LIB.XGDMatrixSliceDMatrix(self.handle,
                                               c_array(ctypes.c_int, rindex),
                                               len(rindex),
                                               ctypes.byref(res.handle)))
        return res


class Booster(object):
    """"A Booster of of XGBoost."""
    def __init__(self, params=None, cache=(), model_file=None):
        # pylint: disable=invalid-name
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
        self.handle = ctypes.c_void_p()
        _check_call(_LIB.XGBoosterCreate(dmats, len(cache), ctypes.byref(self.handle)))
        self.set_param({'seed': 0})
        self.set_param(params or {})
        if model_file is not None:
            self.load_model(model_file)

    def __del__(self):
        _LIB.XGBoosterFree(self.handle)

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
            handle = ctypes.c_void_p()
            _check_call(_LIB.XGBoosterCreate(dmats, 0, ctypes.byref(handle)))
            length = ctypes.c_ulong(len(buf))
            ptr = (ctypes.c_char * len(buf)).from_buffer(buf)
            _check_call(_LIB.XGBoosterLoadModelFromBuffer(handle, ptr, length))
            state['handle'] = handle
        self.__dict__.update(state)
        self.set_param({'seed': 0})

    def __copy__(self):
        return self.__deepcopy__()

    def __deepcopy__(self):
        return Booster(model_file=self.save_raw())

    def copy(self):
        """Copy the booster object.

        Returns
        --------
        a copied booster model
        """
        return self.__copy__()

    def set_param(self, params, value=None):
        """Set parameters into the DMatrix."""
        if isinstance(params, collections.Mapping):
            params = params.items()
        elif isinstance(params, STRING_TYPES) and value is not None:
            params = [(params, value)]
        for key, val in params:
            _check_call(_LIB.XGBoosterSetParam(self.handle, c_str(key), c_str(str(val))))

    def update(self, dtrain, iteration, fobj=None):
        """
        Update (one iteration).

        Parameters
        ----------
        dtrain : DMatrix
            Training data.
        iteration : int
            Current iteration number.
        fobj : function
            Customized objective function.
        """
        if not isinstance(dtrain, DMatrix):
            raise TypeError('invalid training matrix: {}'.format(type(dtrain).__name__))
        if fobj is None:
            _check_call(_LIB.XGBoosterUpdateOneIter(self.handle, iteration, dtrain.handle))
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
        _check_call(_LIB.XGBoosterBoostOneIter(self.handle, dtrain.handle,
                                               c_array(ctypes.c_float, grad),
                                               c_array(ctypes.c_float, hess),
                                               len(grad)))

    def eval_set(self, evals, iteration=0, feval=None):
        # pylint: disable=invalid-name
        """Evaluate  a set of data.

        Parameters
        ----------
        evals : list of tuples (DMatrix, string)
            List of items to be evaluated.
        iteration : int
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
                if not isinstance(d[1], STRING_TYPES):
                    raise TypeError('expected string, got {}'.format(type(d[1]).__name__))
            dmats = c_array(ctypes.c_void_p, [d[0].handle for d in evals])
            evnames = c_array(ctypes.c_char_p, [c_str(d[1]) for d in evals])
            msg = ctypes.c_char_p()
            _check_call(_LIB.XGBoosterEvalOneIter(self.handle, iteration,
                                                  dmats, evnames, len(evals),
                                                  ctypes.byref(msg)))
            return msg.value
        else:
            res = '[%d]' % iteration
            for dmat, evname in evals:
                name, val = feval(self.predict(dmat), dmat)
                res += '\t%s-%s:%f' % (evname, name, val)
            return res

    def eval(self, data, name='eval', iteration=0):
        """Evaluate the model on mat.


        Parameters
        ---------
        data : DMatrix
            The dmatrix storing the input.

        name : str (default = 'eval')
            The name of the dataset


        iteration : int (default = 0)
            The current iteration number
        """
        return self.eval_set([(data, name)], iteration)

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
        preds = ctypes.POINTER(ctypes.c_float)()
        _check_call(_LIB.XGBoosterPredict(self.handle, data.handle,
                                          option_mask, ntree_limit,
                                          ctypes.byref(length),
                                          ctypes.byref(preds)))
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
        if isinstance(fname, STRING_TYPES):  # assume file name
            _check_call(_LIB.XGBoosterSaveModel(self.handle, c_str(fname)))
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
        cptr = ctypes.POINTER(ctypes.c_char)()
        _check_call(_LIB.XGBoosterGetModelRaw(self.handle,
                                              ctypes.byref(length),
                                              ctypes.byref(cptr)))
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
            _LIB.XGBoosterLoadModel(self.handle, c_str(fname))
        else:
            buf = fname
            length = ctypes.c_ulong(len(buf))
            ptr = (ctypes.c_char * len(buf)).from_buffer(buf)
            _check_call(_LIB.XGBoosterLoadModelFromBuffer(self.handle, ptr, length))

    def dump_model(self, fout, fmap='', with_stats=False):
        """
        Dump model into a text file.

        Parameters
        ----------
        foout : string
            Output file name.
        fmap : string, optional
            Name of the file containing feature map names.
        with_stats : bool (optional)
            Controls whether the split statistics are output.
        """
        if isinstance(fout, STRING_TYPES):
            fout = open(fout, 'w')
            need_close = True
        else:
            need_close = False
        ret = self.get_dump(fmap, with_stats)
        for i in range(len(ret)):
            fout.write('booster[{}]:\n'.format(i))
            fout.write(ret[i])
        if need_close:
            fout.close()

    def get_dump(self, fmap='', with_stats=False):
        """
        Returns the dump the model as a list of strings.
        """
        length = ctypes.c_ulong()
        sarr = ctypes.POINTER(ctypes.c_char_p)()
        _check_call(_LIB.XGBoosterDumpModel(self.handle,
                                            c_str(fmap),
                                            int(with_stats),
                                            ctypes.byref(length),
                                            ctypes.byref(sarr)))
        res = []
        for i in range(length.value):
            res.append(str(sarr[i].decode('ascii')))
        return res

    def get_fscore(self, fmap=''):
        """Get feature importance of each feature.

        Parameters
        ----------
        fmap: str (optional)
           The name of feature map file
        """
        trees = self.get_dump(fmap)
        fmap = {}
        for tree in trees:
            for line in tree.split('\n'):
                arr = line.split('[')
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
          early_stopping_rounds=None, evals_result=None):
    # pylint: disable=too-many-statements,too-many-branches, attribute-defined-outside-init
    """Train a booster with given parameters.

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
        if not isinstance(evals_result, dict):
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
                if isinstance(bst_eval_set, STRING_TYPES):
                    msg = bst_eval_set
                else:
                    msg = bst_eval_set.decode()

                sys.stderr.write(msg + '\n')
                if evals_result is not None:
                    res = re.findall(":([0-9.]+).", msg)
                    for key, val in zip(evals_name, res):
                        evals_result[key].append(val)
        return bst

    else:
        # early stopping
        if len(evals) < 1:
            raise ValueError('For early stopping you need at least one set in evals.')

        sys.stderr.write("Will train until {} error hasn't decreased in {} rounds.\n".format(\
                evals[-1][1], early_stopping_rounds))

        # is params a list of tuples? are we using multiple eval metrics?
        if isinstance(params, list):
            if len(params) != len(dict(params).items()):
                raise ValueError('Check your params.'\
                                     'Early stopping works with single eval metric only.')
            params = dict(params)

        # either minimize loss or maximize AUC/MAP/NDCG
        maximize_score = False
        if 'eval_metric' in params:
            maximize_metrics = ('auc', 'map', 'ndcg')
            if any(params['eval_metric'].startswith(x) for x in maximize_metrics):
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

            if isinstance(bst_eval_set, STRING_TYPES):
                msg = bst_eval_set
            else:
                msg = bst_eval_set.decode()

            sys.stderr.write(msg + '\n')

            if evals_result is not None:
                res = re.findall(":([0-9.]+).", msg)
                for key, val in zip(evals_name, res):
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
    """"Auxiliary datastruct to hold one fold of CV."""
    def __init__(self, dtrain, dtest, param):
        """"Initialize the CVPack"""
        self.dtrain = dtrain
        self.dtest = dtest
        self.watchlist = [(dtrain, 'train'), (dtest, 'test')]
        self.bst = Booster(param, [dtrain, dtest])

    def update(self, iteration, fobj):
        """"Update the boosters for one iteration"""
        self.bst.update(self.dtrain, iteration, fobj)

    def eval(self, iteration, feval):
        """"Evaluate the CVPack for one iteration."""
        return self.bst.eval_set(self.watchlist, iteration, feval)


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
    # pylint: disable=invalid-name
    """
    Aggregate cross-validation results.
    """
    cvmap = {}
    ret = rlist[0].split()[0]
    for line in rlist:
        arr = line.split()
        assert ret == arr[0]
        for it in arr[1:]:
            if not isinstance(it, STRING_TYPES):
                it = it.decode()
            k, v = it.split(':')
            if k not in cvmap:
                cvmap[k] = []
            cvmap[k].append(float(v))
    for k, v in sorted(cvmap.items(), key=lambda x: x[0]):
        v = np.array(v)
        if not isinstance(ret, STRING_TYPES):
            ret = ret.decode()
        if show_stdv:
            ret += '\tcv-%s:%f+%f' % (k, np.mean(v), np.std(v))
        else:
            ret += '\tcv-%s:%f' % (k, np.mean(v))
    return ret


def cv(params, dtrain, num_boost_round=10, nfold=3, metrics=(),
       obj=None, feval=None, fpreproc=None, show_stdv=True, seed=0):
    # pylint: disable = invalid-name
    """Cross-validation with given paramaters.

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
        for fold in cvfolds:
            fold.update(i, obj)
        res = aggcv([f.eval(i, feval) for f in cvfolds], show_stdv)
        sys.stderr.write(res + '\n')
        results.append(res)
    return results


# used for compatiblity without sklearn
XGBModelBase = object
XGBClassifierBase = object
XGBRegressorBase = object
if SKLEARN_INSTALLED:
    XGBModelBase = BaseEstimator
    XGBRegressorBase = RegressorMixin
    XGBClassifierBase = ClassifierMixin

class XGBModel(XGBModelBase):
    # pylint: disable=too-many-arguments, too-many-instance-attributes, invalid-name
    """Implementation of the Scikit-Learn API for XGBoost.

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
    def __init__(self, max_depth=3, learning_rate=0.1, n_estimators=100,
                 silent=True, objective="reg:linear",
                 nthread=-1, gamma=0, min_child_weight=1, max_delta_step=0,
                 subsample=1, colsample_bytree=1,
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
        """Get the underlying xgboost Booster of this model.

        This will raise an exception when fit was not called

        Returns
        -------
        booster : a xgboost booster of underlying model
        """
        if self._Booster is None:
            raise XGBoostError('need to call fit beforehand')
        return self._Booster

    def get_params(self, deep=False):
        """Get parameter.s"""
        params = super(XGBModel, self).get_params(deep=deep)
        if params['missing'] is np.nan:
            params['missing'] = None  # sklearn doesn't handle nan. see #4725
        return params

    def get_xgb_params(self):
        """Get xgboost type parameters."""
        xgb_params = self.get_params()

        xgb_params['silent'] = 1 if self.silent else 0

        if self.nthread <= 0:
            xgb_params.pop('nthread', None)
        return xgb_params

    def fit(self, data, y):
        # pylint: disable=missing-docstring,invalid-name
        train_dmatrix = DMatrix(data, label=y, missing=self.missing)
        self._Booster = train(self.get_xgb_params(), train_dmatrix, self.n_estimators)
        return self

    def predict(self, data):
        # pylint: disable=missing-docstring,invalid-name
        test_dmatrix = DMatrix(data, missing=self.missing)
        return self.booster().predict(test_dmatrix)


class XGBClassifier(XGBModel, XGBClassifierBase):
    # pylint: disable=missing-docstring,too-many-arguments,invalid-name
    __doc__ = """
    Implementation of the scikit-learn API for XGBoost classification
    """ + "\n".join(XGBModel.__doc__.split('\n')[2:])

    def __init__(self, max_depth=3, learning_rate=0.1,
                 n_estimators=100, silent=True,
                 objective="binary:logistic",
                 nthread=-1, gamma=0, min_child_weight=1,
                 max_delta_step=0, subsample=1, colsample_bytree=1,
                 base_score=0.5, seed=0, missing=None):
        super(XGBClassifier, self).__init__(max_depth, learning_rate,
                                            n_estimators, silent, objective,
                                            nthread, gamma, min_child_weight,
                                            max_delta_step, subsample,
                                            colsample_bytree,
                                            base_score, seed, missing)

    def fit(self, X, y, sample_weight=None):
        # pylint: disable = attribute-defined-outside-init,arguments-differ
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
            train_dmatrix = DMatrix(X, label=training_labels, weight=sample_weight,
                                    missing=self.missing)
        else:
            train_dmatrix = DMatrix(X, label=training_labels,
                                    missing=self.missing)

        self._Booster = train(xgb_options, train_dmatrix, self.n_estimators)

        return self

    def predict(self, data):
        test_dmatrix = DMatrix(data, missing=self.missing)
        class_probs = self.booster().predict(test_dmatrix)
        if len(class_probs.shape) > 1:
            column_indexes = np.argmax(class_probs, axis=1)
        else:
            column_indexes = np.repeat(0, data.shape[0])
            column_indexes[class_probs > 0.5] = 1
        return self._le.inverse_transform(column_indexes)

    def predict_proba(self, data):
        test_dmatrix = DMatrix(data, missing=self.missing)
        class_probs = self.booster().predict(test_dmatrix)
        if self.objective == "multi:softprob":
            return class_probs
        else:
            classone_probs = class_probs
            classzero_probs = 1.0 - classone_probs
            return np.vstack((classzero_probs, classone_probs)).transpose()

class XGBRegressor(XGBModel, XGBRegressorBase):
    # pylint: disable=missing-docstring
    __doc__ = """
    Implementation of the scikit-learn API for XGBoost regression
    """ + "\n".join(XGBModel.__doc__.split('\n')[2:])
