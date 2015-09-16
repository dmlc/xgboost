# coding: utf-8
# pylint: disable=too-many-locals, too-many-arguments, invalid-name
"""Training Library containing training routines."""
from __future__ import absolute_import

import sys
import re
import numpy as np
from .core import Booster, STRING_TYPES

def train(params, dtrain, num_boost_round=10, evals=(), obj=None, feval=None,
          early_stopping_rounds=None, evals_result=None, verbose_eval=True):
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
    verbose_eval : bool
        If `verbose_eval` then the evaluation metric on the validation set, if
        given, is printed at each boosting stage.

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
            evals_result.update({key: [] for key in evals_name})

    if not early_stopping_rounds:
        for i in range(num_boost_round):
            bst.update(dtrain, i, obj)
            if len(evals) != 0:
                bst_eval_set = bst.eval_set(evals, i, feval)
                if isinstance(bst_eval_set, STRING_TYPES):
                    msg = bst_eval_set
                else:
                    msg = bst_eval_set.decode()

                if verbose_eval:
                    sys.stderr.write(msg + '\n')
                if evals_result is not None:
                    res = re.findall(":-?([0-9.]+).", msg)
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

            if verbose_eval:
                sys.stderr.write(msg + '\n')

            if evals_result is not None:
                res = re.findall(":-?([0-9.]+).", msg)
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

