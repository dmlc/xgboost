# coding: utf-8
# pylint: disable=too-many-locals, too-many-arguments, invalid-name
# pylint: disable=too-many-branches, too-many-statements
"""Training Library containing training routines."""
from __future__ import absolute_import

import warnings
import numpy as np
from .core import Booster, STRING_TYPES, XGBoostError, CallbackEnv, EarlyStopException
from .compat import (SKLEARN_INSTALLED, XGBStratifiedKFold)
from . import rabit
from . import callback


def _train_internal(params, dtrain,
                    num_boost_round=10, evals=(),
                    obj=None, feval=None,
                    xgb_model=None, callbacks=None):
    """internal training function"""
    callbacks = [] if callbacks is None else callbacks
    evals = list(evals)
    if isinstance(params, dict) \
            and 'eval_metric' in params \
            and isinstance(params['eval_metric'], list):
        params = dict((k, v) for k, v in params.items())
        eval_metrics = params['eval_metric']
        params.pop("eval_metric", None)
        params = list(params.items())
        for eval_metric in eval_metrics:
            params += [('eval_metric', eval_metric)]

    bst = Booster(params, [dtrain] + [d[0] for d in evals])
    nboost = 0
    num_parallel_tree = 1

    if xgb_model is not None:
        if not isinstance(xgb_model, STRING_TYPES):
            xgb_model = xgb_model.save_raw()
        bst = Booster(params, [dtrain] + [d[0] for d in evals], model_file=xgb_model)
        nboost = len(bst.get_dump())

    _params = dict(params) if isinstance(params, list) else params

    if 'num_parallel_tree' in _params:
        num_parallel_tree = _params['num_parallel_tree']
        nboost //= num_parallel_tree
    if 'num_class' in _params:
        nboost //= _params['num_class']

    # Distributed code: Load the checkpoint from rabit.
    version = bst.load_rabit_checkpoint()
    assert(rabit.get_world_size() != 1 or version == 0)
    rank = rabit.get_rank()
    start_iteration = int(version / 2)
    nboost += start_iteration

    callbacks_before_iter = [
        cb for cb in callbacks if cb.__dict__.get('before_iteration', False)]
    callbacks_after_iter = [
        cb for cb in callbacks if not cb.__dict__.get('before_iteration', False)]

    for i in range(start_iteration, num_boost_round):
        for cb in callbacks_before_iter:
            cb(CallbackEnv(model=bst,
                           cvfolds=None,
                           iteration=i,
                           begin_iteration=start_iteration,
                           end_iteration=num_boost_round,
                           rank=rank,
                           evaluation_result_list=None))
        # Distributed code: need to resume to this point.
        # Skip the first update if it is a recovery step.
        if version % 2 == 0:
            bst.update(dtrain, i, obj)
            bst.save_rabit_checkpoint()
            version += 1

        assert(rabit.get_world_size() == 1 or version == rabit.version_number())

        nboost += 1
        evaluation_result_list = []
        # check evaluation result.
        if len(evals) != 0:
            bst_eval_set = bst.eval_set(evals, i, feval)
            if isinstance(bst_eval_set, STRING_TYPES):
                msg = bst_eval_set
            else:
                msg = bst_eval_set.decode()
            res = [x.split(':') for x in msg.split()]
            evaluation_result_list = [(k, float(v)) for k, v in res[1:]]
        try:
            for cb in callbacks_after_iter:
                cb(CallbackEnv(model=bst,
                               cvfolds=None,
                               iteration=i,
                               begin_iteration=start_iteration,
                               end_iteration=num_boost_round,
                               rank=rank,
                               evaluation_result_list=evaluation_result_list))
        except EarlyStopException:
            break
        # do checkpoint after evaluation, in case evaluation also updates booster.
        bst.save_rabit_checkpoint()
        version += 1

    if bst.attr('best_score') is not None:
        bst.best_score = float(bst.attr('best_score'))
        bst.best_iteration = int(bst.attr('best_iteration'))
    else:
        bst.best_iteration = nboost - 1
    bst.best_ntree_limit = (bst.best_iteration + 1) * num_parallel_tree
    return bst


def train(params, dtrain, num_boost_round=10, evals=(), obj=None, feval=None,
          maximize=False, early_stopping_rounds=None, evals_result=None,
          verbose_eval=True, xgb_model=None, callbacks=None, learning_rates=None):
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
    evals: list of pairs (DMatrix, string)
        List of items to be evaluated during training, this allows user to watch
        performance on the validation set.
    obj : function
        Customized objective function.
    feval : function
        Customized evaluation function.
    maximize : bool
        Whether to maximize feval.
    early_stopping_rounds: int
        Activates early stopping. Validation error needs to decrease at least
        every <early_stopping_rounds> round(s) to continue training.
        Requires at least one item in evals.
        If there's more than one, will use the last.
        Returns the model from the last iteration (not the best one).
        If early stopping occurs, the model will have three additional fields:
        bst.best_score, bst.best_iteration and bst.best_ntree_limit.
        (Use bst.best_ntree_limit to get the correct value if num_parallel_tree
        and/or num_class appears in the parameters)
    evals_result: dict
        This dictionary stores the evaluation results of all the items in watchlist.
        Example: with a watchlist containing [(dtest,'eval'), (dtrain,'train')] and
        a parameter containing ('eval_metric': 'logloss')
        Returns: {'train': {'logloss': ['0.48253', '0.35953']},
                  'eval': {'logloss': ['0.480385', '0.357756']}}
    verbose_eval : bool or int
        Requires at least one item in evals.
        If `verbose_eval` is True then the evaluation metric on the validation set is
        printed at each boosting stage.
        If `verbose_eval` is an integer then the evaluation metric on the validation set
        is printed at every given `verbose_eval` boosting stage. The last boosting stage
        / the boosting stage found by using `early_stopping_rounds` is also printed.
        Example: with verbose_eval=4 and at least one item in evals, an evaluation metric
        is printed every 4 boosting stages, instead of every boosting stage.
    learning_rates: list or function (deprecated - use callback API instead)
        List of learning rate for each boosting round
        or a customized function that calculates eta in terms of
        current number of round and the total number of boosting round (e.g. yields
        learning rate decay)
    xgb_model : file name of stored xgb model or 'Booster' instance
        Xgb model to be loaded before training (allows training continuation).
    callbacks : list of callback functions
        List of callback functions that are applied at end of each iteration.
        It is possible to use predefined callbacks by using xgb.callback module.
        Example: [xgb.callback.reset_learning_rate(custom_rates)]

    Returns
    -------
    booster : a trained booster model
    """
    callbacks = [] if callbacks is None else callbacks

    # Most of legacy advanced options becomes callbacks
    if isinstance(verbose_eval, bool) and verbose_eval:
        callbacks.append(callback.print_evaluation())
    else:
        if isinstance(verbose_eval, int):
            callbacks.append(callback.print_evaluation(verbose_eval))

    if early_stopping_rounds is not None:
        callbacks.append(callback.early_stop(early_stopping_rounds,
                                             maximize=maximize,
                                             verbose=bool(verbose_eval)))
    if evals_result is not None:
        callbacks.append(callback.record_evaluation(evals_result))

    if learning_rates is not None:
        warnings.warn("learning_rates parameter is deprecated - use callback API instead",
                      DeprecationWarning)
        callbacks.append(callback.reset_learning_rate(learning_rates))

    return _train_internal(params, dtrain,
                           num_boost_round=num_boost_round,
                           evals=evals,
                           obj=obj, feval=feval,
                           xgb_model=xgb_model, callbacks=callbacks)


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


def mknfold(dall, nfold, param, seed, evals=(), fpreproc=None, stratified=False, folds=None):
    """
    Make an n-fold list of CVPack from random indices.
    """
    evals = list(evals)
    np.random.seed(seed)

    if stratified is False and folds is None:
        randidx = np.random.permutation(dall.num_row())
        kstep = int(len(randidx) / nfold)
        idset = [randidx[(i * kstep): min(len(randidx), (i + 1) * kstep)] for i in range(nfold)]
    elif folds is not None and isinstance(folds, list):
        idset = [x[1] for x in folds]
        nfold = len(idset)
    else:
        sfk = XGBStratifiedKFold(n_splits=nfold, shuffle=True, random_state=seed)
        idset = [x[1] for x in sfk.split(X=dall.get_label(), y=dall.get_label())]

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


def aggcv(rlist):
    # pylint: disable=invalid-name
    """
    Aggregate cross-validation results.

    If verbose_eval is true, progress is displayed in every call. If
    verbose_eval is an integer, progress will only be displayed every
    `verbose_eval` trees, tracked via trial.
    """
    cvmap = {}
    idx = rlist[0].split()[0]
    for line in rlist:
        arr = line.split()
        assert idx == arr[0]
        for it in arr[1:]:
            if not isinstance(it, STRING_TYPES):
                it = it.decode()
            k, v = it.split(':')
            if k not in cvmap:
                cvmap[k] = []
            cvmap[k].append(float(v))
    msg = idx
    results = []
    for k, v in sorted(cvmap.items(), key=lambda x: (x[0].startswith('test'), x[0])):
        v = np.array(v)
        if not isinstance(msg, STRING_TYPES):
            msg = msg.decode()
        mean, std = np.mean(v), np.std(v)
        results.extend([(k, mean, std)])
    return results


def cv(params, dtrain, num_boost_round=10, nfold=3, stratified=False, folds=None,
       metrics=(), obj=None, feval=None, maximize=False, early_stopping_rounds=None,
       fpreproc=None, as_pandas=True, verbose_eval=None, show_stdv=True,
       seed=0, callbacks=None):
    # pylint: disable = invalid-name
    """Cross-validation with given parameters.

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
    stratified : bool
        Perform stratified sampling.
    folds : a KFold or StratifiedKFold instance
        Sklearn KFolds or StratifiedKFolds.
    metrics : string or list of strings
        Evaluation metrics to be watched in CV.
    obj : function
        Custom objective function.
    feval : function
        Custom evaluation function.
    maximize : bool
        Whether to maximize feval.
    early_stopping_rounds: int
        Activates early stopping. CV error needs to decrease at least
        every <early_stopping_rounds> round(s) to continue.
        Last entry in evaluation history is the one from best iteration.
    fpreproc : function
        Preprocessing function that takes (dtrain, dtest, param) and returns
        transformed versions of those.
    as_pandas : bool, default True
        Return pd.DataFrame when pandas is installed.
        If False or pandas is not installed, return np.ndarray
    verbose_eval : bool, int, or None, default None
        Whether to display the progress. If None, progress will be displayed
        when np.ndarray is returned. If True, progress will be displayed at
        boosting stage. If an integer is given, progress will be displayed
        at every given `verbose_eval` boosting stage.
    show_stdv : bool, default True
        Whether to display the standard deviation in progress.
        Results are not affected, and always contains std.
    seed : int
        Seed used to generate the folds (passed to numpy.random.seed).
    callbacks : list of callback functions
        List of callback functions that are applied at end of each iteration.
        It is possible to use predefined callbacks by using xgb.callback module.
        Example: [xgb.callback.reset_learning_rate(custom_rates)]

    Returns
    -------
    evaluation history : list(string)
    """
    if stratified is True and not SKLEARN_INSTALLED:
        raise XGBoostError('sklearn needs to be installed in order to use stratified cv')

    if isinstance(metrics, str):
        metrics = [metrics]

    if isinstance(params, list):
        _metrics = [x[1] for x in params if x[0] == 'eval_metric']
        params = dict(params)
        if 'eval_metric' in params:
            params['eval_metric'] = _metrics
    else:
        params = dict((k, v) for k, v in params.items())

    if len(metrics) == 0 and 'eval_metric' in params:
        if isinstance(params['eval_metric'], list):
            metrics = params['eval_metric']
        else:
            metrics = [params['eval_metric']]

    params.pop("eval_metric", None)

    results = {}
    cvfolds = mknfold(dtrain, nfold, params, seed, metrics, fpreproc, stratified, folds)

    # setup callbacks
    callbacks = [] if callbacks is None else callbacks
    if early_stopping_rounds is not None:
        callbacks.append(callback.early_stop(early_stopping_rounds,
                                             maximize=maximize,
                                             verbose=False))

    if isinstance(verbose_eval, bool) and verbose_eval:
        callbacks.append(callback.print_evaluation(show_stdv=show_stdv))
    else:
        if isinstance(verbose_eval, int):
            callbacks.append(callback.print_evaluation(verbose_eval, show_stdv=show_stdv))

    callbacks_before_iter = [
        cb for cb in callbacks if cb.__dict__.get('before_iteration', False)]
    callbacks_after_iter = [
        cb for cb in callbacks if not cb.__dict__.get('before_iteration', False)]

    for i in range(num_boost_round):
        for cb in callbacks_before_iter:
            cb(CallbackEnv(model=None,
                           cvfolds=cvfolds,
                           iteration=i,
                           begin_iteration=0,
                           end_iteration=num_boost_round,
                           rank=0,
                           evaluation_result_list=None))
        for fold in cvfolds:
            fold.update(i, obj)
        res = aggcv([f.eval(i, feval) for f in cvfolds])

        for key, mean, std in res:
            if key + '-mean' not in results:
                results[key + '-mean'] = []
            if key + '-std' not in results:
                results[key + '-std'] = []
            results[key + '-mean'].append(mean)
            results[key + '-std'].append(std)
        try:
            for cb in callbacks_after_iter:
                cb(CallbackEnv(model=None,
                               cvfolds=cvfolds,
                               iteration=i,
                               begin_iteration=0,
                               end_iteration=num_boost_round,
                               rank=0,
                               evaluation_result_list=res))
        except EarlyStopException as e:
            for k in results.keys():
                results[k] = results[k][:(e.best_iteration + 1)]
            break
    if as_pandas:
        try:
            import pandas as pd
            results = pd.DataFrame.from_dict(results)
        except ImportError:
            pass
    return results
