# pylint: disable=too-many-locals, too-many-arguments, invalid-name
# pylint: disable=too-many-branches, too-many-statements
"""Training Library containing training routines."""
import copy
import os
import weakref
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast

import numpy as np

from ._typing import BoosterParam, Callable, FPreProcCallable
from .callback import (
    CallbackContainer,
    EarlyStopping,
    EvaluationMonitor,
    TrainingCallback,
)
from .compat import SKLEARN_INSTALLED, DataFrame, XGBStratifiedKFold
from .core import (
    Booster,
    DMatrix,
    Metric,
    Objective,
    XGBoostError,
    _deprecate_positional_args,
    _RefMixIn,
)

_CVFolds = Sequence["CVPack"]


@_deprecate_positional_args
def train(
    params: Dict[str, Any],
    dtrain: DMatrix,
    num_boost_round: int = 10,
    *,
    evals: Optional[Sequence[Tuple[DMatrix, str]]] = None,
    obj: Optional[Objective] = None,
    maximize: Optional[bool] = None,
    early_stopping_rounds: Optional[int] = None,
    evals_result: Optional[TrainingCallback.EvalsLog] = None,
    verbose_eval: Optional[Union[bool, int]] = True,
    xgb_model: Optional[Union[str, os.PathLike, Booster, bytearray]] = None,
    callbacks: Optional[Sequence[TrainingCallback]] = None,
    custom_metric: Optional[Metric] = None,
) -> Booster:
    """Train a booster with given parameters.

    Parameters
    ----------
    params :
        Booster params.
    dtrain :
        Data to be trained.
    num_boost_round :
        Number of boosting iterations.
    evals :
        List of validation sets for which metrics will evaluated during training.
        Validation metrics will help us track the performance of the model.
    obj
        Custom objective function.  See :doc:`Custom Objective
        </tutorials/custom_metric_obj>` for details.
    maximize :
        Whether to maximize custom_metric.

    early_stopping_rounds :

        Activates early stopping. Validation metric needs to improve at least once in
        every **early_stopping_rounds** round(s) to continue training.

        Requires at least one item in **evals**.

        The method returns the model from the last iteration (not the best one).  Use
        custom callback :py:class:`~xgboost.callback.EarlyStopping` or :py:meth:`model
        slicing <xgboost.Booster.__getitem__>` if the best model is desired.  If there's
        more than one item in **evals**, the last entry will be used for early stopping.

        If there's more than one metric in the **eval_metric** parameter given in
        **params**, the last metric will be used for early stopping.

        If early stopping occurs, the model will have two additional fields:
        ``bst.best_score``, ``bst.best_iteration``.

    evals_result :
        This dictionary stores the evaluation results of all the items in watchlist.

        Example: with a watchlist containing
        ``[(dtest,'eval'), (dtrain,'train')]`` and
        a parameter containing ``('eval_metric': 'logloss')``,
        the **evals_result** returns

        .. code-block:: python

            {'train': {'logloss': ['0.48253', '0.35953']},
             'eval': {'logloss': ['0.480385', '0.357756']}}

    verbose_eval :
        Requires at least one item in **evals**.

        If **verbose_eval** is True then the evaluation metric on the validation set is
        printed at each boosting stage.

        If **verbose_eval** is an integer then the evaluation metric on the validation
        set is printed at every given **verbose_eval** boosting stage. The last boosting
        stage / the boosting stage found by using **early_stopping_rounds** is also
        printed.

        Example: with ``verbose_eval=4`` and at least one item in **evals**, an
        evaluation metric is printed every 4 boosting stages, instead of every boosting
        stage.

    xgb_model :
        Xgb model to be loaded before training (allows training continuation).

    callbacks :
        List of callback functions that are applied at end of each iteration.
        It is possible to use predefined callbacks by using
        :ref:`Callback API <callback_api>`.

        .. note::

           States in callback are not preserved during training, which means callback
           objects can not be reused for multiple training sessions without
           reinitialization or deepcopy.

        .. code-block:: python

            for params in parameters_grid:
                # be sure to (re)initialize the callbacks before each run
                callbacks = [xgb.callback.LearningRateScheduler(custom_rates)]
                xgboost.train(params, Xy, callbacks=callbacks)

    custom_metric:

        .. versionadded 1.6.0

        Custom metric function.  See :doc:`Custom Metric </tutorials/custom_metric_obj>`
        for details. The metric receives transformed prediction (after applying the
        reverse link function) when using a builtin objective, and raw output when using
        a custom objective.

    Returns
    -------
    Booster : a trained booster model

    """

    callbacks = [] if callbacks is None else copy.copy(list(callbacks))
    evals = list(evals) if evals else []

    for va, _ in evals:
        if not isinstance(va, DMatrix):
            raise TypeError("Invalid type for the `evals`.")

        if (
            isinstance(va, _RefMixIn)
            and va.ref is not weakref.ref(dtrain)
            and va is not dtrain
        ):
            raise ValueError(
                "Training dataset should be used as a reference when constructing "
                "the `QuantileDMatrix` for evaluation."
            )

    bst = Booster(params, [dtrain] + [d[0] for d in evals], model_file=xgb_model)
    start_iteration = 0

    if verbose_eval:
        verbose_eval = 1 if verbose_eval is True else verbose_eval
        callbacks.append(EvaluationMonitor(period=verbose_eval))
    if early_stopping_rounds:
        callbacks.append(EarlyStopping(rounds=early_stopping_rounds, maximize=maximize))
    cb_container = CallbackContainer(
        callbacks, metric=custom_metric, output_margin=callable(obj)
    )

    bst = cb_container.before_training(bst)

    for i in range(start_iteration, num_boost_round):
        if cb_container.before_iteration(bst, i, dtrain, evals):
            break
        bst.update(dtrain, iteration=i, fobj=obj)
        if cb_container.after_iteration(bst, i, dtrain, evals):
            break

    bst = cb_container.after_training(bst)

    if evals_result is not None:
        evals_result.update(cb_container.history)

    return bst.reset()


class CVPack:
    """ "Auxiliary datastruct to hold one fold of CV."""

    def __init__(
        self, dtrain: DMatrix, dtest: DMatrix, param: Optional[Union[Dict, List]]
    ) -> None:
        """Initialize the CVPack."""
        self.dtrain = dtrain
        self.dtest = dtest
        self.watchlist = [(dtrain, "train"), (dtest, "test")]
        self.bst = Booster(param, [dtrain, dtest])

    def __getattr__(self, name: str) -> Callable:
        def _inner(*args: Any, **kwargs: Any) -> Any:
            return getattr(self.bst, name)(*args, **kwargs)

        return _inner

    def update(self, iteration: int, fobj: Optional[Objective]) -> None:
        """ "Update the boosters for one iteration"""
        self.bst.update(self.dtrain, iteration, fobj)

    def eval(self, iteration: int, feval: Optional[Metric], output_margin: bool) -> str:
        """ "Evaluate the CVPack for one iteration."""
        return self.bst.eval_set(self.watchlist, iteration, feval, output_margin)


class _PackedBooster:
    def __init__(self, cvfolds: _CVFolds) -> None:
        self.cvfolds = cvfolds

    def update(self, iteration: int, obj: Optional[Objective]) -> None:
        """Iterate through folds for update"""
        for fold in self.cvfolds:
            fold.update(iteration, obj)

    def eval(
        self, iteration: int, feval: Optional[Metric], output_margin: bool
    ) -> List[str]:
        """Iterate through folds for eval"""
        result = [f.eval(iteration, feval, output_margin) for f in self.cvfolds]
        return result

    def set_attr(self, **kwargs: Optional[Any]) -> Any:
        """Iterate through folds for setting attributes"""
        for f in self.cvfolds:
            f.bst.set_attr(**kwargs)

    def attr(self, key: str) -> Optional[str]:
        """Redirect to booster attr."""
        return self.cvfolds[0].bst.attr(key)

    def set_param(
        self,
        params: Union[Dict, Iterable[Tuple[str, Any]], str],
        value: Optional[str] = None,
    ) -> None:
        """Iterate through folds for set_param"""
        for f in self.cvfolds:
            f.bst.set_param(params, value)

    def num_boosted_rounds(self) -> int:
        """Number of boosted rounds."""
        return self.cvfolds[0].num_boosted_rounds()

    @property
    def best_iteration(self) -> int:
        """Get best_iteration"""
        return int(cast(int, self.cvfolds[0].bst.attr("best_iteration")))

    @best_iteration.setter
    def best_iteration(self, iteration: int) -> None:
        """Get best_iteration"""
        self.set_attr(best_iteration=iteration)

    @property
    def best_score(self) -> float:
        """Get best_score."""
        return float(cast(float, self.cvfolds[0].bst.attr("best_score")))

    @best_score.setter
    def best_score(self, score: float) -> None:
        self.set_attr(best_score=score)


def groups_to_rows(groups: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """
    Given group row boundaries, convert ground indexes to row indexes
    :param groups: list of groups for testing
    :param boundaries: rows index limits of each group
    :return: row in group
    """
    return np.concatenate([np.arange(boundaries[g], boundaries[g + 1]) for g in groups])


def mkgroupfold(
    *,
    dall: DMatrix,
    nfold: int,
    param: BoosterParam,
    evals: Sequence[str] = (),
    fpreproc: Optional[FPreProcCallable] = None,
    shuffle: bool = True,
) -> List[CVPack]:
    """
    Make n folds for cross-validation maintaining groups
    :return: cross-validation folds
    """
    # we have groups for pairwise ranking... get a list of the group indexes
    group_boundaries = dall.get_uint_info("group_ptr")
    group_sizes = np.diff(group_boundaries)

    if shuffle is True:
        idx = np.random.permutation(len(group_sizes))
    else:
        idx = np.arange(len(group_sizes))
    # list by fold of test group indexes
    out_group_idset = np.array_split(idx, nfold)
    # list by fold of train group indexes
    in_group_idset = [
        np.concatenate([out_group_idset[i] for i in range(nfold) if k != i])
        for k in range(nfold)
    ]
    # from the group indexes, convert them to row indexes
    in_idset = [
        groups_to_rows(in_groups, group_boundaries) for in_groups in in_group_idset
    ]
    out_idset = [
        groups_to_rows(out_groups, group_boundaries) for out_groups in out_group_idset
    ]

    # build the folds by taking the appropriate slices
    ret = []
    for k in range(nfold):
        # perform the slicing using the indexes determined by the above methods
        dtrain = dall.slice(in_idset[k], allow_groups=True)
        dtrain.set_group(group_sizes[in_group_idset[k]])
        dtest = dall.slice(out_idset[k], allow_groups=True)
        dtest.set_group(group_sizes[out_group_idset[k]])
        # run preprocessing on the data set if needed
        if fpreproc is not None:
            dtrain, dtest, tparam = fpreproc(dtrain, dtest, param.copy())
        else:
            tparam = param
        plst = list(tparam.items()) + [("eval_metric", itm) for itm in evals]
        ret.append(CVPack(dtrain, dtest, plst))
    return ret


def mknfold(
    *,
    dall: DMatrix,
    nfold: int,
    param: BoosterParam,
    seed: int,
    evals: Sequence[str] = (),
    fpreproc: Optional[FPreProcCallable] = None,
    stratified: Optional[bool] = False,
    folds: Optional[XGBStratifiedKFold] = None,
    shuffle: bool = True,
) -> List[CVPack]:
    """
    Make an n-fold list of CVPack from random indices.
    """
    evals = list(evals)
    np.random.seed(seed)

    if stratified is False and folds is None:
        # Do standard k-fold cross validation. Automatically determine the folds.
        if len(dall.get_uint_info("group_ptr")) > 1:
            return mkgroupfold(
                dall=dall,
                nfold=nfold,
                param=param,
                evals=evals,
                fpreproc=fpreproc,
                shuffle=shuffle,
            )

        if shuffle is True:
            idx = np.random.permutation(dall.num_row())
        else:
            idx = np.arange(dall.num_row())
        out_idset = np.array_split(idx, nfold)
        in_idset = [
            np.concatenate([out_idset[i] for i in range(nfold) if k != i])
            for k in range(nfold)
        ]
    elif folds is not None:
        # Use user specified custom split using indices
        try:
            in_idset = [x[0] for x in folds]
            out_idset = [x[1] for x in folds]
        except TypeError:
            # Custom stratification using Sklearn KFoldSplit object
            splits = list(folds.split(X=dall.get_label(), y=dall.get_label()))
            in_idset = [x[0] for x in splits]
            out_idset = [x[1] for x in splits]
        nfold = len(out_idset)
    else:
        # Do standard stratefied shuffle k-fold split
        sfk = XGBStratifiedKFold(n_splits=nfold, shuffle=True, random_state=seed)
        splits = list(sfk.split(X=dall.get_label(), y=dall.get_label()))
        in_idset = [x[0] for x in splits]
        out_idset = [x[1] for x in splits]
        nfold = len(out_idset)

    ret = []
    for k in range(nfold):
        # perform the slicing using the indexes determined by the above methods
        dtrain = dall.slice(in_idset[k])
        dtest = dall.slice(out_idset[k])
        # run preprocessing on the data set if needed
        if fpreproc is not None:
            dtrain, dtest, tparam = fpreproc(dtrain, dtest, param.copy())
        else:
            tparam = param
        plst = list(tparam.items()) + [("eval_metric", itm) for itm in evals]
        ret.append(CVPack(dtrain, dtest, plst))
    return ret


@_deprecate_positional_args
def cv(
    params: BoosterParam,
    dtrain: DMatrix,
    num_boost_round: int = 10,
    *,
    nfold: int = 3,
    stratified: bool = False,
    folds: XGBStratifiedKFold = None,
    metrics: Sequence[str] = (),
    obj: Optional[Objective] = None,
    maximize: Optional[bool] = None,
    early_stopping_rounds: Optional[int] = None,
    fpreproc: Optional[FPreProcCallable] = None,
    as_pandas: bool = True,
    verbose_eval: Optional[Union[int, bool]] = None,
    show_stdv: bool = True,
    seed: int = 0,
    callbacks: Optional[Sequence[TrainingCallback]] = None,
    shuffle: bool = True,
    custom_metric: Optional[Metric] = None,
) -> Union[Dict[str, float], DataFrame]:
    # pylint: disable = invalid-name
    """Cross-validation with given parameters.

    Parameters
    ----------
    params : dict
        Booster params.
    dtrain :
        Data to be trained. Only the :py:class:`DMatrix` without external memory is
        supported.
    num_boost_round :
        Number of boosting iterations.
    nfold : int
        Number of folds in CV.
    stratified : bool
        Perform stratified sampling.
    folds : a KFold or StratifiedKFold instance or list of fold indices
        Sklearn KFolds or StratifiedKFolds object.
        Alternatively may explicitly pass sample indices for each fold.
        For ``n`` folds, **folds** should be a length ``n`` list of tuples.
        Each tuple is ``(in,out)`` where ``in`` is a list of indices to be used
        as the training samples for the ``n`` th fold and ``out`` is a list of
        indices to be used as the testing samples for the ``n`` th fold.
    metrics : string or list of strings
        Evaluation metrics to be watched in CV.
    obj :

        Custom objective function.  See :doc:`Custom Objective
        </tutorials/custom_metric_obj>` for details.

    maximize : bool
        Whether to maximize the evaluataion metric (score or error).

    early_stopping_rounds: int
        Activates early stopping. Cross-Validation metric (average of validation
        metric computed over CV folds) needs to improve at least once in
        every **early_stopping_rounds** round(s) to continue training.
        The last entry in the evaluation history will represent the best iteration.
        If there's more than one metric in the **eval_metric** parameter given in
        **params**, the last metric will be used for early stopping.
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
    callbacks :
        List of callback functions that are applied at end of each iteration.
        It is possible to use predefined callbacks by using
        :ref:`Callback API <callback_api>`.

        .. note::

           States in callback are not preserved during training, which means callback
           objects can not be reused for multiple training sessions without
           reinitialization or deepcopy.

        .. code-block:: python

            for params in parameters_grid:
                # be sure to (re)initialize the callbacks before each run
                callbacks = [xgb.callback.LearningRateScheduler(custom_rates)]
                xgboost.train(params, Xy, callbacks=callbacks)

    shuffle : bool
        Shuffle data before creating folds.
    custom_metric :

        .. versionadded 1.6.0

        Custom metric function.  See :doc:`Custom Metric </tutorials/custom_metric_obj>`
        for details.

    Returns
    -------
    evaluation history : list(string)
    """
    if stratified is True and not SKLEARN_INSTALLED:
        raise XGBoostError(
            "sklearn needs to be installed in order to use stratified cv"
        )
    if isinstance(metrics, str):
        metrics = [metrics]
    if isinstance(dtrain, _RefMixIn):
        raise ValueError("`QuantileDMatrix` is not yet supported.")

    params = params.copy()
    if isinstance(params, list):
        _metrics = [x[1] for x in params if x[0] == "eval_metric"]
        params = dict(params)
        if "eval_metric" in params:
            params["eval_metric"] = _metrics

    if (not metrics) and "eval_metric" in params:
        if isinstance(params["eval_metric"], list):
            metrics = params["eval_metric"]
        else:
            metrics = [params["eval_metric"]]

    params.pop("eval_metric", None)

    results: Dict[str, List[float]] = {}
    cvfolds = mknfold(
        dall=dtrain,
        nfold=nfold,
        param=params,
        seed=seed,
        evals=metrics,
        fpreproc=fpreproc,
        stratified=stratified,
        folds=folds,
        shuffle=shuffle,
    )

    # setup callbacks
    callbacks = [] if callbacks is None else copy.copy(list(callbacks))

    if verbose_eval:
        verbose_eval = 1 if verbose_eval is True else verbose_eval
        callbacks.append(EvaluationMonitor(period=verbose_eval, show_stdv=show_stdv))
    if early_stopping_rounds:
        callbacks.append(EarlyStopping(rounds=early_stopping_rounds, maximize=maximize))
    callbacks_container = CallbackContainer(
        callbacks, metric=custom_metric, is_cv=True, output_margin=callable(obj)
    )

    booster = _PackedBooster(cvfolds)
    callbacks_container.before_training(booster)

    for i in range(num_boost_round):
        if callbacks_container.before_iteration(booster, i, dtrain, None):
            break
        booster.update(i, obj)

        should_break = callbacks_container.after_iteration(booster, i, dtrain, None)
        res = callbacks_container.aggregated_cv
        for key, mean, std in cast(List[Tuple[str, float, float]], res):
            if key + "-mean" not in results:
                results[key + "-mean"] = []
            if key + "-std" not in results:
                results[key + "-std"] = []
            results[key + "-mean"].append(mean)
            results[key + "-std"].append(std)

        if should_break:
            for k in results.keys():  # pylint: disable=consider-iterating-dictionary
                results[k] = results[k][: (booster.best_iteration + 1)]
            break
    if as_pandas:
        try:
            import pandas as pd

            results = pd.DataFrame.from_dict(results)
        except ImportError:
            pass

    callbacks_container.after_training(booster)

    return results
