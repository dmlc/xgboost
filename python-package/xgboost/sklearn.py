# pylint: disable=too-many-arguments, too-many-locals, invalid-name, fixme, too-many-lines
"""Scikit-Learn Wrapper interface for XGBoost."""
import copy
import warnings
import json
import os
from typing import Union, Optional, List, Dict, Callable, Tuple, Any, TypeVar, Type, cast
from typing import Sequence
import numpy as np

from .core import Booster, DMatrix, XGBoostError
from .core import _deprecate_positional_args, _convert_ntree_limit
from .core import Metric
from .training import train
from .callback import TrainingCallback
from .data import _is_cudf_df, _is_cudf_ser, _is_cupy_array
from ._typing import ArrayLike

# Do not use class names on scikit-learn directly.  Re-define the classes on
# .compat to guarantee the behavior without scikit-learn
from .compat import (
    SKLEARN_INSTALLED,
    XGBModelBase,
    XGBClassifierBase,
    XGBRegressorBase,
    XGBoostLabelEncoder,
)


class XGBRankerMixIn:  # pylint: disable=too-few-public-methods
    """MixIn for ranking, defines the _estimator_type usually defined in scikit-learn base
    classes."""

    _estimator_type = "ranker"


def _check_rf_callback(
    early_stopping_rounds: Optional[int],
    callbacks: Optional[Sequence[TrainingCallback]],
) -> None:
    if early_stopping_rounds is not None or callbacks is not None:
        raise NotImplementedError(
            "`early_stopping_rounds` and `callbacks` are not implemented for"
            " random forest."
        )


_SklObjective = Optional[
    Union[
        str, Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
    ]
]


def _objective_decorator(
    func: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
) -> Callable[[np.ndarray, DMatrix], Tuple[np.ndarray, np.ndarray]]:
    """Decorate an objective function

    Converts an objective function using the typical sklearn metrics
    signature so that it is usable with ``xgboost.training.train``

    Parameters
    ----------
    func:
        Expects a callable with signature ``func(y_true, y_pred)``:

        y_true: array_like of shape [n_samples]
            The target values
        y_pred: array_like of shape [n_samples]
            The predicted values

    Returns
    -------
    new_func:
        The new objective function as expected by ``xgboost.training.train``.
        The signature is ``new_func(preds, dmatrix)``:

        preds: array_like, shape [n_samples]
            The predicted values
        dmatrix: ``DMatrix``
            The training set from which the labels will be extracted using
            ``dmatrix.get_label()``
    """
    def inner(preds: np.ndarray, dmatrix: DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        """internal function"""
        labels = dmatrix.get_label()
        return func(labels, preds)
    return inner


def _metric_decorator(func: Callable) -> Metric:
    """Decorate a metric function from sklearn.

    Converts an metric function that uses the typical sklearn metric signature so that it
    is compatible with :py:func:`train`

    """
    def inner(y_score: np.ndarray, dmatrix: DMatrix) -> Tuple[str, float]:
        y_true = dmatrix.get_label()
        return func.__name__, func(y_true, y_score)
    return inner


__estimator_doc = '''
    n_estimators : int
        Number of gradient boosted trees.  Equivalent to number of boosting
        rounds.
'''

__model_doc = f'''
    max_depth :  Optional[int]
        Maximum tree depth for base learners.
    max_leaves :
        Maximum number of leaves; 0 indicates no limit.
    max_bin :
        If using histogram-based algorithm, maximum number of bins per feature
    grow_policy :
        Tree growing policy. 0: favor splitting at nodes closest to the node, i.e. grow
        depth-wise. 1: favor splitting at nodes with highest loss change.
    learning_rate : Optional[float]
        Boosting learning rate (xgb's "eta")
    verbosity : Optional[int]
        The degree of verbosity. Valid values are 0 (silent) - 3 (debug).
    objective : {_SklObjective}
        Specify the learning task and the corresponding learning objective or
        a custom objective function to be used (see note below).
    booster: Optional[str]
        Specify which booster to use: gbtree, gblinear or dart.
    tree_method: Optional[str]
        Specify which tree method to use.  Default to auto.  If this parameter is set to
        default, XGBoost will choose the most conservative option available.  It's
        recommended to study this option from the parameters document :doc:`tree method
        </treemethod>`
    n_jobs : Optional[int]
        Number of parallel threads used to run xgboost.  When used with other Scikit-Learn
        algorithms like grid search, you may choose which algorithm to parallelize and
        balance the threads.  Creating thread contention will significantly slow down both
        algorithms.
    gamma : Optional[float]
        (min_split_loss) Minimum loss reduction required to make a further partition on a
        leaf node of the tree.
    min_child_weight : Optional[float]
        Minimum sum of instance weight(hessian) needed in a child.
    max_delta_step : Optional[float]
        Maximum delta step we allow each tree's weight estimation to be.
    subsample : Optional[float]
        Subsample ratio of the training instance.
    sampling_method :
        Sampling method. Used only by `gpu_hist` tree method.
          - `uniform`: select random training instances uniformly.
          - `gradient_based` select random training instances with higher probability when
            the gradient and hessian are larger. (cf. CatBoost)
    colsample_bytree : Optional[float]
        Subsample ratio of columns when constructing each tree.
    colsample_bylevel : Optional[float]
        Subsample ratio of columns for each level.
    colsample_bynode : Optional[float]
        Subsample ratio of columns for each split.
    reg_alpha : Optional[float]
        L1 regularization term on weights (xgb's alpha).
    reg_lambda : Optional[float]
        L2 regularization term on weights (xgb's lambda).
    scale_pos_weight : Optional[float]
        Balancing of positive and negative weights.
    base_score : Optional[float]
        The initial prediction score of all instances, global bias.
    random_state : Optional[Union[numpy.random.RandomState, int]]
        Random number seed.

        .. note::

           Using gblinear booster with shotgun updater is nondeterministic as
           it uses Hogwild algorithm.

    missing : float, default np.nan
        Value in the data which needs to be present as a missing value.
    num_parallel_tree: Optional[int]
        Used for boosting random forest.
    monotone_constraints : Optional[Union[Dict[str, int], str]]
        Constraint of variable monotonicity.  See :doc:`tutorial </tutorials/monotonic>`
        for more information.
    interaction_constraints : Optional[Union[str, List[Tuple[str]]]]
        Constraints for interaction representing permitted interactions.  The
        constraints must be specified in the form of a nested list, e.g. ``[[0, 1], [2,
        3, 4]]``, where each inner list is a group of indices of features that are
        allowed to interact with each other.  See :doc:`tutorial
        </tutorials/feature_interaction_constraint>` for more information
    importance_type: Optional[str]
        The feature importance type for the feature_importances\\_ property:

        * For tree model, it's either "gain", "weight", "cover", "total_gain" or
          "total_cover".
        * For linear model, only "weight" is defined and it's the normalized coefficients
          without bias.

    gpu_id : Optional[int]
        Device ordinal.
    validate_parameters : Optional[bool]
        Give warnings for unknown parameter.
    predictor : Optional[str]
        Force XGBoost to use specific predictor, available choices are [cpu_predictor,
        gpu_predictor].
    enable_categorical : bool

        .. versionadded:: 1.5.0

        .. note:: This parameter is experimental

        Experimental support for categorical data.  When enabled, cudf/pandas.DataFrame
        should be used to specify categorical data type.  Also, JSON/UBJSON
        serialization format is required.

    max_cat_to_onehot : Optional[int]

        .. versionadded:: 1.6.0

        .. note:: This parameter is experimental

        A threshold for deciding whether XGBoost should use one-hot encoding based split
        for categorical data.  When number of categories is lesser than the threshold
        then one-hot encoding is chosen, otherwise the categories will be partitioned
        into children nodes.  Only relevant for regression and binary classification.
        See :doc:`Categorical Data </tutorials/categorical>` for details.

    eval_metric : Optional[Union[str, List[str], Callable]]

        .. versionadded:: 1.6.0

        Metric used for monitoring the training result and early stopping.  It can be a
        string or list of strings as names of predefined metric in XGBoost (See
        doc/parameter.rst), one of the metrics in :py:mod:`sklearn.metrics`, or any other
        user defined metric that looks like `sklearn.metrics`.

        If custom objective is also provided, then custom metric should implement the
        corresponding reverse link function.

        Unlike the `scoring` parameter commonly used in scikit-learn, when a callable
        object is provided, it's assumed to be a cost function and by default XGBoost will
        minimize the result during early stopping.

        For advanced usage on Early stopping like directly choosing to maximize instead of
        minimize, see :py:obj:`xgboost.callback.EarlyStopping`.

        See :doc:`Custom Objective and Evaluation Metric </tutorials/custom_metric_obj>`
        for more.

        .. note::

             This parameter replaces `eval_metric` in :py:meth:`fit` method.  The old one
             receives un-transformed prediction regardless of whether custom objective is
             being used.

        .. code-block:: python

            from sklearn.datasets import load_diabetes
            from sklearn.metrics import mean_absolute_error
            X, y = load_diabetes(return_X_y=True)
            reg = xgb.XGBRegressor(
                tree_method="hist",
                eval_metric=mean_absolute_error,
            )
            reg.fit(X, y, eval_set=[(X, y)])

    early_stopping_rounds : Optional[int]

        .. versionadded:: 1.6.0

        Activates early stopping. Validation metric needs to improve at least once in
        every **early_stopping_rounds** round(s) to continue training.  Requires at least
        one item in **eval_set** in :py:meth:`fit`.

        The method returns the model from the last iteration (not the best one).  If
        there's more than one item in **eval_set**, the last entry will be used for early
        stopping.  If there's more than one metric in **eval_metric**, the last metric
        will be used for early stopping.

        If early stopping occurs, the model will have three additional fields:
        :py:attr:`best_score`, :py:attr:`best_iteration` and
        :py:attr:`best_ntree_limit`.

        .. note::

            This parameter replaces `early_stopping_rounds` in :py:meth:`fit` method.

    callbacks : Optional[List[TrainingCallback]]
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

    kwargs : dict, optional
        Keyword arguments for XGBoost Booster object.  Full documentation of parameters
        can be found :doc:`here </parameter>`.
        Attempting to set a parameter via the constructor args and \\*\\*kwargs
        dict simultaneously will result in a TypeError.

        .. note:: \\*\\*kwargs unsupported by scikit-learn

            \\*\\*kwargs is unsupported by scikit-learn.  We do not guarantee
            that parameters passed via this argument will interact properly
            with scikit-learn.
'''

__custom_obj_note = '''
        .. note::  Custom objective function

            A custom objective function can be provided for the ``objective``
            parameter. In this case, it should have the signature
            ``objective(y_true, y_pred) -> grad, hess``:

            y_true: array_like of shape [n_samples]
                The target values
            y_pred: array_like of shape [n_samples]
                The predicted values

            grad: array_like of shape [n_samples]
                The value of the gradient for each sample point.
            hess: array_like of shape [n_samples]
                The value of the second derivative for each sample point
'''


def xgboost_model_doc(
    header: str, items: List[str],
    extra_parameters: Optional[str] = None,
    end_note: Optional[str] = None
) -> Callable[[Type], Type]:
    '''Obtain documentation for Scikit-Learn wrappers

    Parameters
    ----------
    header: str
       An introducion to the class.
    items : list
       A list of common doc items.  Available items are:
         - estimators: the meaning of n_estimators
         - model: All the other parameters
         - objective: note for customized objective
    extra_parameters: str
       Document for class specific parameters, placed at the head.
    end_note: str
       Extra notes put to the end.
'''
    def get_doc(item: str) -> str:
        '''Return selected item'''
        __doc = {'estimators': __estimator_doc,
                 'model': __model_doc,
                 'objective': __custom_obj_note}
        return __doc[item]

    def adddoc(cls: Type) -> Type:
        doc = ['''
Parameters
----------
''']
        if extra_parameters:
            doc.append(extra_parameters)
        doc.extend([get_doc(i) for i in items])
        if end_note:
            doc.append(end_note)
        full_doc = [header + '\n\n']
        full_doc.extend(doc)
        cls.__doc__ = ''.join(full_doc)
        return cls
    return adddoc


def _wrap_evaluation_matrices(
    missing: float,
    X: Any,
    y: Any,
    group: Optional[Any],
    qid: Optional[Any],
    sample_weight: Optional[Any],
    base_margin: Optional[Any],
    feature_weights: Optional[Any],
    eval_set: Optional[Sequence[Tuple[Any, Any]]],
    sample_weight_eval_set: Optional[Sequence[Any]],
    base_margin_eval_set: Optional[Sequence[Any]],
    eval_group: Optional[Sequence[Any]],
    eval_qid: Optional[Sequence[Any]],
    create_dmatrix: Callable,
    enable_categorical: bool,
) -> Tuple[Any, List[Tuple[Any, str]]]:
    """Convert array_like evaluation matrices into DMatrix.  Perform validation on the way.

    """
    train_dmatrix = create_dmatrix(
        data=X,
        label=y,
        group=group,
        qid=qid,
        weight=sample_weight,
        base_margin=base_margin,
        feature_weights=feature_weights,
        missing=missing,
        enable_categorical=enable_categorical,
    )

    n_validation = 0 if eval_set is None else len(eval_set)

    def validate_or_none(meta: Optional[Sequence], name: str) -> Sequence:
        if meta is None:
            return [None] * n_validation
        if len(meta) != n_validation:
            raise ValueError(
                f"{name}'s length does not equal `eval_set`'s length, " +
                f"expecting {n_validation}, got {len(meta)}"
            )
        return meta

    if eval_set is not None:
        sample_weight_eval_set = validate_or_none(
            sample_weight_eval_set, "sample_weight_eval_set"
        )
        base_margin_eval_set = validate_or_none(
            base_margin_eval_set, "base_margin_eval_set"
        )
        eval_group = validate_or_none(eval_group, "eval_group")
        eval_qid = validate_or_none(eval_qid, "eval_qid")

        evals = []
        for i, (valid_X, valid_y) in enumerate(eval_set):
            # Skip the duplicated entry.
            if all(
                (
                    valid_X is X, valid_y is y,
                    sample_weight_eval_set[i] is sample_weight,
                    base_margin_eval_set[i] is base_margin,
                    eval_group[i] is group,
                    eval_qid[i] is qid
                )
            ):
                evals.append(train_dmatrix)
            else:
                m = create_dmatrix(
                    data=valid_X,
                    label=valid_y,
                    weight=sample_weight_eval_set[i],
                    group=eval_group[i],
                    qid=eval_qid[i],
                    base_margin=base_margin_eval_set[i],
                    missing=missing,
                    enable_categorical=enable_categorical,
                )
                evals.append(m)
        nevals = len(evals)
        eval_names = [f"validation_{i}" for i in range(nevals)]
        evals = list(zip(evals, eval_names))
    else:
        if any(
            meta is not None
            for meta in [
                sample_weight_eval_set,
                base_margin_eval_set,
                eval_group,
                eval_qid,
            ]
        ):
            raise ValueError(
                "`eval_set` is not set but one of the other evaluation meta info is "
                "not None."
            )
        evals = []

    return train_dmatrix, evals


@xgboost_model_doc("""Implementation of the Scikit-Learn API for XGBoost.""",
                   ['estimators', 'model', 'objective'])
class XGBModel(XGBModelBase):
    # pylint: disable=too-many-arguments, too-many-instance-attributes, missing-docstring
    def __init__(
        self,
        max_depth: Optional[int] = None,
        max_leaves: Optional[int] = None,
        max_bin: Optional[int] = None,
        grow_policy: Optional[str] = None,
        learning_rate: Optional[float] = None,
        n_estimators: int = 100,
        verbosity: Optional[int] = None,
        objective: _SklObjective = None,
        booster: Optional[str] = None,
        tree_method: Optional[str] = None,
        n_jobs: Optional[int] = None,
        gamma: Optional[float] = None,
        min_child_weight: Optional[float] = None,
        max_delta_step: Optional[float] = None,
        subsample: Optional[float] = None,
        sampling_method: Optional[str] = None,
        colsample_bytree: Optional[float] = None,
        colsample_bylevel: Optional[float] = None,
        colsample_bynode: Optional[float] = None,
        reg_alpha: Optional[float] = None,
        reg_lambda: Optional[float] = None,
        scale_pos_weight: Optional[float] = None,
        base_score: Optional[float] = None,
        random_state: Optional[Union[np.random.RandomState, int]] = None,
        missing: float = np.nan,
        num_parallel_tree: Optional[int] = None,
        monotone_constraints: Optional[Union[Dict[str, int], str]] = None,
        interaction_constraints: Optional[Union[str, Sequence[Sequence[str]]]] = None,
        importance_type: Optional[str] = None,
        gpu_id: Optional[int] = None,
        validate_parameters: Optional[bool] = None,
        predictor: Optional[str] = None,
        enable_categorical: bool = False,
        max_cat_to_onehot: Optional[int] = None,
        eval_metric: Optional[Union[str, List[str], Callable]] = None,
        early_stopping_rounds: Optional[int] = None,
        callbacks: Optional[List[TrainingCallback]] = None,
        **kwargs: Any
    ) -> None:
        if not SKLEARN_INSTALLED:
            raise XGBoostError(
                "sklearn needs to be installed in order to use this module"
            )
        self.n_estimators = n_estimators
        self.objective = objective

        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.max_bin = max_bin
        self.grow_policy = grow_policy
        self.learning_rate = learning_rate
        self.verbosity = verbosity
        self.booster = booster
        self.tree_method = tree_method
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.sampling_method = sampling_method
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.base_score = base_score
        self.missing = missing
        self.num_parallel_tree = num_parallel_tree
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.monotone_constraints = monotone_constraints
        self.interaction_constraints = interaction_constraints
        self.importance_type = importance_type
        self.gpu_id = gpu_id
        self.validate_parameters = validate_parameters
        self.predictor = predictor
        self.enable_categorical = enable_categorical
        self.max_cat_to_onehot = max_cat_to_onehot
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.callbacks = callbacks
        if kwargs:
            self.kwargs = kwargs

    def _more_tags(self) -> Dict[str, bool]:
        '''Tags used for scikit-learn data validation.'''
        return {'allow_nan': True, 'no_validation': True}

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "_Booster")

    def get_booster(self) -> Booster:
        """Get the underlying xgboost Booster of this model.

        This will raise an exception when fit was not called

        Returns
        -------
        booster : a xgboost booster of underlying model
        """
        if not self.__sklearn_is_fitted__():
            from sklearn.exceptions import NotFittedError
            raise NotFittedError('need to call fit or load_model beforehand')
        return self._Booster

    def set_params(self, **params: Any) -> "XGBModel":
        """Set the parameters of this estimator.  Modification of the sklearn method to
        allow unknown kwargs. This allows using the full range of xgboost
        parameters that are not defined as member variables in sklearn grid
        search.

        Returns
        -------
        self

        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self

        # this concatenates kwargs into parameters, enabling `get_params` for
        # obtaining parameters from keyword parameters.
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                if not hasattr(self, "kwargs"):
                    self.kwargs = {}
                self.kwargs[key] = value

        if hasattr(self, '_Booster'):
            parameters = self.get_xgb_params()
            self.get_booster().set_param(parameters)

        return self

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        # pylint: disable=attribute-defined-outside-init
        """Get parameters."""
        # Based on: https://stackoverflow.com/questions/59248211
        # The basic flow in `get_params` is:
        # 0. Return parameters in subclass first, by using inspect.
        # 1. Return parameters in `XGBModel` (the base class).
        # 2. Return whatever in `**kwargs`.
        # 3. Merge them.
        params = super().get_params(deep)
        cp = copy.copy(self)
        cp.__class__ = cp.__class__.__bases__[0]
        params.update(cp.__class__.get_params(cp, deep))
        # if kwargs is a dict, update params accordingly
        if hasattr(self, "kwargs") and isinstance(self.kwargs, dict):
            params.update(self.kwargs)
        if isinstance(params['random_state'], np.random.RandomState):
            params['random_state'] = params['random_state'].randint(
                np.iinfo(np.int32).max)

        def parse_parameter(value: Any) -> Optional[Union[int, float, str]]:
            for t in (int, float, str):
                try:
                    ret = t(value)
                    return ret
                except ValueError:
                    continue
            return None

        # Get internal parameter values
        try:
            config = json.loads(self.get_booster().save_config())
            stack = [config]
            internal = {}
            while stack:
                obj = stack.pop()
                for k, v in obj.items():
                    if k.endswith('_param'):
                        for p_k, p_v in v.items():
                            internal[p_k] = p_v
                    elif isinstance(v, dict):
                        stack.append(v)

            for k, v in internal.items():
                if k in params and params[k] is None:
                    params[k] = parse_parameter(v)
        except ValueError:
            pass
        return params

    def get_xgb_params(self) -> Dict[str, Any]:
        """Get xgboost specific parameters."""
        params = self.get_params()
        # Parameters that should not go into native learner.
        wrapper_specific = {
            "importance_type",
            "kwargs",
            "missing",
            "n_estimators",
            "use_label_encoder",
            "enable_categorical",
            "early_stopping_rounds",
            "callbacks",
        }
        filtered = {}
        for k, v in params.items():
            if k not in wrapper_specific and not callable(v):
                filtered[k] = v
        return filtered

    def get_num_boosting_rounds(self) -> int:
        """Gets the number of xgboost boosting rounds."""
        return self.n_estimators

    def _get_type(self) -> str:
        if not hasattr(self, '_estimator_type'):
            raise TypeError(
                "`_estimator_type` undefined.  "
                "Please use appropriate mixin to define estimator type."
            )
        return self._estimator_type  # pylint: disable=no-member

    def save_model(self, fname: Union[str, os.PathLike]) -> None:
        meta = {}
        for k, v in self.__dict__.items():
            if k == '_le':
                meta['_le'] = self._le.to_json()
                continue
            if k == '_Booster':
                continue
            if k == 'classes_':
                # numpy array is not JSON serializable
                meta['classes_'] = self.classes_.tolist()
                continue
            try:
                json.dumps({k: v})
                meta[k] = v
            except TypeError:
                warnings.warn(str(k) + ' is not saved in Scikit-Learn meta.', UserWarning)
        meta['_estimator_type'] = self._get_type()
        meta_str = json.dumps(meta)
        self.get_booster().set_attr(scikit_learn=meta_str)
        self.get_booster().save_model(fname)
        # Delete the attribute after save
        self.get_booster().set_attr(scikit_learn=None)

    save_model.__doc__ = f"""{Booster.save_model.__doc__}"""

    def load_model(self, fname: Union[str, bytearray, os.PathLike]) -> None:
        # pylint: disable=attribute-defined-outside-init
        if not hasattr(self, '_Booster'):
            self._Booster = Booster({'n_jobs': self.n_jobs})
        self.get_booster().load_model(fname)
        meta_str = self.get_booster().attr('scikit_learn')
        if meta_str is None:
            # FIXME(jiaming): This doesn't have to be a problem as most of the needed
            # information like num_class and objective is in Learner class.
            warnings.warn(
                'Loading a native XGBoost model with Scikit-Learn interface.'
            )
            return
        meta = json.loads(meta_str)
        states = {}
        for k, v in meta.items():
            if k == '_le':
                self._le = XGBoostLabelEncoder()
                self._le.from_json(v)
                continue
            # FIXME(jiaming): This can be removed once label encoder is gone since we can
            # generate it from `np.arange(self.n_classes_)`
            if k == 'classes_':
                self.classes_ = np.array(v)
                continue
            if k == "_estimator_type":
                if self._get_type() != v:
                    raise TypeError(
                        "Loading an estimator with different type. "
                        f"Expecting: {self._get_type()}, got: {v}"
                    )
                continue
            states[k] = v
        self.__dict__.update(states)
        # Delete the attribute after load
        self.get_booster().set_attr(scikit_learn=None)

    load_model.__doc__ = f"""{Booster.load_model.__doc__}"""

    # pylint: disable=too-many-branches
    def _configure_fit(
        self,
        booster: Optional[Union[Booster, "XGBModel", str]],
        eval_metric: Optional[Union[Callable, str, Sequence[str]]],
        params: Dict[str, Any],
        early_stopping_rounds: Optional[int],
        callbacks: Optional[Sequence[TrainingCallback]],
    ) -> Tuple[
        Optional[Union[Booster, str, "XGBModel"]],
        Optional[Metric],
        Dict[str, Any],
        Optional[int],
        Optional[Sequence[TrainingCallback]],
    ]:
        """Configure parameters for :py:meth:`fit`."""
        if isinstance(booster, XGBModel):
            model: Optional[Union[Booster, str]] = booster.get_booster()
        else:
            model = booster

        def _deprecated(parameter: str) -> None:
            warnings.warn(
                f"`{parameter}` in `fit` method is deprecated for better compatibility "
                f"with scikit-learn, use `{parameter}` in constructor or`set_params` "
                "instead.",
                UserWarning,
            )

        def _duplicated(parameter: str) -> None:
            raise ValueError(
                f"2 different `{parameter}` are provided.  Use the one in constructor "
                "or `set_params` instead."
            )

        # Configure evaluation metric.
        if eval_metric is not None:
            _deprecated("eval_metric")
        if self.eval_metric is not None and eval_metric is not None:
            _duplicated("eval_metric")
        # - track where does the evaluation metric come from
        if self.eval_metric is not None:
            from_fit = False
            eval_metric = self.eval_metric
        else:
            from_fit = True
        # - configure callable evaluation metric
        metric: Optional[Metric] = None
        if eval_metric is not None:
            if callable(eval_metric) and from_fit:
                # No need to wrap the evaluation function for old parameter.
                metric = eval_metric
            elif callable(eval_metric):
                # Parameter from constructor or set_params
                metric = _metric_decorator(eval_metric)
            else:
                params.update({"eval_metric": eval_metric})

        # Configure early_stopping_rounds
        if early_stopping_rounds is not None:
            _deprecated("early_stopping_rounds")
        if early_stopping_rounds is not None and self.early_stopping_rounds is not None:
            _duplicated("early_stopping_rounds")
        early_stopping_rounds = (
            self.early_stopping_rounds
            if self.early_stopping_rounds is not None
            else early_stopping_rounds
        )

        # Configure callbacks
        if callbacks is not None:
            _deprecated("callbacks")
        if callbacks is not None and self.callbacks is not None:
            _duplicated("callbacks")
        callbacks = self.callbacks if self.callbacks is not None else callbacks

        tree_method = params.get("tree_method", None)
        cat_support = {"gpu_hist", "approx", "hist"}
        if self.enable_categorical and tree_method not in cat_support:
            raise ValueError(
                "Experimental support for categorical data is not implemented for"
                " current tree method yet."
            )

        return model, metric, params, early_stopping_rounds, callbacks

    def _set_evaluation_result(self, evals_result: TrainingCallback.EvalsLog) -> None:
        if evals_result:
            self.evals_result_ = cast(Dict[str, Dict[str, List[float]]], evals_result)

    @_deprecate_positional_args
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        sample_weight: Optional[ArrayLike] = None,
        base_margin: Optional[ArrayLike] = None,
        eval_set: Optional[Sequence[Tuple[ArrayLike, ArrayLike]]] = None,
        eval_metric: Optional[Union[str, Sequence[str], Metric]] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: Optional[bool] = True,
        xgb_model: Optional[Union[Booster, str, "XGBModel"]] = None,
        sample_weight_eval_set: Optional[Sequence[ArrayLike]] = None,
        base_margin_eval_set: Optional[Sequence[ArrayLike]] = None,
        feature_weights: Optional[ArrayLike] = None,
        callbacks: Optional[Sequence[TrainingCallback]] = None
    ) -> "XGBModel":
        # pylint: disable=invalid-name,attribute-defined-outside-init
        """Fit gradient boosting model.

        Note that calling ``fit()`` multiple times will cause the model object to be
        re-fit from scratch. To resume training from a previous checkpoint, explicitly
        pass ``xgb_model`` argument.

        Parameters
        ----------
        X :
            Feature matrix
        y :
            Labels
        sample_weight :
            instance weights
        base_margin :
            global bias for each instance.
        eval_set :
            A list of (X, y) tuple pairs to use as validation sets, for which
            metrics will be computed.
            Validation metrics will help us track the performance of the model.

        eval_metric : str, list of str, or callable, optional
            .. deprecated:: 1.6.0
                Use `eval_metric` in :py:meth:`__init__` or :py:meth:`set_params` instead.

        early_stopping_rounds : int
            .. deprecated:: 1.6.0
                Use `early_stopping_rounds` in :py:meth:`__init__` or
                :py:meth:`set_params` instead.
        verbose :
            If `verbose` and an evaluation set is used, writes the evaluation metric
            measured on the validation set to stderr.
        xgb_model :
            file name of stored XGBoost model or 'Booster' instance XGBoost model to be
            loaded before training (allows training continuation).
        sample_weight_eval_set :
            A list of the form [L_1, L_2, ..., L_n], where each L_i is an array like
            object storing instance weights for the i-th validation set.
        base_margin_eval_set :
            A list of the form [M_1, M_2, ..., M_n], where each M_i is an array like
            object storing base margin for the i-th validation set.
        feature_weights :
            Weight for each feature, defines the probability of each feature being
            selected when colsample is being used.  All values must be greater than 0,
            otherwise a `ValueError` is thrown.

        callbacks :
            .. deprecated:: 1.6.0
                Use `callbacks` in :py:meth:`__init__` or :py:meth:`set_params` instead.
        """
        evals_result: TrainingCallback.EvalsLog = {}
        train_dmatrix, evals = _wrap_evaluation_matrices(
            missing=self.missing,
            X=X,
            y=y,
            group=None,
            qid=None,
            sample_weight=sample_weight,
            base_margin=base_margin,
            feature_weights=feature_weights,
            eval_set=eval_set,
            sample_weight_eval_set=sample_weight_eval_set,
            base_margin_eval_set=base_margin_eval_set,
            eval_group=None,
            eval_qid=None,
            create_dmatrix=lambda **kwargs: DMatrix(nthread=self.n_jobs, **kwargs),
            enable_categorical=self.enable_categorical,
        )
        params = self.get_xgb_params()

        if callable(self.objective):
            obj: Optional[
                Callable[[np.ndarray, DMatrix], Tuple[np.ndarray, np.ndarray]]
            ] = _objective_decorator(self.objective)
            params["objective"] = "reg:squarederror"
        else:
            obj = None

        model, metric, params, early_stopping_rounds, callbacks = self._configure_fit(
            xgb_model, eval_metric, params, early_stopping_rounds, callbacks
        )
        self._Booster = train(
            params,
            train_dmatrix,
            self.get_num_boosting_rounds(),
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            evals_result=evals_result,
            obj=obj,
            custom_metric=metric,
            verbose_eval=verbose,
            xgb_model=model,
            callbacks=callbacks,
        )

        self._set_evaluation_result(evals_result)
        return self

    def _can_use_inplace_predict(self) -> bool:
        # When predictor is explicitly set, using `inplace_predict` might result into
        # error with incompatible data type.
        # Inplace predict doesn't handle as many data types as DMatrix, but it's
        # sufficient for dask interface where input is simpiler.
        predictor = self.get_params().get("predictor", None)
        if predictor in ("auto", None) and self.booster != "gblinear":
            return True
        return False

    def _get_iteration_range(
        self, iteration_range: Optional[Tuple[int, int]]
    ) -> Tuple[int, int]:
        if (iteration_range is None or iteration_range[1] == 0):
            # Use best_iteration if defined.
            try:
                iteration_range = (0, self.best_iteration + 1)
            except AttributeError:
                iteration_range = (0, 0)
        if self.booster == "gblinear":
            iteration_range = (0, 0)
        return iteration_range

    def predict(
        self,
        X: ArrayLike,
        output_margin: bool = False,
        ntree_limit: Optional[int] = None,
        validate_features: bool = True,
        base_margin: Optional[ArrayLike] = None,
        iteration_range: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Predict with `X`.  If the model is trained with early stopping, then `best_iteration`
        is used automatically.  For tree models, when data is on GPU, like cupy array or
        cuDF dataframe and `predictor` is not specified, the prediction is run on GPU
        automatically, otherwise it will run on CPU.

        .. note:: This function is only thread safe for `gbtree` and `dart`.

        Parameters
        ----------
        X :
            Data to predict with.
        output_margin :
            Whether to output the raw untransformed margin value.
        ntree_limit :
            Deprecated, use `iteration_range` instead.
        validate_features :
            When this is True, validate that the Booster's and data's feature_names are
            identical.  Otherwise, it is assumed that the feature_names are the same.
        base_margin :
            Margin added to prediction.
        iteration_range :
            Specifies which layer of trees are used in prediction.  For example, if a
            random forest is trained with 100 rounds.  Specifying ``iteration_range=(10,
            20)``, then only the forests built during [10, 20) (half open set) rounds are
            used in this prediction.

            .. versionadded:: 1.4.0

        Returns
        -------
        prediction

        """
        iteration_range = _convert_ntree_limit(
            self.get_booster(), ntree_limit, iteration_range
        )
        iteration_range = self._get_iteration_range(iteration_range)
        if self._can_use_inplace_predict():
            try:
                predts = self.get_booster().inplace_predict(
                    data=X,
                    iteration_range=iteration_range,
                    predict_type="margin" if output_margin else "value",
                    missing=self.missing,
                    base_margin=base_margin,
                    validate_features=validate_features,
                )
                if _is_cupy_array(predts):
                    import cupy     # pylint: disable=import-error
                    predts = cupy.asnumpy(predts)  # ensure numpy array is used.
                return predts
            except TypeError:
                # coo, csc, dt
                pass

        test = DMatrix(
            X, base_margin=base_margin,
            missing=self.missing,
            nthread=self.n_jobs,
            enable_categorical=self.enable_categorical
        )
        return self.get_booster().predict(
            data=test,
            iteration_range=iteration_range,
            output_margin=output_margin,
            validate_features=validate_features,
        )

    def apply(
        self, X: ArrayLike,
        ntree_limit: int = 0,
        iteration_range: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """Return the predicted leaf every tree for each sample. If the model is trained with
        early stopping, then `best_iteration` is used automatically.

        Parameters
        ----------
        X : array_like, shape=[n_samples, n_features]
            Input features matrix.

        iteration_range :
            See :py:meth:`predict`.

        ntree_limit :
            Deprecated, use ``iteration_range`` instead.

        Returns
        -------
        X_leaves : array_like, shape=[n_samples, n_trees]
            For each datapoint x in X and for each tree, return the index of the
            leaf x ends up in. Leaves are numbered within
            ``[0; 2**(self.max_depth+1))``, possibly with gaps in the numbering.

        """
        iteration_range = _convert_ntree_limit(
            self.get_booster(), ntree_limit, iteration_range
        )
        iteration_range = self._get_iteration_range(iteration_range)
        test_dmatrix = DMatrix(X, missing=self.missing, nthread=self.n_jobs)
        return self.get_booster().predict(
            test_dmatrix,
            pred_leaf=True,
            iteration_range=iteration_range
        )

    def evals_result(self) -> Dict[str, Dict[str, List[float]]]:
        """Return the evaluation results.

        If **eval_set** is passed to the :py:meth:`fit` function, you can call
        ``evals_result()`` to get evaluation results for all passed **eval_sets**.  When
        **eval_metric** is also passed to the :py:meth:`fit` function, the
        **evals_result** will contain the **eval_metrics** passed to the :py:meth:`fit`
        function.

        The returned evaluation result is a dictionary:

        .. code-block:: python

            {'validation_0': {'logloss': ['0.604835', '0.531479']},
             'validation_1': {'logloss': ['0.41965', '0.17686']}}

        Returns
        -------
        evals_result

        """
        if getattr(self, "evals_result_", None) is not None:
            evals_result = self.evals_result_
        else:
            raise XGBoostError(
                "No evaluation result, `eval_set` is not used during training."
            )

        return evals_result

    @property
    def n_features_in_(self) -> int:
        """Number of features seen during :py:meth:`fit`."""
        booster = self.get_booster()
        return booster.num_features()

    @property
    def feature_names_in_(self) -> np.ndarray:
        """Names of features seen during :py:meth:`fit`.  Defined only when `X` has feature
        names that are all strings."""
        feature_names = self.get_booster().feature_names
        if feature_names is None:
            raise AttributeError(
                "`feature_names_in_` is defined only when `X` has feature names that "
                "are all strings."
            )
        return np.array(feature_names)

    def _early_stopping_attr(self, attr: str) -> Union[float, int]:
        booster = self.get_booster()
        try:
            return getattr(booster, attr)
        except AttributeError as e:
            raise AttributeError(
                f'`{attr}` in only defined when early stopping is used.'
            ) from e

    @property
    def best_score(self) -> float:
        """The best score obtained by early stopping."""
        return float(self._early_stopping_attr('best_score'))

    @property
    def best_iteration(self) -> int:
        """The best iteration obtained by early stopping.  This attribute is 0-based,
        for instance if the best iteration is the first round, then best_iteration is 0.

        """
        return int(self._early_stopping_attr('best_iteration'))

    @property
    def best_ntree_limit(self) -> int:
        return int(self._early_stopping_attr('best_ntree_limit'))

    @property
    def feature_importances_(self) -> np.ndarray:
        """
        Feature importances property, return depends on `importance_type` parameter.

        Returns
        -------
        feature_importances_ : array of shape ``[n_features]`` except for multi-class
        linear model, which returns an array with shape `(n_features, n_classes)`

        """
        b: Booster = self.get_booster()

        def dft() -> str:
            return "weight" if self.booster == "gblinear" else "gain"
        score = b.get_score(
            importance_type=self.importance_type if self.importance_type else dft()
        )
        if b.feature_names is None:
            feature_names = [f"f{i}" for i in range(self.n_features_in_)]
        else:
            feature_names = b.feature_names
        # gblinear returns all features so the `get` in next line is only for gbtree.
        all_features = [score.get(f, 0.) for f in feature_names]
        all_features_arr = np.array(all_features, dtype=np.float32)
        total = all_features_arr.sum()
        if total == 0:
            return all_features_arr
        return all_features_arr / total

    @property
    def coef_(self) -> np.ndarray:
        """
        Coefficients property

        .. note:: Coefficients are defined only for linear learners

            Coefficients are only defined when the linear model is chosen as
            base learner (`booster=gblinear`). It is not defined for other base
            learner types, such as tree learners (`booster=gbtree`).

        Returns
        -------
        coef_ : array of shape ``[n_features]`` or ``[n_classes, n_features]``
        """
        if self.get_params()['booster'] != 'gblinear':
            raise AttributeError(
                f"Coefficients are not defined for Booster type {self.booster}"
            )
        b = self.get_booster()
        coef = np.array(json.loads(b.get_dump(dump_format='json')[0])['weight'])
        # Logic for multiclass classification
        n_classes = getattr(self, 'n_classes_', None)
        if n_classes is not None:
            if n_classes > 2:
                assert len(coef.shape) == 1
                assert coef.shape[0] % n_classes == 0
                coef = coef.reshape((n_classes, -1))
        return coef

    @property
    def intercept_(self) -> np.ndarray:
        """
        Intercept (bias) property

        .. note:: Intercept is defined only for linear learners

            Intercept (bias) is only defined when the linear model is chosen as base
            learner (`booster=gblinear`). It is not defined for other base learner types,
            such as tree learners (`booster=gbtree`).

        Returns
        -------
        intercept_ : array of shape ``(1,)`` or ``[n_classes]``
        """
        if self.get_params()['booster'] != 'gblinear':
            raise AttributeError(
                f"Intercept (bias) is not defined for Booster type {self.booster}"
            )
        b = self.get_booster()
        return np.array(json.loads(b.get_dump(dump_format='json')[0])['bias'])


PredtT = TypeVar("PredtT", bound=np.ndarray)


def _cls_predict_proba(n_classes: int, prediction: PredtT, vstack: Callable) -> PredtT:
    assert len(prediction.shape) <= 2
    if len(prediction.shape) == 2 and prediction.shape[1] == n_classes:
        # multi-class
        return prediction
    if (
        len(prediction.shape) == 2
        and n_classes == 2
        and prediction.shape[1] >= n_classes
    ):
        # multi-label
        return prediction
    # binary logistic function
    classone_probs = prediction
    classzero_probs = 1.0 - classone_probs
    return vstack((classzero_probs, classone_probs)).transpose()


@xgboost_model_doc(
    "Implementation of the scikit-learn API for XGBoost classification.",
    ['model', 'objective'], extra_parameters='''
    n_estimators : int
        Number of boosting rounds.
''')
class XGBClassifier(XGBModel, XGBClassifierBase):
    # pylint: disable=missing-docstring,invalid-name,too-many-instance-attributes
    @_deprecate_positional_args
    def __init__(
        self,
        *,
        objective: _SklObjective = "binary:logistic",
        use_label_encoder: bool = False,
        **kwargs: Any
    ) -> None:
        # must match the parameters for `get_params`
        self.use_label_encoder = use_label_encoder
        if use_label_encoder is True:
            raise ValueError("Label encoder was removed in 1.6.")
        super().__init__(objective=objective, **kwargs)

    @_deprecate_positional_args
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        sample_weight: Optional[ArrayLike] = None,
        base_margin: Optional[ArrayLike] = None,
        eval_set: Optional[Sequence[Tuple[ArrayLike, ArrayLike]]] = None,
        eval_metric: Optional[Union[str, Sequence[str], Metric]] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: Optional[bool] = True,
        xgb_model: Optional[Union[Booster, str, XGBModel]] = None,
        sample_weight_eval_set: Optional[Sequence[ArrayLike]] = None,
        base_margin_eval_set: Optional[Sequence[ArrayLike]] = None,
        feature_weights: Optional[ArrayLike] = None,
        callbacks: Optional[Sequence[TrainingCallback]] = None
    ) -> "XGBClassifier":
        # pylint: disable = attribute-defined-outside-init,too-many-statements
        evals_result: TrainingCallback.EvalsLog = {}

        if _is_cudf_df(y) or _is_cudf_ser(y):
            import cupy as cp  # pylint: disable=E0401

            self.classes_ = cp.unique(y.values)
            self.n_classes_ = len(self.classes_)
            expected_classes = cp.arange(self.n_classes_)
        elif _is_cupy_array(y):
            import cupy as cp  # pylint: disable=E0401

            self.classes_ = cp.unique(y)
            self.n_classes_ = len(self.classes_)
            expected_classes = cp.arange(self.n_classes_)
        else:
            self.classes_ = np.unique(np.asarray(y))
            self.n_classes_ = len(self.classes_)
            expected_classes = np.arange(self.n_classes_)
        if (
            self.classes_.shape != expected_classes.shape
            or not (self.classes_ == expected_classes).all()
        ):
            raise ValueError(
                f"Invalid classes inferred from unique values of `y`.  "
                f"Expected: {expected_classes}, got {self.classes_}"
            )

        params = self.get_xgb_params()

        if callable(self.objective):
            obj: Optional[
                Callable[[np.ndarray, DMatrix], Tuple[np.ndarray, np.ndarray]]
            ] = _objective_decorator(self.objective)
            # Use default value. Is it really not used ?
            params["objective"] = "binary:logistic"
        else:
            obj = None

        if self.n_classes_ > 2:
            # Switch to using a multiclass objective in the underlying XGB instance
            if params.get("objective", None) != "multi:softmax":
                params["objective"] = "multi:softprob"
            params["num_class"] = self.n_classes_

        model, metric, params, early_stopping_rounds, callbacks = self._configure_fit(
            xgb_model, eval_metric, params, early_stopping_rounds, callbacks
        )
        train_dmatrix, evals = _wrap_evaluation_matrices(
            missing=self.missing,
            X=X,
            y=y,
            group=None,
            qid=None,
            sample_weight=sample_weight,
            base_margin=base_margin,
            feature_weights=feature_weights,
            eval_set=eval_set,
            sample_weight_eval_set=sample_weight_eval_set,
            base_margin_eval_set=base_margin_eval_set,
            eval_group=None,
            eval_qid=None,
            create_dmatrix=lambda **kwargs: DMatrix(nthread=self.n_jobs, **kwargs),
            enable_categorical=self.enable_categorical,
        )

        self._Booster = train(
            params,
            train_dmatrix,
            self.get_num_boosting_rounds(),
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            evals_result=evals_result,
            obj=obj,
            custom_metric=metric,
            verbose_eval=verbose,
            xgb_model=model,
            callbacks=callbacks,
        )

        if not callable(self.objective):
            self.objective = params["objective"]

        self._set_evaluation_result(evals_result)
        return self

    assert XGBModel.fit.__doc__ is not None
    fit.__doc__ = XGBModel.fit.__doc__.replace(
        'Fit gradient boosting model',
        'Fit gradient boosting classifier', 1)

    def predict(
        self,
        X: ArrayLike,
        output_margin: bool = False,
        ntree_limit: Optional[int] = None,
        validate_features: bool = True,
        base_margin: Optional[ArrayLike] = None,
        iteration_range: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        class_probs = super().predict(
            X=X,
            output_margin=output_margin,
            ntree_limit=ntree_limit,
            validate_features=validate_features,
            base_margin=base_margin,
            iteration_range=iteration_range,
        )
        if output_margin:
            # If output_margin is active, simply return the scores
            return class_probs

        if len(class_probs.shape) > 1 and self.n_classes_ != 2:
            # multi-class, turns softprob into softmax
            column_indexes: np.ndarray = np.argmax(class_probs, axis=1)  # type: ignore
        elif len(class_probs.shape) > 1 and class_probs.shape[1] != 1:
            # multi-label
            column_indexes = np.zeros(class_probs.shape)
            column_indexes[class_probs > 0.5] = 1
        elif self.objective == "multi:softmax":
            return class_probs.astype(np.int32)
        else:
            # turns soft logit into class label
            column_indexes = np.repeat(0, class_probs.shape[0])
            column_indexes[class_probs > 0.5] = 1

        if hasattr(self, '_le'):
            return self._le.inverse_transform(column_indexes)
        return column_indexes

    def predict_proba(
        self,
        X: ArrayLike,
        ntree_limit: Optional[int] = None,
        validate_features: bool = True,
        base_margin: Optional[ArrayLike] = None,
        iteration_range: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """ Predict the probability of each `X` example being of a given class.

        .. note:: This function is only thread safe for `gbtree` and `dart`.

        Parameters
        ----------
        X : array_like
            Feature matrix.
        ntree_limit : int
            Deprecated, use `iteration_range` instead.
        validate_features : bool
            When this is True, validate that the Booster's and data's feature_names are
            identical.  Otherwise, it is assumed that the feature_names are the same.
        base_margin : array_like
            Margin added to prediction.
        iteration_range :
            Specifies which layer of trees are used in prediction.  For example, if a
            random forest is trained with 100 rounds.  Specifying `iteration_range=(10,
            20)`, then only the forests built during [10, 20) (half open set) rounds are
            used in this prediction.

        Returns
        -------
        prediction :
            a numpy array of shape array-like of shape (n_samples, n_classes) with the
            probability of each data example being of a given class.
        """
        # custom obj:      Do nothing as we don't know what to do.
        # softprob:        Do nothing, output is proba.
        # softmax:         Unsupported by predict_proba()
        # binary:logistic: Expand the prob vector into 2-class matrix after predict.
        # binary:logitraw: Unsupported by predict_proba()
        if self.objective == "multi:softmax":
            # We need to run a Python implementation of softmax for it.  Just ask user to
            # use softprob since XGBoost's implementation has mitigation for floating
            # point overflow.  No need to reinvent the wheel.
            raise ValueError(
                "multi:softmax doesn't support `predict_proba`.  "
                "Switch to `multi:softproba` instead"
            )
        class_probs = super().predict(
            X=X,
            ntree_limit=ntree_limit,
            validate_features=validate_features,
            base_margin=base_margin,
            iteration_range=iteration_range
        )
        # If model is loaded from a raw booster there's no `n_classes_`
        return _cls_predict_proba(getattr(self, "n_classes_", 0), class_probs, np.vstack)


@xgboost_model_doc(
    "scikit-learn API for XGBoost random forest classification.",
    ['model', 'objective'],
    extra_parameters='''
    n_estimators : int
        Number of trees in random forest to fit.
''')
class XGBRFClassifier(XGBClassifier):
    # pylint: disable=missing-docstring
    @_deprecate_positional_args
    def __init__(
        self, *,
        learning_rate: float = 1.0,
        subsample: float = 0.8,
        colsample_bynode: float = 0.8,
        reg_lambda: float = 1e-5,
        **kwargs: Any
    ):
        super().__init__(learning_rate=learning_rate,
                         subsample=subsample,
                         colsample_bynode=colsample_bynode,
                         reg_lambda=reg_lambda,
                         **kwargs)
        _check_rf_callback(self.early_stopping_rounds, self.callbacks)

    def get_xgb_params(self) -> Dict[str, Any]:
        params = super().get_xgb_params()
        params['num_parallel_tree'] = self.n_estimators
        return params

    def get_num_boosting_rounds(self) -> int:
        return 1

    # pylint: disable=unused-argument
    @_deprecate_positional_args
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        sample_weight: Optional[ArrayLike] = None,
        base_margin: Optional[ArrayLike] = None,
        eval_set: Optional[Sequence[Tuple[ArrayLike, ArrayLike]]] = None,
        eval_metric: Optional[Union[str, Sequence[str], Metric]] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: Optional[bool] = True,
        xgb_model: Optional[Union[Booster, str, XGBModel]] = None,
        sample_weight_eval_set: Optional[Sequence[ArrayLike]] = None,
        base_margin_eval_set: Optional[Sequence[ArrayLike]] = None,
        feature_weights: Optional[ArrayLike] = None,
        callbacks: Optional[Sequence[TrainingCallback]] = None
    ) -> "XGBRFClassifier":
        args = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        _check_rf_callback(early_stopping_rounds, callbacks)
        super().fit(**args)
        return self


@xgboost_model_doc(
    "Implementation of the scikit-learn API for XGBoost regression.",
    ['estimators', 'model', 'objective'])
class XGBRegressor(XGBModel, XGBRegressorBase):
    # pylint: disable=missing-docstring
    @_deprecate_positional_args
    def __init__(
        self, *, objective: _SklObjective = "reg:squarederror", **kwargs: Any
    ) -> None:
        super().__init__(objective=objective, **kwargs)


@xgboost_model_doc(
    "scikit-learn API for XGBoost random forest regression.",
    ['model', 'objective'], extra_parameters='''
    n_estimators : int
        Number of trees in random forest to fit.
''')
class XGBRFRegressor(XGBRegressor):
    # pylint: disable=missing-docstring
    @_deprecate_positional_args
    def __init__(
        self,
        *,
        learning_rate: float = 1.0,
        subsample: float = 0.8,
        colsample_bynode: float = 0.8,
        reg_lambda: float = 1e-5,
        **kwargs: Any
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bynode=colsample_bynode,
            reg_lambda=reg_lambda,
            **kwargs
        )
        _check_rf_callback(self.early_stopping_rounds, self.callbacks)

    def get_xgb_params(self) -> Dict[str, Any]:
        params = super().get_xgb_params()
        params["num_parallel_tree"] = self.n_estimators
        return params

    def get_num_boosting_rounds(self) -> int:
        return 1

    # pylint: disable=unused-argument
    @_deprecate_positional_args
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        sample_weight: Optional[ArrayLike] = None,
        base_margin: Optional[ArrayLike] = None,
        eval_set: Optional[Sequence[Tuple[ArrayLike, ArrayLike]]] = None,
        eval_metric: Optional[Union[str, Sequence[str], Metric]] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: Optional[bool] = True,
        xgb_model: Optional[Union[Booster, str, XGBModel]] = None,
        sample_weight_eval_set: Optional[Sequence[ArrayLike]] = None,
        base_margin_eval_set: Optional[Sequence[ArrayLike]] = None,
        feature_weights: Optional[ArrayLike] = None,
        callbacks: Optional[Sequence[TrainingCallback]] = None
    ) -> "XGBRFRegressor":
        args = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        _check_rf_callback(early_stopping_rounds, callbacks)
        super().fit(**args)
        return self


@xgboost_model_doc(
    'Implementation of the Scikit-Learn API for XGBoost Ranking.',
    ['estimators', 'model'],
    end_note='''
        .. note::

            A custom objective function is currently not supported by XGBRanker.
            Likewise, a custom metric function is not supported either.

        .. note::

            Query group information is required for ranking tasks by either using the
            `group` parameter or `qid` parameter in `fit` method.

        Before fitting the model, your data need to be sorted by query group. When fitting
        the model, you need to provide an additional array that contains the size of each
        query group.

        For example, if your original data look like:

        +-------+-----------+---------------+
        |   qid |   label   |   features    |
        +-------+-----------+---------------+
        |   1   |   0       |   x_1         |
        +-------+-----------+---------------+
        |   1   |   1       |   x_2         |
        +-------+-----------+---------------+
        |   1   |   0       |   x_3         |
        +-------+-----------+---------------+
        |   2   |   0       |   x_4         |
        +-------+-----------+---------------+
        |   2   |   1       |   x_5         |
        +-------+-----------+---------------+
        |   2   |   1       |   x_6         |
        +-------+-----------+---------------+
        |   2   |   1       |   x_7         |
        +-------+-----------+---------------+

        then your group array should be ``[3, 4]``.  Sometimes using query id (`qid`)
        instead of group can be more convenient.
''')
class XGBRanker(XGBModel, XGBRankerMixIn):
    # pylint: disable=missing-docstring,too-many-arguments,invalid-name
    @_deprecate_positional_args
    def __init__(self, *, objective: str = "rank:pairwise", **kwargs: Any):
        super().__init__(objective=objective, **kwargs)
        if callable(self.objective):
            raise ValueError("custom objective function not supported by XGBRanker")
        if "rank:" not in objective:
            raise ValueError("please use XGBRanker for ranking task")

    @_deprecate_positional_args
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        group: Optional[ArrayLike] = None,
        qid: Optional[ArrayLike] = None,
        sample_weight: Optional[ArrayLike] = None,
        base_margin: Optional[ArrayLike] = None,
        eval_set: Optional[Sequence[Tuple[ArrayLike, ArrayLike]]] = None,
        eval_group: Optional[Sequence[ArrayLike]] = None,
        eval_qid: Optional[Sequence[ArrayLike]] = None,
        eval_metric: Optional[Union[str, Sequence[str], Metric]] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: Optional[bool] = False,
        xgb_model: Optional[Union[Booster, str, XGBModel]] = None,
        sample_weight_eval_set: Optional[Sequence[ArrayLike]] = None,
        base_margin_eval_set: Optional[Sequence[ArrayLike]] = None,
        feature_weights: Optional[ArrayLike] = None,
        callbacks: Optional[Sequence[TrainingCallback]] = None
    ) -> "XGBRanker":
        # pylint: disable = attribute-defined-outside-init,arguments-differ
        """Fit gradient boosting ranker

        Note that calling ``fit()`` multiple times will cause the model object to be
        re-fit from scratch. To resume training from a previous checkpoint, explicitly
        pass ``xgb_model`` argument.

        Parameters
        ----------
        X :
            Feature matrix
        y :
            Labels
        group :
            Size of each query group of training data. Should have as many elements as the
            query groups in the training data.  If this is set to None, then user must
            provide qid.
        qid :
            Query ID for each training sample.  Should have the size of n_samples.  If
            this is set to None, then user must provide group.
        sample_weight :
            Query group weights

            .. note:: Weights are per-group for ranking tasks

                In ranking task, one weight is assigned to each query group/id (not each
                data point). This is because we only care about the relative ordering of
                data points within each group, so it doesn't make sense to assign weights
                to individual data points.
        base_margin :
            Global bias for each instance.
        eval_set :
            A list of (X, y) tuple pairs to use as validation sets, for which
            metrics will be computed.
            Validation metrics will help us track the performance of the model.
        eval_group :
            A list in which ``eval_group[i]`` is the list containing the sizes of all
            query groups in the ``i``-th pair in **eval_set**.
        eval_qid :
            A list in which ``eval_qid[i]`` is the array containing query ID of ``i``-th
            pair in **eval_set**.

        eval_metric : str, list of str, optional
            .. deprecated:: 1.6.0
                use `eval_metric` in :py:meth:`__init__` or :py:meth:`set_params` instead.

        early_stopping_rounds : int
            .. deprecated:: 1.6.0
                use `early_stopping_rounds` in :py:meth:`__init__` or
                :py:meth:`set_params` instead.

        verbose :
            If `verbose` and an evaluation set is used, writes the evaluation metric
            measured on the validation set to stderr.
        xgb_model :
            file name of stored XGBoost model or 'Booster' instance XGBoost model to be
            loaded before training (allows training continuation).
        sample_weight_eval_set :
            A list of the form [L_1, L_2, ..., L_n], where each L_i is a list of
            group weights on the i-th validation set.

            .. note:: Weights are per-group for ranking tasks

                In ranking task, one weight is assigned to each query group (not each
                data point). This is because we only care about the relative ordering of
                data points within each group, so it doesn't make sense to assign
                weights to individual data points.
        base_margin_eval_set :
            A list of the form [M_1, M_2, ..., M_n], where each M_i is an array like
            object storing base margin for the i-th validation set.
        feature_weights :
            Weight for each feature, defines the probability of each feature being
            selected when colsample is being used.  All values must be greater than 0,
            otherwise a `ValueError` is thrown.

        callbacks :
            .. deprecated:: 1.6.0
                Use `callbacks` in :py:meth:`__init__` or :py:meth:`set_params` instead.
        """
        # check if group information is provided
        if group is None and qid is None:
            raise ValueError("group or qid is required for ranking task")

        if eval_set is not None:
            if eval_group is None and eval_qid is None:
                raise ValueError(
                    "eval_group or eval_qid is required if eval_set is not None")
        train_dmatrix, evals = _wrap_evaluation_matrices(
            missing=self.missing,
            X=X,
            y=y,
            group=group,
            qid=qid,
            sample_weight=sample_weight,
            base_margin=base_margin,
            feature_weights=feature_weights,
            eval_set=eval_set,
            sample_weight_eval_set=sample_weight_eval_set,
            base_margin_eval_set=base_margin_eval_set,
            eval_group=eval_group,
            eval_qid=eval_qid,
            create_dmatrix=lambda **kwargs: DMatrix(nthread=self.n_jobs, **kwargs),
            enable_categorical=self.enable_categorical,
        )

        evals_result: TrainingCallback.EvalsLog = {}
        params = self.get_xgb_params()

        model, metric, params, early_stopping_rounds, callbacks = self._configure_fit(
            xgb_model, eval_metric, params, early_stopping_rounds, callbacks
        )
        if callable(metric):
            raise ValueError(
                'Custom evaluation metric is not yet supported for XGBRanker.'
            )

        self._Booster = train(
            params,
            train_dmatrix,
            self.get_num_boosting_rounds(),
            early_stopping_rounds=early_stopping_rounds,
            evals=evals,
            evals_result=evals_result,
            custom_metric=metric,
            verbose_eval=verbose, xgb_model=model,
            callbacks=callbacks
        )

        self.objective = params["objective"]

        self._set_evaluation_result(evals_result)
        return self
