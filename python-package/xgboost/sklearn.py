# pylint: disable=too-many-arguments, too-many-locals, invalid-name, fixme, too-many-lines
"""Scikit-Learn Wrapper interface for XGBoost."""

import collections
import copy
import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from inspect import signature
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np
from scipy.special import softmax

from ._data_utils import Categories
from ._typing import (
    ArrayLike,
    EvalsLog,
    FeatureNames,
    FeatureTypes,
    IterationRange,
    ModelIn,
)
from .callback import TrainingCallback

# Do not use class names on scikit-learn directly.  Re-define the classes on
# .compat to guarantee the behavior without scikit-learn
from .compat import (
    SKLEARN_INSTALLED,
    XGBClassifierBase,
    XGBModelBase,
    XGBRegressorBase,
    _sklearn_Tags,
    _sklearn_version,
    import_cupy,
    is_dataframe,
)
from .config import config_context
from .core import (
    Booster,
    DMatrix,
    Metric,
    Objective,
    QuantileDMatrix,
    XGBoostError,
    _deprecate_positional_args,
    _parse_eval_str,
    _parse_version,
    _py_version,
)
from .data import (
    CAT_T,
    _is_cudf_df,
    _is_cudf_ser,
    _is_cupy_alike,
    _is_pandas_df,
    _is_polars_lazyframe,
)
from .training import train


class XGBRankerMixIn:
    """MixIn for ranking, defines the _estimator_type usually defined in scikit-learn
    base classes.

    """

    _estimator_type = "ranker"


def _check_rf_callback(
    early_stopping_rounds: Optional[int],
    callbacks: Optional[Sequence[TrainingCallback]],
) -> None:
    if early_stopping_rounds is not None or callbacks is not None:
        raise NotImplementedError(
            "`early_stopping_rounds` and `callbacks` are not implemented for"
            " the sklearn random forest estimator interface."
        )


def _can_use_qdm(tree_method: Optional[str], device: Optional[str]) -> bool:
    not_sycl = (device is None) or (not device.startswith("sycl"))
    return tree_method in ("hist", None, "auto") and not_sycl


class _SklObjWProto(Protocol):
    def __call__(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        sample_weight: Optional[ArrayLike],
    ) -> Tuple[ArrayLike, ArrayLike]: ...


_SklObjProto = Callable[[ArrayLike, ArrayLike], Tuple[np.ndarray, np.ndarray]]
SklObjective = Optional[Union[str, _SklObjWProto, _SklObjProto]]


def _objective_decorator(func: Union[_SklObjWProto, _SklObjProto]) -> Objective:
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
        sample_weight :
            Optional sample weight, None or a ndarray.

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

    parameters = signature(func).parameters
    supports_sw = "sample_weight" in parameters

    def inner(preds: np.ndarray, dmatrix: DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        """Internal function."""
        sample_weight = dmatrix.get_weight()
        labels = dmatrix.get_label()

        if sample_weight.size > 0 and not supports_sw:
            raise ValueError(
                "Custom objective doesn't have the `sample_weight` parameter while"
                " sample_weight is used."
            )
        if sample_weight.size > 0:
            fnw = cast(_SklObjWProto, func)
            return fnw(labels, preds, sample_weight=sample_weight)

        fn = cast(_SklObjProto, func)
        return fn(labels, preds)

    return inner


def _metric_decorator(func: Callable) -> Metric:
    """Decorate a metric function from sklearn.

    Converts an metric function that uses the typical sklearn metric signature so that
    it is compatible with :py:func:`train`

    """

    def inner(y_score: np.ndarray, dmatrix: DMatrix) -> Tuple[str, float]:
        y_true = dmatrix.get_label()
        weight = dmatrix.get_weight()
        if weight.size == 0:
            return func.__name__, func(y_true, y_score)
        return func.__name__, func(y_true, y_score, sample_weight=weight)

    return inner


def ltr_metric_decorator(func: Callable, n_jobs: Optional[int]) -> Metric:
    """Decorate a learning to rank metric."""

    def inner(y_score: np.ndarray, dmatrix: DMatrix) -> Tuple[str, float]:
        y_true = dmatrix.get_label()
        group_ptr = dmatrix.get_uint_info("group_ptr")
        if group_ptr.size < 2:
            raise ValueError(
                "Invalid `group_ptr`. Likely caused by invalid qid or group."
            )
        scores = np.empty(group_ptr.size - 1)
        futures = []
        weight = dmatrix.get_group()
        no_weight = weight.size == 0

        def task(i: int) -> float:
            begin = group_ptr[i - 1]
            end = group_ptr[i]
            gy = y_true[begin:end]
            gp = y_score[begin:end]
            if gy.size == 1:
                # Maybe there's a better default? 1.0 because many ranking score
                # functions have output in range [0, 1].
                return 1.0
            return func(gy, gp)

        workers = n_jobs if n_jobs is not None else os.cpu_count()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for i in range(1, group_ptr.size):
                f = executor.submit(task, i)
                futures.append(f)

            for i, f in enumerate(futures):
                scores[i] = f.result()

        if no_weight:
            return func.__name__, scores.mean()

        return func.__name__, np.average(scores, weights=weight)

    return inner


__estimator_doc = f"""
    n_estimators : {Optional[int]}
        Number of gradient boosted trees.  Equivalent to number of boosting
        rounds.
"""

__model_doc = f"""
    max_depth :  {Optional[int]}

        Maximum tree depth for base learners.

    max_leaves : {Optional[int]}

        Maximum number of leaves; 0 indicates no limit.

    max_bin : {Optional[int]}

        If using histogram-based algorithm, maximum number of bins per feature

    grow_policy : {Optional[str]}

        Tree growing policy.

        - depthwise: Favors splitting at nodes closest to the node,
        - lossguide: Favors splitting at nodes with highest loss change.

    learning_rate : {Optional[float]}

        Boosting learning rate (xgb's "eta")

    verbosity : {Optional[int]}

        The degree of verbosity. Valid values are 0 (silent) - 3 (debug).

    objective : {SklObjective}

        Specify the learning task and the corresponding learning objective or a custom
        objective function to be used.

        For custom objective, see :doc:`/tutorials/custom_metric_obj` and
        :ref:`custom-obj-metric` for more information, along with the end note for
        function signatures.

    booster: {Optional[str]}

        Specify which booster to use: ``gbtree``, ``gblinear`` or ``dart``.

    tree_method : {Optional[str]}

        Specify which tree method to use.  Default to auto.  If this parameter is set to
        default, XGBoost will choose the most conservative option available.  It's
        recommended to study this option from the parameters document :doc:`tree method
        </treemethod>`

    n_jobs : {Optional[int]}

        Number of parallel threads used to run xgboost.  When used with other
        Scikit-Learn algorithms like grid search, you may choose which algorithm to
        parallelize and balance the threads.  Creating thread contention will
        significantly slow down both algorithms.

    gamma : {Optional[float]}

        (min_split_loss) Minimum loss reduction required to make a further partition on
        a leaf node of the tree.

    min_child_weight : {Optional[float]}

        Minimum sum of instance weight(hessian) needed in a child.

    max_delta_step : {Optional[float]}

        Maximum delta step we allow each tree's weight estimation to be.

    subsample : {Optional[float]}

        Subsample ratio of the training instance.

    sampling_method : {Optional[str]}

        Sampling method. Used only by the GPU version of ``hist`` tree method.

        - ``uniform``: Select random training instances uniformly.
        - ``gradient_based``: Select random training instances with higher probability
            when the gradient and hessian are larger. (cf. CatBoost)

    colsample_bytree : {Optional[float]}

        Subsample ratio of columns when constructing each tree.

    colsample_bylevel : {Optional[float]}

        Subsample ratio of columns for each level.

    colsample_bynode : {Optional[float]}

        Subsample ratio of columns for each split.

    reg_alpha : {Optional[float]}

        L1 regularization term on weights (xgb's alpha).

    reg_lambda : {Optional[float]}

        L2 regularization term on weights (xgb's lambda).

    scale_pos_weight : {Optional[float]}
        Balancing of positive and negative weights.

    base_score : {Optional[Union[float, List[float]]]}

        The initial prediction score of all instances, global bias.

    random_state : {Optional[Union[np.random.RandomState, np.random.Generator, int]]}

        Random number seed.

        .. note::

           Using gblinear booster with shotgun updater is nondeterministic as
           it uses Hogwild algorithm.

    missing : float

        Value in the data which needs to be present as a missing value. Default to
        :py:data:`numpy.nan`.

    num_parallel_tree: {Optional[int]}

        Used for boosting random forest.

    monotone_constraints : {Optional[Union[Dict[str, int], str]]}

        Constraint of variable monotonicity.  See :doc:`tutorial </tutorials/monotonic>`
        for more information.

    interaction_constraints : {Optional[Union[str, List[Tuple[str]]]]}

        Constraints for interaction representing permitted interactions.  The
        constraints must be specified in the form of a nested list, e.g. ``[[0, 1], [2,
        3, 4]]``, where each inner list is a group of indices of features that are
        allowed to interact with each other.  See :doc:`tutorial
        </tutorials/feature_interaction_constraint>` for more information

    importance_type: {Optional[str]}

        The feature importance type for the feature_importances\\_ property:

        * For tree model, it's either "gain", "weight", "cover", "total_gain" or
          "total_cover".
        * For linear model, only "weight" is defined and it's the normalized
          coefficients without bias.

    device : {Optional[str]}

        .. versionadded:: 2.0.0

        Device ordinal, available options are `cpu`, `cuda`, and `gpu`.

    validate_parameters : {Optional[bool]}

        Give warnings for unknown parameter.

    enable_categorical : bool

        See the same parameter of :py:class:`DMatrix` for details.

    feature_types : {Optional[FeatureTypes]}

        .. versionadded:: 1.7.0

        Used for specifying feature types without constructing a dataframe. See
        the :py:class:`DMatrix` for details.

    feature_weights : Optional[ArrayLike]

        Weight for each feature, defines the probability of each feature being selected
        when colsample is being used.  All values must be greater than 0, otherwise a
        `ValueError` is thrown.

    max_cat_to_onehot : Optional[int]

        .. versionadded:: 1.6.0

        .. note:: This parameter is experimental

        A threshold for deciding whether XGBoost should use one-hot encoding based split
        for categorical data.  When number of categories is lesser than the threshold
        then one-hot encoding is chosen, otherwise the categories will be partitioned
        into children nodes. Also, `enable_categorical` needs to be set to have
        categorical feature support. See :doc:`Categorical Data
        </tutorials/categorical>` and :ref:`cat-param` for details.

    max_cat_threshold : {Optional[int]}

        .. versionadded:: 1.7.0

        .. note:: This parameter is experimental

        Maximum number of categories considered for each split. Used only by
        partition-based splits for preventing over-fitting. Also, `enable_categorical`
        needs to be set to have categorical feature support. See :doc:`Categorical Data
        </tutorials/categorical>` and :ref:`cat-param` for details.

    multi_strategy : {Optional[str]}

        .. versionadded:: 2.0.0

        .. note:: This parameter is working-in-progress.

        The strategy used for training multi-target models, including multi-target
        regression and multi-class classification. See :doc:`/tutorials/multioutput` for
        more information.

        - ``one_output_per_tree``: One model for each target.
        - ``multi_output_tree``:  Use multi-target trees.

    eval_metric : {Optional[Union[str, List[Union[str, Callable]], Callable]]}

        .. versionadded:: 1.6.0

        Metric used for monitoring the training result and early stopping.  It can be a
        string or list of strings as names of predefined metric in XGBoost (See
        :doc:`/parameter`), one of the metrics in :py:mod:`sklearn.metrics`, or any
        other user defined metric that looks like `sklearn.metrics`.

        If custom objective is also provided, then custom metric should implement the
        corresponding reverse link function.

        Unlike the `scoring` parameter commonly used in scikit-learn, when a callable
        object is provided, it's assumed to be a cost function and by default XGBoost
        will minimize the result during early stopping.

        For advanced usage on Early stopping like directly choosing to maximize instead
        of minimize, see :py:obj:`xgboost.callback.EarlyStopping`.

        See :doc:`/tutorials/custom_metric_obj` and :ref:`custom-obj-metric` for more
        information.

        .. code-block:: python

            from sklearn.datasets import load_diabetes
            from sklearn.metrics import mean_absolute_error
            X, y = load_diabetes(return_X_y=True)
            reg = xgb.XGBRegressor(
                tree_method="hist",
                eval_metric=mean_absolute_error,
            )
            reg.fit(X, y, eval_set=[(X, y)])

    early_stopping_rounds : {Optional[int]}

        .. versionadded:: 1.6.0

        - Activates early stopping. Validation metric needs to improve at least once in
          every **early_stopping_rounds** round(s) to continue training.  Requires at
          least one item in **eval_set** in :py:meth:`fit`.

        - If early stopping occurs, the model will have two additional attributes:
          :py:attr:`best_score` and :py:attr:`best_iteration`. These are used by the
          :py:meth:`predict` and :py:meth:`apply` methods to determine the optimal
          number of trees during inference. If users want to access the full model
          (including trees built after early stopping), they can specify the
          `iteration_range` in these inference methods. In addition, other utilities
          like model plotting can also use the entire model.

        - If you prefer to discard the trees after `best_iteration`, consider using the
          callback function :py:class:`xgboost.callback.EarlyStopping`.

        - If there's more than one item in **eval_set**, the last entry will be used for
          early stopping.  If there's more than one metric in **eval_metric**, the last
          metric will be used for early stopping.

    callbacks : {Optional[List[TrainingCallback]]}

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
                reg = xgboost.XGBRegressor(**params, callbacks=callbacks)
                reg.fit(X, y)

    kwargs : {Optional[Any]}

        Keyword arguments for XGBoost Booster object.  Full documentation of parameters
        can be found :doc:`here </parameter>`.
        Attempting to set a parameter via the constructor args and \\*\\*kwargs
        dict simultaneously will result in a TypeError.

        .. note:: \\*\\*kwargs unsupported by scikit-learn

            \\*\\*kwargs is unsupported by scikit-learn.  We do not guarantee
            that parameters passed via this argument will interact properly
            with scikit-learn.
"""

__custom_obj_note = """
        .. note::  Custom objective function

            A custom objective function can be provided for the ``objective``
            parameter. In this case, it should have the signature ``objective(y_true,
            y_pred) -> [grad, hess]`` or ``objective(y_true, y_pred, *, sample_weight)
            -> [grad, hess]``:

            y_true: array_like of shape [n_samples]
                The target values
            y_pred: array_like of shape [n_samples]
                The predicted values
            sample_weight :
                Optional sample weights.

            grad: array_like of shape [n_samples]
                The value of the gradient for each sample point.
            hess: array_like of shape [n_samples]
                The value of the second derivative for each sample point

            Note that, if the custom objective produces negative values for
            the Hessian, these will be clipped. If the objective is non-convex,
            one might also consider using the expected Hessian (Fisher
            information) instead.
"""

TDoc = TypeVar("TDoc", bound=Type)


def xgboost_model_doc(
    header: str,
    items: List[str],
    extra_parameters: Optional[str] = None,
    end_note: Optional[str] = None,
) -> Callable[[TDoc], TDoc]:
    """Obtain documentation for Scikit-Learn wrappers

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
       Extra notes put to the end."""

    def get_doc(item: str) -> str:
        """Return selected item"""
        __doc = {
            "estimators": __estimator_doc,
            "model": __model_doc,
            "objective": __custom_obj_note,
        }
        return __doc[item]

    def adddoc(cls: TDoc) -> TDoc:
        doc = [
            """
Parameters
----------
"""
        ]
        if extra_parameters:
            doc.append(extra_parameters)
        doc.extend([get_doc(i) for i in items])
        if end_note:
            doc.append(end_note)
        full_doc = [
            header + "\nSee :doc:`/python/sklearn_estimator` for more information.\n"
        ]
        full_doc.extend(doc)
        cls.__doc__ = "".join(full_doc)
        return cls

    return adddoc


def get_model_categories(
    X: ArrayLike,
    model: Optional[Union[Booster, str]],
    feature_types: Optional[FeatureTypes],
) -> Tuple[Optional[Union[Booster, str]], Optional[Union[FeatureTypes, Categories]]]:
    """Extract the optional reference categories from the booster. Used for training
    continuation. The result should be passed to the :py:func:`pick_ref_categories`.

    """
    # Skip if it's not a dataframe as there's no new encoding to be recoded.
    #
    # This function helps override the `feature_types` parameter. The `feature_types`
    # from user is not useful when input is a dataframe as the real feature type should
    # be encoded into the DF.
    if model is None or not is_dataframe(X):
        return model, feature_types

    if isinstance(model, str):
        model = Booster(model_file=model)

    categories = model.get_categories()
    if not categories.empty():
        # override the `feature_types`.
        return model, categories
    # Convert empty into None.
    return model, feature_types


def pick_ref_categories(
    X: Any,
    model_cats: Optional[Union[FeatureTypes, Categories]],
    Xy_cats: Optional[Categories],
) -> Optional[Union[FeatureTypes, Categories]]:
    """Use the reference categories from the model. If none, then use the reference
    categories from the training DMatrix.

    Parameters
    ----------
    X :
        Input feature matrix.

    model_cats :
        Optional categories stored in the previous model (training continuation). This
        should come from the :py:func:`get_model_categories`.

    Xy_cats :
        Optional categories from the training DMatrix. Used for re-coding the validation
        dataset.

    """
    categories: Optional[Categories] = None
    if not isinstance(model_cats, Categories) and is_dataframe(X):
        categories = Xy_cats
    if categories is not None and not categories.empty():
        model_cats = categories

    return model_cats


def _wrap_evaluation_matrices(
    *,
    missing: float,
    X: Any,
    y: Any,
    group: Optional[Any],
    qid: Optional[Any],
    sample_weight: Optional[Any],
    base_margin: Optional[Any],
    feature_weights: Optional[ArrayLike],
    eval_set: Optional[Sequence[Tuple[Any, Any]]],
    sample_weight_eval_set: Optional[Sequence[Any]],
    base_margin_eval_set: Optional[Sequence[Any]],
    eval_group: Optional[Sequence[Any]],
    eval_qid: Optional[Sequence[Any]],
    create_dmatrix: Callable,
    enable_categorical: bool,
    feature_types: Optional[Union[FeatureTypes, Categories]],
) -> Tuple[Any, List[Tuple[Any, str]]]:
    """Convert array_like evaluation matrices into DMatrix. Perform sanity checks on the
    way.

    """
    # Feature_types contains the optional reference categories from the booster object.
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
        feature_types=feature_types,
        ref=None,
    )

    n_validation = 0 if eval_set is None else len(eval_set)
    if hasattr(train_dmatrix, "get_categories"):
        Xy_cats = train_dmatrix.get_categories()
    else:
        Xy_cats = None

    def validate_or_none(meta: Optional[Sequence], name: str) -> Sequence:
        if meta is None:
            return [None] * n_validation
        if len(meta) != n_validation:
            raise ValueError(
                f"{name}'s length does not equal `eval_set`'s length, "
                + f"expecting {n_validation}, got {len(meta)}"
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
            # Skip the entry if it's the training DMatrix.
            if all(
                (
                    valid_X is X,
                    valid_y is y,
                    sample_weight_eval_set[i] is sample_weight,
                    base_margin_eval_set[i] is base_margin,
                    eval_group[i] is group,
                    eval_qid[i] is qid,
                )
            ):
                evals.append(train_dmatrix)
                continue

            feature_types = pick_ref_categories(valid_X, feature_types, Xy_cats)
            m = create_dmatrix(
                data=valid_X,
                label=valid_y,
                weight=sample_weight_eval_set[i],
                group=eval_group[i],
                qid=eval_qid[i],
                base_margin=base_margin_eval_set[i],
                missing=missing,
                enable_categorical=enable_categorical,
                feature_types=feature_types,
                ref=train_dmatrix,
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


DEFAULT_N_ESTIMATORS = 100


@xgboost_model_doc(
    """Implementation of the Scikit-Learn API for XGBoost.""",
    ["estimators", "model", "objective"],
)
class XGBModel(XGBModelBase):
    # pylint: disable=too-many-arguments, too-many-instance-attributes, missing-docstring
    @_deprecate_positional_args
    def __init__(
        self,
        *,
        max_depth: Optional[int] = None,
        max_leaves: Optional[int] = None,
        max_bin: Optional[int] = None,
        grow_policy: Optional[str] = None,
        learning_rate: Optional[float] = None,
        n_estimators: Optional[int] = None,
        verbosity: Optional[int] = None,
        objective: SklObjective = None,
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
        base_score: Optional[Union[float, List[float]]] = None,
        random_state: Optional[
            Union[np.random.RandomState, np.random.Generator, int]
        ] = None,
        missing: float = np.nan,
        num_parallel_tree: Optional[int] = None,
        monotone_constraints: Optional[Union[Dict[str, int], str]] = None,
        interaction_constraints: Optional[Union[str, Sequence[Sequence[str]]]] = None,
        importance_type: Optional[str] = None,
        device: Optional[str] = None,
        validate_parameters: Optional[bool] = None,
        enable_categorical: bool = False,
        feature_types: Optional[FeatureTypes] = None,
        feature_weights: Optional[ArrayLike] = None,
        max_cat_to_onehot: Optional[int] = None,
        max_cat_threshold: Optional[int] = None,
        multi_strategy: Optional[str] = None,
        eval_metric: Optional[Union[str, List[Union[str, Callable]], Callable]] = None,
        early_stopping_rounds: Optional[int] = None,
        callbacks: Optional[List[TrainingCallback]] = None,
        **kwargs: Any,
    ) -> None:
        if not SKLEARN_INSTALLED:
            raise ImportError(
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
        self.device = device
        self.validate_parameters = validate_parameters
        self.enable_categorical = enable_categorical
        self.feature_types = feature_types
        if isinstance(self.feature_types, Categories):
            raise TypeError(
                "If you are training with a prior model (training continuation), "
                "The scikit-learn interface can automatically reuse the categories from"
                " that model."
            )
        self.feature_weights = feature_weights
        self.max_cat_to_onehot = max_cat_to_onehot
        self.max_cat_threshold = max_cat_threshold
        self.multi_strategy = multi_strategy
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.callbacks = callbacks
        if kwargs:
            self.kwargs = kwargs

    def _more_tags(self) -> Dict[str, bool]:
        """Tags used for scikit-learn data validation."""
        tags = {"allow_nan": True, "no_validation": True, "sparse": True}
        if hasattr(self, "kwargs") and self.kwargs.get("updater") == "shotgun":
            tags["non_deterministic"] = True

        tags["categorical"] = self.enable_categorical
        tags["string"] = self.enable_categorical
        return tags

    @staticmethod
    def _update_sklearn_tags_from_dict(
        *,
        tags: _sklearn_Tags,
        tags_dict: Dict[str, bool],
    ) -> _sklearn_Tags:
        """Update ``sklearn.utils.Tags`` inherited from ``scikit-learn`` base classes.

        ``scikit-learn`` 1.6 introduced a dataclass-based interface for estimator tags.
        ref: https://github.com/scikit-learn/scikit-learn/pull/29677

        This method handles updating that instance based on the values in
        ``self._more_tags()``.

        """
        tags.non_deterministic = tags_dict.get("non_deterministic", False)
        tags.no_validation = tags_dict["no_validation"]
        tags.input_tags.allow_nan = tags_dict["allow_nan"]
        tags.input_tags.sparse = tags_dict["sparse"]
        tags.input_tags.categorical = tags_dict["categorical"]
        return tags

    def __sklearn_tags__(self) -> _sklearn_Tags:
        # XGBModelBase.__sklearn_tags__() cannot be called unconditionally,
        # because that method isn't defined for scikit-learn<1.6
        if not hasattr(XGBModelBase, "__sklearn_tags__"):
            err_msg = (
                "__sklearn_tags__() should not be called when using scikit-learn<1.6. "
                f"Detected version: {_sklearn_version}"
            )
            raise AttributeError(err_msg)

        # take whatever tags are provided by BaseEstimator, then modify
        # them with XGBoost-specific values
        return self._update_sklearn_tags_from_dict(
            tags=super().__sklearn_tags__(),  # pylint: disable=no-member
            tags_dict=self._more_tags(),
        )

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "_Booster")

    @property
    def _doc_link_module(self) -> str:
        return "xgboost"

    @property
    def _doc_link_template(self) -> str:
        ver = _py_version()
        (major, minor, _), post = _parse_version(ver)

        if post == "dev":
            rel = "latest"
        else:
            # RTD tracks the release branch. We don't have independent branches for
            # patch releases.
            rel = f"release_{major}.{minor}.0"

        module = self.__class__.__module__
        # All sklearn estimators are forwarded to the top level module in both source
        # code and sphinx api doc.
        if module == "xgboost.sklearn":
            module = module.split(".")[0]
        name = self.__class__.__name__

        base = "https://xgboost.readthedocs.io/en"
        return f"{base}/{rel}/python/python_api.html#{module}.{name}"

    def _wrapper_params(self) -> Set[str]:
        wrapper_specific = {
            "importance_type",
            "kwargs",
            "missing",
            "n_estimators",
            "enable_categorical",
            "early_stopping_rounds",
            "callbacks",
            "feature_types",
            "feature_weights",
        }
        return wrapper_specific

    def get_booster(self) -> Booster:
        """Get the underlying xgboost Booster of this model.

        This will raise an exception when fit was not called

        Returns
        -------
        booster : a xgboost booster of underlying model
        """
        if not self.__sklearn_is_fitted__():
            from sklearn.exceptions import NotFittedError

            raise NotFittedError("need to call fit or load_model beforehand")
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

        if self.__sklearn_is_fitted__():
            parameters = self.get_xgb_params()
            self.get_booster().set_param(parameters)

        return self

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        # pylint: disable=attribute-defined-outside-init
        """Get parameters."""
        # Based on: https://stackoverflow.com/questions/59248211
        # The basic flow in `get_params` is:
        # 0. Return parameters in subclass (self.__class__) first, by using inspect.
        # 1. Return parameters in all parent classes (especially `XGBModel`).
        # 2. Return whatever in `**kwargs`.
        # 3. Merge them.
        #
        # This needs to accommodate being called recursively in the following
        # inheritance graphs (and similar for classification and ranking):
        #
        #   XGBRFRegressor -> XGBRegressor -> XGBModel -> BaseEstimator
        #                     XGBRegressor -> XGBModel -> BaseEstimator
        #                                     XGBModel -> BaseEstimator
        #
        params = super().get_params(deep)
        cp = copy.copy(self)
        # If the immediate parent defines get_params(), use that.
        if callable(getattr(cp.__class__.__bases__[0], "get_params", None)):
            cp.__class__ = cp.__class__.__bases__[0]
        # Otherwise, skip it and assume the next class will have it.
        # This is here primarily for cases where the first class in MRO is a scikit-learn mixin.
        else:
            cp.__class__ = cp.__class__.__bases__[1]
        params.update(cp.__class__.get_params(cp, deep))
        # if kwargs is a dict, update params accordingly
        if hasattr(self, "kwargs") and isinstance(self.kwargs, dict):
            params.update(self.kwargs)
        if isinstance(params["random_state"], np.random.RandomState):
            params["random_state"] = params["random_state"].randint(
                np.iinfo(np.int32).max
            )
        elif isinstance(params["random_state"], np.random.Generator):
            params["random_state"] = int(
                params["random_state"].integers(np.iinfo(np.int32).max)
            )

        return params

    def get_xgb_params(self) -> Dict[str, Any]:
        """Get xgboost specific parameters."""
        params: Dict[str, Any] = self.get_params()

        # Parameters that should not go into native learner.
        wrapper_specific = self._wrapper_params()
        filtered = {}
        for k, v in params.items():
            if k not in wrapper_specific and not callable(v):
                filtered[k] = v

        return filtered

    def get_num_boosting_rounds(self) -> int:
        """Gets the number of xgboost boosting rounds."""
        return DEFAULT_N_ESTIMATORS if self.n_estimators is None else self.n_estimators

    def _get_type(self) -> str:
        if not hasattr(self, "_estimator_type"):
            raise TypeError(
                "`_estimator_type` undefined.  "
                "Please use appropriate mixin to define estimator type."
            )
        return self._estimator_type  # pylint: disable=no-member

    def save_model(self, fname: Union[str, os.PathLike]) -> None:
        meta: Dict[str, Any] = {}
        # For validation.
        meta["_estimator_type"] = self._get_type()
        meta_str = json.dumps(meta)
        self.get_booster().set_attr(scikit_learn=meta_str)
        self.get_booster().save_model(fname)
        self.get_booster().set_attr(scikit_learn=None)

    save_model.__doc__ = f"""{Booster.save_model.__doc__}"""

    def load_model(self, fname: ModelIn) -> None:
        # pylint: disable=attribute-defined-outside-init
        if not self.__sklearn_is_fitted__():
            self._Booster = Booster({"n_jobs": self.n_jobs})
        self.get_booster().load_model(fname)

        meta_str = self.get_booster().attr("scikit_learn")
        if meta_str is not None:
            meta = json.loads(meta_str)
            t = meta.get("_estimator_type", None)
            if t is not None and t != self._get_type():
                raise TypeError(
                    "Loading an estimator with different type. Expecting: "
                    f"{self._get_type()}, got: {t}"
                )

        self.get_booster().set_attr(scikit_learn=None)
        config = json.loads(self.get_booster().save_config())
        self._load_model_attributes(config)

    load_model.__doc__ = f"""{Booster.load_model.__doc__}"""

    def _load_model_attributes(self, config: dict) -> None:
        """Load model attributes without hyper-parameters."""
        from sklearn.base import is_classifier

        booster = self.get_booster()

        self.objective = config["learner"]["objective"]["name"]
        self.booster = config["learner"]["gradient_booster"]["name"]
        self.base_score = json.loads(
            config["learner"]["learner_model_param"]["base_score"]
        )
        self.feature_types = booster.feature_types
        self.enable_categorical = self.feature_types is not None and any(
            ft == CAT_T for ft in self.feature_types
        )

        if is_classifier(self):
            self.n_classes_ = int(config["learner"]["learner_model_param"]["num_class"])
            # binary classification is treated as regression in XGBoost.
            self.n_classes_ = 2 if self.n_classes_ < 2 else self.n_classes_

    # pylint: disable=too-many-branches
    def _configure_fit(
        self,
        booster: Optional[Union[Booster, "XGBModel", str]],
        params: Dict[str, Any],
        feature_weights: Optional[ArrayLike],
    ) -> Tuple[
        Optional[Union[Booster, str, "XGBModel"]],
        Optional[Metric],
        Dict[str, Any],
        Optional[ArrayLike],
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

        # - configure callable evaluation metric
        metric: Optional[Metric] = None

        def custom_metric(m: Callable) -> Metric:
            if self._get_type() == "ranker":
                wrapped = ltr_metric_decorator(m, self.n_jobs)
            else:
                wrapped = _metric_decorator(m)
            return wrapped

        def invalid_type(m: Any) -> None:
            msg = f"Invalid type for the `eval_metric`: {type(m)}"
            raise TypeError(msg)

        if self.eval_metric is not None:
            if callable(self.eval_metric):
                metric = custom_metric(self.eval_metric)
            elif isinstance(self.eval_metric, str):
                params.update({"eval_metric": self.eval_metric})
            else:
                # A sequence of metrics
                if not isinstance(self.eval_metric, collections.abc.Sequence):
                    invalid_type(self.eval_metric)
                # Could be a list of strings or callables
                builtin_metrics: List[str] = []
                for m in self.eval_metric:
                    if callable(m):
                        if metric is not None:
                            raise NotImplementedError(
                                "Using multiple custom metrics is not yet supported."
                            )
                        metric = custom_metric(m)
                    elif isinstance(m, str):
                        builtin_metrics.append(m)
                    else:
                        invalid_type(m)
                if builtin_metrics:
                    params.update({"eval_metric": builtin_metrics})

        if feature_weights is not None:
            _deprecated("feature_weights")
        if feature_weights is not None and self.feature_weights is not None:
            _duplicated("feature_weights")
        feature_weights = (
            self.feature_weights
            if self.feature_weights is not None
            else feature_weights
        )

        tree_method = params.get("tree_method", None)
        if self.enable_categorical and tree_method == "exact":
            raise ValueError(
                "Experimental support for categorical data is not implemented for"
                " current tree method yet."
            )
        return model, metric, params, feature_weights

    def _create_dmatrix(self, ref: Optional[DMatrix], **kwargs: Any) -> DMatrix:
        # Use `QuantileDMatrix` to save memory.
        if _can_use_qdm(self.tree_method, self.device) and self.booster != "gblinear":
            try:
                return QuantileDMatrix(
                    **kwargs, ref=ref, nthread=self.n_jobs, max_bin=self.max_bin
                )
            except TypeError:  # `QuantileDMatrix` supports lesser types than DMatrix
                pass
        return DMatrix(**kwargs, nthread=self.n_jobs)

    def _set_evaluation_result(self, evals_result: EvalsLog) -> None:
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
        verbose: Optional[Union[bool, int]] = True,
        xgb_model: Optional[Union[Booster, str, "XGBModel"]] = None,
        sample_weight_eval_set: Optional[Sequence[ArrayLike]] = None,
        base_margin_eval_set: Optional[Sequence[ArrayLike]] = None,
        feature_weights: Optional[ArrayLike] = None,
    ) -> "XGBModel":
        # pylint: disable=invalid-name,attribute-defined-outside-init
        """Fit gradient boosting model.

        Note that calling ``fit()`` multiple times will cause the model object to be
        re-fit from scratch. To resume training from a previous checkpoint, explicitly
        pass ``xgb_model`` argument.

        Parameters
        ----------
        X :
            Input feature matrix. See :ref:`py-data` for a list of supported types.

            When the ``tree_method`` is set to ``hist``, internally, the
            :py:class:`QuantileDMatrix` will be used instead of the :py:class:`DMatrix`
            for conserving memory. However, this has performance implications when the
            device of input data is not matched with algorithm. For instance, if the
            input is a numpy array on CPU but ``cuda`` is used for training, then the
            data is first processed on CPU then transferred to GPU.
        y :
            Labels
        sample_weight :
            instance weights
        base_margin :
            Global bias for each instance. See :doc:`/tutorials/intercept` for details.
        eval_set :
            A list of (X, y) tuple pairs to use as validation sets, for which
            metrics will be computed.
            Validation metrics will help us track the performance of the model.

        verbose :
            If `verbose` is True and an evaluation set is used, the evaluation metric
            measured on the validation set is printed to stdout at each boosting stage.
            If `verbose` is an integer, the evaluation metric is printed at each
            `verbose` boosting stage. The last boosting stage / the boosting stage found
            by using `early_stopping_rounds` is also printed.
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

            .. deprecated:: 3.0.0

            Use `feature_weights` in :py:meth:`__init__` or :py:meth:`set_params`
            instead.

        """
        with config_context(verbosity=self.verbosity):
            params = self.get_xgb_params()
            model, metric, params, feature_weights = self._configure_fit(
                xgb_model, params, feature_weights
            )
            model, feature_types = get_model_categories(X, model, self.feature_types)

            evals_result: EvalsLog = {}
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
                create_dmatrix=self._create_dmatrix,
                enable_categorical=self.enable_categorical,
                feature_types=feature_types,
            )

            if callable(self.objective):
                obj: Optional[Objective] = _objective_decorator(self.objective)
                params["objective"] = "reg:squarederror"
            else:
                obj = None

            self._Booster = train(
                params,
                train_dmatrix,
                self.get_num_boosting_rounds(),
                evals=evals,
                early_stopping_rounds=self.early_stopping_rounds,
                evals_result=evals_result,
                obj=obj,
                custom_metric=metric,
                verbose_eval=verbose,
                xgb_model=model,
                callbacks=self.callbacks,
            )

            self._set_evaluation_result(evals_result)
            return self

    def _can_use_inplace_predict(self) -> bool:
        return self.booster != "gblinear"

    def _get_iteration_range(
        self, iteration_range: Optional[IterationRange]
    ) -> IterationRange:
        if iteration_range is None or iteration_range[1] == 0:
            # Use best_iteration if defined.
            try:
                iteration_range = (0, self.best_iteration + 1)
            except AttributeError:
                iteration_range = (0, 0)
        if self.booster == "gblinear":
            iteration_range = (0, 0)
        return iteration_range

    @_deprecate_positional_args
    def predict(
        self,
        X: ArrayLike,
        *,
        output_margin: bool = False,
        validate_features: bool = True,
        base_margin: Optional[ArrayLike] = None,
        iteration_range: Optional[IterationRange] = None,
    ) -> ArrayLike:
        """Predict with `X`.  If the model is trained with early stopping, then
        :py:attr:`best_iteration` is used automatically. The estimator uses
        `inplace_predict` by default and falls back to using :py:class:`DMatrix` if
        devices between the data and the estimator don't match.

        .. note:: This function is only thread safe for `gbtree` and `dart`.

        Parameters
        ----------
        X :
            Data to predict with. See :ref:`py-data` for a list of supported types.
        output_margin :
            Whether to output the raw untransformed margin value.
        validate_features :
            When this is True, validate that the Booster's and data's feature_names are
            identical.  Otherwise, it is assumed that the feature_names are the same.
        base_margin :
            Global bias for each instance. See :doc:`/tutorials/intercept` for details.
        iteration_range :
            Specifies which layer of trees are used in prediction.  For example, if a
            random forest is trained with 100 rounds.  Specifying ``iteration_range=(10,
            20)``, then only the forests built during [10, 20) (half open set) rounds
            are used in this prediction.

            .. versionadded:: 1.4.0

        Returns
        -------
        prediction

        """
        with config_context(verbosity=self.verbosity):
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
                    if _is_cupy_alike(predts):
                        cp = import_cupy()

                        predts = cp.asnumpy(predts)  # ensure numpy array is used.
                    return predts
                except TypeError:
                    # coo, csc, dt
                    pass

            test = DMatrix(
                X,
                base_margin=base_margin,
                missing=self.missing,
                nthread=self.n_jobs,
                feature_types=self.feature_types,
                enable_categorical=self.enable_categorical,
            )
            return self.get_booster().predict(
                data=test,
                iteration_range=iteration_range,
                output_margin=output_margin,
                validate_features=validate_features,
            )

    def apply(
        self,
        X: ArrayLike,
        iteration_range: Optional[IterationRange] = None,
    ) -> np.ndarray:
        """Return the predicted leaf every tree for each sample. If the model is trained
        with early stopping, then :py:attr:`best_iteration` is used automatically.

        Parameters
        ----------
        X :
            Input features matrix. See :ref:`py-data` for a list of supported types.

        iteration_range :
            See :py:meth:`predict`.

        Returns
        -------
        X_leaves : array_like, shape=[n_samples, n_trees]
            For each datapoint x in X and for each tree, return the index of the
            leaf x ends up in. Leaves are numbered within
            ``[0; 2**(self.max_depth+1))``, possibly with gaps in the numbering.

        """
        with config_context(verbosity=self.verbosity):
            iteration_range = self._get_iteration_range(iteration_range)
            test_dmatrix = DMatrix(
                X,
                missing=self.missing,
                feature_types=self.feature_types,
                nthread=self.n_jobs,
                enable_categorical=self.enable_categorical,
            )
            return self.get_booster().predict(
                test_dmatrix, pred_leaf=True, iteration_range=iteration_range
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
        """Names of features seen during :py:meth:`fit`.  Defined only when `X` has
        feature names that are all strings.

        """
        feature_names = self.get_booster().feature_names
        if feature_names is None:
            raise AttributeError(
                "`feature_names_in_` is defined only when `X` has feature names that "
                "are all strings."
            )
        return np.array(feature_names)

    @property
    def best_score(self) -> float:
        """The best score obtained by early stopping."""
        return self.get_booster().best_score

    @property
    def best_iteration(self) -> int:
        """The best iteration obtained by early stopping.  This attribute is 0-based,
        for instance if the best iteration is the first round, then best_iteration is 0.

        """
        return self.get_booster().best_iteration

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances property, return depends on `importance_type`
        parameter. When model trained with multi-class/multi-label/multi-target dataset,
        the feature importance is "averaged" over all targets. The "average" is defined
        based on the importance type. For instance, if the importance type is
        "total_gain", then the score is sum of loss change for each split from all
        trees.

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
            feature_names: FeatureNames = [f"f{i}" for i in range(self.n_features_in_)]
        else:
            feature_names = b.feature_names
        # gblinear returns all features so the `get` in next line is only for gbtree.
        all_features = [score.get(f, 0.0) for f in feature_names]
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
        if self.get_xgb_params()["booster"] != "gblinear":
            raise AttributeError(
                f"Coefficients are not defined for Booster type {self.booster}"
            )
        b = self.get_booster()
        coef = np.array(json.loads(b.get_dump(dump_format="json")[0])["weight"])
        # Logic for multiclass classification
        n_classes = getattr(self, "n_classes_", None)
        if n_classes is not None:
            if n_classes > 2:
                assert len(coef.shape) == 1
                assert coef.shape[0] % n_classes == 0
                coef = coef.reshape((n_classes, -1))
        return coef

    @property
    def intercept_(self) -> np.ndarray:
        """Intercept (bias) property

        For tree-based model, the returned value is the `base_score`.

        Returns
        -------
        intercept_ : array of shape ``(1,)`` or ``[n_classes]``

        """
        booster_config = self.get_xgb_params()["booster"]
        b = self.get_booster()
        if booster_config != "gblinear":  # gbtree, dart
            config = json.loads(b.save_config())
            intercept = json.loads(
                config["learner"]["learner_model_param"]["base_score"]
            )
            return np.array(intercept, dtype=np.float32)

        return np.array(
            json.loads(b.get_dump(dump_format="json")[0])["bias"], dtype=np.float32
        )


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
    ["model", "objective"],
    extra_parameters="""
    n_estimators : Optional[int]
        Number of boosting rounds.
""",
)
class XGBClassifier(XGBClassifierBase, XGBModel):
    # pylint: disable=missing-docstring,invalid-name,too-many-instance-attributes
    @_deprecate_positional_args
    def __init__(
        self,
        *,
        objective: SklObjective = "binary:logistic",
        **kwargs: Any,
    ) -> None:
        super().__init__(objective=objective, **kwargs)

    def _more_tags(self) -> Dict[str, bool]:
        tags = super()._more_tags()
        tags["multilabel"] = True
        return tags

    def __sklearn_tags__(self) -> _sklearn_Tags:
        tags = super().__sklearn_tags__()
        tags_dict = self._more_tags()
        tags.classifier_tags.multi_label = tags_dict["multilabel"]
        return tags

    @_deprecate_positional_args
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        sample_weight: Optional[ArrayLike] = None,
        base_margin: Optional[ArrayLike] = None,
        eval_set: Optional[Sequence[Tuple[ArrayLike, ArrayLike]]] = None,
        verbose: Optional[Union[bool, int]] = True,
        xgb_model: Optional[Union[Booster, str, XGBModel]] = None,
        sample_weight_eval_set: Optional[Sequence[ArrayLike]] = None,
        base_margin_eval_set: Optional[Sequence[ArrayLike]] = None,
        feature_weights: Optional[ArrayLike] = None,
    ) -> "XGBClassifier":
        # pylint: disable = attribute-defined-outside-init,too-many-statements
        with config_context(verbosity=self.verbosity):
            # We keep the n_classes_ as a simple member instead of loading it from
            # booster in a Python property. This way we can have efficient and
            # thread-safe prediction.
            if _is_polars_lazyframe(y):
                y = y.collect()
            if _is_cudf_df(y) or _is_cudf_ser(y):
                cp = import_cupy()

                classes = cp.unique(y.values)
                self.n_classes_ = len(classes)
                expected_classes = cp.array(self.classes_)
            elif _is_cupy_alike(y):
                cp = import_cupy()

                classes = cp.unique(y)
                self.n_classes_ = len(classes)
                expected_classes = cp.array(self.classes_)
            else:
                classes = np.unique(np.asarray(y))
                self.n_classes_ = len(classes)
                expected_classes = self.classes_
            if (
                classes.shape != expected_classes.shape
                or not (classes == expected_classes).all()
            ):
                raise ValueError(
                    f"Invalid classes inferred from unique values of `y`.  "
                    f"Expected: {expected_classes}, got {classes}"
                )

            params = self.get_xgb_params()

            if callable(self.objective):
                obj: Optional[Objective] = _objective_decorator(self.objective)
                # Use default value. Is it really not used ?
                params["objective"] = "binary:logistic"
            else:
                obj = None

            if self.n_classes_ > 2:
                # Switch to using a multiclass objective in the underlying XGB instance
                if params.get("objective", None) != "multi:softmax":
                    params["objective"] = "multi:softprob"
                params["num_class"] = self.n_classes_

            model, metric, params, feature_weights = self._configure_fit(
                xgb_model, params, feature_weights
            )
            model, feature_types = get_model_categories(X, model, self.feature_types)

            evals_result: EvalsLog = {}
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
                create_dmatrix=self._create_dmatrix,
                enable_categorical=self.enable_categorical,
                feature_types=feature_types,
            )

            self._Booster = train(
                params,
                train_dmatrix,
                self.get_num_boosting_rounds(),
                evals=evals,
                early_stopping_rounds=self.early_stopping_rounds,
                evals_result=evals_result,
                obj=obj,
                custom_metric=metric,
                verbose_eval=verbose,
                xgb_model=model,
                callbacks=self.callbacks,
            )

            if not callable(self.objective):
                self.objective = params["objective"]

            self._set_evaluation_result(evals_result)
            return self

    assert XGBModel.fit.__doc__ is not None
    fit.__doc__ = XGBModel.fit.__doc__.replace(
        "Fit gradient boosting model", "Fit gradient boosting classifier", 1
    )

    @_deprecate_positional_args
    def predict(
        self,
        X: ArrayLike,
        *,
        output_margin: bool = False,
        validate_features: bool = True,
        base_margin: Optional[ArrayLike] = None,
        iteration_range: Optional[IterationRange] = None,
    ) -> ArrayLike:
        with config_context(verbosity=self.verbosity):
            class_probs = super().predict(
                X=X,
                output_margin=output_margin,
                validate_features=validate_features,
                base_margin=base_margin,
                iteration_range=iteration_range,
            )
            if output_margin:
                # If output_margin is active, simply return the scores
                return class_probs

            if len(class_probs.shape) > 1 and self.n_classes_ != 2:
                # multi-class, turns softprob into softmax
                column_indexes: np.ndarray = np.argmax(class_probs, axis=1)
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

            return column_indexes

    def predict_proba(
        self,
        X: ArrayLike,
        validate_features: bool = True,
        base_margin: Optional[ArrayLike] = None,
        iteration_range: Optional[IterationRange] = None,
    ) -> np.ndarray:
        """Predict the probability of each `X` example being of a given class. If the
        model is trained with early stopping, then :py:attr:`best_iteration` is used
        automatically. The estimator uses `inplace_predict` by default and falls back to
        using :py:class:`DMatrix` if devices between the data and the estimator don't
        match.

        .. note:: This function is only thread safe for `gbtree` and `dart`.

        Parameters
        ----------
        X :
            Feature matrix. See :ref:`py-data` for a list of supported types.
        validate_features :
            When this is True, validate that the Booster's and data's feature_names are
            identical.  Otherwise, it is assumed that the feature_names are the same.
        base_margin :
            Global bias for each instance. See :doc:`/tutorials/intercept` for details.
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
        # softmax:         Use softmax from scipy
        # binary:logistic: Expand the prob vector into 2-class matrix after predict.
        # binary:logitraw: Unsupported by predict_proba()
        if self.objective == "multi:softmax":
            raw_predt = super().predict(
                X=X,
                validate_features=validate_features,
                base_margin=base_margin,
                iteration_range=iteration_range,
                output_margin=True,
            )
            class_prob = softmax(raw_predt, axis=1)
            return class_prob
        class_probs = super().predict(
            X=X,
            validate_features=validate_features,
            base_margin=base_margin,
            iteration_range=iteration_range,
        )
        return _cls_predict_proba(self.n_classes_, class_probs, np.vstack)

    @property
    def classes_(self) -> np.ndarray:
        return np.arange(self.n_classes_)


@xgboost_model_doc(
    "scikit-learn API for XGBoost random forest classification.",
    ["model", "objective"],
    extra_parameters="""
    n_estimators : Optional[int]
        Number of trees in random forest to fit.
""",
)
class XGBRFClassifier(XGBClassifier):
    # pylint: disable=missing-docstring
    @_deprecate_positional_args
    def __init__(
        self,
        *,
        learning_rate: float = 1.0,
        subsample: float = 0.8,
        colsample_bynode: float = 0.8,
        reg_lambda: float = 1e-5,
        **kwargs: Any,
    ):
        super().__init__(
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bynode=colsample_bynode,
            reg_lambda=reg_lambda,
            **kwargs,
        )
        _check_rf_callback(self.early_stopping_rounds, self.callbacks)

    def get_xgb_params(self) -> Dict[str, Any]:
        params = super().get_xgb_params()
        params["num_parallel_tree"] = super().get_num_boosting_rounds()
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
        verbose: Optional[Union[bool, int]] = True,
        xgb_model: Optional[Union[Booster, str, XGBModel]] = None,
        sample_weight_eval_set: Optional[Sequence[ArrayLike]] = None,
        base_margin_eval_set: Optional[Sequence[ArrayLike]] = None,
        feature_weights: Optional[ArrayLike] = None,
    ) -> "XGBRFClassifier":
        args = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        _check_rf_callback(self.early_stopping_rounds, self.callbacks)
        super().fit(**args)
        return self


@xgboost_model_doc(
    "Implementation of the scikit-learn API for XGBoost regression.",
    ["estimators", "model", "objective"],
)
class XGBRegressor(XGBRegressorBase, XGBModel):
    # pylint: disable=missing-docstring
    @_deprecate_positional_args
    def __init__(
        self, *, objective: SklObjective = "reg:squarederror", **kwargs: Any
    ) -> None:
        super().__init__(objective=objective, **kwargs)

    def _more_tags(self) -> Dict[str, bool]:
        tags = super()._more_tags()
        tags["multioutput"] = True
        tags["multioutput_only"] = False
        return tags

    def __sklearn_tags__(self) -> _sklearn_Tags:
        tags = super().__sklearn_tags__()
        tags_dict = self._more_tags()
        tags.target_tags.multi_output = tags_dict["multioutput"]
        tags.target_tags.single_output = not tags_dict["multioutput_only"]
        return tags


@xgboost_model_doc(
    "scikit-learn API for XGBoost random forest regression.",
    ["model", "objective"],
    extra_parameters="""
    n_estimators : Optional[int]
        Number of trees in random forest to fit.
""",
)
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
        **kwargs: Any,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bynode=colsample_bynode,
            reg_lambda=reg_lambda,
            **kwargs,
        )
        _check_rf_callback(self.early_stopping_rounds, self.callbacks)

    def get_xgb_params(self) -> Dict[str, Any]:
        params = super().get_xgb_params()
        params["num_parallel_tree"] = super().get_num_boosting_rounds()
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
        verbose: Optional[Union[bool, int]] = True,
        xgb_model: Optional[Union[Booster, str, XGBModel]] = None,
        sample_weight_eval_set: Optional[Sequence[ArrayLike]] = None,
        base_margin_eval_set: Optional[Sequence[ArrayLike]] = None,
        feature_weights: Optional[ArrayLike] = None,
    ) -> "XGBRFRegressor":
        args = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        _check_rf_callback(self.early_stopping_rounds, self.callbacks)
        super().fit(**args)
        return self


def _get_qid(
    X: ArrayLike, qid: Optional[ArrayLike]
) -> Tuple[ArrayLike, Optional[ArrayLike]]:
    """Get the special qid column from X if exists."""
    if (_is_pandas_df(X) or _is_cudf_df(X)) and hasattr(X, "qid"):
        if qid is not None:
            raise ValueError(
                "Found both the special column `qid` in `X` and the `qid` from the"
                "`fit` method. Please remove one of them."
            )
        q_x = X.qid
        X = X.drop("qid", axis=1)
        return X, q_x
    return X, qid


@xgboost_model_doc(
    """Implementation of the Scikit-Learn API for XGBoost Ranking.

See :doc:`Learning to Rank </tutorials/learning_to_rank>` for an introducion.

    """,
    ["estimators", "model"],
    end_note="""
        .. note::

            A custom objective function is currently not supported by XGBRanker.

        .. note::

            Query group information is only required for ranking training but not
            prediction. Multiple groups can be predicted on a single call to
            :py:meth:`predict`.

        When fitting the model with the `group` parameter, your data need to be sorted
        by the query group first. `group` is an array that contains the size of each
        query group.

        Similarly, when fitting the model with the `qid` parameter, the data should be
        sorted according to query index and `qid` is an array that contains the query
        index for each training sample.

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

        then :py:meth:`fit` method can be called with either `group` array as ``[3, 4]``
        or with `qid` as ``[1, 1, 1, 2, 2, 2, 2]``, that is the qid column.  Also, the
        `qid` can be a special column of input `X` instead of a separated parameter, see
        :py:meth:`fit` for more info.""",
)
class XGBRanker(XGBRankerMixIn, XGBModel):
    # pylint: disable=missing-docstring,too-many-arguments,invalid-name
    @_deprecate_positional_args
    def __init__(self, *, objective: str = "rank:ndcg", **kwargs: Any):
        super().__init__(objective=objective, **kwargs)
        if callable(self.objective):
            raise ValueError("custom objective function not supported by XGBRanker")
        if "rank:" not in objective:
            raise ValueError("please use XGBRanker for ranking task")

    def _create_ltr_dmatrix(
        self, ref: Optional[DMatrix], data: ArrayLike, qid: ArrayLike, **kwargs: Any
    ) -> DMatrix:
        data, qid = _get_qid(data, qid)

        if kwargs.get("group", None) is None and qid is None:
            raise ValueError("Either `group` or `qid` is required for ranking task")

        return super()._create_dmatrix(ref=ref, data=data, qid=qid, **kwargs)

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
        verbose: Optional[Union[bool, int]] = False,
        xgb_model: Optional[Union[Booster, str, XGBModel]] = None,
        sample_weight_eval_set: Optional[Sequence[ArrayLike]] = None,
        base_margin_eval_set: Optional[Sequence[ArrayLike]] = None,
        feature_weights: Optional[ArrayLike] = None,
    ) -> "XGBRanker":
        # pylint: disable = attribute-defined-outside-init,arguments-differ
        """Fit gradient boosting ranker

        Note that calling ``fit()`` multiple times will cause the model object to be
        re-fit from scratch. To resume training from a previous checkpoint, explicitly
        pass ``xgb_model`` argument.

        Parameters
        ----------
        X :
            Feature matrix. See :ref:`py-data` for a list of supported types.

            When this is a :py:class:`pandas.DataFrame` or a :py:class:`cudf.DataFrame`,
            it may contain a special column called ``qid`` for specifying the query
            index. Using a special column is the same as using the `qid` parameter,
            except for being compatible with sklearn utility functions like
            :py:func:`sklearn.model_selection.cross_validation`. The same convention
            applies to the :py:meth:`XGBRanker.score` and :py:meth:`XGBRanker.predict`.

            +-----+----------------+----------------+
            | qid | feat_0         | feat_1         |
            +-----+----------------+----------------+
            | 0   | :math:`x_{00}` | :math:`x_{01}` |
            +-----+----------------+----------------+
            | 1   | :math:`x_{10}` | :math:`x_{11}` |
            +-----+----------------+----------------+
            | 1   | :math:`x_{20}` | :math:`x_{21}` |
            +-----+----------------+----------------+

            When the ``tree_method`` is set to ``hist``, internally, the
            :py:class:`QuantileDMatrix` will be used instead of the :py:class:`DMatrix`
            for conserving memory. However, this has performance implications when the
            device of input data is not matched with algorithm. For instance, if the
            input is a numpy array on CPU but ``cuda`` is used for training, then the
            data is first processed on CPU then transferred to GPU.
        y :
            Labels
        group :
            Size of each query group of training data. Should have as many elements as
            the query groups in the training data.  If this is set to None, then user
            must provide qid.
        qid :
            Query ID for each training sample.  Should have the size of n_samples.  If
            this is set to None, then user must provide group or a special column in X.
        sample_weight :
            Query group weights

            .. note:: Weights are per-group for ranking tasks

                In ranking task, one weight is assigned to each query group/id (not each
                data point). This is because we only care about the relative ordering of
                data points within each group, so it doesn't make sense to assign
                weights to individual data points.

        base_margin :
            Global bias for each instance. See :doc:`/tutorials/intercept` for details.
        eval_set :
            A list of (X, y) tuple pairs to use as validation sets, for which
            metrics will be computed.
            Validation metrics will help us track the performance of the model.
        eval_group :
            A list in which ``eval_group[i]`` is the list containing the sizes of all
            query groups in the ``i``-th pair in **eval_set**.
        eval_qid :
            A list in which ``eval_qid[i]`` is the array containing query ID of ``i``-th
            pair in **eval_set**. The special column convention in `X` applies to
            validation datasets as well.

        verbose :
            If `verbose` is True and an evaluation set is used, the evaluation metric
            measured on the validation set is printed to stdout at each boosting stage.
            If `verbose` is an integer, the evaluation metric is printed at each
            `verbose` boosting stage. The last boosting stage / the boosting stage found
            by using `early_stopping_rounds` is also printed.
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

        """
        with config_context(verbosity=self.verbosity):
            params = self.get_xgb_params()

            model, metric, params, feature_weights = self._configure_fit(
                xgb_model, params, feature_weights
            )
            model, feature_types = get_model_categories(X, model, self.feature_types)

            evals_result: EvalsLog = {}
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
                create_dmatrix=self._create_ltr_dmatrix,
                enable_categorical=self.enable_categorical,
                feature_types=feature_types,
            )

            self._Booster = train(
                params,
                train_dmatrix,
                num_boost_round=self.get_num_boosting_rounds(),
                early_stopping_rounds=self.early_stopping_rounds,
                evals=evals,
                evals_result=evals_result,
                custom_metric=metric,
                verbose_eval=verbose,
                xgb_model=model,
                callbacks=self.callbacks,
            )

            self.objective = params["objective"]

            self._set_evaluation_result(evals_result)
            return self

    @_deprecate_positional_args
    def predict(
        self,
        X: ArrayLike,
        *,
        output_margin: bool = False,
        validate_features: bool = True,
        base_margin: Optional[ArrayLike] = None,
        iteration_range: Optional[IterationRange] = None,
    ) -> ArrayLike:
        X, _ = _get_qid(X, None)
        return super().predict(
            X,
            output_margin=output_margin,
            validate_features=validate_features,
            base_margin=base_margin,
            iteration_range=iteration_range,
        )

    def apply(
        self,
        X: ArrayLike,
        iteration_range: Optional[IterationRange] = None,
    ) -> ArrayLike:
        X, _ = _get_qid(X, None)
        return super().apply(X, iteration_range)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Evaluate score for data using the last evaluation metric. If the model is
        trained with early stopping, then :py:attr:`best_iteration` is used
        automatically.

        Parameters
        ----------
        X : Union[pd.DataFrame, cudf.DataFrame]
          Feature matrix. A DataFrame with a special `qid` column.

        y :
          Labels

        Returns
        -------
        score :
          The result of the first evaluation metric for the ranker.

        """
        X, qid = _get_qid(X, None)
        # fixme(jiamingy): base margin and group weight is not yet supported. We might
        # need to make extra special fields in the dataframe.
        Xyq = DMatrix(
            X,
            y,
            qid=qid,
            missing=self.missing,
            enable_categorical=self.enable_categorical,
            nthread=self.n_jobs,
            feature_types=self.feature_types,
        )
        if callable(self.eval_metric):
            metric = ltr_metric_decorator(self.eval_metric, self.n_jobs)
            result_str = self.get_booster().eval_set([(Xyq, "eval")], feval=metric)
        else:
            result_str = self.get_booster().eval(Xyq)

        metric_score = _parse_eval_str(result_str)
        return metric_score[-1][1]
