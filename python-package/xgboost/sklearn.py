# coding: utf-8
# pylint: disable=too-many-arguments, too-many-locals, invalid-name, fixme, E0012, R0912, C0302
"""Scikit-Learn Wrapper interface for XGBoost."""
import copy
import warnings
import json
from typing import Union, Optional, List, Dict, Callable, Tuple, Any, TypeVar
import numpy as np
from .core import Booster, DMatrix, XGBoostError
from .core import _deprecate_positional_args, _convert_ntree_limit
from .core import Metric
from .training import train
from .data import _is_cudf_df, _is_cudf_ser, _is_cupy_array

# Do not use class names on scikit-learn directly.  Re-define the classes on
# .compat to guarantee the behavior without scikit-learn
from .compat import (SKLEARN_INSTALLED, XGBModelBase,
                     XGBClassifierBase, XGBRegressorBase, XGBoostLabelEncoder)


class XGBRankerMixIn:           # pylint: disable=too-few-public-methods
    """MixIn for ranking, defines the _estimator_type usually defined in scikit-learn base
    classes."""
    _estimator_type = "ranker"


def _objective_decorator(func):
    """Decorate an objective function

    Converts an objective function using the typical sklearn metrics
    signature so that it is usable with ``xgboost.training.train``

    Parameters
    ----------
    func: callable
        Expects a callable with signature ``func(y_true, y_pred)``:

        y_true: array_like of shape [n_samples]
            The target values
        y_pred: array_like of shape [n_samples]
            The predicted values

    Returns
    -------
    new_func: callable
        The new objective function as expected by ``xgboost.training.train``.
        The signature is ``new_func(preds, dmatrix)``:

        preds: array_like, shape [n_samples]
            The predicted values
        dmatrix: ``DMatrix``
            The training set from which the labels will be extracted using
            ``dmatrix.get_label()``
    """
    def inner(preds, dmatrix):
        """internal function"""
        labels = dmatrix.get_label()
        return func(labels, preds)
    return inner


__estimator_doc = '''
    n_estimators : int
        Number of gradient boosted trees.  Equivalent to number of boosting
        rounds.
'''

__model_doc = '''
    max_depth : int
        Maximum tree depth for base learners.
    learning_rate : float
        Boosting learning rate (xgb's "eta")
    verbosity : int
        The degree of verbosity. Valid values are 0 (silent) - 3 (debug).
    objective : string or callable
        Specify the learning task and the corresponding learning objective or
        a custom objective function to be used (see note below).
    booster: string
        Specify which booster to use: gbtree, gblinear or dart.
    tree_method: string
        Specify which tree method to use.  Default to auto.  If this parameter
        is set to default, XGBoost will choose the most conservative option
        available.  It's recommended to study this option from parameters
        document.
    n_jobs : int
        Number of parallel threads used to run xgboost.  When used with other Scikit-Learn
        algorithms like grid search, you may choose which algorithm to parallelize and
        balance the threads.  Creating thread contention will significantly slow down both
        algorithms.
    gamma : float
        Minimum loss reduction required to make a further partition on a leaf
        node of the tree.
    min_child_weight : float
        Minimum sum of instance weight(hessian) needed in a child.
    max_delta_step : float
        Maximum delta step we allow each tree's weight estimation to be.
    subsample : float
        Subsample ratio of the training instance.
    colsample_bytree : float
        Subsample ratio of columns when constructing each tree.
    colsample_bylevel : float
        Subsample ratio of columns for each level.
    colsample_bynode : float
        Subsample ratio of columns for each split.
    reg_alpha : float (xgb's alpha)
        L1 regularization term on weights
    reg_lambda : float (xgb's lambda)
        L2 regularization term on weights
    scale_pos_weight : float
        Balancing of positive and negative weights.
    base_score:
        The initial prediction score of all instances, global bias.
    random_state : int
        Random number seed.

        .. note::

           Using gblinear booster with shotgun updater is nondeterministic as
           it uses Hogwild algorithm.

    missing : float, default np.nan
        Value in the data which needs to be present as a missing value.
    num_parallel_tree: int
        Used for boosting random forest.
    monotone_constraints : str
        Constraint of variable monotonicity.  See tutorial for more
        information.
    interaction_constraints : str
        Constraints for interaction representing permitted interactions.  The
        constraints must be specified in the form of a nest list, e.g. [[0, 1],
        [2, 3, 4]], where each inner list is a group of indices of features
        that are allowed to interact with each other.  See tutorial for more
        information
    importance_type: string, default "gain"
        The feature importance type for the feature_importances\\_ property:
        either "gain", "weight", "cover", "total_gain" or "total_cover".
    gpu_id :
        Device ordinal.
    validate_parameters :
        Give warnings for unknown parameter.

    \\*\\*kwargs : dict, optional
        Keyword arguments for XGBoost Booster object.  Full documentation of
        parameters can be found here:
        https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst.
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


def xgboost_model_doc(header, items, extra_parameters=None, end_note=None):
    '''Obtain documentation for Scikit-Learn wrappers

    Parameters
    ----------
    header: str
       An introducion to the class.
    items : list
       A list of commom doc items.  Available items are:
         - estimators: the meaning of n_estimators
         - model: All the other parameters
         - objective: note for customized objective
    extra_parameters: str
       Document for class specific parameters, placed at the head.
    end_note: str
       Extra notes put to the end.
'''
    def get_doc(item):
        '''Return selected item'''
        __doc = {'estimators': __estimator_doc,
                 'model': __model_doc,
                 'objective': __custom_obj_note}
        return __doc[item]

    def adddoc(cls):
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
    eval_set: Optional[List[Tuple[Any, Any]]],
    sample_weight_eval_set: Optional[List[Any]],
    base_margin_eval_set: Optional[List[Any]],
    eval_group: Optional[List[Any]],
    eval_qid: Optional[List[Any]],
    create_dmatrix: Callable,
    label_transform: Callable = lambda x: x,
) -> Tuple[Any, Optional[List[Tuple[Any, str]]]]:
    """Convert array_like evaluation matrices into DMatrix.  Perform validation on the way.

    """
    train_dmatrix = create_dmatrix(
        data=X,
        label=label_transform(y),
        group=group,
        qid=qid,
        weight=sample_weight,
        base_margin=base_margin,
        feature_weights=feature_weights,
        missing=missing,
    )

    def validate_or_none(meta: Optional[List], name: str) -> List:
        if meta is None:
            return [None] * len(eval_set)
        if len(meta) != len(eval_set):
            raise ValueError(
                f"{name}'s length does not eqaul to `eval_set`, " +
                f"expecting {len(eval_set)}, got {len(meta)}"
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
                    label=label_transform(valid_y),
                    weight=sample_weight_eval_set[i],
                    group=eval_group[i],
                    qid=eval_qid[i],
                    base_margin=base_margin_eval_set[i],
                    missing=missing,
                )
                evals.append(m)
        nevals = len(evals)
        eval_names = ["validation_{}".format(i) for i in range(nevals)]
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
        max_depth=None,
        learning_rate=None,
        n_estimators=100,
        verbosity=None,
        objective=None,
        booster=None,
        tree_method=None,
        n_jobs=None,
        gamma=None,
        min_child_weight=None,
        max_delta_step=None,
        subsample=None,
        colsample_bytree=None,
        colsample_bylevel=None,
        colsample_bynode=None,
        reg_alpha=None,
        reg_lambda=None,
        scale_pos_weight=None,
        base_score=None,
        random_state=None,
        missing=np.nan,
        num_parallel_tree=None,
        monotone_constraints=None,
        interaction_constraints=None,
        importance_type="gain",
        gpu_id=None,
        validate_parameters=None,
        **kwargs
    ):
        if not SKLEARN_INSTALLED:
            raise XGBoostError(
                "sklearn needs to be installed in order to use this module"
            )
        self.n_estimators = n_estimators
        self.objective = objective

        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.verbosity = verbosity
        self.booster = booster
        self.tree_method = tree_method
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.base_score = base_score
        self.missing = missing
        self.num_parallel_tree = num_parallel_tree
        self.kwargs = kwargs
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.monotone_constraints = monotone_constraints
        self.interaction_constraints = interaction_constraints
        self.importance_type = importance_type
        self.gpu_id = gpu_id
        self.validate_parameters = validate_parameters

    def _more_tags(self):
        '''Tags used for scikit-learn data validation.'''
        return {'allow_nan': True, 'no_validation': True}

    def get_booster(self):
        """Get the underlying xgboost Booster of this model.

        This will raise an exception when fit was not called

        Returns
        -------
        booster : a xgboost booster of underlying model
        """
        if not hasattr(self, '_Booster'):
            from sklearn.exceptions import NotFittedError
            raise NotFittedError('need to call fit or load_model beforehand')
        return self._Booster

    def set_params(self, **params):
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
                self.kwargs[key] = value

        if hasattr(self, '_Booster'):
            parameters = self.get_xgb_params()
            self.get_booster().set_param(parameters)

        return self

    def get_params(self, deep=True):
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
        if isinstance(self.kwargs, dict):
            params.update(self.kwargs)
        if isinstance(params['random_state'], np.random.RandomState):
            params['random_state'] = params['random_state'].randint(
                np.iinfo(np.int32).max)

        def parse_parameter(value):
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
                if k in params.keys() and params[k] is None:
                    params[k] = parse_parameter(v)
        except ValueError:
            pass
        return params

    def get_xgb_params(self):
        """Get xgboost specific parameters."""
        params = self.get_params()
        # Parameters that should not go into native learner.
        wrapper_specific = {
            'importance_type', 'kwargs', 'missing', 'n_estimators', 'use_label_encoder'}
        filtered = dict()
        for k, v in params.items():
            if k not in wrapper_specific and not callable(v):
                filtered[k] = v
        return filtered

    def get_num_boosting_rounds(self):
        """Gets the number of xgboost boosting rounds."""
        return self.n_estimators

    def _get_type(self) -> str:
        if not hasattr(self, '_estimator_type'):
            raise TypeError(
                "`_estimator_type` undefined.  "
                "Please use appropriate mixin to define estimator type."
            )
        return self._estimator_type  # pylint: disable=no-member

    def save_model(self, fname: str):
        """Save the model to a file.

        The model is saved in an XGBoost internal format which is universal
        among the various XGBoost interfaces. Auxiliary attributes of the
        Python Booster object (such as feature names) will not be saved.

          .. note::

            See:

            https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html

        Parameters
        ----------
        fname : string
            Output file name

        """
        meta = dict()
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

    def load_model(self, fname):
        # pylint: disable=attribute-defined-outside-init
        """Load the model from a file.

        The model is loaded from an XGBoost internal format which is universal
        among the various XGBoost interfaces. Auxiliary attributes of the
        Python Booster object (such as feature names) will not be loaded.

        Parameters
        ----------
        fname : string
            Input file name.

        """
        if not hasattr(self, '_Booster'):
            self._Booster = Booster({'n_jobs': self.n_jobs})
        self._Booster.load_model(fname)
        meta = self._Booster.attr('scikit_learn')
        if meta is None:
            # FIXME(jiaming): This doesn't have to be a problem as most of the needed
            # information like num_class and objective is in Learner class.
            warnings.warn(
                'Loading a native XGBoost model with Scikit-Learn interface.')
            return
        meta = json.loads(meta)
        states = dict()
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
            if k == 'use_label_encoder':
                self.use_label_encoder = bool(v)
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

    def _configure_fit(
        self,
        booster: Optional[Union[Booster, "XGBModel"]],
        eval_metric: Optional[Union[Callable, str, List[str]]],
        params: Dict[str, Any],
    ) -> Tuple[Booster, Optional[Metric], Dict[str, Any]]:
        # pylint: disable=protected-access, no-self-use
        model = booster
        if hasattr(model, '_Booster'):
            model = model._Booster  # Handle the case when xgb_model is a sklearn model object
        feval = eval_metric if callable(eval_metric) else None
        if eval_metric is not None:
            if callable(eval_metric):
                eval_metric = None
            else:
                params.update({"eval_metric": eval_metric})
        return model, feval, params

    def _set_evaluation_result(self, evals_result: Optional[dict]) -> None:
        if evals_result:
            for val in evals_result.items():
                evals_result_key = list(val[1].keys())[0]
                evals_result[val[0]][evals_result_key] = val[1][evals_result_key]
            self.evals_result_ = evals_result

    @_deprecate_positional_args
    def fit(
        self,
        X,
        y,
        *,
        sample_weight=None,
        base_margin=None,
        eval_set=None,
        eval_metric=None,
        early_stopping_rounds=None,
        verbose=True,
        xgb_model: Optional[Union[Booster, str, "XGBModel"]] = None,
        sample_weight_eval_set=None,
        base_margin_eval_set=None,
        feature_weights=None,
        callbacks=None
    ):
        # pylint: disable=invalid-name,attribute-defined-outside-init
        """Fit gradient boosting model.

        Note that calling ``fit()`` multiple times will cause the model object to be
        re-fit from scratch. To resume training from a previous checkpoint, explicitly
        pass ``xgb_model`` argument.

        Parameters
        ----------
        X : array_like
            Feature matrix
        y : array_like
            Labels
        sample_weight : array_like
            instance weights
        base_margin : array_like
            global bias for each instance.
        eval_set : list, optional
            A list of (X, y) tuple pairs to use as validation sets, for which
            metrics will be computed.
            Validation metrics will help us track the performance of the model.
        eval_metric : str, list of str, or callable, optional
            If a str, should be a built-in evaluation metric to use. See
            doc/parameter.rst.
            If a list of str, should be the list of multiple built-in evaluation metrics
            to use.
            If callable, a custom evaluation metric. The call signature is
            ``func(y_predicted, y_true)`` where ``y_true`` will be a DMatrix object such
            that you may need to call the ``get_label`` method. It must return a str,
            value pair where the str is a name for the evaluation and value is the value
            of the evaluation function. The callable custom objective is always minimized.
        early_stopping_rounds : int
            Activates early stopping. Validation metric needs to improve at least once in
            every **early_stopping_rounds** round(s) to continue training.
            Requires at least one item in **eval_set**.
            The method returns the model from the last iteration (not the best one).
            If there's more than one item in **eval_set**, the last entry will be used
            for early stopping.
            If there's more than one metric in **eval_metric**, the last metric will be
            used for early stopping.
            If early stopping occurs, the model will have three additional fields:
            ``clf.best_score``, ``clf.best_iteration`` and ``clf.best_ntree_limit``.
        verbose : bool
            If `verbose` and an evaluation set is used, writes the evaluation metric
            measured on the validation set to stderr.
        xgb_model :
            file name of stored XGBoost model or 'Booster' instance XGBoost model to be
            loaded before training (allows training continuation).
        sample_weight_eval_set : list, optional
            A list of the form [L_1, L_2, ..., L_n], where each L_i is an array like
            object storing instance weights for the i-th validation set.
        base_margin_eval_set : list, optional
            A list of the form [M_1, M_2, ..., M_n], where each M_i is an array like
            object storing base margin for the i-th validation set.
        feature_weights: array_like
            Weight for each feature, defines the probability of each feature being
            selected when colsample is being used.  All values must be greater than 0,
            otherwise a `ValueError` is thrown.  Only available for `hist`, `gpu_hist` and
            `exact` tree methods.
        callbacks : list of callback functions
            List of callback functions that are applied at end of each iteration.
            It is possible to use predefined callbacks by using :ref:`callback_api`.
            Example:

            .. code-block:: python

                callbacks = [xgb.callback.EarlyStopping(rounds=early_stopping_rounds,
                                                        save_best=True)]

        """
        evals_result = {}

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
        )
        params = self.get_xgb_params()

        if callable(self.objective):
            obj = _objective_decorator(self.objective)
            params["objective"] = "reg:squarederror"
        else:
            obj = None

        model, feval, params = self._configure_fit(xgb_model, eval_metric, params)
        self._Booster = train(
            params,
            train_dmatrix,
            self.get_num_boosting_rounds(),
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            evals_result=evals_result,
            obj=obj,
            feval=feval,
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
        params = self.get_params()
        if params.get("predictor", None) is None and self.booster != "gblinear":
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
        X,
        output_margin=False,
        ntree_limit=None,
        validate_features=True,
        base_margin=None,
        iteration_range=None,
    ):
        """
        Predict with `X`.

        .. note:: This function is only thread safe for `gbtree` and `dart`.

        Parameters
        ----------
        X : array_like
            Data to predict with
        output_margin : bool
            Whether to output the raw untransformed margin value.
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

            .. versionadded:: 1.4.0
        Returns
        -------
        prediction : numpy array
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
            X, base_margin=base_margin, missing=self.missing, nthread=self.n_jobs
        )
        return self.get_booster().predict(
            data=test,
            iteration_range=iteration_range,
            output_margin=output_margin,
            validate_features=validate_features,
        )

    def apply(
        self, X, ntree_limit: int = 0, iteration_range: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """Return the predicted leaf every tree for each sample.

        Parameters
        ----------
        X : array_like, shape=[n_samples, n_features]
            Input features matrix.

        ntree_limit : int
            Limit number of trees in the prediction; defaults to 0 (use all trees).

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

    def evals_result(self):
        """Return the evaluation results.

        If **eval_set** is passed to the `fit` function, you can call
        ``evals_result()`` to get evaluation results for all passed **eval_sets**.
        When **eval_metric** is also passed to the `fit` function, the
        **evals_result** will contain the **eval_metrics** passed to the `fit` function.

        Returns
        -------
        evals_result : dictionary

        Example
        -------

        .. code-block:: python

            param_dist = {'objective':'binary:logistic', 'n_estimators':2}

            clf = xgb.XGBModel(**param_dist)

            clf.fit(X_train, y_train,
                    eval_set=[(X_train, y_train), (X_test, y_test)],
                    eval_metric='logloss',
                    verbose=True)

            evals_result = clf.evals_result()

        The variable **evals_result** will contain:

        .. code-block:: python

            {'validation_0': {'logloss': ['0.604835', '0.531479']},
            'validation_1': {'logloss': ['0.41965', '0.17686']}}
        """
        if self.evals_result_:
            evals_result = self.evals_result_
        else:
            raise XGBoostError('No results.')

        return evals_result

    @property
    def n_features_in_(self) -> int:
        booster = self.get_booster()
        return booster.num_features()

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
        return float(self._early_stopping_attr('best_score'))

    @property
    def best_iteration(self) -> int:
        return int(self._early_stopping_attr('best_iteration'))

    @property
    def best_ntree_limit(self) -> int:
        return int(self._early_stopping_attr('best_ntree_limit'))

    @property
    def feature_importances_(self):
        """
        Feature importances property

        .. note:: Feature importance is defined only for tree boosters

            Feature importance is only defined when the decision tree model is chosen as base
            learner (`booster=gbtree`). It is not defined for other base learner types, such
            as linear learners (`booster=gblinear`).

        Returns
        -------
        feature_importances_ : array of shape ``[n_features]``

        """
        if self.get_params()['booster'] not in {'gbtree', 'dart'}:
            raise AttributeError(
                'Feature importance is not defined for Booster type {}'
                .format(self.booster))
        b: Booster = self.get_booster()
        score = b.get_score(importance_type=self.importance_type)
        if b.feature_names is None:
            feature_names = ["f{0}".format(i) for i in range(self.n_features_in_)]
        else:
            feature_names = b.feature_names
        all_features = [score.get(f, 0.) for f in feature_names]
        all_features = np.array(all_features, dtype=np.float32)
        total = all_features.sum()
        if total == 0:
            return all_features
        return all_features / total

    @property
    def coef_(self):
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
                'Coefficients are not defined for Booster type {}'
                .format(self.booster))
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
    def intercept_(self):
        """
        Intercept (bias) property

        .. note:: Intercept is defined only for linear learners

            Intercept (bias) is only defined when the linear model is chosen as base
            learner (`booster=gblinear`). It is not defined for other base learner types, such
            as tree learners (`booster=gbtree`).

        Returns
        -------
        intercept_ : array of shape ``(1,)`` or ``[n_classes]``
        """
        if self.get_params()['booster'] != 'gblinear':
            raise AttributeError(
                'Intercept (bias) is not defined for Booster type {}'
                .format(self.booster))
        b = self.get_booster()
        return np.array(json.loads(b.get_dump(dump_format='json')[0])['bias'])


PredtT = TypeVar("PredtT")


def _cls_predict_proba(n_classes: int, prediction: PredtT, vstack: Callable) -> PredtT:
    assert len(prediction.shape) <= 2
    if len(prediction.shape) == 2 and prediction.shape[1] == n_classes:
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
    use_label_encoder : bool
        (Deprecated) Use the label encoder from scikit-learn to encode the labels. For new
        code, we recommend that you set this parameter to False.
''')
class XGBClassifier(XGBModel, XGBClassifierBase):
    # pylint: disable=missing-docstring,invalid-name,too-many-instance-attributes
    @_deprecate_positional_args
    def __init__(self, *, objective="binary:logistic", use_label_encoder=True, **kwargs):
        self.use_label_encoder = use_label_encoder
        super().__init__(objective=objective, **kwargs)

    @_deprecate_positional_args
    def fit(
        self,
        X,
        y,
        *,
        sample_weight=None,
        base_margin=None,
        eval_set=None,
        eval_metric=None,
        early_stopping_rounds=None,
        verbose=True,
        xgb_model=None,
        sample_weight_eval_set=None,
        base_margin_eval_set=None,
        feature_weights=None,
        callbacks=None
    ):
        # pylint: disable = attribute-defined-outside-init,too-many-statements
        can_use_label_encoder = True
        label_encoding_check_error = (
            "The label must consist of integer "
            "labels of form 0, 1, 2, ..., [num_class - 1]."
        )
        label_encoder_deprecation_msg = (
            "The use of label encoder in XGBClassifier is deprecated and will be "
            "removed in a future release. To remove this warning, do the "
            "following: 1) Pass option use_label_encoder=False when constructing "
            "XGBClassifier object; and 2) Encode your labels (y) as integers "
            "starting with 0, i.e. 0, 1, 2, ..., [num_class - 1]."
        )

        evals_result = {}
        if _is_cudf_df(y) or _is_cudf_ser(y):
            import cupy as cp  # pylint: disable=E0401

            self.classes_ = cp.unique(y.values)
            self.n_classes_ = len(self.classes_)
            can_use_label_encoder = False
            expected_classes = cp.arange(self.n_classes_)
            if (
                self.classes_.shape != expected_classes.shape
                or not (self.classes_ == expected_classes).all()
            ):
                raise ValueError(label_encoding_check_error)
        elif _is_cupy_array(y):
            import cupy as cp  # pylint: disable=E0401

            self.classes_ = cp.unique(y)
            self.n_classes_ = len(self.classes_)
            can_use_label_encoder = False
            expected_classes = cp.arange(self.n_classes_)
            if (
                self.classes_.shape != expected_classes.shape
                or not (self.classes_ == expected_classes).all()
            ):
                raise ValueError(label_encoding_check_error)
        else:
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
            if not self.use_label_encoder and (
                not np.array_equal(self.classes_, np.arange(self.n_classes_))
            ):
                raise ValueError(label_encoding_check_error)

        params = self.get_xgb_params()

        if callable(self.objective):
            obj = _objective_decorator(self.objective)
            # Use default value. Is it really not used ?
            params["objective"] = "binary:logistic"
        else:
            obj = None

        if self.n_classes_ > 2:
            # Switch to using a multiclass objective in the underlying
            # XGB instance
            params["objective"] = "multi:softprob"
            params["num_class"] = self.n_classes_

        if self.use_label_encoder:
            if not can_use_label_encoder:
                raise ValueError('The option use_label_encoder=True is incompatible with inputs ' +
                                 'of type cuDF or cuPy. Please set use_label_encoder=False when ' +
                                 'constructing XGBClassifier object. NOTE: ' +
                                 label_encoder_deprecation_msg)
            warnings.warn(label_encoder_deprecation_msg, UserWarning)
            self._le = XGBoostLabelEncoder().fit(y)
            label_transform = self._le.transform
        else:
            label_transform = lambda x: x

        model, feval, params = self._configure_fit(xgb_model, eval_metric, params)
        if len(X.shape) != 2:
            # Simply raise an error here since there might be many
            # different ways of reshaping
            raise ValueError("Please reshape the input data X into 2-dimensional matrix.")

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
            label_transform=label_transform,
        )

        self._Booster = train(
            params,
            train_dmatrix,
            self.get_num_boosting_rounds(),
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            evals_result=evals_result,
            obj=obj,
            feval=feval,
            verbose_eval=verbose,
            xgb_model=model,
            callbacks=callbacks,
        )

        if not callable(self.objective):
            self.objective = params["objective"]

        self._set_evaluation_result(evals_result)
        return self

    fit.__doc__ = XGBModel.fit.__doc__.replace(
        'Fit gradient boosting model',
        'Fit gradient boosting classifier', 1)

    def predict(
        self,
        X,
        output_margin=False,
        ntree_limit=None,
        validate_features=True,
        base_margin=None,
        iteration_range: Optional[Tuple[int, int]] = None,
    ):
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

        if len(class_probs.shape) > 1:
            # turns softprob into softmax
            column_indexes = np.argmax(class_probs, axis=1)
        else:
            # turns soft logit into class label
            column_indexes = np.repeat(0, class_probs.shape[0])
            column_indexes[class_probs > 0.5] = 1

        if hasattr(self, '_le'):
            return self._le.inverse_transform(column_indexes)
        return column_indexes

    def predict_proba(
        self,
        X,
        ntree_limit=None,
        validate_features=False,
        base_margin=None,
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
        prediction : numpy array
            a numpy array of shape array-like of shape (n_samples, n_classes) with the
            probability of each data example being of a given class.
        """
        # custom obj:      Do nothing as we don't know what to do.
        # softprob:        Do nothing, output is proba.
        # softmax:         Use output margin to remove the argmax in PredTransform.
        # binary:logistic: Expand the prob vector into 2-class matrix after predict.
        # binary:logitraw: Unsupported by predict_proba()
        class_probs = super().predict(
            X=X,
            output_margin=self.objective == "multi:softmax",
            ntree_limit=ntree_limit,
            validate_features=validate_features,
            base_margin=base_margin,
            iteration_range=iteration_range
        )
        # If model is loaded from a raw booster there's no `n_classes_`
        return _cls_predict_proba(
            getattr(self, "n_classes_", None), class_probs, np.vstack
        )

    def evals_result(self):
        """Return the evaluation results.

        If **eval_set** is passed to the `fit` function, you can call
        ``evals_result()`` to get evaluation results for all passed **eval_sets**.
        When **eval_metric** is also passed to the `fit` function, the
        **evals_result** will contain the **eval_metrics** passed to the `fit` function.

        Returns
        -------
        evals_result : dictionary

        Example
        -------

        .. code-block:: python

            param_dist = {'objective':'binary:logistic', 'n_estimators':2}

            clf = xgb.XGBClassifier(**param_dist)

            clf.fit(X_train, y_train,
                    eval_set=[(X_train, y_train), (X_test, y_test)],
                    eval_metric='logloss',
                    verbose=True)

            evals_result = clf.evals_result()

        The variable **evals_result** will contain

        .. code-block:: python

            {'validation_0': {'logloss': ['0.604835', '0.531479']},
            'validation_1': {'logloss': ['0.41965', '0.17686']}}
        """
        if self.evals_result_:
            evals_result = self.evals_result_
        else:
            raise XGBoostError('No results.')

        return evals_result


@xgboost_model_doc(
    "scikit-learn API for XGBoost random forest classification.",
    ['model', 'objective'],
    extra_parameters='''
    n_estimators : int
        Number of trees in random forest to fit.
    use_label_encoder : bool
        (Deprecated) Use the label encoder from scikit-learn to encode the labels. For new
        code, we recommend that you set this parameter to False.
''')
class XGBRFClassifier(XGBClassifier):
    # pylint: disable=missing-docstring
    @_deprecate_positional_args
    def __init__(self, *,
                 learning_rate=1,
                 subsample=0.8,
                 colsample_bynode=0.8,
                 reg_lambda=1e-5,
                 use_label_encoder=True,
                 **kwargs):
        super().__init__(learning_rate=learning_rate,
                         subsample=subsample,
                         colsample_bynode=colsample_bynode,
                         reg_lambda=reg_lambda,
                         use_label_encoder=use_label_encoder,
                         **kwargs)

    def get_xgb_params(self):
        params = super().get_xgb_params()
        params['num_parallel_tree'] = self.n_estimators
        return params

    def get_num_boosting_rounds(self):
        return 1


@xgboost_model_doc(
    "Implementation of the scikit-learn API for XGBoost regression.",
    ['estimators', 'model', 'objective'])
class XGBRegressor(XGBModel, XGBRegressorBase):
    # pylint: disable=missing-docstring
    @_deprecate_positional_args
    def __init__(self, *, objective="reg:squarederror", **kwargs):
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
    def __init__(self, *, learning_rate=1, subsample=0.8, colsample_bynode=0.8,
                 reg_lambda=1e-5, **kwargs):
        super().__init__(learning_rate=learning_rate, subsample=subsample,
                         colsample_bynode=colsample_bynode,
                         reg_lambda=reg_lambda, **kwargs)

    def get_xgb_params(self):
        params = super().get_xgb_params()
        params['num_parallel_tree'] = self.n_estimators
        return params

    def get_num_boosting_rounds(self):
        return 1


@xgboost_model_doc(
    'Implementation of the Scikit-Learn API for XGBoost Ranking.',
    ['estimators', 'model'],
    end_note='''
        Note
        ----
        A custom objective function is currently not supported by XGBRanker.
        Likewise, a custom metric function is not supported either.

        Note
        ----
        Query group information is required for ranking tasks by either using the `group`
        parameter or `qid` parameter in `fit` method.

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
    def __init__(self, *, objective="rank:pairwise", **kwargs):
        super().__init__(objective=objective, **kwargs)
        if callable(self.objective):
            raise ValueError("custom objective function not supported by XGBRanker")
        if "rank:" not in self.objective:
            raise ValueError("please use XGBRanker for ranking task")

    @_deprecate_positional_args
    def fit(
        self,
        X,
        y,
        *,
        group=None,
        qid=None,
        sample_weight=None,
        base_margin=None,
        eval_set=None,
        eval_group=None,
        eval_qid=None,
        eval_metric=None,
        early_stopping_rounds=None,
        verbose=False,
        xgb_model: Optional[Union[Booster, str, XGBModel]] = None,
        sample_weight_eval_set=None,
        base_margin_eval_set=None,
        feature_weights=None,
        callbacks=None
    ) -> "XGBRanker":
        # pylint: disable = attribute-defined-outside-init,arguments-differ
        """Fit gradient boosting ranker

        Note that calling ``fit()`` multiple times will cause the model object to be
        re-fit from scratch. To resume training from a previous checkpoint, explicitly
        pass ``xgb_model`` argument.

        Parameters
        ----------
        X : array_like
            Feature matrix
        y : array_like
            Labels
        group : array_like
            Size of each query group of training data. Should have as many elements as the
            query groups in the training data.  If this is set to None, then user must
            provide qid.
        qid : array_like
            Query ID for each training sample.  Should have the size of n_samples.  If
            this is set to None, then user must provide group.
        sample_weight : array_like
            Query group weights

            .. note:: Weights are per-group for ranking tasks

                In ranking task, one weight is assigned to each query group/id (not each
                data point). This is because we only care about the relative ordering of
                data points within each group, so it doesn't make sense to assign weights
                to individual data points.
        base_margin : array_like
            Global bias for each instance.
        eval_set : list, optional
            A list of (X, y) tuple pairs to use as validation sets, for which
            metrics will be computed.
            Validation metrics will help us track the performance of the model.
        eval_group : list of arrays, optional
            A list in which ``eval_group[i]`` is the list containing the sizes of all
            query groups in the ``i``-th pair in **eval_set**.
        eval_qid : list of array_like, optional
            A list in which ``eval_qid[i]`` is the array containing query ID of ``i``-th
            pair in **eval_set**.
        eval_metric : str, list of str, optional
            If a str, should be a built-in evaluation metric to use. See
            doc/parameter.rst.
            If a list of str, should be the list of multiple built-in evaluation metrics
            to use. The custom evaluation metric is not yet supported for the ranker.
        early_stopping_rounds : int
            Activates early stopping. Validation metric needs to improve at least once in
            every **early_stopping_rounds** round(s) to continue training.  Requires at
            least one item in **eval_set**.
            The method returns the model from the last iteration (not the best one).  If
            there's more than one item in **eval_set**, the last entry will be used for
            early stopping.
            If there's more than one metric in **eval_metric**, the last metric will be
            used for early stopping.
            If early stopping occurs, the model will have three additional fields:
            ``clf.best_score``, ``clf.best_iteration`` and ``clf.best_ntree_limit``.
        verbose : bool
            If `verbose` and an evaluation set is used, writes the evaluation metric
            measured on the validation set to stderr.
        xgb_model :
            file name of stored XGBoost model or 'Booster' instance XGBoost model to be
            loaded before training (allows training continuation).
        sample_weight_eval_set : list, optional
            A list of the form [L_1, L_2, ..., L_n], where each L_i is a list of
            group weights on the i-th validation set.

            .. note:: Weights are per-group for ranking tasks

                In ranking task, one weight is assigned to each query group (not each
                data point). This is because we only care about the relative ordering of
                data points within each group, so it doesn't make sense to assign
                weights to individual data points.
        base_margin_eval_set : list, optional
            A list of the form [M_1, M_2, ..., M_n], where each M_i is an array like
            object storing base margin for the i-th validation set.
        feature_weights: array_like
            Weight for each feature, defines the probability of each feature being
            selected when colsample is being used.  All values must be greater than 0,
            otherwise a `ValueError` is thrown.  Only available for `hist`, `gpu_hist` and
            `exact` tree methods.
        callbacks : list of callback functions
            List of callback functions that are applied at end of each
            iteration.  It is possible to use predefined callbacks by using
            :ref:`callback_api`.  Example:

            .. code-block:: python

                callbacks = [xgb.callback.EarlyStopping(rounds=early_stopping_rounds,
                                                        save_best=True)]

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
        )

        evals_result = {}
        params = self.get_xgb_params()

        model, feval, params = self._configure_fit(xgb_model, eval_metric, params)
        if callable(feval):
            raise ValueError(
                'Custom evaluation metric is not yet supported for XGBRanker.'
            )

        self._Booster = train(
            params, train_dmatrix,
            self.n_estimators,
            early_stopping_rounds=early_stopping_rounds,
            evals=evals,
            evals_result=evals_result, feval=feval,
            verbose_eval=verbose, xgb_model=model,
            callbacks=callbacks
        )

        self.objective = params["objective"]

        self._set_evaluation_result(evals_result)
        return self
