# coding: utf-8
# pylint: disable=too-many-arguments, too-many-locals, invalid-name, fixme, E0012, R0912, C0302
"""Scikit-Learn Wrapper interface for XGBoost."""
import copy
import warnings
import json
import numpy as np
from .core import Booster, DMatrix, XGBoostError
from .training import train

# Do not use class names on scikit-learn directly.  Re-define the classes on
# .compat to guarantee the behavior without scikit-learn
from .compat import (SKLEARN_INSTALLED, XGBModelBase,
                     XGBClassifierBase, XGBRegressorBase, XGBoostLabelEncoder)


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
        Number of parallel threads used to run xgboost.
    gamma : float
        Minimum loss reduction required to make a further partition on a leaf
        node of the tree.
    min_child_weight : int
        Minimum sum of instance weight(hessian) needed in a child.
    max_delta_step : int
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

    missing : float, optional
        Value in the data which needs to be present as a missing value. If
        None, defaults to np.nan.
    num_parallel_tree: int
        Used for boosting random forest.
    monotone_constraints : str
        Constraint of variable monotonicity.  See tutorial for more
        information.c
    interaction_constraints : str
        Constraints for interaction representing permitted interactions.  The
        constraints must be specified in the form of a nest list, e.g. [[0, 1],
        [2, 3, 4]], where each inner list is a group of indices of features
        that are allowed to interact with each other.  See tutorial for more
        information
    importance_type: string, default "gain"
        The feature importance type for the feature_importances\\_ property:
        either "gain", "weight", "cover", "total_gain" or "total_cover".

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


@xgboost_model_doc("""Implementation of the Scikit-Learn API for XGBoost.""",
                   ['estimators', 'model', 'objective'])
class XGBModel(XGBModelBase):
    # pylint: disable=too-many-arguments, too-many-instance-attributes, missing-docstring
    def __init__(self, max_depth=None, learning_rate=None, n_estimators=100,
                 verbosity=None, objective=None, booster=None,
                 tree_method=None, n_jobs=None, gamma=None,
                 min_child_weight=None, max_delta_step=None, subsample=None,
                 colsample_bytree=None, colsample_bylevel=None,
                 colsample_bynode=None, reg_alpha=None, reg_lambda=None,
                 scale_pos_weight=None, base_score=None, random_state=None,
                 missing=None, num_parallel_tree=None,
                 monotone_constraints=None, interaction_constraints=None,
                 importance_type="gain", gpu_id=None,
                 validate_parameters=False, **kwargs):
        if not SKLEARN_INSTALLED:
            raise XGBoostError(
                'sklearn needs to be installed in order to use this module')
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
        self.missing = missing if missing is not None else np.nan
        self.num_parallel_tree = num_parallel_tree
        self.kwargs = kwargs
        self._Booster = None
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.monotone_constraints = monotone_constraints
        self.interaction_constraints = interaction_constraints
        self.importance_type = importance_type
        self.gpu_id = gpu_id
        # Parameter validation is not working with Scikit-Learn interface, as
        # it passes all paraemters into XGBoost core, whether they are used or
        # not.
        self.validate_parameters = validate_parameters

    def __setstate__(self, state):
        # backward compatibility code
        # load booster from raw if it is raw
        # the booster now support pickle
        bst = state["_Booster"]
        if bst is not None and not isinstance(bst, Booster):
            state["_Booster"] = Booster(model_file=bst)
        self.__dict__.update(state)

    def get_booster(self):
        """Get the underlying xgboost Booster of this model.

        This will raise an exception when fit was not called

        Returns
        -------
        booster : a xgboost booster of underlying model
        """
        if self._Booster is None:
            raise XGBoostError('need to call fit or load_model beforehand')
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

        # this concatenates kwargs into paraemters, enabling `get_params` for
        # obtaining parameters from keyword paraemters.
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value

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
        if hasattr(self, '__copy__'):
            warnings.warn('Calling __copy__ on Scikit-Learn wrapper, ' +
                          'which may disable data cache and result in ' +
                          'lower performance.')
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
            for t in (int, float):
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
        except XGBoostError:
            pass
        return params

    def get_xgb_params(self):
        """Get xgboost type parameters."""
        xgb_params = self.get_params()
        return xgb_params

    def get_num_boosting_rounds(self):
        """Gets the number of xgboost boosting rounds."""
        return self.n_estimators

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
                warnings.warn(str(k) + ' is not saved in Scikit-Learn meta.')
        meta['type'] = type(self).__name__
        meta = json.dumps(meta)
        self.get_booster().set_attr(scikit_learn=meta)
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
        if self._Booster is None:
            self._Booster = Booster({'n_jobs': self.n_jobs})
        self._Booster.load_model(fname)
        meta = self._Booster.attr('scikit_learn')
        if meta is None:
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
            if k == 'classes_':
                self.classes_ = np.array(v)
                continue
            if k == 'type' and type(self).__name__ != v:
                msg = 'Current model type: {}, '.format(type(self).__name__) + \
                      'type of model in file: {}'.format(v)
                raise TypeError(msg)
            if k == 'type':
                continue
            states[k] = v
        self.__dict__.update(states)
        # Delete the attribute after load
        self.get_booster().set_attr(scikit_learn=None)

    def fit(self, X, y, sample_weight=None, base_margin=None,
            eval_set=None, eval_metric=None, early_stopping_rounds=None,
            verbose=True, xgb_model=None, sample_weight_eval_set=None,
            callbacks=None):
        # pylint: disable=invalid-name,attribute-defined-outside-init
        """Fit gradient boosting model

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
        sample_weight_eval_set : list, optional
            A list of the form [L_1, L_2, ..., L_n], where each L_i is a list of
            instance weights on the i-th validation set.
        eval_metric : str, list of str, or callable, optional
            If a str, should be a built-in evaluation metric to use. See
            doc/parameter.rst.
            If a list of str, should be the list of multiple built-in evaluation metrics
            to use.
            If callable, a custom evaluation metric. The call
            signature is ``func(y_predicted, y_true)`` where ``y_true`` will be a
            DMatrix object such that you may need to call the ``get_label``
            method. It must return a str, value pair where the str is a name
            for the evaluation and value is the value of the evaluation
            function. The callable custom objective is always minimized.
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
            If `verbose` and an evaluation set is used, writes the evaluation
            metric measured on the validation set to stderr.
        xgb_model : str
            file name of stored XGBoost model or 'Booster' instance XGBoost model to be
            loaded before training (allows training continuation).
        callbacks : list of callback functions
            List of callback functions that are applied at end of each iteration.
            It is possible to use predefined callbacks by using :ref:`callback_api`.
            Example:

            .. code-block:: python

                [xgb.callback.reset_learning_rate(custom_rates)]
        """
        train_dmatrix = DMatrix(data=X, label=y, weight=sample_weight,
                                base_margin=base_margin,
                                missing=self.missing,
                                nthread=self.n_jobs)

        evals_result = {}

        if eval_set is not None:
            if not isinstance(eval_set[0], (list, tuple)):
                raise TypeError('Unexpected input type for `eval_set`')
            if sample_weight_eval_set is None:
                sample_weight_eval_set = [None] * len(eval_set)
            evals = list(
                DMatrix(eval_set[i][0], label=eval_set[i][1], missing=self.missing,
                        weight=sample_weight_eval_set[i], nthread=self.n_jobs)
                for i in range(len(eval_set)))
            evals = list(zip(evals, ["validation_{}".format(i) for i in
                                     range(len(evals))]))
        else:
            evals = ()

        params = self.get_xgb_params()

        if callable(self.objective):
            obj = _objective_decorator(self.objective)
            params["objective"] = "reg:squarederror"
        else:
            obj = None

        feval = eval_metric if callable(eval_metric) else None
        if eval_metric is not None:
            if callable(eval_metric):
                eval_metric = None
            else:
                params.update({'eval_metric': eval_metric})

        self._Booster = train(params, train_dmatrix,
                              self.get_num_boosting_rounds(), evals=evals,
                              early_stopping_rounds=early_stopping_rounds,
                              evals_result=evals_result, obj=obj, feval=feval,
                              verbose_eval=verbose, xgb_model=xgb_model,
                              callbacks=callbacks)

        if evals_result:
            for val in evals_result.items():
                evals_result_key = list(val[1].keys())[0]
                evals_result[val[0]][evals_result_key] = val[1][evals_result_key]
            self.evals_result_ = evals_result

        if early_stopping_rounds is not None:
            self.best_score = self._Booster.best_score
            self.best_iteration = self._Booster.best_iteration
            self.best_ntree_limit = self._Booster.best_ntree_limit
        return self

    def predict(self, data, output_margin=False, ntree_limit=None,
                validate_features=True, base_margin=None):
        """
        Predict with `data`.

        .. note:: This function is not thread safe.

          For each booster object, predict can only be called from one thread.
          If you want to run prediction using multiple thread, call ``xgb.copy()`` to make copies
          of model object and then call ``predict()``.

          .. code-block:: python

            preds = bst.predict(dtest, ntree_limit=num_round)

        Parameters
        ----------
        data : numpy.array/scipy.sparse
            Data to predict with
        output_margin : bool
            Whether to output the raw untransformed margin value.
        ntree_limit : int
            Limit number of trees in the prediction; defaults to best_ntree_limit if defined
            (i.e. it has been trained with early stopping), otherwise 0 (use all trees).
        validate_features : bool
            When this is True, validate that the Booster's and data's feature_names are identical.
            Otherwise, it is assumed that the feature_names are the same.
        Returns
        -------
        prediction : numpy array
        """
        # pylint: disable=missing-docstring,invalid-name
        test_dmatrix = DMatrix(data, base_margin=base_margin,
                               missing=self.missing, nthread=self.n_jobs)
        # get ntree_limit to use - if none specified, default to
        # best_ntree_limit if defined, otherwise 0.
        if ntree_limit is None:
            ntree_limit = getattr(self, "best_ntree_limit", 0)
        return self.get_booster().predict(test_dmatrix,
                                          output_margin=output_margin,
                                          ntree_limit=ntree_limit,
                                          validate_features=validate_features)

    def apply(self, X, ntree_limit=0):
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
        test_dmatrix = DMatrix(X, missing=self.missing, nthread=self.n_jobs)
        return self.get_booster().predict(test_dmatrix,
                                          pred_leaf=True,
                                          ntree_limit=ntree_limit)

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
        if getattr(self, 'booster', None) is not None and self.booster not in {
                'gbtree', 'dart'}:
            raise AttributeError('Feature importance is not defined for Booster type {}'
                                 .format(self.booster))
        b = self.get_booster()
        score = b.get_score(importance_type=self.importance_type)
        all_features = [score.get(f, 0.) for f in b.feature_names]
        all_features = np.array(all_features, dtype=np.float32)
        return all_features / all_features.sum()

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
        if getattr(self, 'booster', None) is not None and self.booster != 'gblinear':
            raise AttributeError('Coefficients are not defined for Booster type {}'
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
        if getattr(self, 'booster', None) is not None and self.booster != 'gblinear':
            raise AttributeError('Intercept (bias) is not defined for Booster type {}'
                                 .format(self.booster))
        b = self.get_booster()
        return np.array(json.loads(b.get_dump(dump_format='json')[0])['bias'])


@xgboost_model_doc(
    "Implementation of the scikit-learn API for XGBoost classification.",
    ['model', 'objective'])
class XGBClassifier(XGBModel, XGBClassifierBase):
    # pylint: disable=missing-docstring,invalid-name,too-many-instance-attributes
    def __init__(self, objective="binary:logistic", **kwargs):
        super().__init__(objective=objective, **kwargs)

    def fit(self, X, y, sample_weight=None, base_margin=None,
            eval_set=None, eval_metric=None,
            early_stopping_rounds=None, verbose=True, xgb_model=None,
            sample_weight_eval_set=None, callbacks=None):
        # pylint: disable = attribute-defined-outside-init,arguments-differ

        evals_result = {}
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        xgb_options = self.get_xgb_params()

        if callable(self.objective):
            obj = _objective_decorator(self.objective)
            # Use default value. Is it really not used ?
            xgb_options["objective"] = "binary:logistic"
        else:
            obj = None

        if self.n_classes_ > 2:
            # Switch to using a multiclass objective in the underlying
            # XGB instance
            xgb_options['objective'] = 'multi:softprob'
            xgb_options['num_class'] = self.n_classes_

        feval = eval_metric if callable(eval_metric) else None
        if eval_metric is not None:
            if callable(eval_metric):
                eval_metric = None
            else:
                xgb_options.update({"eval_metric": eval_metric})

        self._le = XGBoostLabelEncoder().fit(y)
        training_labels = self._le.transform(y)

        if eval_set is not None:
            if sample_weight_eval_set is None:
                sample_weight_eval_set = [None] * len(eval_set)
            evals = list(
                DMatrix(eval_set[i][0],
                        label=self._le.transform(eval_set[i][1]),
                        missing=self.missing, weight=sample_weight_eval_set[i],
                        nthread=self.n_jobs)
                for i in range(len(eval_set))
            )
            nevals = len(evals)
            eval_names = ["validation_{}".format(i) for i in range(nevals)]
            evals = list(zip(evals, eval_names))
        else:
            evals = ()

        if len(X.shape) != 2:
            # Simply raise an error here since there might be many
            # different ways of reshaping
            raise ValueError(
                'Please reshape the input data X into 2-dimensional matrix.')
        self._features_count = X.shape[1]
        train_dmatrix = DMatrix(X, label=training_labels, weight=sample_weight,
                                base_margin=base_margin,
                                missing=self.missing, nthread=self.n_jobs)

        self._Booster = train(xgb_options, train_dmatrix,
                              self.get_num_boosting_rounds(),
                              evals=evals,
                              early_stopping_rounds=early_stopping_rounds,
                              evals_result=evals_result, obj=obj, feval=feval,
                              verbose_eval=verbose, xgb_model=xgb_model,
                              callbacks=callbacks)

        self.objective = xgb_options["objective"]
        if evals_result:
            for val in evals_result.items():
                evals_result_key = list(val[1].keys())[0]
                evals_result[val[0]][
                    evals_result_key] = val[1][evals_result_key]
            self.evals_result_ = evals_result

        if early_stopping_rounds is not None:
            self.best_score = self._Booster.best_score
            self.best_iteration = self._Booster.best_iteration
            self.best_ntree_limit = self._Booster.best_ntree_limit

        return self

    fit.__doc__ = XGBModel.fit.__doc__.replace(
        'Fit gradient boosting model',
        'Fit gradient boosting classifier', 1)

    def predict(self, data, output_margin=False, ntree_limit=None,
                validate_features=True, base_margin=None):
        """
        Predict with `data`.

        .. note:: This function is not thread safe.

          For each booster object, predict can only be called from one thread.
          If you want to run prediction using multiple thread, call
          ``xgb.copy()`` to make copies of model object and then call
          ``predict()``.

          .. code-block:: python

            preds = bst.predict(dtest, ntree_limit=num_round)

        Parameters
        ----------
        data : array_like
            The dmatrix storing the input.
        output_margin : bool
            Whether to output the raw untransformed margin value.
        ntree_limit : int
            Limit number of trees in the prediction; defaults to
            best_ntree_limit if defined (i.e. it has been trained with early
            stopping), otherwise 0 (use all trees).
        validate_features : bool
            When this is True, validate that the Booster's and data's
            feature_names are identical.  Otherwise, it is assumed that the
            feature_names are the same.

        Returns
        -------
        prediction : numpy array
        """
        test_dmatrix = DMatrix(data, base_margin=base_margin,
                               missing=self.missing, nthread=self.n_jobs)
        if ntree_limit is None:
            ntree_limit = getattr(self, "best_ntree_limit", 0)
        class_probs = self.get_booster().predict(
            test_dmatrix,
            output_margin=output_margin,
            ntree_limit=ntree_limit,
            validate_features=validate_features)
        if output_margin:
            # If output_margin is active, simply return the scores
            return class_probs

        if len(class_probs.shape) > 1:
            column_indexes = np.argmax(class_probs, axis=1)
        else:
            column_indexes = np.repeat(0, class_probs.shape[0])
            column_indexes[class_probs > 0.5] = 1

        if hasattr(self, '_le'):
            return self._le.inverse_transform(column_indexes)
        warnings.warn(
            'Label encoder is not defined.  Returning class probability.')
        return class_probs

    def predict_proba(self, data, ntree_limit=None, validate_features=True,
                      base_margin=None):
        """
        Predict the probability of each `data` example being of a given class.

        .. note:: This function is not thread safe

            For each booster object, predict can only be called from one
            thread.  If you want to run prediction using multiple thread, call
            ``xgb.copy()`` to make copies of model object and then call predict

        Parameters
        ----------
        data : DMatrix
            The dmatrix storing the input.
        ntree_limit : int
            Limit number of trees in the prediction; defaults to best_ntree_limit if defined
            (i.e. it has been trained with early stopping), otherwise 0 (use all trees).
        validate_features : bool
            When this is True, validate that the Booster's and data's feature_names are identical.
            Otherwise, it is assumed that the feature_names are the same.

        Returns
        -------
        prediction : numpy array
            a numpy array with the probability of each data example being of a given class.
        """
        test_dmatrix = DMatrix(data, base_margin=base_margin,
                               missing=self.missing, nthread=self.n_jobs)
        if ntree_limit is None:
            ntree_limit = getattr(self, "best_ntree_limit", 0)
        class_probs = self.get_booster().predict(test_dmatrix,
                                                 ntree_limit=ntree_limit,
                                                 validate_features=validate_features)
        if self.objective == "multi:softprob":
            return class_probs
        classone_probs = class_probs
        classzero_probs = 1.0 - classone_probs
        return np.vstack((classzero_probs, classone_probs)).transpose()

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
''')
class XGBRFClassifier(XGBClassifier):
    # pylint: disable=missing-docstring
    def __init__(self,
                 learning_rate=1,
                 subsample=0.8,
                 colsample_bynode=0.8,
                 reg_lambda=1e-5,
                 **kwargs):
        super().__init__(learning_rate=learning_rate,
                         subsample=subsample,
                         colsample_bynode=colsample_bynode,
                         reg_lambda=reg_lambda,
                         **kwargs)

    def get_xgb_params(self):
        params = super(XGBRFClassifier, self).get_xgb_params()
        params['num_parallel_tree'] = self.n_estimators
        return params

    def get_num_boosting_rounds(self):
        return 1


@xgboost_model_doc(
    "Implementation of the scikit-learn API for XGBoost regression.",
    ['estimators', 'model', 'objective'])
class XGBRegressor(XGBModel, XGBRegressorBase):
    # pylint: disable=missing-docstring
    def __init__(self, objective="reg:squarederror", **kwargs):
        super().__init__(objective=objective, **kwargs)


@xgboost_model_doc(
    "scikit-learn API for XGBoost random forest regression.",
    ['model', 'objective'])
class XGBRFRegressor(XGBRegressor):
    # pylint: disable=missing-docstring
    def __init__(self, learning_rate=1, subsample=0.8, colsample_bynode=0.8,
                 reg_lambda=1e-5, **kwargs):
        super().__init__(learning_rate=learning_rate, subsample=subsample,
                         colsample_bynode=colsample_bynode,
                         reg_lambda=reg_lambda, **kwargs)

    def get_xgb_params(self):
        params = super(XGBRFRegressor, self).get_xgb_params()
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
        Query group information is required for ranking tasks.

        Before fitting the model, your data need to be sorted by query
        group. When fitting the model, you need to provide an additional array
        that contains the size of each query group.

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

        then your group array should be ``[3, 4]``.
''')
class XGBRanker(XGBModel):
    # pylint: disable=missing-docstring,too-many-arguments,invalid-name
    def __init__(self, objective='rank:pairwise', **kwargs):
        super().__init__(objective=objective, **kwargs)
        if callable(self.objective):
            raise ValueError(
                "custom objective function not supported by XGBRanker")
        if "rank:" not in self.objective:
            raise ValueError("please use XGBRanker for ranking task")

    def fit(self, X, y, group, sample_weight=None, base_margin=None,
            eval_set=None,
            sample_weight_eval_set=None, eval_group=None, eval_metric=None,
            early_stopping_rounds=None, verbose=False, xgb_model=None,
            callbacks=None):
        # pylint: disable = attribute-defined-outside-init,arguments-differ
        """Fit gradient boosting ranker

        Parameters
        ----------
        X : array_like
            Feature matrix
        y : array_like
            Labels
        group : array_like
            Size of each query group of training data. Should have as many
            elements as the query groups in the training data
        sample_weight : array_like
            Query group weights

            .. note:: Weights are per-group for ranking tasks

                In ranking task, one weight is assigned to each query group
                (not each data point). This is because we only care about the
                relative ordering of data points within each group, so it
                doesn't make sense to assign weights to individual data points.

        base_margin : array_like
            Global bias for each instance.
        eval_set : list, optional
            A list of (X, y) tuple pairs to use as validation sets, for which
            metrics will be computed.
            Validation metrics will help us track the performance of the model.
        sample_weight_eval_set : list, optional
            A list of the form [L_1, L_2, ..., L_n], where each L_i is a list of
            group weights on the i-th validation set.

            .. note:: Weights are per-group for ranking tasks

                In ranking task, one weight is assigned to each query group (not each
                data point). This is because we only care about the relative ordering of
                data points within each group, so it doesn't make sense to assign
                weights to individual data points.

        eval_group : list of arrays, optional
            A list in which ``eval_group[i]`` is the list containing the sizes of all
            query groups in the ``i``-th pair in **eval_set**.
        eval_metric : str, list of str, optional
            If a str, should be a built-in evaluation metric to use. See
            doc/parameter.rst.
            If a list of str, should be the list of multiple built-in evaluation metrics
            to use. The custom evaluation metric is not yet supported for the ranker.
        early_stopping_rounds : int
            Activates early stopping. Validation metric needs to improve at least once in
            every **early_stopping_rounds** round(s) to continue training.
            Requires at least one item in **eval_set**.
            The method returns the model from the last iteration (not the best one).
            If there's more than one item in **eval_set**, the last entry will be used
            for early stopping.
            If there's more than one metric in **eval_metric**, the last metric
            will be used for early stopping.
            If early stopping occurs, the model will have three additional
            fields: ``clf.best_score``, ``clf.best_iteration`` and
            ``clf.best_ntree_limit``.
        verbose : bool
            If `verbose` and an evaluation set is used, writes the evaluation
            metric measured on the validation set to stderr.
        xgb_model : str
            file name of stored XGBoost model or 'Booster' instance XGBoost
            model to be loaded before training (allows training continuation).
        callbacks : list of callback functions
            List of callback functions that are applied at end of each
            iteration.  It is possible to use predefined callbacks by using
            :ref:`callback_api`.  Example:

            .. code-block:: python

                [xgb.callback.reset_learning_rate(custom_rates)]

        """
        # check if group information is provided
        if group is None:
            raise ValueError("group is required for ranking task")

        if eval_set is not None:
            if eval_group is None:
                raise ValueError(
                    "eval_group is required if eval_set is not None")
            if len(eval_group) != len(eval_set):
                raise ValueError(
                    "length of eval_group should match that of eval_set")
            if any(group is None for group in eval_group):
                raise ValueError(
                    "group is required for all eval datasets for ranking task")

        def _dmat_init(group, **params):
            ret = DMatrix(**params)
            ret.set_group(group)
            return ret

        train_dmatrix = DMatrix(data=X, label=y, weight=sample_weight,
                                base_margin=base_margin,
                                missing=self.missing, nthread=self.n_jobs)
        train_dmatrix.set_group(group)

        evals_result = {}

        if eval_set is not None:
            if sample_weight_eval_set is None:
                sample_weight_eval_set = [None] * len(eval_set)
            evals = [_dmat_init(eval_group[i],
                                data=eval_set[i][0],
                                label=eval_set[i][1],
                                missing=self.missing,
                                weight=sample_weight_eval_set[i],
                                nthread=self.n_jobs)
                     for i in range(len(eval_set))]
            nevals = len(evals)
            eval_names = ["eval_{}".format(i) for i in range(nevals)]
            evals = list(zip(evals, eval_names))
        else:
            evals = ()

        params = self.get_xgb_params()

        feval = eval_metric if callable(eval_metric) else None
        if eval_metric is not None:
            if callable(eval_metric):
                raise ValueError(
                    'Custom evaluation metric is not yet supported for XGBRanker.')
            params.update({'eval_metric': eval_metric})

        self._Booster = train(params, train_dmatrix,
                              self.n_estimators,
                              early_stopping_rounds=early_stopping_rounds,
                              evals=evals,
                              evals_result=evals_result, feval=feval,
                              verbose_eval=verbose, xgb_model=xgb_model,
                              callbacks=callbacks)

        self.objective = params["objective"]

        if evals_result:
            for val in evals_result.items():
                evals_result_key = list(val[1].keys())[0]
                evals_result[val[0]][evals_result_key] = val[1][evals_result_key]
            self.evals_result = evals_result

        if early_stopping_rounds is not None:
            self.best_score = self._Booster.best_score
            self.best_iteration = self._Booster.best_iteration
            self.best_ntree_limit = self._Booster.best_ntree_limit

        return self

    def predict(self, data, output_margin=False,
                ntree_limit=0, validate_features=True, base_margin=None):

        test_dmatrix = DMatrix(data, base_margin=base_margin,
                               missing=self.missing)
        if ntree_limit is None:
            ntree_limit = getattr(self, "best_ntree_limit", 0)

        return self.get_booster().predict(test_dmatrix,
                                          output_margin=output_margin,
                                          ntree_limit=ntree_limit,
                                          validate_features=validate_features)

    predict.__doc__ = XGBModel.predict.__doc__
