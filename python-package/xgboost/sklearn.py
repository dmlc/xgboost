# coding: utf-8
# pylint: disable=too-many-arguments, too-many-locals, invalid-name, fixme
"""Scikit-Learn Wrapper interface for XGBoost."""
from __future__ import absolute_import

import numpy as np
from .core import Booster, DMatrix, XGBoostError
from .training import train

try:
    from sklearn.base import BaseEstimator
    from sklearn.base import RegressorMixin, ClassifierMixin
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_INSTALLED = True
except ImportError:
    SKLEARN_INSTALLED = False

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
        if not params.get('eval_metric', True):
            del params['eval_metric']  # don't give as None param to Booster
        return params

    def get_xgb_params(self):
        """Get xgboost type parameters."""
        xgb_params = self.get_params()

        xgb_params['silent'] = 1 if self.silent else 0

        if self.nthread <= 0:
            xgb_params.pop('nthread', None)
        return xgb_params

    def fit(self, X, y, eval_set=None, eval_metric=None,
            early_stopping_rounds=None, verbose=True):
        # pylint: disable=missing-docstring,invalid-name,attribute-defined-outside-init
        """
        Fit the gradient boosting model

        Parameters
        ----------
        X : array_like
            Feature matrix
        y : array_like
            Labels
        eval_set : list, optional
            A list of (X, y) tuple pairs to use as a validation set for
            early-stopping
        eval_metric : str, callable, optional
            If a str, should be a built-in evaluation metric to use. See
            doc/parameter.md. If callable, a custom evaluation metric. The call
            signature is func(y_predicted, y_true) where y_true will be a
            DMatrix object such that you may need to call the get_label
            method. It must return a str, value pair where the str is a name
            for the evaluation and value is the value of the evaluation
            function. This objective is always minimized.
        early_stopping_rounds : int
            Activates early stopping. Validation error needs to decrease at
            least every <early_stopping_rounds> round(s) to continue training.
            Requires at least one item in evals.  If there's more than one,
            will use the last. Returns the model from the last iteration
            (not the best one). If early stopping occurs, the model will
            have two additional fields: bst.best_score and bst.best_iteration.
        verbose : bool
            If `verbose` and an evaluation set is used, writes the evaluation
            metric measured on the validation set to stderr.
        """
        trainDmatrix = DMatrix(X, label=y, missing=self.missing)

        eval_results = {}
        if eval_set is not None:
            evals = list(DMatrix(x[0], label=x[1]) for x in eval_set)
            evals = list(zip(evals, ["validation_{}".format(i) for i in
                                     range(len(evals))]))
        else:
            evals = ()

        params = self.get_xgb_params()

        feval = eval_metric if callable(eval_metric) else None
        if eval_metric is not None:
            if callable(eval_metric):
                eval_metric = None
            else:
                params.update({'eval_metric': eval_metric})

        self._Booster = train(params, trainDmatrix,
                              self.n_estimators, evals=evals,
                              early_stopping_rounds=early_stopping_rounds,
                              evals_result=eval_results, feval=feval,
                              verbose_eval=verbose)
        if eval_results:
            eval_results = {k: np.array(v, dtype=float)
                            for k, v in eval_results.items()}
            eval_results = {k: np.array(v) for k, v in eval_results.items()}
            self.eval_results = eval_results

        if early_stopping_rounds is not None:
            self.best_score = self._Booster.best_score
            self.best_iteration = self._Booster.best_iteration
        return self

    def predict(self, data):
        # pylint: disable=missing-docstring,invalid-name
        test_dmatrix = DMatrix(data, missing=self.missing)
        return self.booster().predict(test_dmatrix)


class XGBClassifier(XGBModel, XGBClassifierBase):
    # pylint: disable=missing-docstring,too-many-arguments,invalid-name
    __doc__ = """Implementation of the scikit-learn API for XGBoost classification.

    """ + '\n'.join(XGBModel.__doc__.split('\n')[2:])

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

    def fit(self, X, y, sample_weight=None, eval_set=None, eval_metric=None,
            early_stopping_rounds=None, verbose=True):
        # pylint: disable = attribute-defined-outside-init,arguments-differ
        """
        Fit gradient boosting classifier

        Parameters
        ----------
        X : array_like
            Feature matrix
        y : array_like
            Labels
        sample_weight : array_like
            Weight for each instance
        eval_set : list, optional
            A list of (X, y) pairs to use as a validation set for
            early-stopping
        eval_metric : str, callable, optional
            If a str, should be a built-in evaluation metric to use. See
            doc/parameter.md. If callable, a custom evaluation metric. The call
            signature is func(y_predicted, y_true) where y_true will be a
            DMatrix object such that you may need to call the get_label
            method. It must return a str, value pair where the str is a name
            for the evaluation and value is the value of the evaluation
            function. This objective is always minimized.
        early_stopping_rounds : int, optional
            Activates early stopping. Validation error needs to decrease at
            least every <early_stopping_rounds> round(s) to continue training.
            Requires at least one item in evals.  If there's more than one,
            will use the last. Returns the model from the last iteration
            (not the best one). If early stopping occurs, the model will
            have two additional fields: bst.best_score and bst.best_iteration.
        verbose : bool
            If `verbose` and an evaluation set is used, writes the evaluation
            metric measured on the validation set to stderr.
        """
        eval_results = {}
        self.classes_ = list(np.unique(y))
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ > 2:
            # Switch to using a multiclass objective in the underlying XGB instance
            self.objective = "multi:softprob"
            xgb_options = self.get_xgb_params()
            xgb_options['num_class'] = self.n_classes_
        else:
            xgb_options = self.get_xgb_params()

        feval = eval_metric if callable(eval_metric) else None
        if eval_metric is not None:
            if callable(eval_metric):
                eval_metric = None
            else:
                xgb_options.update({"eval_metric": eval_metric})

        if eval_set is not None:
            # TODO: use sample_weight if given?
            evals = list(DMatrix(x[0], label=x[1]) for x in eval_set)
            nevals = len(evals)
            eval_names = ["validation_{}".format(i) for i in range(nevals)]
            evals = list(zip(evals, eval_names))
        else:
            evals = ()

        self._le = LabelEncoder().fit(y)
        training_labels = self._le.transform(y)

        if sample_weight is not None:
            train_dmatrix = DMatrix(X, label=training_labels, weight=sample_weight,
                                    missing=self.missing)
        else:
            train_dmatrix = DMatrix(X, label=training_labels,
                                    missing=self.missing)

        self._Booster = train(xgb_options, train_dmatrix, self.n_estimators,
                              evals=evals,
                              early_stopping_rounds=early_stopping_rounds,
                              evals_result=eval_results, feval=feval,
                              verbose_eval=verbose)

        if eval_results:
            eval_results = {k: np.array(v, dtype=float)
                            for k, v in eval_results.items()}
            self.eval_results = eval_results

        if early_stopping_rounds is not None:
            self.best_score = self._Booster.best_score
            self.best_iteration = self._Booster.best_iteration

        return self

    def predict(self, data):
        test_dmatrix = DMatrix(data, missing=self.missing)
        class_probs = self.booster().predict(test_dmatrix)
        if len(class_probs.shape) > 1:
            column_indexes = np.argmax(class_probs, axis=1)
        else:
            column_indexes = np.repeat(0, class_probs.shape[0])
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
    __doc__ = """Implementation of the scikit-learn API for XGBoost regression.
    """ + '\n'.join(XGBModel.__doc__.split('\n')[2:])
