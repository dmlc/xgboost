# coding: utf-8
# pylint: disable=too-many-locals, too-many-arguments, invalid-name
# pylint: disable=too-many-branches, too-many-statements
"""AutoML Library containing useful routines."""
from __future__ import absolute_import

import warnings


class ParamError(Exception):
    """
    Excetion raised for errors in hyper-parameter configuration.
    """

class ConvergenceTester():
    """Convergence tester.

        The rule for convergence is idx + 1 + n < size * c, where idx is the
        index best metric value so far (0-based) and size is the current number of
        points in the series, n and c are parameters for the test.
    """

    def __init__(self, min_num_points=-1, n=0, c=1.0):
        """
        Parameters
        ----------
        min_num_points : int
            The minimum number of points to be considered convergence.
        n : int
            The n.
        c : float
            A constant factor in [0, 1]
        """
        self.min_num_points = min_num_points
        self.n = n
        self.c = c
        self.measure_list = []
        self.best_idx = -1

    def reset(self, maximize=False):
        """
        Resets this tester.

        Parameters
        ----------
        maximize : bool
            Whether to maximize the metric.
        """
        self.maximize = maximize
        self.measure_list = []
        self.best_idx = -1
        if maximize:
            self.best_so_far = float('-inf')
        else:
            self.best_so_far = float('inf')

    @staticmethod
    def parse(cc):
        """
        ConvergenceTester configuration parser.

        Returns
        -------
        result : A configured ConvergenceTester.
        """
        min_num_points = -1
        n = 0
        c = 1.0
        if cc and cc.strip():
            strs = cc.split(":")
            if len(strs) > 0:
                min_num_points = int(strs[0])
            if len(strs) > 1:
                n = int(strs[1])
            if len(strs) > 2:
                c = float(strs[2])
        return ConvergenceTester(min_num_points, n, c)

    def size(self):
        """
        Returns the size of the series.

        Returns
        -------
        result : the size of the series.
        """
        return len(self.measure_list)

    def is_first_better(self, a, b):
        """Returns whether to the 1st metric value is better than the 2nd one.

        Parameters
        ----------
        a : float
            the 1st metric value.
        b : float
            the 2nd metric value.

        Returns
        -------
        result : whether to the 1st metric value is better than the 2nd one.
        """
        if self.maximize:
            return a > b
        else:
            return a < b

    def add(self, v):
        """Adds a metric value.

        Parameters
        ----------
        v : float
            The metric value to add.
        """
        self.measure_list.append(v)

        if self.is_first_better(v, self.best_so_far):
            self.best_so_far = v
            self.best_idx = self.size() - 1

    def get_best_idx(self):
        """
        Returns the index for best metric value so far.

        Returns
        -------
        result : the index for best metric value so far.
        """
        return self.best_idx

    def get_best_so_far(self):
        """
        Returns the best metric value so far.

        Returns
        -------
        result : the best metric value so far.
        """
        return self.best_so_far

    def is_maximize(self):
        """
        Returns whether to maximize the metric.

        Returns
        -------
        result : whether to maximize the metric.
        """
        return self.maximize

    def is_converged(self):
        """
        Returns true if the series is converged.

        Returns
        -------
        result : true if the series is converged.
        """
        return self.min_num_points >= 0 and \
               self.size() > self.min_num_points and \
               self.best_idx >= 0 and \
               self.best_idx + 1 + self.n < self.size() * self.c

    def print_params(self):
        """
        Returns the string representation of this convergence tester.

        Returns
        -------
        result : the string representation of this convergence tester.
        """
        return "maximize = {0}, min_num_points = {1}, n = {2}, c = {3}".format( \
            self.maximize, self.min_num_points, self.n, self.c)

def param_undefined_info(param_name, default_value):
    """
    Returns info warning an undefined parameter.
    """
    return "Parameter warning: the hyper-parameter " + \
           "'{}' is not specified, will use default value '{}'" \
           .format(param_name, default_value)

def param_invalid_value_info(param_name, default_value):
    """
    Returns info warning an invalid parameter configuration.
    """
    return "Parameter warning: the configuration of hyper-parameter "+ \
           "'{}' is not valid, will use default value '{}'" \
           .format(param_name, default_value)

# Available values and bounds of hyper-parameters
OBJECTIVE_LIST = ['reg:linear', 'reg:logistic', 'binary:logistic', 'binary:logitraw', \
     'binary:hinge', 'count:poisson', 'multi:softmax', 'multi:softprob',\
     'reg:gamma', 'reg:tweedie', 'rank:ndcg', 'rank:pairwise', \
     'rank:map', 'survival:cox']
METRIC_LIST = ['rmse', 'mae', 'logloss', 'error', 'merror', 'mlogloss', 'auc', \
  'poisson-nloglik', 'gamma-nloglik', 'gamma-deviance', 'tweedie-nloglik', \
  'ndcg', 'map', 'aucpr', 'ndcg-', 'map-', 'cox-nloglik']
METRIC_CHECKLIST = ['error@', 'ndcg@', 'map@']
MAXIMIZE_METRICS = ('auc', 'map', 'ndcg')
MAX_DEPTH_DEFAULT_VALUE = 6
MAX_DEPTH_LOWER_BOUND = 0
ETA_DEFAULT_VALUE = 0.3
ETA_LOWER_BOUND = 0.0
ETA_UPPER_BOUND = 1.0
NUM_TREE_DEFAULT_VALUE = 100
NUM_TREE_LOWER_BOUND = 0
DEFAULT_METRIC = {'binary': 'auc', 'multi': 'merror', 'reg': 'rmse', 'rank': 'ndcg',\
                  'survival': 'cox-nloglik'}

def get_optimization_direction(params):
    maximize_score = False
    metric = params['eval_metric']
    if any(metric.startswith(x) for x in MAXIMIZE_METRICS):
        maximize_score = True

    return maximize_score

def xgb_parameter_checker(params, num_round, num_class=None, skip_list=[]):
    """
    XGBoost hyper-parameter checker.

    This function checks if the input hyper-parameters setting
    satisfies specific requirements and use default values for
    parameters not satisfying requirements.

    Parameters
    ----------
    params : dict
        XGBoost hyper-parameter configuration.
    num_round: int
        number of boosting rounds.
    num_class: int
        number of classes in the target feature.
    skip_list: list
        list of parameters that can be skipped checking.

    Returns
    -------
    result:
        A checked and corrected XGBoost hyper-parameter
        configuration.
    """
    # check objective function
    if 'objective' in params:
        objective = params['objective']
        if objective not in OBJECTIVE_LIST:
            raise ParamError("The objective function '{}' is not supported.".format(objective))
    else:
        raise ParamError("The objective function is not specified.")

    objective_type = objective.split(':')[0]
    # check number of classes in label
    if not num_class and 'num_class' in params:
        try:
            num_class = int(params['num_class'])
        except Exception:
            raise ParamError("Parameter num_class is ill-formed: " + params['num_class'])
    if num_class:
        num_class = int(num_class)
        if objective_type == 'binary' and num_class != 2:
            raise ParamError("Binary objective function can not handle more than " +\
                              "2 classes label. Recommend using 'multi:softmax'")
        if objective_type != 'binary' and num_class == 2:
            warnings.warn("Only 2 classes in label. Will use 'binary:logistic' as " +\
                              "the objective function")
            params['objective'] = 'binary:logistic'
        params['num_class'] = num_class
    if objective_type == 'binary':
        params.pop('num_class', None)

    if 'max_depth' not in skip_list:
        try:
            max_depth = params['max_depth']
            max_depth = int(max_depth)
            if  max_depth <= MAX_DEPTH_LOWER_BOUND:
                warnings.warn(param_invalid_value_info('max_depth', MAX_DEPTH_DEFAULT_VALUE))
                params['max_depth'] = MAX_DEPTH_DEFAULT_VALUE
        except KeyError:
            warnings.warn(param_undefined_info('max_depth', MAX_DEPTH_DEFAULT_VALUE))
            params['max_depth'] = MAX_DEPTH_DEFAULT_VALUE
        except ValueError:
            warnings.warn(param_invalid_value_info('max_depth', MAX_DEPTH_DEFAULT_VALUE))
            params['max_depth'] = MAX_DEPTH_DEFAULT_VALUE

    if 'eta' not in skip_list:
        try:
            learning_rate_key = 'eta'
            if learning_rate_key not in params:
                learning_rate_key = 'learning_rate'
            eta = params[learning_rate_key]
            eta = float(eta)
            if eta <= ETA_LOWER_BOUND or eta > ETA_UPPER_BOUND:
                warnings.warn(param_invalid_value_info('eta (alias: learning_rate)', ETA_DEFAULT_VALUE))
                params['eta'] = ETA_DEFAULT_VALUE
        except KeyError:
            warnings.warn(param_undefined_info('eta (alias: learning_rate)', ETA_DEFAULT_VALUE))
            params['eta'] = ETA_DEFAULT_VALUE
        except ValueError:
            warnings.warn(param_invalid_value_info('eta (alias: learning_rate)', ETA_DEFAULT_VALUE))
            params['eta'] = ETA_DEFAULT_VALUE

    try:
        num_tree = int(num_round)
        if num_tree <= NUM_TREE_LOWER_BOUND:
            warnings.warn(param_invalid_value_info('num_round', NUM_TREE_DEFAULT_VALUE))
            params['num_round'] = NUM_TREE_DEFAULT_VALUE
    except KeyError:
        warnings.warn(param_undefined_info('num_round', NUM_TREE_DEFAULT_VALUE))
        params['num_round'] = NUM_TREE_DEFAULT_VALUE
    except ValueError:
        warnings.warn(param_invalid_value_info('num_round', NUM_TREE_DEFAULT_VALUE))
        params['num_round'] = NUM_TREE_DEFAULT_VALUE

    if 'eval_metric' not in params:
        objective_type = objective.split(':')[0]
        warnings.warn(param_undefined_info('eval_metric', DEFAULT_METRIC[objective_type]))
        params['eval_metric'] = DEFAULT_METRIC[objective_type]
    if params['eval_metric'] not in METRIC_LIST:
        checked = False
        for metric_to_check in METRIC_CHECKLIST:
            if metric_to_check in params['eval_metric']:
                mid_num = params['eval_metric'].strip(metric_to_check).rstrip('-')
                try:
                    mid_num = float(mid_num)
                except ValueError:
                    raise ParamError("A number required after '{}'".format(metric_to_check))
                checked = True
        if not checked:
            objective_type = objective.split(':')[0]
            warnings.warn(param_invalid_value_info('eval_metric', DEFAULT_METRIC[objective_type]))
            params['eval_metric'] = DEFAULT_METRIC[objective_type]

    maximize_score = get_optimization_direction(params)
    params['maximize_eval_metric'] = str(maximize_score)

    return params
