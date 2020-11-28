# coding: utf-8
# pylint: disable=invalid-name, too-many-statements, no-self-use
# pylint: disable=too-many-arguments
"""Training Library containing training routines."""
from abc import ABC
import collections
import os
import pickle
from typing import Callable, List
import numpy

from . import rabit
from .core import EarlyStopException, CallbackEnv, Booster, XGBoostError
from .compat import STRING_TYPES


def _get_callback_context(env):
    """return whether the current callback context is cv or train"""
    if env.model is not None and env.cvfolds is None:
        context = 'train'
    elif env.model is None and env.cvfolds is not None:
        context = 'cv'
    return context


def _fmt_metric(value, show_stdv=True):
    """format metric string"""
    if len(value) == 2:
        return '{0}:{1:.5f}'.format(value[0], value[1])
    if len(value) == 3:
        if show_stdv:
            return '{0}:{1:.5f}+{2:.5f}'.format(value[0], value[1], value[2])
        return '{0}:{1:.5f}'.format(value[0], value[1])
    raise ValueError("wrong metric value", value)


def print_evaluation(period=1, show_stdv=True):
    """Create a callback that print evaluation result.

    We print the evaluation results every **period** iterations
    and on the first and the last iterations.

    Parameters
    ----------
    period : int
        The period to log the evaluation results

    show_stdv : bool, optional
         Whether show stdv if provided

    Returns
    -------
    callback : function
        A callback that print evaluation every period iterations.
    """
    def callback(env):
        """internal function"""
        if env.rank != 0 or (not env.evaluation_result_list) or period is False or period == 0:
            return
        i = env.iteration
        if i % period == 0 or i + 1 == env.begin_iteration or i + 1 == env.end_iteration:
            msg = '\t'.join([_fmt_metric(x, show_stdv) for x in env.evaluation_result_list])
            rabit.tracker_print('[%d]\t%s\n' % (i, msg))
    return callback


def record_evaluation(eval_result):
    """Create a call back that records the evaluation history into **eval_result**.

    Parameters
    ----------
    eval_result : dict
       A dictionary to store the evaluation results.

    Returns
    -------
    callback : function
        The requested callback function.
    """
    if not isinstance(eval_result, dict):
        raise TypeError('eval_result has to be a dictionary')
    eval_result.clear()

    def init(env):
        """internal function"""
        for k, _ in env.evaluation_result_list:
            pos = k.index('-')
            key = k[:pos]
            metric = k[pos + 1:]
            if key not in eval_result:
                eval_result[key] = {}
            if metric not in eval_result[key]:
                eval_result[key][metric] = []

    def callback(env):
        """internal function"""
        if not eval_result:
            init(env)
        for k, v in env.evaluation_result_list:
            pos = k.index('-')
            key = k[:pos]
            metric = k[pos + 1:]
            eval_result[key][metric].append(v)
    return callback


def reset_learning_rate(learning_rates):
    """Reset learning rate after iteration 1

    NOTE: the initial learning rate will still take in-effect on first iteration.

    Parameters
    ----------
    learning_rates: list or function
        List of learning rate for each boosting round
        or a customized function that calculates eta in terms of
        current number of round and the total number of boosting round (e.g.
        yields learning rate decay)

        * list ``l``: ``eta = l[boosting_round]``
        * function ``f``: ``eta = f(boosting_round, num_boost_round)``

    Returns
    -------
    callback : function
        The requested callback function.
    """
    def get_learning_rate(i, n, learning_rates):
        """helper providing the learning rate"""
        if isinstance(learning_rates, list):
            if len(learning_rates) != n:
                raise ValueError("Length of list 'learning_rates' has to equal 'num_boost_round'.")
            new_learning_rate = learning_rates[i]
        else:
            new_learning_rate = learning_rates(i, n)
        return new_learning_rate

    def callback(env):
        """internal function"""
        context = _get_callback_context(env)

        if context == 'train':
            bst, i, n = env.model, env.iteration, env.end_iteration
            bst.set_param(
                'learning_rate', get_learning_rate(i, n, learning_rates))
        elif context == 'cv':
            i, n = env.iteration, env.end_iteration
            for cvpack in env.cvfolds:
                bst = cvpack.bst
                bst.set_param(
                    'learning_rate', get_learning_rate(i, n, learning_rates))

    callback.before_iteration = False
    return callback


def early_stop(stopping_rounds, maximize=False, verbose=True):
    """Create a callback that activates early stoppping.

    Validation error needs to decrease at least
    every **stopping_rounds** round(s) to continue training.
    Requires at least one item in **evals**.
    If there's more than one, will use the last.
    Returns the model from the last iteration (not the best one).
    If early stopping occurs, the model will have three additional fields:
    ``bst.best_score``, ``bst.best_iteration`` and ``bst.best_ntree_limit``.
    (Use ``bst.best_ntree_limit`` to get the correct value if ``num_parallel_tree``
    and/or ``num_class`` appears in the parameters)

    Parameters
    ----------
    stopping_rounds : int
       The stopping rounds before the trend occur.

    maximize : bool
        Whether to maximize evaluation metric.

    verbose : optional, bool
        Whether to print message about early stopping information.

    Returns
    -------
    callback : function
        The requested callback function.
    """
    state = {}

    def init(env):
        """internal function"""
        bst = env.model

        if not env.evaluation_result_list:
            raise ValueError('For early stopping you need at least one set in evals.')
        if len(env.evaluation_result_list) > 1 and verbose:
            msg = ("Multiple eval metrics have been passed: "
                   "'{0}' will be used for early stopping.\n\n")
            rabit.tracker_print(msg.format(env.evaluation_result_list[-1][0]))
        maximize_metrics = ('auc', 'aucpr', 'map', 'ndcg')
        maximize_at_n_metrics = ('auc@', 'aucpr@', 'map@', 'ndcg@')
        maximize_score = maximize
        metric_label = env.evaluation_result_list[-1][0]
        metric = metric_label.split('-', 1)[-1]

        if any(metric.startswith(x) for x in maximize_at_n_metrics):
            maximize_score = True

        if any(metric.split(":")[0] == x for x in maximize_metrics):
            maximize_score = True

        if verbose and env.rank == 0:
            msg = "Will train until {} hasn't improved in {} rounds.\n"
            rabit.tracker_print(msg.format(metric_label, stopping_rounds))

        state['maximize_score'] = maximize_score
        state['best_iteration'] = 0
        if maximize_score:
            state['best_score'] = float('-inf')
        else:
            state['best_score'] = float('inf')
        msg = '[%d]\t%s' % (
            env.iteration,
            '\t'.join([_fmt_metric(x) for x in env.evaluation_result_list]))
        state['best_msg'] = msg

        if bst is not None:
            if bst.attr('best_score') is not None:
                state['best_score'] = float(bst.attr('best_score'))
                state['best_iteration'] = int(bst.attr('best_iteration'))
                state['best_msg'] = bst.attr('best_msg')
            else:
                bst.set_attr(best_iteration=str(state['best_iteration']))
                bst.set_attr(best_score=str(state['best_score']))
        else:
            assert env.cvfolds is not None

    def callback(env):
        """internal function"""
        if not state:
            init(env)
        score = env.evaluation_result_list[-1][1]
        best_score = state['best_score']
        best_iteration = state['best_iteration']
        maximize_score = state['maximize_score']
        if (maximize_score and score > best_score) or \
                (not maximize_score and score < best_score):
            msg = '[%d]\t%s' % (
                env.iteration,
                '\t'.join([_fmt_metric(x) for x in env.evaluation_result_list]))
            state['best_msg'] = msg
            state['best_score'] = score
            state['best_iteration'] = env.iteration
            # save the property to attributes, so they will occur in checkpoint.
            if env.model is not None:
                env.model.set_attr(best_score=str(state['best_score']),
                                   best_iteration=str(state['best_iteration']),
                                   best_msg=state['best_msg'])
        elif env.iteration - best_iteration >= stopping_rounds:
            best_msg = state['best_msg']
            if verbose and env.rank == 0:
                msg = "Stopping. Best iteration:\n{}\n\n"
                rabit.tracker_print(msg.format(best_msg))
            raise EarlyStopException(best_iteration)
    return callback


# The new implementation of callback functions.
# Breaking:
# - reset learning rate no longer accepts total boosting rounds

# pylint: disable=unused-argument
class TrainingCallback(ABC):
    '''Interface for training callback.

    .. versionadded:: 1.3.0

    '''
    def __init__(self):
        pass

    def before_training(self, model):
        '''Run before training starts.'''
        return model

    def after_training(self, model):
        '''Run after training is finished.'''
        return model

    def before_iteration(self, model, epoch, evals_log):
        '''Run before each iteration.  Return True when training should stop.'''
        return False

    def after_iteration(self, model, epoch, evals_log):
        '''Run after each iteration.  Return True when training should stop.'''
        return False


def _aggcv(rlist):
    # pylint: disable=invalid-name
    """Aggregate cross-validation results.

    """
    cvmap = {}
    idx = rlist[0].split()[0]
    for line in rlist:
        arr = line.split()
        assert idx == arr[0]
        for metric_idx, it in enumerate(arr[1:]):
            if not isinstance(it, STRING_TYPES):
                it = it.decode()
            k, v = it.split(':')
            if (metric_idx, k) not in cvmap:
                cvmap[(metric_idx, k)] = []
            cvmap[(metric_idx, k)].append(float(v))
    msg = idx
    results = []
    for (metric_idx, k), v in sorted(cvmap.items(), key=lambda x: x[0][0]):
        v = numpy.array(v)
        if not isinstance(msg, STRING_TYPES):
            msg = msg.decode()
        mean, std = numpy.mean(v), numpy.std(v)
        results.extend([(k, mean, std)])
    return results


def _allreduce_metric(score):
    '''Helper function for computing customized metric in distributed
    environment.  Not strictly correct as many functions don't use mean value
    as final result.

    '''
    world = rabit.get_world_size()
    assert world != 0
    if world == 1:
        return score
    if isinstance(score, tuple):  # has mean and stdv
        raise ValueError(
            'xgboost.cv function should not be used in distributed environment.')
    score = numpy.array([score])
    score = rabit.allreduce(score, rabit.Op.SUM) / world
    return score[0]


class CallbackContainer:
    '''A special callback for invoking a list of other callbacks.

    .. versionadded:: 1.3.0

    '''
    def __init__(self, callbacks: List[TrainingCallback],
                 metric: Callable = None, is_cv: bool = False):
        self.callbacks = set(callbacks)
        if metric is not None:
            msg = 'metric must be callable object for monitoring.  For ' + \
                'builtin metrics, passing them in training parameter' + \
                ' will invoke monitor automatically.'
            assert callable(metric), msg
        self.metric = metric
        self.history = collections.OrderedDict()
        self.is_cv = is_cv

        if self.is_cv:
            self.aggregated_cv = None

    def before_training(self, model):
        '''Function called before training.'''
        for c in self.callbacks:
            model = c.before_training(model=model)
            msg = 'before_training should return the model'
            if self.is_cv:
                assert isinstance(model.cvfolds, list), msg
            else:
                assert isinstance(model, Booster), msg
        return model

    def after_training(self, model):
        '''Function called after training.'''
        for c in self.callbacks:
            model = c.after_training(model=model)
            msg = 'after_training should return the model'
            if self.is_cv:
                assert isinstance(model.cvfolds, list), msg
            else:
                assert isinstance(model, Booster), msg
        return model

    def before_iteration(self, model, epoch, dtrain, evals):
        '''Function called before training iteration.'''
        return any(c.before_iteration(model, epoch, self.history)
                   for c in self.callbacks)

    def _update_history(self, score, epoch):
        for d in score:
            name, s = d[0], float(d[1])
            if self.is_cv:
                std = float(d[2])
                s = (s, std)
            splited_names = name.split('-')
            data_name = splited_names[0]
            metric_name = '-'.join(splited_names[1:])
            s = _allreduce_metric(s)
            if data_name in self.history:
                data_history = self.history[data_name]
                if metric_name in data_history:
                    data_history[metric_name].append(s)
                else:
                    data_history[metric_name] = [s]
            else:
                self.history[data_name] = collections.OrderedDict()
                self.history[data_name][metric_name] = [s]
        return False

    def after_iteration(self, model, epoch, dtrain, evals):
        '''Function called after training iteration.'''
        if self.is_cv:
            scores = model.eval(epoch, self.metric)
            scores = _aggcv(scores)
            self.aggregated_cv = scores
            self._update_history(scores, epoch)
        else:
            evals = [] if evals is None else evals
            for _, name in evals:
                assert name.find('-') == -1, 'Dataset name should not contain `-`'
            score = model.eval_set(evals, epoch, self.metric)
            score = score.split()[1:]  # into datasets
            # split up `test-error:0.1234`
            score = [tuple(s.split(':')) for s in score]
            self._update_history(score, epoch)
        ret = any(c.after_iteration(model, epoch, self.history)
                  for c in self.callbacks)
        return ret


class LearningRateScheduler(TrainingCallback):
    '''Callback function for scheduling learning rate.

    .. versionadded:: 1.3.0

    Parameters
    ----------

    learning_rates : callable/collections.Sequence
        If it's a callable object, then it should accept an integer parameter
        `epoch` and returns the corresponding learning rate.  Otherwise it
        should be a sequence like list or tuple with the same size of boosting
        rounds.

    '''
    def __init__(self, learning_rates):
        assert callable(learning_rates) or \
            isinstance(learning_rates, collections.abc.Sequence)
        if callable(learning_rates):
            self.learning_rates = learning_rates
        else:
            self.learning_rates = lambda epoch: learning_rates[epoch]
        super().__init__()

    def after_iteration(self, model, epoch, evals_log):
        model.set_param('learning_rate', self.learning_rates(epoch))


# pylint: disable=too-many-instance-attributes
class EarlyStopping(TrainingCallback):
    ''' Callback function for early stopping

    .. versionadded:: 1.3.0

    Parameters
    ----------
    rounds : int
        Early stopping rounds.
    metric_name : str
        Name of metric that is used for early stopping.
    data_name: str
        Name of dataset that is used for early stopping.
    maximize : bool
        Whether to maximize evaluation metric.  None means auto (discouraged).
    save_best : bool
        Whether training should return the best model or the last model.
    '''
    def __init__(self,
                 rounds,
                 metric_name=None,
                 data_name=None,
                 maximize=None,
                 save_best=False):
        self.data = data_name
        self.metric_name = metric_name
        self.rounds = rounds
        self.save_best = save_best
        self.maximize = maximize
        self.stopping_history = {}

        if self.maximize is not None:
            if self.maximize:
                self.improve_op = lambda x, y: x > y
            else:
                self.improve_op = lambda x, y: x < y

        self.current_rounds = 0
        self.best_scores = {}
        super().__init__()

    def _update_rounds(self, score, name, metric, model, epoch):
        # Just to be compatibility with old behavior before 1.3.  We should let
        # user to decide.
        if self.maximize is None:
            maximize_metrics = ('auc', 'aucpr', 'map', 'ndcg', 'auc@',
                                'aucpr@', 'map@', 'ndcg@')
            if any(metric.startswith(x) for x in maximize_metrics):
                self.improve_op = lambda x, y: x > y
                self.maximize = True
            else:
                self.improve_op = lambda x, y: x < y
                self.maximize = False

        if not self.stopping_history:  # First round
            self.current_rounds = 0
            self.stopping_history[name] = {}
            self.stopping_history[name][metric] = [score]
            self.best_scores[name] = {}
            self.best_scores[name][metric] = [score]
            model.set_attr(best_score=str(score), best_iteration=str(epoch))
        elif not self.improve_op(score, self.best_scores[name][metric][-1]):
            # Not improved
            self.stopping_history[name][metric].append(score)
            self.current_rounds += 1
        else:  # Improved
            self.stopping_history[name][metric].append(score)
            self.best_scores[name][metric].append(score)
            record = self.stopping_history[name][metric][-1]
            model.set_attr(best_score=str(record), best_iteration=str(epoch))
            self.current_rounds = 0  # reset

        if self.current_rounds >= self.rounds:
            # Should stop
            return True
        return False

    def after_iteration(self, model: Booster, epoch, evals_log):
        msg = 'Must have at least 1 validation dataset for early stopping.'
        assert len(evals_log.keys()) >= 1, msg
        data_name = ''
        if self.data:
            for d, _ in evals_log.items():
                if d == self.data:
                    data_name = d
            if not data_name:
                raise ValueError('No dataset named:', self.data)
        else:
            # Use the last one as default.
            data_name = list(evals_log.keys())[-1]
        assert isinstance(data_name, str) and data_name
        data_log = evals_log[data_name]

        # Filter out scores that can not be used for early stopping.
        if self.metric_name:
            metric_name = self.metric_name
        else:
            # Use last metric by default.
            assert isinstance(data_log, collections.OrderedDict)
            metric_name = list(data_log.keys())[-1]
        score = data_log[metric_name][-1]
        return self._update_rounds(score, data_name, metric_name, model, epoch)

    def after_training(self, model: Booster):
        try:
            if self.save_best:
                model = model[: int(model.attr('best_iteration'))]
        except XGBoostError as e:
            raise XGBoostError('`save_best` is not applicable to current booster') from e
        return model


class EvaluationMonitor(TrainingCallback):
    '''Print the evaluation result at each iteration.

    .. versionadded:: 1.3.0

    Parameters
    ----------

    metric : callable
        Extra user defined metric.
    rank : int
        Which worker should be used for printing the result.
    period : int
        How many epoches between printing.
    show_stdv : bool
        Used in cv to show standard deviation.  Users should not specify it.
    '''
    def __init__(self, rank=0, period=1, show_stdv=False):
        self.printer_rank = rank
        self.show_stdv = show_stdv
        self.period = period
        assert period > 0
        # last error message, useful when early stopping and period are used together.
        self._latest = None
        super().__init__()

    def _fmt_metric(self, data, metric, score, std):
        if std is not None and self.show_stdv:
            msg = '\t{0}:{1:.5f}+{2:.5f}'.format(data + '-' + metric, score, std)
        else:
            msg = '\t{0}:{1:.5f}'.format(data + '-' + metric, score)
        return msg

    def after_iteration(self, model, epoch, evals_log):
        if not evals_log:
            return False

        msg = f'[{epoch}]'
        if rabit.get_rank() == self.printer_rank:
            for data, metric in evals_log.items():
                for metric_name, log in metric.items():
                    if isinstance(log[-1], tuple):
                        score = log[-1][0]
                        stdv = log[-1][1]
                    else:
                        score = log[-1]
                        stdv = None
                    msg += self._fmt_metric(data, metric_name, score, stdv)
            msg += '\n'

            if (epoch % self.period) != 0 or self.period == 1:
                rabit.tracker_print(msg)
                self._latest = None
            else:
                # There is skipped message
                self._latest = msg
        return False

    def after_training(self, model):
        if rabit.get_rank() == self.printer_rank and self._latest is not None:
            rabit.tracker_print(self._latest)
        return model


class TrainingCheckPoint(TrainingCallback):
    '''Checkpointing operation.

    .. versionadded:: 1.3.0

    Parameters
    ----------

    directory : os.PathLike
        Output model directory.
    name : str
        pattern of output model file.  Models will be saved as name_0.json, name_1.json,
        name_2.json ....
    as_pickle : boolean
        When set to Ture, all training parameters will be saved in pickle format, instead
        of saving only the model.
    iterations : int
        Interval of checkpointing.  Checkpointing is slow so setting a larger number can
        reduce performance hit.

    '''
    def __init__(self, directory: os.PathLike, name: str = 'model',
                 as_pickle=False, iterations: int = 100):
        self._path = directory
        self._name = name
        self._as_pickle = as_pickle
        self._iterations = iterations
        self._epoch = 0
        super().__init__()

    def after_iteration(self, model, epoch, evals_log):
        if self._epoch == self._iterations:
            path = os.path.join(self._path, self._name + '_' + str(epoch) +
                                ('.pkl' if self._as_pickle else '.json'))
            self._epoch = 0
            if rabit.get_rank() == 0:
                if self._as_pickle:
                    with open(path, 'wb') as fd:
                        pickle.dump(model, fd)
                else:
                    model.save_model(path)
        self._epoch += 1


class LegacyCallbacks:
    '''Adapter for legacy callback functions.

    .. versionadded:: 1.3.0

    Parameters
    ----------

    callbacks : Sequence
        A sequence of legacy callbacks (callbacks that are not instance of
        TrainingCallback)
    start_iteration : int
        Begining iteration.
    end_iteration : int
        End iteration, normally is the number of boosting rounds.
    evals : Sequence
        Sequence of evaluation dataset tuples.
    feval : Custom evaluation metric.
    '''
    def __init__(self, callbacks, start_iteration, end_iteration,
                 feval, cvfolds=None):
        self.callbacks_before_iter = [
            cb for cb in callbacks
            if cb.__dict__.get('before_iteration', False)]
        self.callbacks_after_iter = [
            cb for cb in callbacks
            if not cb.__dict__.get('before_iteration', False)]

        self.start_iteration = start_iteration
        self.end_iteration = end_iteration
        self.cvfolds = cvfolds

        self.feval = feval
        assert self.feval is None or callable(self.feval)

        if cvfolds is not None:
            self.aggregated_cv = None

        super().__init__()

    def before_training(self, model):
        '''Nothing to do for legacy callbacks'''
        return model

    def after_training(self, model):
        '''Nothing to do for legacy callbacks'''
        return model

    def before_iteration(self, model, epoch, dtrain, evals):
        '''Called before each iteration.'''
        for cb in self.callbacks_before_iter:
            rank = rabit.get_rank()
            cb(CallbackEnv(model=model,
                           cvfolds=self.cvfolds,
                           iteration=epoch,
                           begin_iteration=self.start_iteration,
                           end_iteration=self.end_iteration,
                           rank=rank,
                           evaluation_result_list=None))
        return False

    def after_iteration(self, model, epoch, dtrain, evals):
        '''Called after each iteration.'''
        evaluation_result_list = []
        if self.cvfolds is not None:
            scores = model.eval(epoch, self.feval)
            self.aggregated_cv = _aggcv(scores)
            evaluation_result_list = self.aggregated_cv

        if evals:
            # When cv is used, evals are embedded into folds.
            assert self.cvfolds is None
            bst_eval_set = model.eval_set(evals, epoch, self.feval)
            if isinstance(bst_eval_set, STRING_TYPES):
                msg = bst_eval_set
            else:
                msg = bst_eval_set.decode()
            res = [x.split(':') for x in msg.split()]
            evaluation_result_list = [(k, float(v)) for k, v in res[1:]]

        try:
            for cb in self.callbacks_after_iter:
                rank = rabit.get_rank()
                cb(CallbackEnv(model=model,
                               cvfolds=self.cvfolds,
                               iteration=epoch,
                               begin_iteration=self.start_iteration,
                               end_iteration=self.end_iteration,
                               rank=rank,
                               evaluation_result_list=evaluation_result_list))
        except EarlyStopException:
            return True

        return False
