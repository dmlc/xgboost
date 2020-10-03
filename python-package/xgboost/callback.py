# coding: utf-8
# pylint: disable=invalid-name, too-many-statements, no-self-use
# pylint: disable=too-many-arguments
"""Training Library containing training routines."""
from abc import ABC
import collections
import os
import numpy

from . import rabit
from .core import EarlyStopException, DMatrix, CallbackEnv
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
            return  '{0}:{1:.5f}+{2:.5f}'.format(value[0], value[1], value[2])
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


#
# The new implementation of callback functions.
#
# TODOs
# - eval_set
# - cv
# - tests
# - doc
# - enforced best_xxx
# - merged functionality of es and mon.

# pylint: disable=unused-argument
class TrainingCallback(ABC):
    '''Interface for training callback.

    .. versionadded:: 1.3.0

    '''
    def __init__(self):
        self.history = {}

    def before_training(self, model):
        '''Run before training starts.'''

    def after_training(self, model):
        '''Run after training is finished.'''

    def before_iteration(self, model, epoch, dtrain, evals):
        '''Run before each iteration.'''
        return False

    def after_iteration(self, model, epoch, dtrain, evals):
        '''Run after each iteration.'''
        return False


class CallbackContainer(TrainingCallback):
    '''A container for list of callbacks.

    .. versionadded:: 1.3.0

    '''
    def __init__(self, callbacks):
        self.callbacks = callbacks
        super().__init__()

    def before_training(self, model):
        '''Function called before training.'''
        for c in self.callbacks:
            c.before_training(model)

    def after_training(self, model):
        '''Function called after training.'''
        for c in self.callbacks:
            c.after_training(model)

    def before_iteration(self, model, epoch, dtrain, evals):
        '''Function called before training iteration.'''
        return any(c.before_iteration(model, epoch, dtrain, evals)
                   for c in self.callbacks)

    def after_iteration(self, model, epoch, dtrain, evals):
        '''Function called after training iteration.'''
        return any(c.after_iteration(model, epoch, dtrain, evals)
                   for c in self.callbacks)


class LearningRateScheduler(TrainingCallback):
    '''Callback function for scheduling learning rate.

    .. versionadded:: 1.3.0

    Parameters
    ----------

    learning_rates : callable/collections.Sequence
        If it's a callable object, then it should accept an integer parameter
        `epoch` and returns the corresponding learning rate.  Otherwise it
        shoule be a sequence like list or tuple with the same size of boosting
        rounds.

    '''
    def __init__(self, learning_rates):
        assert callable(learning_rates) or \
            isinstance(learning_rates, collections.Sequence)
        if callable(learning_rates):
            self.learning_rates = learning_rates
        else:
            self.learning_rates = lambda epoch: learning_rates[epoch]
        super().__init__()

    def after_iteration(self, model, epoch, dtrain, evals):
        model.set_param('learning_rate', self.learning_rates(epoch))


def _allreduce_metric(score, maximize):
    score = numpy.array([score])
    if maximize:
        score = rabit.allreduce(score, rabit.Op.MAX)
    else:
        score = rabit.allreduce(score, rabit.Op.MIN)
    return score


# pylint: disable=too-many-instance-attributes
class EarlyStopping(TrainingCallback):
    ''' Callback function for early stopping

    .. versionadded:: 1.3.0

    Parameters
    ----------
    data
        data for evaluation.
    name : str
        Name of data.
    metric : str/callable
        Name of metric.  Use the default metric if not specified.
    metric_name : str
        Name of metric, used when metric is a callable object.
    rounds : int
        Early stopping rounds.
    maximize : bool
        Whether to maximize evaluation metric.
    missing : float
        Same as missing for DMatrix, used when input is not a DMatrix.
    wegiht
        Same as label for DMatrix, used when input is not a DMatrix.
    label
        Same as weight for DMatrix, used when input is not a DMatrix.
    '''
    def __init__(self, name, rounds, metric=None, metric_name='metric',
                 maximize=False):
        self.name = name
        self.metric = metric
        self.rounds = rounds
        if callable(self.metric):
            self.metric_name = metric_name
        else:
            self.metric_name = self.metric
        self.maximize = maximize

        if self.maximize:
            self.improve_op = lambda x, y: x > y
        else:
            self.improve_op = lambda x, y: x < y

        self.current_rounds = 0
        self.best_scores = {}
        super().__init__()

    def before_training(self, model):
        if not callable(self.metric):
            model.set_param({'eval_metric': self.metric})

    def after_training(self, model):
        model.best_iteration = self.rounds
        model.set_attr(best_iteration=str(self.rounds))

    def _update_rounds(self, scores, model, epoch):
        assert len(scores) == 1
        score = scores[0]
        metric, s = score[0], score[1]
        if not self.history:    # First round
            self.current_rounds = 0
            self.history[self.name] = {}
            self.history[self.name][metric] = [s]
            self.best_scores[self.name] = {}
            self.best_scores[self.name][metric] = [s]
        elif not self.improve_op(s, self.best_scores[self.name][metric][-1]):
            # Not improved
            self.history[self.name][metric].append(s)
            self.current_rounds += 1
        else:                   # Improved
            self.history[self.name][metric].append(s)
            self.best_scores[self.name][metric].append(s)
            record = self.history[self.name][metric][-1]
            model.set_attr(best_score=str(record),
                           best_iteration=str(epoch))
            self.current_rounds = 0  # reset

        if self.current_rounds >= self.rounds:
            return True
        return False

    def after_iteration(self, model, epoch, dtrain, evals):
        assert not rabit.is_distributed(), '''
Use distributed version instead.  For dask users:

>>>  from xgboost.dask import EarlyStopping
'''
        msg = 'Must have at least 1 validation dataset for early stopping.'
        assert len(evals) >= 1, msg
        stopping_data = evals[-1][1]
        if callable(self.metric):
            label = stopping_data.get_label()
            predt = model.predict(stopping_data)
            score = self.metric(label, predt)
            score = _allreduce_metric(score, self.maximize)
            score = [(self.metric_name, score)]
        else:
            score = model.eval(stopping_data)
            score = [s.split(':') for s in score.split()]
            score = [(k, float(v)) for k, v in score[1:]]

        return self._update_rounds(score, model, epoch)


class EvaluationMonitor(TrainingCallback):
    '''Print the evaluation result at each iteration.

    .. versionadded:: 1.3.0

    Parameters
    ----------

    data
        Data for evaluation.
    name : str
        Name of data.
    metric : str
        Name of metric
    rank : int
        Which worker should be used for printing the result.
    missing : float
        Used when data is not a DMatrix.
    weight
        Used when data is not a DMatrix.
    label
        Used when data is not a DMatrix.
    '''
    def __init__(self, name, metric=None, rank=0):
        self.name = name
        self.metric = metric
        self.printer_rank = rank
        super().__init__()

    def before_training(self, model):
        model.set_param({'eval_metric': self.metric})

    def _update_history(self, score, epoch):
        score = [s.split(':') for s in score.split()]
        score = [(k, float(v)) for k, v in score[1:]]

        if rabit.get_rank() == self.printer_rank:
            msg = _fmt_metric(score[0])
            rabit.tracker_print('[%d]\t%s\n' % (epoch, msg))

        def metric_name():
            if self.metric:
                return self.metric
            name = score[0][0]
            pos = name.index('-')
            name = name[pos+1:]
            return name

        if not self.history:
            self.history[metric_name()] = [score[0][1]]
        else:
            self.history[metric_name()].append(score[0][1])

        return False

    def after_iteration(self, model, epoch, dtrain, evals):
        assert not rabit.is_distributed()
        score = model.eval(self.data, self.name)
        return self._update_history(score, epoch)


class TrainingCheckPoint(TrainingCallback):
    '''Checkpointing operation.

    .. versionadded:: 1.3.0

    Parameters
    ----------

    path : os.PathLike
        Output model path.
    iterations : int
        Interval of checkpointing.
    '''
    def __init__(self, path: os.PathLike, iterations=10):
        self._path = path
        self._iterations = iterations
        self._epoch = 0

    def after_iteration(self, model, epoch, dtrain, evals):
        self._epoch += 1
        if self._epoch == 10:
            self._epoch = 0
            if rabit.get_rank() == 0:
                model.save_model(self._path)


class LegacyCallbacks(TrainingCallback):
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
                 evals, feval):
        self.callbacks_before_iter = [
            cb for cb in callbacks
            if cb.__dict__.get('before_iteration', False)]
        self.callbacks_after_iter = [
            cb for cb in callbacks
            if not cb.__dict__.get('before_iteration', False)]

        self.start_iteration = start_iteration
        self.end_iteration = end_iteration

        self.evals = evals
        self.feval = feval
        assert self.feval is None or callable(self.feval)

        super().__init__()

    def before_iteration(self, model, epoch, dtrain, evals):
        for cb in self.callbacks_before_iter:
            rank = rabit.get_rank()
            cb(CallbackEnv(model=model,
                           cvfolds=None,
                           iteration=epoch,
                           begin_iteration=self.start_iteration,
                           end_iteration=self.end_iteration,
                           rank=rank,
                           evaluation_result_list=None))
        return False

    def after_iteration(self, model, epoch, dtrain, evals):
        evaluation_result_list = []
        if self.evals:
            bst_eval_set = model.eval_set(self.evals, epoch, self.feval)
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
                               cvfolds=None,
                               iteration=epoch,
                               begin_iteration=self.start_iteration,
                               end_iteration=self.end_iteration,
                               rank=rank,
                               evaluation_result_list=evaluation_result_list))
        except EarlyStopException:
            return True

        return False
