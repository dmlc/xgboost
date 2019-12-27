# coding: utf-8
# pylint: disable=invalid-name, too-many-statements
"""Training Library containing training routines."""

from . import rabit
from .core import EarlyStopException


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
    raise ValueError("wrong metric value")


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
    stopp_rounds : int
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
