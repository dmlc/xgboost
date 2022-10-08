# coding: utf-8
# pylint: disable=invalid-name, too-many-statements
# pylint: disable=too-many-arguments
"""Callback library containing training routines.  See :doc:`Callback Functions
</python/callbacks>` for a quick introduction.

"""

from abc import ABC
import collections
import os
import pickle
from typing import Callable, List, Optional, Union, Dict, Tuple, TypeVar, cast, Sequence, Any
import numpy

from . import collective
from .core import Booster, DMatrix, XGBoostError, _get_booster_layer_trees


__all__ = [
    "TrainingCallback",
    "LearningRateScheduler",
    "EarlyStopping",
    "EvaluationMonitor",
    "TrainingCheckPoint",
    "CallbackContainer"
]

_Score = Union[float, Tuple[float, float]]
_ScoreList = Union[List[float], List[Tuple[float, float]]]

_Model = Any  # real type is Union[Booster, CVPack]; need more work


# pylint: disable=unused-argument
class TrainingCallback(ABC):
    '''Interface for training callback.

    .. versionadded:: 1.3.0

    '''

    EvalsLog = Dict[str, Dict[str, _ScoreList]]

    def __init__(self) -> None:
        pass

    def before_training(self, model: _Model) -> _Model:
        '''Run before training starts.'''
        return model

    def after_training(self, model: _Model) -> _Model:
        '''Run after training is finished.'''
        return model

    def before_iteration(self, model: _Model, epoch: int, evals_log: EvalsLog) -> bool:
        '''Run before each iteration.  Return True when training should stop.'''
        return False

    def after_iteration(self, model: _Model, epoch: int, evals_log: EvalsLog) -> bool:
        '''Run after each iteration.  Return True when training should stop.'''
        return False


def _aggcv(rlist: List[str]) -> List[Tuple[str, float, float]]:
    # pylint: disable=invalid-name, too-many-locals
    """Aggregate cross-validation results.

    """
    cvmap: Dict[Tuple[int, str], List[float]] = {}
    idx = rlist[0].split()[0]
    for line in rlist:
        arr: List[str] = line.split()
        assert idx == arr[0]
        for metric_idx, it in enumerate(arr[1:]):
            if not isinstance(it, str):
                it = it.decode()
            k, v = it.split(':')
            if (metric_idx, k) not in cvmap:
                cvmap[(metric_idx, k)] = []
            cvmap[(metric_idx, k)].append(float(v))
    msg = idx
    results = []
    for (_, name), s in sorted(cvmap.items(), key=lambda x: x[0][0]):
        as_arr = numpy.array(s)
        if not isinstance(msg, str):
            msg = msg.decode()
        mean, std = numpy.mean(as_arr), numpy.std(as_arr)
        results.extend([(name, mean, std)])
    return results


# allreduce type
_ART = TypeVar("_ART")


def _allreduce_metric(score: _ART) -> _ART:
    '''Helper function for computing customized metric in distributed
    environment.  Not strictly correct as many functions don't use mean value
    as final result.

    '''
    world = collective.get_world_size()
    assert world != 0
    if world == 1:
        return score
    if isinstance(score, tuple):  # has mean and stdv
        raise ValueError(
            'xgboost.cv function should not be used in distributed environment.')
    arr = numpy.array([score])
    arr = collective.allreduce(arr, collective.Op.SUM) / world
    return arr[0]


class CallbackContainer:
    '''A special internal callback for invoking a list of other callbacks.

    .. versionadded:: 1.3.0

    '''

    EvalsLog = TrainingCallback.EvalsLog

    def __init__(
        self,
        callbacks: Sequence[TrainingCallback],
        metric: Callable = None,
        output_margin: bool = True,
        is_cv: bool = False
    ) -> None:
        self.callbacks = set(callbacks)
        if metric is not None:
            msg = 'metric must be callable object for monitoring.  For ' + \
                'builtin metrics, passing them in training parameter' + \
                ' will invoke monitor automatically.'
            assert callable(metric), msg
        self.metric = metric
        self.history: TrainingCallback.EvalsLog = collections.OrderedDict()
        self._output_margin = output_margin
        self.is_cv = is_cv

        if self.is_cv:
            self.aggregated_cv = None

    def before_training(self, model: _Model) -> _Model:
        '''Function called before training.'''
        for c in self.callbacks:
            model = c.before_training(model=model)
            msg = 'before_training should return the model'
            if self.is_cv:
                assert isinstance(model.cvfolds, list), msg
            else:
                assert isinstance(model, Booster), msg
        return model

    def after_training(self, model: _Model) -> _Model:
        '''Function called after training.'''
        for c in self.callbacks:
            model = c.after_training(model=model)
            msg = 'after_training should return the model'
            if self.is_cv:
                assert isinstance(model.cvfolds, list), msg
            else:
                assert isinstance(model, Booster), msg

        if not self.is_cv:
            num_parallel_tree, _ = _get_booster_layer_trees(model)
            if model.attr('best_score') is not None:
                model.best_score = float(cast(str, model.attr('best_score')))
                model.best_iteration = int(cast(str, model.attr('best_iteration')))
                # num_class is handled internally
                model.set_attr(
                    best_ntree_limit=str((model.best_iteration + 1) * num_parallel_tree)
                )
                model.best_ntree_limit = int(cast(str, model.attr("best_ntree_limit")))
            else:
                # Due to compatibility with version older than 1.4, these attributes are
                # added to Python object even if early stopping is not used.
                model.best_iteration = model.num_boosted_rounds() - 1
                model.set_attr(best_iteration=str(model.best_iteration))
                model.best_ntree_limit = (model.best_iteration + 1) * num_parallel_tree
                model.set_attr(best_ntree_limit=str(model.best_ntree_limit))

        return model

    def before_iteration(
        self, model: _Model, epoch: int, dtrain: DMatrix, evals: Optional[List[Tuple[DMatrix, str]]]
    ) -> bool:
        '''Function called before training iteration.'''
        return any(c.before_iteration(model, epoch, self.history)
                   for c in self.callbacks)

    def _update_history(
        self,
        score: Union[List[Tuple[str, float]], List[Tuple[str, float, float]]],
        epoch: int
    ) -> None:
        for d in score:
            name: str = d[0]
            s: float = d[1]
            if self.is_cv:
                std = float(cast(Tuple[str, float, float], d)[2])
                x: _Score = (s, std)
            else:
                x = s
            splited_names = name.split('-')
            data_name = splited_names[0]
            metric_name = '-'.join(splited_names[1:])
            x = _allreduce_metric(x)
            if data_name not in self.history:
                self.history[data_name] = collections.OrderedDict()
            data_history = self.history[data_name]
            if metric_name not in data_history:
                data_history[metric_name] = cast(_ScoreList, [])
            metric_history = data_history[metric_name]
            if self.is_cv:
                cast(List[Tuple[float, float]], metric_history).append(
                    cast(Tuple[float, float], x)
                )
            else:
                cast(List[float], metric_history).append(cast(float, x))

    def after_iteration(
        self,
        model: _Model,
        epoch: int,
        dtrain: DMatrix,
        evals: Optional[List[Tuple[DMatrix, str]]],
    ) -> bool:
        '''Function called after training iteration.'''
        if self.is_cv:
            scores = model.eval(epoch, self.metric, self._output_margin)
            scores = _aggcv(scores)
            self.aggregated_cv = scores
            self._update_history(scores, epoch)
        else:
            evals = [] if evals is None else evals
            for _, name in evals:
                assert name.find('-') == -1, 'Dataset name should not contain `-`'
            score: str = model.eval_set(evals, epoch, self.metric, self._output_margin)
            splited = score.split()[1:]  # into datasets
            # split up `test-error:0.1234`
            metric_score_str = [tuple(s.split(':')) for s in splited]
            # convert to float
            metric_score = [(n, float(s)) for n, s in metric_score_str]
            self._update_history(metric_score, epoch)
        ret = any(c.after_iteration(model, epoch, self.history)
                  for c in self.callbacks)
        return ret


class LearningRateScheduler(TrainingCallback):
    """Callback function for scheduling learning rate.

    .. versionadded:: 1.3.0

    Parameters
    ----------

    learning_rates :
        If it's a callable object, then it should accept an integer parameter
        `epoch` and returns the corresponding learning rate.  Otherwise it
        should be a sequence like list or tuple with the same size of boosting
        rounds.

    """

    def __init__(
        self, learning_rates: Union[Callable[[int], float], Sequence[float]]
    ) -> None:
        assert callable(learning_rates) or isinstance(
            learning_rates, collections.abc.Sequence
        )
        if callable(learning_rates):
            self.learning_rates = learning_rates
        else:
            self.learning_rates = lambda epoch: cast(Sequence, learning_rates)[epoch]
        super().__init__()

    def after_iteration(
        self, model: _Model, epoch: int, evals_log: TrainingCallback.EvalsLog
    ) -> bool:
        model.set_param("learning_rate", self.learning_rates(epoch))
        return False


# pylint: disable=too-many-instance-attributes
class EarlyStopping(TrainingCallback):
    """Callback function for early stopping

    .. versionadded:: 1.3.0

    Parameters
    ----------
    rounds :
        Early stopping rounds.
    metric_name :
        Name of metric that is used for early stopping.
    data_name :
        Name of dataset that is used for early stopping.
    maximize :
        Whether to maximize evaluation metric.  None means auto (discouraged).
    save_best :
        Whether training should return the best model or the last model.
    min_delta :
        Minimum absolute change in score to be qualified as an improvement.

        .. versionadded:: 1.5.0

        .. code-block:: python

            clf = xgboost.XGBClassifier(tree_method="gpu_hist")
            es = xgboost.callback.EarlyStopping(
                rounds=2,
                abs_tol=1e-3,
                save_best=True,
                maximize=False,
                data_name="validation_0",
                metric_name="mlogloss",
            )

            X, y = load_digits(return_X_y=True)
            clf.fit(X, y, eval_set=[(X, y)], callbacks=[es])
    """
    def __init__(
        self,
        rounds: int,
        metric_name: Optional[str] = None,
        data_name: Optional[str] = None,
        maximize: Optional[bool] = None,
        save_best: Optional[bool] = False,
        min_delta: float = 0.0
    ) -> None:
        self.data = data_name
        self.metric_name = metric_name
        self.rounds = rounds
        self.save_best = save_best
        self.maximize = maximize
        self.stopping_history: TrainingCallback.EvalsLog = {}
        self._min_delta = min_delta
        if self._min_delta < 0:
            raise ValueError("min_delta must be greater or equal to 0.")

        self.current_rounds: int = 0
        self.best_scores: dict = {}
        self.starting_round: int = 0
        super().__init__()

    def before_training(self, model: _Model) -> _Model:
        self.starting_round = model.num_boosted_rounds()
        return model

    def _update_rounds(
        self, score: _Score, name: str, metric: str, model: _Model, epoch: int
    ) -> bool:
        def get_s(x: _Score) -> float:
            """get score if it's cross validation history."""
            return x[0] if isinstance(x, tuple) else x

        def maximize(new: _Score, best: _Score) -> bool:
            """New score should be greater than the old one."""
            return numpy.greater(get_s(new) - self._min_delta, get_s(best))

        def minimize(new: _Score, best: _Score) -> bool:
            """New score should be smaller than the old one."""
            return numpy.greater(get_s(best) - self._min_delta, get_s(new))

        if self.maximize is None:
            # Just to be compatibility with old behavior before 1.3.  We should let
            # user to decide.
            maximize_metrics = ('auc', 'aucpr', 'map', 'ndcg', 'auc@',
                                'aucpr@', 'map@', 'ndcg@')
            if metric != 'mape' and any(metric.startswith(x) for x in maximize_metrics):
                self.maximize = True
            else:
                self.maximize = False

        if self.maximize:
            improve_op = maximize
        else:
            improve_op = minimize

        assert improve_op

        if not self.stopping_history:  # First round
            self.current_rounds = 0
            self.stopping_history[name] = {}
            self.stopping_history[name][metric] = cast(_ScoreList, [score])
            self.best_scores[name] = {}
            self.best_scores[name][metric] = [score]
            model.set_attr(best_score=str(score), best_iteration=str(epoch))
        elif not improve_op(score, self.best_scores[name][metric][-1]):
            # Not improved
            self.stopping_history[name][metric].append(score)  # type: ignore
            self.current_rounds += 1
        else:  # Improved
            self.stopping_history[name][metric].append(score)  # type: ignore
            self.best_scores[name][metric].append(score)
            record = self.stopping_history[name][metric][-1]
            model.set_attr(best_score=str(record), best_iteration=str(epoch))
            self.current_rounds = 0  # reset

        if self.current_rounds >= self.rounds:
            # Should stop
            return True
        return False

    def after_iteration(self, model: _Model, epoch: int,
                        evals_log: TrainingCallback.EvalsLog) -> bool:
        epoch += self.starting_round  # training continuation
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

    def after_training(self, model: _Model) -> _Model:
        try:
            if self.save_best:
                model = model[: int(model.attr("best_iteration")) + 1]
        except XGBoostError as e:
            raise XGBoostError(
                "`save_best` is not applicable to current booster"
            ) from e
        return model


class EvaluationMonitor(TrainingCallback):
    '''Print the evaluation result at each iteration.

    .. versionadded:: 1.3.0

    Parameters
    ----------

    metric :
        Extra user defined metric.
    rank :
        Which worker should be used for printing the result.
    period :
        How many epoches between printing.
    show_stdv :
        Used in cv to show standard deviation.  Users should not specify it.
    '''
    def __init__(self, rank: int = 0, period: int = 1, show_stdv: bool = False) -> None:
        self.printer_rank = rank
        self.show_stdv = show_stdv
        self.period = period
        assert period > 0
        # last error message, useful when early stopping and period are used together.
        self._latest: Optional[str] = None
        super().__init__()

    def _fmt_metric(
        self, data: str, metric: str, score: float, std: Optional[float]
    ) -> str:
        if std is not None and self.show_stdv:
            msg = f"\t{data + '-' + metric}:{score:.5f}+{std:.5f}"
        else:
            msg = f"\t{data + '-' + metric}:{score:.5f}"
        return msg

    def after_iteration(self, model: _Model, epoch: int,
                        evals_log: TrainingCallback.EvalsLog) -> bool:
        if not evals_log:
            return False

        msg: str = f'[{epoch}]'
        if collective.get_rank() == self.printer_rank:
            for data, metric in evals_log.items():
                for metric_name, log in metric.items():
                    stdv: Optional[float] = None
                    if isinstance(log[-1], tuple):
                        score = log[-1][0]
                        stdv = log[-1][1]
                    else:
                        score = log[-1]
                    msg += self._fmt_metric(data, metric_name, score, stdv)
            msg += '\n'

            if (epoch % self.period) == 0 or self.period == 1:
                collective.communicator_print(msg)
                self._latest = None
            else:
                # There is skipped message
                self._latest = msg
        return False

    def after_training(self, model: _Model) -> _Model:
        if collective.get_rank() == self.printer_rank and self._latest is not None:
            collective.communicator_print(self._latest)
        return model


class TrainingCheckPoint(TrainingCallback):
    '''Checkpointing operation.

    .. versionadded:: 1.3.0

    Parameters
    ----------

    directory :
        Output model directory.
    name :
        pattern of output model file.  Models will be saved as name_0.json, name_1.json,
        name_2.json ....
    as_pickle :
        When set to True, all training parameters will be saved in pickle format, instead
        of saving only the model.
    iterations :
        Interval of checkpointing.  Checkpointing is slow so setting a larger number can
        reduce performance hit.

    '''
    def __init__(
        self,
        directory: Union[str, os.PathLike],
        name: str = 'model',
        as_pickle: bool = False,
        iterations: int = 100
    ) -> None:
        self._path = os.fspath(directory)
        self._name = name
        self._as_pickle = as_pickle
        self._iterations = iterations
        self._epoch = 0
        super().__init__()

    def after_iteration(self, model: _Model, epoch: int,
                        evals_log: TrainingCallback.EvalsLog) -> bool:
        if self._epoch == self._iterations:
            path = os.path.join(self._path, self._name + '_' + str(epoch) +
                                ('.pkl' if self._as_pickle else '.json'))
            self._epoch = 0
            if collective.get_rank() == 0:
                if self._as_pickle:
                    with open(path, 'wb') as fd:
                        pickle.dump(model, fd)
                else:
                    model.save_model(path)
        self._epoch += 1
        return False
