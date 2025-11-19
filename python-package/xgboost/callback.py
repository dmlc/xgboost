"""Callback library containing training routines.  See :doc:`Callback Functions
</python/callbacks>` for a quick introduction.

"""

import collections
import os
import pickle
from abc import ABC
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
    TypeVar,
    Union,
    cast,
)

import numpy

from . import collective
from ._typing import EvalsLog, _ScoreList
from .core import (
    Booster,
    DMatrix,
    XGBoostError,
    _deprecate_positional_args,
    _parse_eval_str,
)

__all__ = [
    "TrainingCallback",
    "LearningRateScheduler",
    "EarlyStopping",
    "EvaluationMonitor",
    "TrainingCheckPoint",
    "CallbackContainer",
]

_Score = Union[float, Tuple[float, float]]

_Model = Any  # real type is Union[Booster, CVPack]; need more work


# pylint: disable=unused-argument
class TrainingCallback(ABC):
    """Interface for training callback.

    .. versionadded:: 1.3.0

    """

    # pylint: disable=invalid-name
    EvalsLog: TypeAlias = EvalsLog

    def __init__(self) -> None:
        pass

    def before_training(self, model: _Model) -> _Model:
        """Run before training starts."""
        return model

    def after_training(self, model: _Model) -> _Model:
        """Run after training is finished."""
        return model

    def before_iteration(self, model: _Model, epoch: int, evals_log: EvalsLog) -> bool:
        """Run before each iteration.  Returns True when training should stop. See
        :py:meth:`after_iteration` for details.

        """
        return False

    def after_iteration(self, model: _Model, epoch: int, evals_log: EvalsLog) -> bool:
        """Run after each iteration.  Returns `True` when training should stop.

        Parameters
        ----------

        model :
            Eeither a :py:class:`~xgboost.Booster` object or a CVPack if the cv function
            in xgboost is being used.
        epoch :
            The current training iteration.
        evals_log :
            A dictionary containing the evaluation history:

            .. code-block:: python

                {"data_name": {"metric_name": [0.5, ...]}}

        """
        return False


def _aggcv(rlist: List[str]) -> List[Tuple[str, float, float]]:
    # pylint: disable=invalid-name, too-many-locals
    """Aggregate cross-validation results."""
    cvmap: Dict[Tuple[int, str], List[float]] = {}
    idx = rlist[0].split()[0]
    for line in rlist:
        arr: List[str] = line.split()
        assert idx == arr[0]
        for metric_idx, it in enumerate(arr[1:]):
            if not isinstance(it, str):
                it = it.decode()
            k, v = it.split(":")
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
    """Helper function for computing customized metric in distributed
    environment.  Not strictly correct as many functions don't use mean value
    as final result.

    """
    world = collective.get_world_size()
    assert world != 0
    if world == 1:
        return score
    if isinstance(score, tuple):  # has mean and stdv
        raise ValueError(
            "xgboost.cv function should not be used in distributed environment."
        )
    arr = numpy.array([score])
    arr = collective.allreduce(arr, collective.Op.SUM) / world
    return arr[0]


class CallbackContainer:
    """A special internal callback for invoking a list of other callbacks.

    .. versionadded:: 1.3.0

    """

    def __init__(
        self,
        callbacks: Sequence[TrainingCallback],
        metric: Optional[Callable] = None,
        output_margin: bool = True,
        is_cv: bool = False,
    ) -> None:
        self.callbacks = list(dict.fromkeys(callbacks))
        for cb in callbacks:
            if not isinstance(cb, TrainingCallback):
                raise TypeError("callback must be an instance of `TrainingCallback`.")

        msg = (
            "metric must be callable object for monitoring.  For builtin metrics"
            ", passing them in training parameter invokes monitor automatically."
        )
        if metric is not None and not callable(metric):
            raise TypeError(msg)

        self.metric = metric
        self.history: EvalsLog = collections.OrderedDict()
        self._output_margin = output_margin
        self.is_cv = is_cv

        if self.is_cv:
            self.aggregated_cv: Optional[list[tuple[str, float, float]]] = None

    def before_training(self, model: _Model) -> _Model:
        """Function called before training."""
        for c in self.callbacks:
            model = c.before_training(model=model)
            msg = "before_training should return the model"
            if self.is_cv:
                assert isinstance(model.cvfolds, list), msg
            else:
                assert isinstance(model, Booster), msg
        return model

    def after_training(self, model: _Model) -> _Model:
        """Function called after training."""
        for c in self.callbacks:
            model = c.after_training(model=model)
            msg = "after_training should return the model"
            if self.is_cv:
                assert isinstance(model.cvfolds, list), msg
            else:
                assert isinstance(model, Booster), msg

        return model

    def before_iteration(
        self,
        model: _Model,
        epoch: int,
        dtrain: DMatrix,
        evals: Optional[List[Tuple[DMatrix, str]]],
    ) -> bool:
        """Function called before training iteration."""
        return any(
            c.before_iteration(model, epoch, self.history) for c in self.callbacks
        )

    def _update_history(
        self,
        score: Union[List[Tuple[str, float]], List[Tuple[str, float, float]]],
        epoch: int,
    ) -> None:
        for d in score:
            name: str = d[0]
            s: float = d[1]
            if self.is_cv:
                std = float(cast(Tuple[str, float, float], d)[2])
                x: _Score = (s, std)
            else:
                x = s
            splited_names = name.split("-")
            data_name = splited_names[0]
            metric_name = "-".join(splited_names[1:])
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
        """Function called after training iteration."""
        if self.is_cv:
            scores = model.eval(epoch, self.metric, self._output_margin)
            scores = _aggcv(scores)
            self.aggregated_cv = scores
            self._update_history(scores, epoch)
        else:
            evals = [] if evals is None else evals
            for _, name in evals:
                assert name.find("-") == -1, "Dataset name should not contain `-`"
            score: str = model.eval_set(evals, epoch, self.metric, self._output_margin)
            metric_score = _parse_eval_str(score)
            self._update_history(metric_score, epoch)
        ret = any(c.after_iteration(model, epoch, self.history) for c in self.callbacks)
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
        if not callable(learning_rates) and not isinstance(
            learning_rates, collections.abc.Sequence
        ):
            raise TypeError(
                "Invalid learning rates, expecting callable or sequence, got: "
                f"{type(learning_rates)}"
            )

        if callable(learning_rates):
            self.learning_rates = learning_rates
        else:
            self.learning_rates = lambda epoch: cast(Sequence, learning_rates)[epoch]
        super().__init__()

    def after_iteration(self, model: _Model, epoch: int, evals_log: EvalsLog) -> bool:
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
        Whether training should return the best model or the last model. If set to
        `True`, it will only keep the boosting rounds up to the detected best iteration,
        discarding the ones that come after. This is only supported with tree methods
        (not `gblinear`). Also, the `cv` function doesn't return a model, the parameter
        is not applicable.
    min_delta :

        .. versionadded:: 1.5.0

        Minimum absolute change in score to be qualified as an improvement.

    Examples
    --------

    .. code-block:: python

        es = xgboost.callback.EarlyStopping(
            rounds=2,
            min_delta=1e-3,
            save_best=True,
            maximize=False,
            data_name="validation_0",
            metric_name="mlogloss",
        )
        clf = xgboost.XGBClassifier(tree_method="hist", device="cuda", callbacks=[es])

        X, y = load_digits(return_X_y=True)
        clf.fit(X, y, eval_set=[(X, y)])
    """

    # pylint: disable=too-many-arguments
    @_deprecate_positional_args
    def __init__(
        self,
        *,
        rounds: int,
        metric_name: Optional[str] = None,
        data_name: Optional[str] = None,
        maximize: Optional[bool] = None,
        save_best: Optional[bool] = False,
        min_delta: float = 0.0,
    ) -> None:
        self.data = data_name
        self.metric_name = metric_name
        self.rounds = rounds
        self.save_best = save_best
        self.maximize = maximize
        self.stopping_history: EvalsLog = {}
        self._min_delta = min_delta
        if self._min_delta < 0:
            raise ValueError("min_delta must be greater or equal to 0.")

        self.current_rounds: int = 0
        self.best_scores: dict = {}
        self.starting_round: int = 0
        super().__init__()

    def before_training(self, model: _Model) -> _Model:
        self.starting_round = model.num_boosted_rounds()
        if not isinstance(model, Booster) and self.save_best:
            raise ValueError(
                "`save_best` is not applicable to the `cv` function as it doesn't"
                " return a model."
            )
        return model

    def _update_rounds(
        self, *, score: _Score, name: str, metric: str, model: _Model, epoch: int
    ) -> bool:
        def get_s(value: _Score) -> float:
            """get score if it's cross validation history."""
            return value[0] if isinstance(value, tuple) else value

        def maximize(new: _Score, best: _Score) -> bool:
            """New score should be greater than the old one."""
            return numpy.greater(get_s(new) - self._min_delta, get_s(best))

        def minimize(new: _Score, best: _Score) -> bool:
            """New score should be lesser than the old one."""
            return numpy.greater(get_s(best) - self._min_delta, get_s(new))

        if self.maximize is None:
            # Just to be compatibility with old behavior before 1.3.  We should let
            # user to decide.
            maximize_metrics = (
                "auc",
                "aucpr",
                "pre",
                "pre@",
                "map",
                "ndcg",
                "auc@",
                "aucpr@",
                "map@",
                "ndcg@",
            )
            if metric != "mape" and any(metric.startswith(x) for x in maximize_metrics):
                self.maximize = True
            else:
                self.maximize = False

        if self.maximize:
            improve_op = maximize
        else:
            improve_op = minimize

        if not self.stopping_history:  # First round
            self.current_rounds = 0
            self.stopping_history[name] = {}
            self.stopping_history[name][metric] = cast(_ScoreList, [score])
            self.best_scores[name] = {}
            self.best_scores[name][metric] = [score]
            model.set_attr(best_score=str(get_s(score)), best_iteration=str(epoch))
        elif not improve_op(score, self.best_scores[name][metric][-1]):
            # Not improved
            self.stopping_history[name][metric].append(score)  # type: ignore
            self.current_rounds += 1
        else:  # Improved
            self.stopping_history[name][metric].append(score)  # type: ignore
            self.best_scores[name][metric].append(score)
            record = self.stopping_history[name][metric][-1]
            model.set_attr(best_score=str(get_s(record)), best_iteration=str(epoch))
            self.current_rounds = 0  # reset

        if self.current_rounds >= self.rounds:
            # Should stop
            return True
        return False

    def after_iteration(self, model: _Model, epoch: int, evals_log: EvalsLog) -> bool:
        epoch += self.starting_round  # training continuation
        msg = "Must have at least 1 validation dataset for early stopping."
        if len(evals_log.keys()) < 1:
            raise ValueError(msg)

        # Get data name
        if self.data:
            data_name = self.data
        else:
            # Use the last one as default.
            data_name = list(evals_log.keys())[-1]
        if data_name not in evals_log:
            raise ValueError(f"No dataset named: {data_name}")

        if not isinstance(data_name, str):
            raise TypeError(
                f"The name of the dataset should be a string. Got: {type(data_name)}"
            )
        data_log = evals_log[data_name]

        # Get metric name
        if self.metric_name:
            metric_name = self.metric_name
        else:
            # Use last metric by default.
            metric_name = list(data_log.keys())[-1]
        if metric_name not in data_log:
            raise ValueError(f"No metric named: {metric_name}")

        # The latest score
        score = data_log[metric_name][-1]
        return self._update_rounds(
            score=score, name=data_name, metric=metric_name, model=model, epoch=epoch
        )

    def after_training(self, model: _Model) -> _Model:
        if not self.save_best:
            return model

        try:
            best_iteration = model.best_iteration
            best_score = model.best_score
            assert best_iteration is not None and best_score is not None
            model = model[: best_iteration + 1]
            model.best_iteration = best_iteration
            model.best_score = best_score
        except XGBoostError as e:
            raise XGBoostError(
                "`save_best` is not applicable to the current booster"
            ) from e

        return model


class EvaluationMonitor(TrainingCallback):
    """Print the evaluation result at each iteration.

    .. versionadded:: 1.3.0

    Parameters
    ----------

    rank :
        Which worker should be used for printing the result.
    period :
        How many epoches between printing.
    show_stdv :
        Used in cv to show standard deviation.  Users should not specify it.
    logger :
        A callable used for logging evaluation result.

    """

    def __init__(
        self,
        rank: int = 0,
        period: int = 1,
        show_stdv: bool = False,
        logger: Callable[[str], None] = collective.communicator_print,
    ):
        self.printer_rank = rank
        self.show_stdv = show_stdv
        self.period = period
        self._logger = logger
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

    def after_iteration(self, model: _Model, epoch: int, evals_log: EvalsLog) -> bool:
        if not evals_log:
            return False

        msg: str = f"[{epoch}]"
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
            msg += "\n"

            if (epoch % self.period) == 0 or self.period == 1:
                self._logger(msg)
                self._latest = None
            else:
                # There is skipped message
                self._latest = msg
        return False

    def after_training(self, model: _Model) -> _Model:
        if collective.get_rank() == self.printer_rank and self._latest is not None:
            self._logger(self._latest)
        return model


class TrainingCheckPoint(TrainingCallback):
    """Checkpointing operation. Users are encouraged to create their own callbacks for
    checkpoint as XGBoost doesn't handle distributed file systems. When checkpointing on
    distributed systems, be sure to know the rank of the worker to avoid multiple
    workers checkpointing to the same place.

    .. versionadded:: 1.3.0

    Since XGBoost 2.1.0, the default format is changed to UBJSON.

    Parameters
    ----------

    directory :
        Output model directory.
    name :
        pattern of output model file.  Models will be saved as name_0.ubj, name_1.ubj,
        name_2.ubj ....
    as_pickle :
        When set to True, all training parameters will be saved in pickle format,
        instead of saving only the model.
    interval :
        Interval of checkpointing.  Checkpointing is slow so setting a larger number can
        reduce performance hit.

    """

    default_format = "ubj"

    def __init__(
        self,
        directory: Union[str, os.PathLike],
        name: str = "model",
        as_pickle: bool = False,
        interval: int = 100,
    ) -> None:
        self._path = os.fspath(directory)
        self._name = name
        self._as_pickle = as_pickle
        self._iterations = interval
        self._epoch = 0  # counter for iterval
        self._start = 0  # beginning iteration
        super().__init__()

    def before_training(self, model: _Model) -> _Model:
        self._start = model.num_boosted_rounds()
        return model

    def after_iteration(self, model: _Model, epoch: int, evals_log: EvalsLog) -> bool:
        if self._epoch == self._iterations:
            path = os.path.join(
                self._path,
                self._name
                + "_"
                + (str(epoch + self._start))
                + (".pkl" if self._as_pickle else f".{self.default_format}"),
            )
            self._epoch = 0  # reset counter
            if collective.get_rank() == 0:
                # checkpoint using the first worker
                if self._as_pickle:
                    with open(path, "wb") as fd:
                        pickle.dump(model, fd)
                else:
                    model.save_model(path)
        self._epoch += 1
        return False
