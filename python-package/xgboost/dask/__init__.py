# pylint: disable=too-many-arguments, too-many-locals
# pylint: disable=missing-class-docstring, invalid-name
# pylint: disable=too-many-lines
"""
Dask extensions for distributed training
----------------------------------------

See :doc:`Distributed XGBoost with Dask </tutorials/dask>` for simple tutorial.  Also
:doc:`/python/dask-examples/index` for some examples.

There are two sets of APIs in this module, one is the functional API including
``train`` and ``predict`` methods.  Another is stateful Scikit-Learner wrapper
inherited from single-node Scikit-Learn interface.

The implementation is heavily influenced by dask_xgboost:
https://github.com/dask/dask-xgboost

Optional dask configuration
===========================

- **coll_cfg**:
    Specify the scheduler address along with communicator configurations. This can be
    used as a replacement of the existing global Dask configuration
    `xgboost.scheduler_address` (see below). See :ref:`tracker-ip` for more info. The
    `tracker_host_ip` should specify the IP address of the Dask scheduler node.

  .. versionadded:: 3.0.0

  .. code-block:: python

    from xgboost import dask as dxgb
    from xgboost.collective import Config

    coll_cfg = Config(
        retry=1, timeout=20, tracker_host_ip="10.23.170.98", tracker_port=0
    )

    clf = dxgb.DaskXGBClassifier(coll_cfg=coll_cfg)
    # or
    dxgb.train(client, {}, Xy, num_boost_round=10, coll_cfg=coll_cfg)

- **xgboost.scheduler_address**: Specify the scheduler address

  .. versionadded:: 1.6.0

  .. deprecated:: 3.0.0

  .. code-block:: python

      dask.config.set({"xgboost.scheduler_address": "192.0.0.100"})
      # We can also specify the port.
      dask.config.set({"xgboost.scheduler_address": "192.0.0.100:12345"})

"""
import logging
from collections import defaultdict
from contextlib import contextmanager
from functools import cache, partial, update_wrapper
from threading import Thread
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    ParamSpec,
    Sequence,
    Set,
    Tuple,
    TypeAlias,
    TypedDict,
    TypeGuard,
    TypeVar,
    Union,
)

import dask
import distributed
import numpy
from dask import array as da
from dask import bag as db
from dask import dataframe as dd
from dask.delayed import Delayed
from distributed import Future
from packaging.version import Version
from packaging.version import parse as parse_version

from .. import collective, config
from .._typing import FeatureNames, FeatureTypes, IterationRange
from ..callback import TrainingCallback
from ..collective import Config as CollConfig
from ..collective import _Args as CollArgs
from ..collective import _ArgVals as CollArgsVals
from ..compat import DataFrame, lazy_isinstance
from ..core import (
    Booster,
    DMatrix,
    Metric,
    Objective,
    XGBoostError,
    _check_distributed_params,
    _deprecate_positional_args,
    _expect,
)
from ..data import _is_cudf_ser, _is_cupy_alike
from ..sklearn import (
    XGBClassifier,
    XGBClassifierBase,
    XGBModel,
    XGBRanker,
    XGBRankerMixIn,
    XGBRegressorBase,
    _can_use_qdm,
    _check_rf_callback,
    _cls_predict_proba,
    _objective_decorator,
    _wrap_evaluation_matrices,
    xgboost_model_doc,
)
from ..tracker import RabitTracker
from ..training import train as worker_train
from .data import _create_dmatrix, _create_quantile_dmatrix, no_group_split
from .utils import get_address_from_user, get_n_threads

_DaskCollection: TypeAlias = Union[da.Array, dd.DataFrame, dd.Series]
_DataT: TypeAlias = Union[da.Array, dd.DataFrame]  # do not use series as predictor
TrainReturnT = TypedDict(
    "TrainReturnT",
    {
        "booster": Booster,
        "history": Dict,
    },
)

__all__ = [
    "CommunicatorContext",
    "DaskDMatrix",
    "DaskQuantileDMatrix",
    "DaskXGBRegressor",
    "DaskXGBClassifier",
    "DaskXGBRanker",
    "DaskXGBRFRegressor",
    "DaskXGBRFClassifier",
    "train",
    "predict",
    "inplace_predict",
]

# TODOs:
#   - CV
#
# Note for developers:
#
#   As of writing asyncio is still a new feature of Python and in depth documentation is
#   rare.  Best examples of various asyncio tricks are in dask (luckily).  Classes like
#   Client, Worker are awaitable.  Some general rules for the implementation here:
#
#     - Synchronous world is different from asynchronous one, and they don't mix well.
#     - Write everything with async, then use distributed Client sync function to do the
#       switch.
#     - Use Any for type hint when the return value can be union of Awaitable and plain
#       value.  This is caused by Client.sync can return both types depending on
#       context.  Right now there's no good way to silent:
#
#         await train(...)
#
#       if train returns an Union type.


LOGGER = logging.getLogger("[xgboost.dask]")


@cache
def _DASK_VERSION() -> Version:
    return parse_version(dask.__version__)


@cache
def _DASK_2024_12_1() -> bool:
    return _DASK_VERSION() >= parse_version("2024.12.1")


@cache
def _DASK_2025_3_0() -> bool:
    return _DASK_VERSION() >= parse_version("2025.3.0")


def _try_start_tracker(
    n_workers: int,
    addrs: List[Union[Optional[str], Optional[Tuple[str, int]]]],
    timeout: Optional[int],
) -> CollArgs:
    env: CollArgs = {}
    try:
        if isinstance(addrs[0], tuple):
            host_ip = addrs[0][0]
            port = addrs[0][1]
            rabit_tracker = RabitTracker(
                n_workers=n_workers,
                host_ip=host_ip,
                port=port,
                sortby="task",
                timeout=0 if timeout is None else timeout,
            )
        else:
            addr = addrs[0]
            assert isinstance(addr, str) or addr is None
            rabit_tracker = RabitTracker(
                n_workers=n_workers,
                host_ip=addr,
                sortby="task",
                timeout=0 if timeout is None else timeout,
            )

        rabit_tracker.start()
        # No timeout since we don't want to abort the training
        thread = Thread(target=rabit_tracker.wait_for)
        thread.daemon = True
        thread.start()
        env.update(rabit_tracker.worker_args())

    except XGBoostError as e:
        if len(addrs) < 2:
            raise
        LOGGER.warning(
            "Failed to bind address '%s', trying to use '%s' instead. Error:\n %s",
            str(addrs[0]),
            str(addrs[1]),
            str(e),
        )
        env = _try_start_tracker(n_workers, addrs[1:], timeout)

    return env


def _start_tracker(
    n_workers: int,
    addr_from_dask: Optional[str],
    addr_from_user: Optional[Tuple[str, int]],
    timeout: Optional[int],
) -> CollArgs:
    """Start Rabit tracker, recurse to try different addresses."""
    env = _try_start_tracker(n_workers, [addr_from_user, addr_from_dask], timeout)
    return env


class CommunicatorContext(collective.CommunicatorContext):
    """A context controlling collective communicator initialization and finalization."""

    def __init__(self, **args: CollArgsVals) -> None:
        super().__init__(**args)

        worker = distributed.get_worker()
        # We use task ID for rank assignment which makes the RABIT rank consistent (but
        # not the same as task ID is string and "10" is sorted before "2") with dask
        # worker name. This outsources the rank assignment to dask and prevents
        # non-deterministic issue.
        self.args["DMLC_TASK_ID"] = f"[xgboost.dask-{worker.name}]:{worker.address}"


def _get_client(client: Optional["distributed.Client"]) -> "distributed.Client":
    """Simple wrapper around testing None."""
    if not isinstance(client, (type(distributed.get_client()), type(None))):
        raise TypeError(
            _expect([type(distributed.get_client()), type(None)], type(client))
        )
    ret = distributed.get_client() if client is None else client
    return ret


# From the implementation point of view, DaskDMatrix complicates a lots of
# things.  A large portion of the code base is about syncing and extracting
# stuffs from DaskDMatrix.  But having an independent data structure gives us a
# chance to perform some specialized optimizations, like building histogram
# index directly.


class DaskDMatrix:
    # pylint: disable=too-many-instance-attributes
    """DMatrix holding on references to Dask DataFrame or Dask Array.  Constructing a
    `DaskDMatrix` forces all lazy computation to be carried out.  Wait for the input
    data explicitly if you want to see actual computation of constructing `DaskDMatrix`.

    See doc for :py:obj:`xgboost.DMatrix` constructor for other parameters.  DaskDMatrix
    accepts only dask collection.

    .. note::

        `DaskDMatrix` does not repartition or move data between workers.  It's the
        caller's responsibility to balance the data.

    .. note::

        For aligning partitions with ranking query groups, use the
        :py:class:`DaskXGBRanker` and its ``allow_group_split`` option.

    .. versionadded:: 1.0.0

    Parameters
    ----------
    client :
        Specify the dask client used for training.  Use default client returned from
        dask if it's set to None.

    """

    @_deprecate_positional_args
    def __init__(
        self,
        client: Optional["distributed.Client"],
        data: _DataT,
        label: Optional[_DaskCollection] = None,
        *,
        weight: Optional[_DaskCollection] = None,
        base_margin: Optional[_DaskCollection] = None,
        missing: Optional[float] = None,
        silent: bool = False,  # pylint: disable=unused-argument
        feature_names: Optional[FeatureNames] = None,
        feature_types: Optional[FeatureTypes] = None,
        group: Optional[_DaskCollection] = None,
        qid: Optional[_DaskCollection] = None,
        label_lower_bound: Optional[_DaskCollection] = None,
        label_upper_bound: Optional[_DaskCollection] = None,
        feature_weights: Optional[_DaskCollection] = None,
        enable_categorical: bool = False,
    ) -> None:
        client = _get_client(client)

        self.feature_names = feature_names
        self.feature_types = feature_types
        self.missing = missing if missing is not None else numpy.nan
        self.enable_categorical = enable_categorical

        if qid is not None and weight is not None:
            raise NotImplementedError("per-group weight is not implemented.")
        if group is not None:
            raise NotImplementedError(
                "group structure is not implemented, use qid instead."
            )

        if len(data.shape) != 2:
            raise ValueError(f"Expecting 2 dimensional input, got: {data.shape}")

        if not isinstance(data, (dd.DataFrame, da.Array)):
            raise TypeError(_expect((dd.DataFrame, da.Array), type(data)))
        if not isinstance(label, (dd.DataFrame, da.Array, dd.Series, type(None))):
            raise TypeError(_expect((dd.DataFrame, da.Array, dd.Series), type(label)))

        self._n_cols = data.shape[1]
        assert isinstance(self._n_cols, int)
        self.worker_map: Dict[str, List[Future]] = defaultdict(list)
        self.is_quantile: bool = False

        self._init = client.sync(
            self._map_local_data,
            client=client,
            data=data,
            label=label,
            weights=weight,
            base_margin=base_margin,
            qid=qid,
            feature_weights=feature_weights,
            label_lower_bound=label_lower_bound,
            label_upper_bound=label_upper_bound,
        )

    def __await__(self) -> Generator:
        return self._init.__await__()

    async def _map_local_data(
        self,
        *,
        client: "distributed.Client",
        data: _DataT,
        label: Optional[_DaskCollection] = None,
        weights: Optional[_DaskCollection] = None,
        base_margin: Optional[_DaskCollection] = None,
        qid: Optional[_DaskCollection] = None,
        feature_weights: Optional[_DaskCollection] = None,
        label_lower_bound: Optional[_DaskCollection] = None,
        label_upper_bound: Optional[_DaskCollection] = None,
    ) -> "DaskDMatrix":
        """Obtain references to local data."""

        def inconsistent(
            left: List[Any], left_name: str, right: List[Any], right_name: str
        ) -> str:
            msg = (
                f"Partitions between {left_name} and {right_name} are not "
                f"consistent: {len(left)} != {len(right)}.  "
                f"Please try to repartition/rechunk your data."
            )
            return msg

        def to_futures(d: _DaskCollection) -> List[Future]:
            """Breaking data into partitions."""
            d = client.persist(d)
            if (
                hasattr(d.partitions, "shape")
                and len(d.partitions.shape) > 1
                and d.partitions.shape[1] > 1
            ):
                raise ValueError(
                    "Data should be"
                    " partitioned by row. To avoid this specify the number"
                    " of columns for your dask Array explicitly. e.g."
                    " chunks=(partition_size, -1])"
                )
            return client.futures_of(d)

        def flatten_meta(meta: Optional[_DaskCollection]) -> Optional[List[Future]]:
            if meta is not None:
                meta_parts: List[Future] = to_futures(meta)
                return meta_parts
            return None

        X_parts = to_futures(data)
        y_parts = flatten_meta(label)
        w_parts = flatten_meta(weights)
        margin_parts = flatten_meta(base_margin)
        qid_parts = flatten_meta(qid)
        ll_parts = flatten_meta(label_lower_bound)
        lu_parts = flatten_meta(label_upper_bound)

        parts: Dict[str, List[Future]] = {"data": X_parts}

        def append_meta(m_parts: Optional[List[Future]], name: str) -> None:
            if m_parts is not None:
                assert len(X_parts) == len(m_parts), inconsistent(
                    X_parts, "X", m_parts, name
                )
                parts[name] = m_parts

        append_meta(y_parts, "label")
        append_meta(w_parts, "weight")
        append_meta(margin_parts, "base_margin")
        append_meta(qid_parts, "qid")
        append_meta(ll_parts, "label_lower_bound")
        append_meta(lu_parts, "label_upper_bound")
        # At this point, `parts` looks like:
        # [(x0, x1, ..), (y0, y1, ..), ..] in future form

        # turn into list of dictionaries.
        packed_parts: List[Dict[str, Future]] = []
        for i in range(len(X_parts)):
            part_dict: Dict[str, Future] = {}
            for key, value in parts.items():
                part_dict[key] = value[i]
            packed_parts.append(part_dict)

        # delay the zipped result
        # pylint: disable=no-member
        delayed_parts: List[Delayed] = list(map(dask.delayed, packed_parts))
        # At this point, the mental model should look like:
        # [{"data": x0, "label": y0, ..}, {"data": x1, "label": y1, ..}, ..]

        # Convert delayed objects into futures and make sure they are realized
        #
        # This also makes partitions to align (co-locate) on workers (X_0, y_0 should be
        # on the same worker).
        fut_parts: List[Future] = client.compute(delayed_parts)
        await distributed.wait(fut_parts)  # async wait for parts to be computed

        for part in fut_parts:
            # Each part is [{"data": x0, "label": y0, ..}, ...] in future form.
            assert part.status == "finished", part.status

        # Preserving the partition order for prediction.
        self.partition_order = {}
        for i, part in enumerate(fut_parts):
            self.partition_order[part.key] = i

        key_to_partition = {part.key: part for part in fut_parts}
        who_has: Dict[str, Tuple[str, ...]] = await client.scheduler.who_has(
            keys=[part.key for part in fut_parts]
        )

        worker_map: Dict[str, List[Future]] = defaultdict(list)

        for key, workers in who_has.items():
            worker_map[next(iter(workers))].append(key_to_partition[key])

        self.worker_map = worker_map

        if feature_weights is None:
            self.feature_weights = None
        else:
            self.feature_weights = await client.compute(feature_weights).result()

        return self

    def _create_fn_args(self, worker_addr: str) -> Dict[str, Any]:
        """Create a dictionary of objects that can be pickled for function
        arguments.

        """
        return {
            "feature_names": self.feature_names,
            "feature_types": self.feature_types,
            "feature_weights": self.feature_weights,
            "missing": self.missing,
            "enable_categorical": self.enable_categorical,
            "parts": self.worker_map.get(worker_addr, None),
            "is_quantile": self.is_quantile,
        }

    def num_col(self) -> int:
        """Get the number of columns (features) in the DMatrix.

        Returns
        -------
        number of columns
        """
        return self._n_cols


_MapRetT = TypeVar("_MapRetT")
_P = ParamSpec("_P")


async def map_worker_partitions(
    client: Optional["distributed.Client"],
    func: Callable[_P, _MapRetT],
    *refs: Any,
    workers: Sequence[str],
) -> _MapRetT:
    """Map a function onto partitions of each worker."""
    # Note for function purity:
    # XGBoost is sensitive to data partition and uses random number generator.
    client = _get_client(client)
    futures = []
    for addr in workers:
        args = []
        for ref in refs:
            if isinstance(ref, DaskDMatrix):
                # pylint: disable=protected-access
                args.append(ref._create_fn_args(addr))
            else:
                args.append(ref)

        def fn(_address: str, *args: _P.args, **kwargs: _P.kwargs) -> List[_MapRetT]:
            worker = distributed.get_worker()

            if worker.address != _address:
                raise ValueError(
                    f"Invalid worker address: {worker.address}, expecting {_address}. "
                    "This is likely caused by one of the workers died and Dask "
                    "re-scheduled a different one. Resilience is not yet supported."
                )
            # Turn result into a list for bag construction
            return [func(*args, **kwargs)]

        # XGBoost requires all workers running training tasks to be unique. Meaning, we
        # can't run 2 training jobs on the same node. This at best leads to an error
        # (NCCL unique check), at worst leads to extremely slow training performance
        # without any warning.
        #
        # See disitributed.scheduler.decide_worker for `allow_other_workers`. In
        # summary, the scheduler chooses a worker from the valid set that has the task
        # dependencies. Each XGBoost's training task has all dependencies in a single
        # worker. As a result, the right worker should be picked by the scheduler even
        # if `allow_other_workers` is set to True.
        #
        # In addition, the scheduler only discards the valid set (the `workers` arg) if
        # there's no candidate can be found. This is likely caused by killed workers. In
        # that case, the check in `fn` should be able to stop the task. If we don't
        # relax the constraint and prevent Dask from choosing an invalid worker, the
        # task will simply hangs. We prefer a quick error here.
        #
        fut = client.submit(
            update_wrapper(partial(fn, addr), fn),
            *args,
            pure=False,
            workers=[addr],
            allow_other_workers=True,
        )
        futures.append(fut)

    def first_valid(results: Iterable[Optional[_MapRetT]]) -> Optional[_MapRetT]:
        for v in results:
            if v is not None:
                return v
        return None

    bag = db.from_delayed(futures)
    fut = await bag.reduction(first_valid, first_valid)
    result = await client.compute(fut).result()

    return result


class DaskQuantileDMatrix(DaskDMatrix):
    """A dask version of :py:class:`QuantileDMatrix`. See :py:class:`DaskDMatrix` for
    parameter documents.

    """

    @_deprecate_positional_args
    def __init__(
        self,
        client: Optional["distributed.Client"],
        data: _DataT,
        label: Optional[_DaskCollection] = None,
        *,
        weight: Optional[_DaskCollection] = None,
        base_margin: Optional[_DaskCollection] = None,
        missing: Optional[float] = None,
        silent: bool = False,  # disable=unused-argument
        feature_names: Optional[FeatureNames] = None,
        feature_types: Optional[Union[Any, List[Any]]] = None,
        max_bin: Optional[int] = None,
        ref: Optional[DaskDMatrix] = None,
        group: Optional[_DaskCollection] = None,
        qid: Optional[_DaskCollection] = None,
        label_lower_bound: Optional[_DaskCollection] = None,
        label_upper_bound: Optional[_DaskCollection] = None,
        feature_weights: Optional[_DaskCollection] = None,
        enable_categorical: bool = False,
        max_quantile_batches: Optional[int] = None,
    ) -> None:
        super().__init__(
            client=client,
            data=data,
            label=label,
            weight=weight,
            base_margin=base_margin,
            group=group,
            qid=qid,
            label_lower_bound=label_lower_bound,
            label_upper_bound=label_upper_bound,
            missing=missing,
            silent=silent,
            feature_weights=feature_weights,
            feature_names=feature_names,
            feature_types=feature_types,
            enable_categorical=enable_categorical,
        )
        self.max_bin = max_bin
        self.max_quantile_batches = max_quantile_batches
        self.is_quantile = True
        self._ref: Optional[int] = id(ref) if ref is not None else None

    def _create_fn_args(self, worker_addr: str) -> Dict[str, Any]:
        args = super()._create_fn_args(worker_addr)
        args["max_bin"] = self.max_bin
        args["max_quantile_batches"] = self.max_quantile_batches
        if self._ref is not None:
            args["ref"] = self._ref
        return args


def _dmatrix_from_list_of_parts(is_quantile: bool, **kwargs: Any) -> DMatrix:
    if is_quantile:
        return _create_quantile_dmatrix(**kwargs)
    return _create_dmatrix(**kwargs)


async def _get_rabit_args(
    client: "distributed.Client",
    n_workers: int,
    dconfig: Optional[Dict[str, Any]] = None,
    coll_cfg: Optional[CollConfig] = None,
) -> Dict[str, Union[str, int]]:
    """Get rabit context arguments from data distribution in DaskDMatrix."""
    # There are 3 possible different addresses:
    # 1. Provided by user via dask.config
    # 2. Guessed by xgboost `get_host_ip` function
    # 3. From dask scheduler
    # We try 1 and 3 if 1 is available, otherwise 2 and 3.

    # See if user config is available
    coll_cfg = CollConfig() if coll_cfg is None else coll_cfg
    host_ip: Optional[str] = None
    port: int = 0
    host_ip, port = get_address_from_user(dconfig, coll_cfg)

    if host_ip is not None:
        user_addr = (host_ip, port)
    else:
        user_addr = None

    # Try address from dask scheduler, this might not work, see
    # https://github.com/dask/dask-xgboost/pull/40
    try:
        sched_addr = distributed.comm.get_address_host(client.scheduler.address)
        sched_addr = sched_addr.strip("/:")
    except Exception:  # pylint: disable=broad-except
        sched_addr = None

    # We assume the scheduler is a fair process and run the tracker there.
    env = await client.run_on_scheduler(
        _start_tracker, n_workers, sched_addr, user_addr, coll_cfg.tracker_timeout
    )
    env = coll_cfg.get_comm_config(env)
    return env


def _get_dask_config() -> Optional[Dict[str, Any]]:
    return dask.config.get("xgboost", default=None)


# train and predict methods are supposed to be "functional", which meets the
# dask paradigm.  But as a side effect, the `evals_result` in single-node API
# is no longer supported since it mutates the input parameter, and it's not
# intuitive to sync the mutation result.  Therefore, a dictionary containing
# evaluation history is instead returned.


def _get_workers_from_data(
    dtrain: DaskDMatrix, evals: Optional[Sequence[Tuple[DaskDMatrix, str]]]
) -> List[str]:
    X_worker_map: Set[str] = set(dtrain.worker_map.keys())
    if evals:
        for e in evals:
            assert len(e) == 2
            assert isinstance(e[0], DaskDMatrix) and isinstance(e[1], str)
            if e[0] is dtrain:
                continue
            worker_map = set(e[0].worker_map.keys())
            X_worker_map = X_worker_map.union(worker_map)
    return list(X_worker_map)


async def _check_workers_are_alive(
    workers: List[str], client: "distributed.Client"
) -> None:
    info = await client.scheduler.identity()
    current_workers = info["workers"].keys()
    missing_workers = set(workers) - current_workers
    if missing_workers:
        raise RuntimeError(f"Missing required workers: {missing_workers}")


def _get_dmatrices(
    train_ref: dict,
    train_id: int,
    *refs: dict,
    evals_id: Sequence[int],
    evals_name: Sequence[str],
    n_threads: int,
) -> Tuple[DMatrix, List[Tuple[DMatrix, str]]]:
    # Create training DMatrix
    Xy = _dmatrix_from_list_of_parts(**train_ref, nthread=n_threads)
    # Create evaluation DMatrices
    evals: List[Tuple[DMatrix, str]] = []
    for i, ref in enumerate(refs):
        # Same DMatrix as the training
        if evals_id[i] == train_id:
            evals.append((Xy, evals_name[i]))
            continue
        if ref.get("ref", None) is not None:
            if ref["ref"] != train_id:
                raise ValueError(
                    "The training DMatrix should be used as a reference to evaluation"
                    " `QuantileDMatrix`."
                )
            del ref["ref"]
            eval_Xy = _dmatrix_from_list_of_parts(**ref, nthread=n_threads, ref=Xy)
        else:
            eval_Xy = _dmatrix_from_list_of_parts(**ref, nthread=n_threads)
        evals.append((eval_Xy, evals_name[i]))
    return Xy, evals


async def _train_async(
    *,
    client: "distributed.Client",
    global_config: Dict[str, Any],
    dconfig: Optional[Dict[str, Any]],
    params: Dict[str, Any],
    dtrain: DaskDMatrix,
    num_boost_round: int,
    evals: Optional[Sequence[Tuple[DaskDMatrix, str]]],
    obj: Optional[Objective],
    early_stopping_rounds: Optional[int],
    verbose_eval: Union[int, bool],
    xgb_model: Optional[Booster],
    callbacks: Optional[Sequence[TrainingCallback]],
    custom_metric: Optional[Metric],
    coll_cfg: Optional[CollConfig],
) -> Optional[TrainReturnT]:
    workers = _get_workers_from_data(dtrain, evals)
    await _check_workers_are_alive(workers, client)
    coll_args = await _get_rabit_args(
        client, len(workers), dconfig=dconfig, coll_cfg=coll_cfg
    )
    _check_distributed_params(params)

    # This function name is displayed in the Dask dashboard task status, let's make it
    # clear that it's XGBoost training.
    def do_train(  # pylint: disable=too-many-positional-arguments
        parameters: Dict,
        coll_args: Dict[str, Union[str, int]],
        train_id: int,
        evals_name: List[str],
        evals_id: List[int],
        train_ref: dict,
        *refs: dict,
    ) -> Optional[TrainReturnT]:
        worker = distributed.get_worker()
        local_param = parameters.copy()
        n_threads = get_n_threads(local_param, worker)
        local_param.update({"nthread": n_threads, "n_jobs": n_threads})

        local_history: TrainingCallback.EvalsLog = {}
        global_config.update({"nthread": n_threads})

        with CommunicatorContext(**coll_args), config.config_context(**global_config):
            Xy, evals = _get_dmatrices(
                train_ref,
                train_id,
                *refs,
                evals_id=evals_id,
                evals_name=evals_name,
                n_threads=n_threads,
            )

            booster = worker_train(
                params=local_param,
                dtrain=Xy,
                num_boost_round=num_boost_round,
                evals_result=local_history,
                evals=evals if len(evals) != 0 else None,
                obj=obj,
                custom_metric=custom_metric,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=verbose_eval,
                xgb_model=xgb_model,
                callbacks=callbacks,
            )
        # Don't return the boosters from empty workers. It's quite difficult to
        # guarantee everything is in sync in the present of empty workers, especially
        # with complex objectives like quantile.
        if Xy.num_row() != 0:
            ret: Optional[TrainReturnT] = {
                "booster": booster,
                "history": local_history,
            }
        else:
            ret = None
        return ret

    async with distributed.MultiLock(workers, client):
        if evals is not None:
            evals_data = [d for d, n in evals]
            evals_name = [n for d, n in evals]
            evals_id = [id(d) for d in evals_data]
        else:
            evals_data = []
            evals_name = []
            evals_id = []

        result = await map_worker_partitions(
            client,
            do_train,
            # extra function parameters
            params,
            coll_args,
            id(dtrain),
            evals_name,
            evals_id,
            *([dtrain] + evals_data),
            # workers to be used for training
            workers=workers,
        )
        return result


@_deprecate_positional_args
def train(  # pylint: disable=unused-argument
    client: "distributed.Client",
    params: Dict[str, Any],
    dtrain: DaskDMatrix,
    num_boost_round: int = 10,
    *,
    evals: Optional[Sequence[Tuple[DaskDMatrix, str]]] = None,
    obj: Optional[Objective] = None,
    early_stopping_rounds: Optional[int] = None,
    xgb_model: Optional[Booster] = None,
    verbose_eval: Union[int, bool] = True,
    callbacks: Optional[Sequence[TrainingCallback]] = None,
    custom_metric: Optional[Metric] = None,
    coll_cfg: Optional[CollConfig] = None,
) -> Any:
    """Train XGBoost model.

    .. versionadded:: 1.0.0

    .. note::

        Other parameters are the same as :py:func:`xgboost.train` except for
        `evals_result`, which is returned as part of function return value instead of
        argument.

    Parameters
    ----------
    client :
        Specify the dask client used for training.  Use default client returned from
        dask if it's set to None.

    coll_cfg :
        Configuration for the communicator used during training. See
        :py:class:`~xgboost.collective.Config`.

    Returns
    -------
    results: dict
        A dictionary containing trained booster and evaluation history.  `history` field
        is the same as `eval_result` from `xgboost.train`.

        .. code-block:: python

            {'booster': xgboost.Booster,
             'history': {'train': {'logloss': ['0.48253', '0.35953']},
                         'eval': {'logloss': ['0.480385', '0.357756']}}}

    """
    client = _get_client(client)
    return client.sync(
        _train_async,
        global_config=config.get_config(),
        dconfig=_get_dask_config(),
        **locals(),
    )


def _can_output_df(is_df: bool, output_shape: Tuple) -> bool:
    return is_df and len(output_shape) <= 2


def _maybe_dataframe(
    data: Any, prediction: Any, columns: List[int], is_df: bool
) -> Any:
    """Return dataframe for prediction when applicable."""
    if _can_output_df(is_df, prediction.shape):
        # Need to preserve the index for dataframe.
        # See issue: https://github.com/dmlc/xgboost/issues/6939
        # In older versions of dask, the partition is actually a numpy array when input
        # is dataframe.
        index = getattr(data, "index", None)
        if lazy_isinstance(data, "cudf.core.dataframe", "DataFrame"):
            import cudf

            if prediction.size == 0:
                return cudf.DataFrame({}, columns=columns, dtype=numpy.float32)

            prediction = cudf.DataFrame(
                prediction, columns=columns, dtype=numpy.float32, index=index
            )
        else:
            if prediction.size == 0:
                return DataFrame({}, columns=columns, dtype=numpy.float32, index=index)

            prediction = DataFrame(
                prediction, columns=columns, dtype=numpy.float32, index=index
            )
    return prediction


async def _direct_predict_impl(  # pylint: disable=too-many-branches
    *,
    mapped_predict: Callable,
    booster: "distributed.Future",
    data: _DataT,
    base_margin: Optional[_DaskCollection],
    output_shape: Tuple[int, ...],
    meta: Dict[int, str],
) -> _DaskCollection:
    columns = tuple(meta.keys())
    if len(output_shape) >= 3 and isinstance(data, dd.DataFrame):
        # Without this check, dask will finish the prediction silently even if output
        # dimension is greater than 3.  But during map_partitions, dask passes a
        # `dd.DataFrame` as local input to xgboost, which is converted to csr_matrix by
        # `_convert_unknown_data` since dd.DataFrame is not known to xgboost native
        # binding.
        raise ValueError(
            "Use `da.Array` or `DaskDMatrix` when output has more than 2 dimensions."
        )
    if _can_output_df(isinstance(data, dd.DataFrame), output_shape):
        if base_margin is not None and isinstance(base_margin, da.Array):
            # Easier for map_partitions
            base_margin_df: Optional[Union[dd.DataFrame, dd.Series]] = (
                base_margin.to_dask_dataframe()
            )
        else:
            base_margin_df = base_margin
        predictions = dd.map_partitions(
            mapped_predict,
            booster,
            data,
            True,
            columns,
            base_margin_df,
            meta=dd.utils.make_meta(meta),
        )
        # classification can return a dataframe, drop 1 dim when it's reg/binary
        if len(output_shape) == 1:
            predictions = predictions.iloc[:, 0]
    else:
        if base_margin is not None and isinstance(
            base_margin, (dd.Series, dd.DataFrame)
        ):
            # Easier for map_blocks
            base_margin_array: Optional[da.Array] = base_margin.to_dask_array()
        else:
            base_margin_array = base_margin
        # Input data is 2-dim array, output can be 1(reg, binary)/2(multi-class,
        # contrib)/3(contrib, interaction)/4(interaction) dims.
        if len(output_shape) == 1:
            drop_axis: Union[int, List[int]] = [1]  # drop from 2 to 1 dim.
            new_axis: Union[int, List[int]] = []
        else:
            drop_axis = []
            if isinstance(data, dd.DataFrame):
                new_axis = list(range(len(output_shape) - 2))
            else:
                new_axis = [i + 2 for i in range(len(output_shape) - 2)]
        if len(output_shape) == 2:
            # Somehow dask fail to infer output shape change for 2-dim prediction, and
            #  `chunks = (None, output_shape[1])` doesn't work due to None is not
            #  supported in map_blocks.

            # data must be an array here as dataframe + 2-dim output predict will return
            # a dataframe instead.
            chunks: Optional[List[Tuple]] = list(data.chunks)
            assert isinstance(chunks, list)
            chunks[1] = (output_shape[1],)
        else:
            chunks = None
        predictions = da.map_blocks(
            mapped_predict,
            booster,
            data,
            False,
            columns,
            base_margin_array,
            chunks=chunks,
            drop_axis=drop_axis,
            new_axis=new_axis,
            dtype=numpy.float32,
        )
    return predictions


def _infer_predict_output(
    booster: Booster, features: int, is_df: bool, inplace: bool, **kwargs: Any
) -> Tuple[Tuple[int, ...], Dict[int, str]]:
    """Create a dummy test sample to infer output shape for prediction."""
    assert isinstance(features, int)
    rng = numpy.random.RandomState(1994)
    test_sample = rng.randn(1, features)
    if inplace:
        kwargs = kwargs.copy()
        if kwargs.pop("predict_type") == "margin":
            kwargs["output_margin"] = True
    m = DMatrix(test_sample, enable_categorical=True)
    # generated DMatrix doesn't have feature name, so no validation.
    test_predt = booster.predict(m, validate_features=False, **kwargs)
    n_columns = test_predt.shape[1] if len(test_predt.shape) > 1 else 1
    meta: Dict[int, str] = {}
    if _can_output_df(is_df, test_predt.shape):
        for i in range(n_columns):
            meta[i] = "f4"
    return test_predt.shape, meta


async def _get_model_future(
    client: "distributed.Client", model: Union[Booster, Dict, "distributed.Future"]
) -> "distributed.Future":
    # See https://github.com/dask/dask/issues/11179#issuecomment-2168094529 for the use
    # of hash.
    # https://github.com/dask/distributed/pull/8796 Don't use broadcast in the `scatter`
    # call, otherwise, the predict function might hang.
    if isinstance(model, Booster):
        booster = await client.scatter(model, hash=False)
    elif isinstance(model, dict):
        booster = await client.scatter(model["booster"], hash=False)
    elif isinstance(model, distributed.Future):
        booster = model
        t = booster.type
        if t is not Booster:
            raise TypeError(
                f"Underlying type of model future should be `Booster`, got {t}"
            )
    else:
        raise TypeError(_expect([Booster, dict, distributed.Future], type(model)))
    return booster


# pylint: disable=too-many-statements
async def _predict_async(
    client: "distributed.Client",
    global_config: Dict[str, Any],
    model: Union[Booster, Dict, "distributed.Future"],
    data: _DataT,
    *,
    output_margin: bool,
    missing: float,
    pred_leaf: bool,
    pred_contribs: bool,
    approx_contribs: bool,
    pred_interactions: bool,
    validate_features: bool,
    iteration_range: IterationRange,
    strict_shape: bool,
) -> _DaskCollection:
    _booster = await _get_model_future(client, model)
    if not isinstance(data, (DaskDMatrix, da.Array, dd.DataFrame)):
        raise TypeError(_expect([DaskDMatrix, da.Array, dd.DataFrame], type(data)))

    def mapped_predict(
        booster: Booster, partition: Any, is_df: bool, columns: List[int], _: Any
    ) -> Any:
        with config.config_context(**global_config):
            m = DMatrix(
                data=partition,
                missing=missing,
                enable_categorical=True,
            )
            predt = booster.predict(
                data=m,
                output_margin=output_margin,
                pred_leaf=pred_leaf,
                pred_contribs=pred_contribs,
                approx_contribs=approx_contribs,
                pred_interactions=pred_interactions,
                validate_features=validate_features,
                iteration_range=iteration_range,
                strict_shape=strict_shape,
            )
            predt = _maybe_dataframe(partition, predt, columns, is_df)
            return predt

    # Predict on dask collection directly.
    if isinstance(data, (da.Array, dd.DataFrame)):
        _output_shape, meta = await client.compute(
            client.submit(
                _infer_predict_output,
                _booster,
                features=data.shape[1],
                is_df=isinstance(data, dd.DataFrame),
                inplace=False,
                output_margin=output_margin,
                pred_leaf=pred_leaf,
                pred_contribs=pred_contribs,
                approx_contribs=approx_contribs,
                pred_interactions=pred_interactions,
                strict_shape=strict_shape,
            )
        )
        return await _direct_predict_impl(
            mapped_predict=mapped_predict,
            booster=_booster,
            data=data,
            base_margin=None,
            output_shape=_output_shape,
            meta=meta,
        )

    output_shape, _ = await client.compute(
        client.submit(
            _infer_predict_output,
            booster=_booster,
            features=data.num_col(),
            is_df=False,
            inplace=False,
            output_margin=output_margin,
            pred_leaf=pred_leaf,
            pred_contribs=pred_contribs,
            approx_contribs=approx_contribs,
            pred_interactions=pred_interactions,
            strict_shape=strict_shape,
        )
    )
    # Prediction on dask DMatrix.
    partition_order = data.partition_order
    feature_names = data.feature_names
    feature_types = data.feature_types
    missing = data.missing

    def dispatched_predict(booster: Booster, part: Dict[str, Any]) -> numpy.ndarray:
        data = part["data"]
        base_margin = part.get("base_margin", None)
        with config.config_context(**global_config):
            m = DMatrix(
                data,
                missing=missing,
                base_margin=base_margin,
                feature_names=feature_names,
                feature_types=feature_types,
                enable_categorical=True,
            )
            predt = booster.predict(
                m,
                output_margin=output_margin,
                pred_leaf=pred_leaf,
                pred_contribs=pred_contribs,
                approx_contribs=approx_contribs,
                pred_interactions=pred_interactions,
                validate_features=validate_features,
                iteration_range=iteration_range,
                strict_shape=strict_shape,
            )
            return predt

    all_parts = []
    all_orders = []
    all_shapes = []
    all_workers: List[str] = []
    workers_address = list(data.worker_map.keys())
    for worker_addr in workers_address:
        list_of_parts = data.worker_map[worker_addr]
        all_parts.extend(list_of_parts)
        all_workers.extend(len(list_of_parts) * [worker_addr])
        all_orders.extend([partition_order[part.key] for part in list_of_parts])
    for w, part in zip(all_workers, all_parts):
        s = client.submit(lambda part: part["data"].shape[0], part, workers=[w])
        all_shapes.append(s)

    parts_with_order = list(zip(all_parts, all_shapes, all_orders, all_workers))
    parts_with_order = sorted(parts_with_order, key=lambda p: p[2])
    all_parts = [part for part, shape, order, w in parts_with_order]
    all_shapes = [shape for part, shape, order, w in parts_with_order]
    all_workers = [w for part, shape, order, w in parts_with_order]

    futures = []
    for w, part in zip(all_workers, all_parts):
        f = client.submit(dispatched_predict, _booster, part, workers=[w])
        futures.append(f)

    # Constructing a dask array from list of numpy arrays
    # See https://docs.dask.org/en/latest/array-creation.html
    arrays = []
    all_shapes = await client.gather(all_shapes)
    for i, rows in enumerate(all_shapes):
        arrays.append(
            da.from_delayed(
                futures[i], shape=(rows,) + output_shape[1:], dtype=numpy.float32
            )
        )
    predictions = da.concatenate(arrays, axis=0)
    return predictions


@_deprecate_positional_args
def predict(  # pylint: disable=unused-argument
    client: Optional["distributed.Client"],
    model: Union[TrainReturnT, Booster, "distributed.Future"],
    data: Union[DaskDMatrix, _DataT],
    *,
    output_margin: bool = False,
    missing: float = numpy.nan,
    pred_leaf: bool = False,
    pred_contribs: bool = False,
    approx_contribs: bool = False,
    pred_interactions: bool = False,
    validate_features: bool = True,
    iteration_range: IterationRange = (0, 0),
    strict_shape: bool = False,
) -> Any:
    """Run prediction with a trained booster.

    .. note::

        Using ``inplace_predict`` might be faster when some features are not needed.
        See :py:meth:`xgboost.Booster.predict` for details on various parameters.  When
        output has more than 2 dimensions (shap value, leaf with strict_shape), input
        should be ``da.Array`` or ``DaskDMatrix``.

    .. versionadded:: 1.0.0

    Parameters
    ----------
    client:
        Specify the dask client used for training.  Use default client
        returned from dask if it's set to None.
    model:
        The trained model.  It can be a distributed.Future so user can
        pre-scatter it onto all workers.
    data:
        Input data used for prediction.  When input is a dataframe object,
        prediction output is a series.
    missing:
        Used when input data is not DaskDMatrix.  Specify the value
        considered as missing.

    Returns
    -------
    prediction: dask.array.Array/dask.dataframe.Series
        When input data is ``dask.array.Array`` or ``DaskDMatrix``, the return value is
        an array, when input data is ``dask.dataframe.DataFrame``, return value can be
        ``dask.dataframe.Series``, ``dask.dataframe.DataFrame``, depending on the output
        shape.

    """
    client = _get_client(client)
    return client.sync(_predict_async, global_config=config.get_config(), **locals())


async def _inplace_predict_async(  # pylint: disable=too-many-branches
    *,
    client: "distributed.Client",
    global_config: Dict[str, Any],
    model: Union[Booster, Dict, "distributed.Future"],
    data: _DataT,
    iteration_range: IterationRange,
    predict_type: str,
    missing: float,
    validate_features: bool,
    base_margin: Optional[_DaskCollection],
    strict_shape: bool,
) -> _DaskCollection:
    client = _get_client(client)
    booster = await _get_model_future(client, model)
    if not isinstance(data, (da.Array, dd.DataFrame)):
        raise TypeError(_expect([da.Array, dd.DataFrame], type(data)))
    if base_margin is not None and not isinstance(
        data, (da.Array, dd.DataFrame, dd.Series)
    ):
        raise TypeError(_expect([da.Array, dd.DataFrame, dd.Series], type(base_margin)))

    def mapped_predict(
        booster: Booster,
        partition: Any,
        is_df: bool,
        columns: List[int],
        base_margin: Any,
    ) -> Any:
        with config.config_context(**global_config):
            prediction = booster.inplace_predict(
                partition,
                iteration_range=iteration_range,
                predict_type=predict_type,
                missing=missing,
                base_margin=base_margin,
                validate_features=validate_features,
                strict_shape=strict_shape,
            )
        prediction = _maybe_dataframe(partition, prediction, columns, is_df)
        return prediction

    # await turns future into value.
    shape, meta = await client.compute(
        client.submit(
            _infer_predict_output,
            booster,
            features=data.shape[1],
            is_df=isinstance(data, dd.DataFrame),
            inplace=True,
            predict_type=predict_type,
            iteration_range=iteration_range,
            strict_shape=strict_shape,
        )
    )
    return await _direct_predict_impl(
        mapped_predict=mapped_predict,
        booster=booster,
        data=data,
        base_margin=base_margin,
        output_shape=shape,
        meta=meta,
    )


@_deprecate_positional_args
def inplace_predict(  # pylint: disable=unused-argument
    client: Optional["distributed.Client"],
    model: Union[TrainReturnT, Booster, "distributed.Future"],
    data: _DataT,
    *,
    iteration_range: IterationRange = (0, 0),
    predict_type: str = "value",
    missing: float = numpy.nan,
    validate_features: bool = True,
    base_margin: Optional[_DaskCollection] = None,
    strict_shape: bool = False,
) -> Any:
    """Inplace prediction. See doc in :py:meth:`xgboost.Booster.inplace_predict` for
    details.

    .. versionadded:: 1.1.0

    Parameters
    ----------
    client:
        Specify the dask client used for training.  Use default client
        returned from dask if it's set to None.
    model:
        See :py:func:`xgboost.dask.predict` for details.
    data :
        dask collection.
    iteration_range:
        See :py:meth:`xgboost.Booster.predict` for details.
    predict_type:
        See :py:meth:`xgboost.Booster.inplace_predict` for details.
    missing:
        Value in the input data which needs to be present as a missing
        value. If None, defaults to np.nan.
    base_margin:
        See :py:obj:`xgboost.DMatrix` for details.

        .. versionadded:: 1.4.0

    strict_shape:
        See :py:meth:`xgboost.Booster.predict` for details.

        .. versionadded:: 1.4.0

    Returns
    -------
    prediction :
        When input data is ``dask.array.Array``, the return value is an array, when
        input data is ``dask.dataframe.DataFrame``, return value can be
        ``dask.dataframe.Series``, ``dask.dataframe.DataFrame``, depending on the output
        shape.

    """
    client = _get_client(client)
    # When used in asynchronous environment, the `client` object should have
    # `asynchronous` attribute as True.  When invoked by the skl interface, it's
    # responsible for setting up the client.
    return client.sync(
        _inplace_predict_async, global_config=config.get_config(), **locals()
    )


async def _async_wrap_evaluation_matrices(
    client: Optional["distributed.Client"],
    device: Optional[str],
    tree_method: Optional[str],
    max_bin: Optional[int],
    **kwargs: Any,
) -> Tuple[DaskDMatrix, Optional[List[Tuple[DaskDMatrix, str]]]]:
    """A switch function for async environment."""

    def _dispatch(ref: Optional[DaskDMatrix], **kwargs: Any) -> DaskDMatrix:
        if _can_use_qdm(tree_method, device):
            return DaskQuantileDMatrix(
                client=client, ref=ref, max_bin=max_bin, **kwargs
            )
        return DaskDMatrix(client=client, **kwargs)

    train_dmatrix, evals = _wrap_evaluation_matrices(create_dmatrix=_dispatch, **kwargs)
    train_dmatrix = await train_dmatrix
    if evals is None:
        return train_dmatrix, evals
    awaited = []
    for e in evals:
        if e[0] is train_dmatrix:  # already awaited
            awaited.append(e)
            continue
        awaited.append((await e[0], e[1]))
    return train_dmatrix, awaited


@contextmanager
def _set_worker_client(
    model: "DaskScikitLearnBase", client: "distributed.Client"
) -> Generator:
    """Temporarily set the client for sklearn model."""
    try:
        model.client = client
        yield model
    finally:
        model.client = None  # type:ignore


class DaskScikitLearnBase(XGBModel):
    """Base class for implementing scikit-learn interface with Dask"""

    _client = None

    def __init__(self, *, coll_cfg: Optional[CollConfig] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.coll_cfg = coll_cfg

    async def _predict_async(
        self,
        data: _DataT,
        *,
        output_margin: bool,
        validate_features: bool,
        base_margin: Optional[_DaskCollection],
        iteration_range: Optional[IterationRange],
    ) -> Any:
        iteration_range = self._get_iteration_range(iteration_range)
        # Dask doesn't support gblinear and accepts only Dask collection types (array
        # and dataframe). We can perform inplace predict.
        assert self._can_use_inplace_predict()
        predts = await inplace_predict(
            client=self.client,
            model=self.get_booster(),
            data=data,
            iteration_range=iteration_range,
            predict_type="margin" if output_margin else "value",
            missing=self.missing,
            base_margin=base_margin,
            validate_features=validate_features,
        )
        if isinstance(predts, dd.DataFrame):
            predts = predts.to_dask_array()
            # Make sure the booster is part of the task graph implicitly
            # only needed for certain versions of dask.
            if _DASK_2024_12_1() and not _DASK_2025_3_0():
                # Fixes this issue for dask>=2024.1.1,<2025.3.0
                # Dask==2025.3.0 fails with:
                #     RuntimeError: Attempting to use an asynchronous
                #     Client in a synchronous context of `dask.compute`
                #
                # Dask==2025.4.0 fails with:
                #     TypeError: Value type is not supported for data
                #     iterator:<class 'distributed.client.Future'>
                predts = predts.persist()
        return predts

    @_deprecate_positional_args
    def predict(
        self,
        X: _DataT,
        *,
        output_margin: bool = False,
        validate_features: bool = True,
        base_margin: Optional[_DaskCollection] = None,
        iteration_range: Optional[IterationRange] = None,
    ) -> Any:
        return self.client.sync(
            self._predict_async,
            X,
            output_margin=output_margin,
            validate_features=validate_features,
            base_margin=base_margin,
            iteration_range=iteration_range,
        )

    async def _apply_async(
        self,
        X: _DataT,
        iteration_range: Optional[IterationRange] = None,
    ) -> Any:
        iteration_range = self._get_iteration_range(iteration_range)
        test_dmatrix: DaskDMatrix = await DaskDMatrix(  # type: ignore
            self.client,
            data=X,
            missing=self.missing,
            feature_types=self.feature_types,
        )
        predts = await predict(
            self.client,
            model=self.get_booster(),
            data=test_dmatrix,
            pred_leaf=True,
            iteration_range=iteration_range,
        )
        return predts

    def apply(
        self,
        X: _DataT,
        iteration_range: Optional[IterationRange] = None,
    ) -> Any:
        return self.client.sync(self._apply_async, X, iteration_range=iteration_range)

    def __await__(self) -> Awaitable[Any]:
        # Generate a coroutine wrapper to make this class awaitable.
        async def _() -> Awaitable[Any]:
            return self

        return self._client_sync(_).__await__()

    def __getstate__(self) -> Dict:
        this = self.__dict__.copy()
        if "_client" in this:
            del this["_client"]
        return this

    @property
    def client(self) -> "distributed.Client":
        """The dask client used in this model.  The `Client` object can not be
        serialized for transmission, so if task is launched from a worker instead of
        directly from the client process, this attribute needs to be set at that worker.

        """

        client = _get_client(self._client)
        return client

    @client.setter
    def client(self, clt: "distributed.Client") -> None:
        # calling `worker_client' doesn't return the correct `asynchronous` attribute,
        # so we have to pass it ourselves.
        self._asynchronous = clt.asynchronous if clt is not None else False
        self._client = clt

    def _client_sync(self, func: Callable, **kwargs: Any) -> Any:
        """Get the correct client, when method is invoked inside a worker we
        should use `worker_client' instead of default client.

        """

        if self._client is None:
            asynchronous = getattr(self, "_asynchronous", False)
            try:
                distributed.get_worker()
                in_worker = True
            except ValueError:
                in_worker = False
            if in_worker:
                with distributed.worker_client() as client:
                    with _set_worker_client(self, client) as this:
                        ret = this.client.sync(
                            func, **kwargs, asynchronous=asynchronous
                        )
                        return ret
                    return ret

        return self.client.sync(func, **kwargs, asynchronous=self.client.asynchronous)


@xgboost_model_doc(
    """Implementation of the Scikit-Learn API for XGBoost.""", ["estimators", "model"]
)
class DaskXGBRegressor(XGBRegressorBase, DaskScikitLearnBase):
    """dummy doc string to workaround pylint, replaced by the decorator."""

    async def _fit_async(
        self,
        X: _DataT,
        y: _DaskCollection,
        *,
        sample_weight: Optional[_DaskCollection],
        base_margin: Optional[_DaskCollection],
        eval_set: Optional[Sequence[Tuple[_DaskCollection, _DaskCollection]]],
        sample_weight_eval_set: Optional[Sequence[_DaskCollection]],
        base_margin_eval_set: Optional[Sequence[_DaskCollection]],
        verbose: Union[int, bool],
        xgb_model: Optional[Union[Booster, XGBModel]],
        feature_weights: Optional[_DaskCollection],
    ) -> _DaskCollection:
        params = self.get_xgb_params()
        model, metric, params, feature_weights = self._configure_fit(
            xgb_model, params, feature_weights
        )

        dtrain, evals = await _async_wrap_evaluation_matrices(
            client=self.client,
            device=self.device,
            tree_method=self.tree_method,
            max_bin=self.max_bin,
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
            missing=self.missing,
            enable_categorical=self.enable_categorical,
            feature_types=self.feature_types,
        )

        if callable(self.objective):
            obj: Optional[Callable] = _objective_decorator(self.objective)
        else:
            obj = None
        results = await self.client.sync(
            _train_async,
            asynchronous=True,
            client=self.client,
            global_config=config.get_config(),
            dconfig=_get_dask_config(),
            params=params,
            dtrain=dtrain,
            num_boost_round=self.get_num_boosting_rounds(),
            evals=evals,
            obj=obj,
            custom_metric=metric,
            verbose_eval=verbose,
            early_stopping_rounds=self.early_stopping_rounds,
            callbacks=self.callbacks,
            coll_cfg=self.coll_cfg,
            xgb_model=model,
        )
        self._Booster = results["booster"]
        self._set_evaluation_result(results["history"])
        return self

    # pylint: disable=missing-docstring, disable=unused-argument
    @_deprecate_positional_args
    def fit(
        self,
        X: _DataT,
        y: _DaskCollection,
        *,
        sample_weight: Optional[_DaskCollection] = None,
        base_margin: Optional[_DaskCollection] = None,
        eval_set: Optional[Sequence[Tuple[_DaskCollection, _DaskCollection]]] = None,
        verbose: Optional[Union[int, bool]] = True,
        xgb_model: Optional[Union[Booster, str, XGBModel]] = None,
        sample_weight_eval_set: Optional[Sequence[_DaskCollection]] = None,
        base_margin_eval_set: Optional[Sequence[_DaskCollection]] = None,
        feature_weights: Optional[_DaskCollection] = None,
    ) -> "DaskXGBRegressor":
        args = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        return self._client_sync(self._fit_async, **args)


@xgboost_model_doc(
    "Implementation of the scikit-learn API for XGBoost classification.",
    ["estimators", "model"],
)
class DaskXGBClassifier(XGBClassifierBase, DaskScikitLearnBase):
    # pylint: disable=missing-class-docstring
    async def _fit_async(
        self,
        X: _DataT,
        y: _DaskCollection,
        *,
        sample_weight: Optional[_DaskCollection],
        base_margin: Optional[_DaskCollection],
        eval_set: Optional[Sequence[Tuple[_DaskCollection, _DaskCollection]]],
        sample_weight_eval_set: Optional[Sequence[_DaskCollection]],
        base_margin_eval_set: Optional[Sequence[_DaskCollection]],
        verbose: Union[int, bool],
        xgb_model: Optional[Union[Booster, XGBModel]],
        feature_weights: Optional[_DaskCollection],
    ) -> "DaskXGBClassifier":
        params = self.get_xgb_params()
        model, metric, params, feature_weights = self._configure_fit(
            xgb_model, params, feature_weights
        )

        dtrain, evals = await _async_wrap_evaluation_matrices(
            self.client,
            device=self.device,
            tree_method=self.tree_method,
            max_bin=self.max_bin,
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
            missing=self.missing,
            enable_categorical=self.enable_categorical,
            feature_types=self.feature_types,
        )

        # pylint: disable=attribute-defined-outside-init
        if isinstance(y, da.Array):
            self.classes_ = await self.client.compute(da.unique(y))
        else:
            self.classes_ = await self.client.compute(y.drop_duplicates())
        if _is_cudf_ser(self.classes_):
            self.classes_ = self.classes_.to_cupy()
        if _is_cupy_alike(self.classes_):
            self.classes_ = self.classes_.get()
        self.classes_ = numpy.array(self.classes_)
        self.n_classes_ = len(self.classes_)

        if self.n_classes_ > 2:
            params["objective"] = "multi:softprob"
            params["num_class"] = self.n_classes_
        else:
            params["objective"] = "binary:logistic"

        if callable(self.objective):
            obj: Optional[Callable] = _objective_decorator(self.objective)
        else:
            obj = None
        results = await self.client.sync(
            _train_async,
            asynchronous=True,
            client=self.client,
            global_config=config.get_config(),
            dconfig=_get_dask_config(),
            params=params,
            dtrain=dtrain,
            num_boost_round=self.get_num_boosting_rounds(),
            evals=evals,
            obj=obj,
            custom_metric=metric,
            verbose_eval=verbose,
            early_stopping_rounds=self.early_stopping_rounds,
            callbacks=self.callbacks,
            coll_cfg=self.coll_cfg,
            xgb_model=model,
        )
        self._Booster = results["booster"]
        if not callable(self.objective):
            self.objective = params["objective"]
        self._set_evaluation_result(results["history"])
        return self

    # pylint: disable=unused-argument
    def fit(
        self,
        X: _DataT,
        y: _DaskCollection,
        *,
        sample_weight: Optional[_DaskCollection] = None,
        base_margin: Optional[_DaskCollection] = None,
        eval_set: Optional[Sequence[Tuple[_DaskCollection, _DaskCollection]]] = None,
        verbose: Optional[Union[int, bool]] = True,
        xgb_model: Optional[Union[Booster, str, XGBModel]] = None,
        sample_weight_eval_set: Optional[Sequence[_DaskCollection]] = None,
        base_margin_eval_set: Optional[Sequence[_DaskCollection]] = None,
        feature_weights: Optional[_DaskCollection] = None,
    ) -> "DaskXGBClassifier":
        args = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        return self._client_sync(self._fit_async, **args)

    async def _predict_proba_async(
        self,
        X: _DataT,
        validate_features: bool,
        base_margin: Optional[_DaskCollection],
        iteration_range: Optional[IterationRange],
    ) -> _DaskCollection:
        if self.objective == "multi:softmax":
            raise ValueError(
                "multi:softmax doesn't support `predict_proba`.  "
                "Switch to `multi:softproba` instead"
            )
        predts = await super()._predict_async(
            data=X,
            output_margin=False,
            validate_features=validate_features,
            base_margin=base_margin,
            iteration_range=iteration_range,
        )
        vstack = update_wrapper(
            partial(da.vstack, allow_unknown_chunksizes=True), da.vstack
        )
        return _cls_predict_proba(getattr(self, "n_classes_", 0), predts, vstack)

    # pylint: disable=missing-function-docstring
    def predict_proba(
        self,
        X: _DaskCollection,
        validate_features: bool = True,
        base_margin: Optional[_DaskCollection] = None,
        iteration_range: Optional[IterationRange] = None,
    ) -> Any:
        return self._client_sync(
            self._predict_proba_async,
            X=X,
            validate_features=validate_features,
            base_margin=base_margin,
            iteration_range=iteration_range,
        )

    predict_proba.__doc__ = XGBClassifier.predict_proba.__doc__

    async def _predict_async(
        self,
        data: _DataT,
        *,
        output_margin: bool,
        validate_features: bool,
        base_margin: Optional[_DaskCollection],
        iteration_range: Optional[IterationRange],
    ) -> _DaskCollection:
        pred_probs = await super()._predict_async(
            data,
            output_margin=output_margin,
            validate_features=validate_features,
            base_margin=base_margin,
            iteration_range=iteration_range,
        )
        if output_margin:
            return pred_probs

        if len(pred_probs.shape) == 1:
            preds = (pred_probs > 0.5).astype(int)
        else:
            assert len(pred_probs.shape) == 2
            assert isinstance(pred_probs, da.Array)
            # when using da.argmax directly, dask will construct a numpy based return
            # array, which runs into error when computing GPU based prediction.

            def _argmax(x: Any) -> Any:
                return x.argmax(axis=1)

            preds = da.map_blocks(_argmax, pred_probs, drop_axis=1)
        return preds


@xgboost_model_doc(
    """Implementation of the Scikit-Learn API for XGBoost Ranking.

    .. versionadded:: 1.4.0

""",
    ["estimators", "model"],
    extra_parameters="""
    allow_group_split :

        .. versionadded:: 3.0.0

        Whether a query group can be split among multiple workers. When set to `False`,
        inputs must be Dask dataframes or series. If you have many small query groups,
        this can significantly increase the fragmentation of the data, and the internal
        DMatrix construction can take longer.

""",
    end_note="""
        .. note::

            For the dask implementation, group is not supported, use qid instead.
""",
)
class DaskXGBRanker(XGBRankerMixIn, DaskScikitLearnBase):
    @_deprecate_positional_args
    def __init__(
        self,
        *,
        objective: str = "rank:pairwise",
        allow_group_split: bool = False,
        coll_cfg: Optional[CollConfig] = None,
        **kwargs: Any,
    ) -> None:
        if callable(objective):
            raise ValueError("Custom objective function not supported by XGBRanker.")
        self.allow_group_split = allow_group_split
        super().__init__(objective=objective, coll_cfg=coll_cfg, **kwargs)

    def _wrapper_params(self) -> Set[str]:
        params = super()._wrapper_params()
        params.add("allow_group_split")
        return params

    async def _fit_async(
        self,
        X: _DataT,
        y: _DaskCollection,
        *,
        qid: Optional[_DaskCollection],
        sample_weight: Optional[_DaskCollection],
        base_margin: Optional[_DaskCollection],
        eval_set: Optional[Sequence[Tuple[_DaskCollection, _DaskCollection]]],
        sample_weight_eval_set: Optional[Sequence[_DaskCollection]],
        base_margin_eval_set: Optional[Sequence[_DaskCollection]],
        eval_qid: Optional[Sequence[_DaskCollection]],
        verbose: Union[int, bool],
        xgb_model: Optional[Union[XGBModel, Booster]],
        feature_weights: Optional[_DaskCollection],
    ) -> "DaskXGBRanker":
        params = self.get_xgb_params()
        model, metric, params, feature_weights = self._configure_fit(
            xgb_model, params, feature_weights
        )
        dtrain, evals = await _async_wrap_evaluation_matrices(
            self.client,
            device=self.device,
            tree_method=self.tree_method,
            max_bin=self.max_bin,
            X=X,
            y=y,
            group=None,
            qid=qid,
            sample_weight=sample_weight,
            base_margin=base_margin,
            feature_weights=feature_weights,
            eval_set=eval_set,
            sample_weight_eval_set=sample_weight_eval_set,
            base_margin_eval_set=base_margin_eval_set,
            eval_group=None,
            eval_qid=eval_qid,
            missing=self.missing,
            enable_categorical=self.enable_categorical,
            feature_types=self.feature_types,
        )
        results = await self.client.sync(
            _train_async,
            asynchronous=True,
            client=self.client,
            global_config=config.get_config(),
            dconfig=_get_dask_config(),
            params=params,
            dtrain=dtrain,
            num_boost_round=self.get_num_boosting_rounds(),
            evals=evals,
            obj=None,
            custom_metric=metric,
            verbose_eval=verbose,
            early_stopping_rounds=self.early_stopping_rounds,
            callbacks=self.callbacks,
            xgb_model=model,
            coll_cfg=self.coll_cfg,
        )
        self._Booster = results["booster"]
        self.evals_result_ = results["history"]
        return self

    # pylint: disable=unused-argument, arguments-differ
    @_deprecate_positional_args
    def fit(
        self,
        X: _DataT,
        y: _DaskCollection,
        *,
        group: Optional[_DaskCollection] = None,
        qid: Optional[_DaskCollection] = None,
        sample_weight: Optional[_DaskCollection] = None,
        base_margin: Optional[_DaskCollection] = None,
        eval_set: Optional[Sequence[Tuple[_DaskCollection, _DaskCollection]]] = None,
        eval_group: Optional[Sequence[_DaskCollection]] = None,
        eval_qid: Optional[Sequence[_DaskCollection]] = None,
        verbose: Optional[Union[int, bool]] = False,
        xgb_model: Optional[Union[XGBModel, str, Booster]] = None,
        sample_weight_eval_set: Optional[Sequence[_DaskCollection]] = None,
        base_margin_eval_set: Optional[Sequence[_DaskCollection]] = None,
        feature_weights: Optional[_DaskCollection] = None,
    ) -> "DaskXGBRanker":
        msg = "Use the `qid` instead of the `group` with the dask interface."
        if not (group is None and eval_group is None):
            raise ValueError(msg)
        if qid is None:
            raise ValueError("`qid` is required for ranking.")

        def check_df(X: _DaskCollection) -> TypeGuard[dd.DataFrame]:
            if not isinstance(X, dd.DataFrame):
                raise TypeError(
                    "When `allow_group_split` is set to False, X is required to be"
                    " a dataframe."
                )
            return True

        def check_ser(
            qid: Optional[_DaskCollection], name: str
        ) -> TypeGuard[Optional[dd.Series]]:
            if not isinstance(qid, dd.Series) and qid is not None:
                raise TypeError(
                    f"When `allow_group_split` is set to False, {name} is required to be"
                    " a series."
                )
            return True

        if not self.allow_group_split:
            assert (
                check_df(X)
                and check_ser(qid, "qid")
                and check_ser(y, "y")
                and check_ser(sample_weight, "sample_weight")
                and check_ser(base_margin, "base_margin")
            )
            assert qid is not None and y is not None
            X_id = id(X)
            X, qid, y, sample_weight, base_margin = no_group_split(
                self.device,
                X,
                qid,
                y=y,
                sample_weight=sample_weight,
                base_margin=base_margin,
            )

            if eval_set is not None:
                new_eval_set = []
                new_eval_qid = []
                new_sample_weight_eval_set = []
                new_base_margin_eval_set = []
                assert eval_qid
                for i, (Xe, ye) in enumerate(eval_set):
                    we = sample_weight_eval_set[i] if sample_weight_eval_set else None
                    be = base_margin_eval_set[i] if base_margin_eval_set else None
                    assert check_df(Xe)
                    assert eval_qid
                    qe = eval_qid[i]
                    assert (
                        eval_qid
                        and check_ser(qe, "qid")
                        and check_ser(ye, "y")
                        and check_ser(we, "sample_weight")
                        and check_ser(be, "base_margin")
                    )
                    assert qe is not None and ye is not None
                    if id(Xe) != X_id:
                        Xe, qe, ye, we, be = no_group_split(
                            self.device, Xe, qe, ye, we, be
                        )
                    else:
                        Xe, qe, ye, we, be = X, qid, y, sample_weight, base_margin

                    new_eval_set.append((Xe, ye))
                    new_eval_qid.append(qe)

                    if we is not None:
                        new_sample_weight_eval_set.append(we)
                    if be is not None:
                        new_base_margin_eval_set.append(be)

                eval_set = new_eval_set
                eval_qid = new_eval_qid
                sample_weight_eval_set = (
                    new_sample_weight_eval_set if new_sample_weight_eval_set else None
                )
                base_margin_eval_set = (
                    new_base_margin_eval_set if new_base_margin_eval_set else None
                )

        return self._client_sync(
            self._fit_async,
            X=X,
            y=y,
            qid=qid,
            sample_weight=sample_weight,
            base_margin=base_margin,
            eval_set=eval_set,
            eval_qid=eval_qid,
            verbose=verbose,
            xgb_model=xgb_model,
            sample_weight_eval_set=sample_weight_eval_set,
            base_margin_eval_set=base_margin_eval_set,
            feature_weights=feature_weights,
        )

    # FIXME(trivialfis): arguments differ due to additional parameters like group and
    # qid.
    fit.__doc__ = XGBRanker.fit.__doc__


@xgboost_model_doc(
    """Implementation of the Scikit-Learn API for XGBoost Random Forest Regressor.

    .. versionadded:: 1.4.0

""",
    ["model", "objective"],
    extra_parameters="""
    n_estimators : int
        Number of trees in random forest to fit.
""",
)
class DaskXGBRFRegressor(DaskXGBRegressor):
    @_deprecate_positional_args
    def __init__(
        self,
        *,
        learning_rate: Optional[float] = 1,
        subsample: Optional[float] = 0.8,
        colsample_bynode: Optional[float] = 0.8,
        reg_lambda: Optional[float] = 1e-5,
        coll_cfg: Optional[CollConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bynode=colsample_bynode,
            reg_lambda=reg_lambda,
            coll_cfg=coll_cfg,
            **kwargs,
        )

    def get_xgb_params(self) -> Dict[str, Any]:
        params = super().get_xgb_params()
        params["num_parallel_tree"] = self.n_estimators
        return params

    def get_num_boosting_rounds(self) -> int:
        return 1

    # pylint: disable=unused-argument
    def fit(
        self,
        X: _DataT,
        y: _DaskCollection,
        *,
        sample_weight: Optional[_DaskCollection] = None,
        base_margin: Optional[_DaskCollection] = None,
        eval_set: Optional[Sequence[Tuple[_DaskCollection, _DaskCollection]]] = None,
        verbose: Optional[Union[int, bool]] = True,
        xgb_model: Optional[Union[Booster, str, XGBModel]] = None,
        sample_weight_eval_set: Optional[Sequence[_DaskCollection]] = None,
        base_margin_eval_set: Optional[Sequence[_DaskCollection]] = None,
        feature_weights: Optional[_DaskCollection] = None,
    ) -> "DaskXGBRFRegressor":
        args = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        _check_rf_callback(self.early_stopping_rounds, self.callbacks)
        super().fit(**args)
        return self


@xgboost_model_doc(
    """Implementation of the Scikit-Learn API for XGBoost Random Forest Classifier.

    .. versionadded:: 1.4.0

""",
    ["model", "objective"],
    extra_parameters="""
    n_estimators : int
        Number of trees in random forest to fit.
""",
)
class DaskXGBRFClassifier(DaskXGBClassifier):
    @_deprecate_positional_args
    def __init__(
        self,
        *,
        learning_rate: Optional[float] = 1,
        subsample: Optional[float] = 0.8,
        colsample_bynode: Optional[float] = 0.8,
        reg_lambda: Optional[float] = 1e-5,
        coll_cfg: Optional[CollConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bynode=colsample_bynode,
            reg_lambda=reg_lambda,
            coll_cfg=coll_cfg,
            **kwargs,
        )

    def get_xgb_params(self) -> Dict[str, Any]:
        params = super().get_xgb_params()
        params["num_parallel_tree"] = self.n_estimators
        return params

    def get_num_boosting_rounds(self) -> int:
        return 1

    # pylint: disable=unused-argument
    def fit(
        self,
        X: _DataT,
        y: _DaskCollection,
        *,
        sample_weight: Optional[_DaskCollection] = None,
        base_margin: Optional[_DaskCollection] = None,
        eval_set: Optional[Sequence[Tuple[_DaskCollection, _DaskCollection]]] = None,
        verbose: Optional[Union[int, bool]] = True,
        xgb_model: Optional[Union[Booster, str, XGBModel]] = None,
        sample_weight_eval_set: Optional[Sequence[_DaskCollection]] = None,
        base_margin_eval_set: Optional[Sequence[_DaskCollection]] = None,
        feature_weights: Optional[_DaskCollection] = None,
    ) -> "DaskXGBRFClassifier":
        args = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        _check_rf_callback(self.early_stopping_rounds, self.callbacks)
        super().fit(**args)
        return self
