# pylint: disable=too-many-arguments, too-many-locals, no-name-in-module
# pylint: disable=missing-class-docstring, invalid-name
# pylint: disable=too-many-lines, fixme
# pylint: disable=import-error
"""Dask extensions for distributed training. See
https://xgboost.readthedocs.io/en/latest/tutorials/dask.html for simple
tutorial.  Also xgboost/demo/dask for some examples.

There are two sets of APIs in this module, one is the functional API including
``train`` and ``predict`` methods.  Another is stateful Scikit-Learner wrapper
inherited from single-node Scikit-Learn interface.

The implementation is heavily influenced by dask_xgboost:
https://github.com/dask/dask-xgboost

"""
import platform
import logging
from collections import defaultdict
from collections.abc import Sequence
from threading import Thread
from typing import TYPE_CHECKING, List, Tuple, Callable, Optional, Any, Union, Dict, Set
from typing import Awaitable, Generator, TypeVar

import numpy

from . import rabit, config

from .callback import TrainingCallback

from .compat import LazyLoader
from .compat import sparse, scipy_sparse
from .compat import PANDAS_INSTALLED, DataFrame, Series, pandas_concat
from .compat import lazy_isinstance

from .core import DMatrix, DeviceQuantileDMatrix, Booster, _expect, DataIter
from .core import Objective, Metric
from .core import _deprecate_positional_args
from .training import train as worker_train
from .tracker import RabitTracker, get_host_ip
from .sklearn import XGBModel, XGBRegressorBase, XGBClassifierBase
from .sklearn import xgboost_model_doc, _objective_decorator
from .sklearn import _cls_predict_proba
from .sklearn import XGBRanker


if TYPE_CHECKING:
    from dask import dataframe as dd
    from dask import array as da
    import dask
    import distributed
else:
    dd = LazyLoader('dd', globals(), 'dask.dataframe')
    da = LazyLoader('da', globals(), 'dask.array')
    dask = LazyLoader('dask', globals(), 'dask')
    distributed = LazyLoader('distributed', globals(), 'dask.distributed')

_DaskCollection = Union["da.Array", "dd.DataFrame", "dd.Series"]

try:
    from mypy_extensions import TypedDict
    TrainReturnT = TypedDict('TrainReturnT', {
        'booster': Booster,
        'history': Dict,
    })
except ImportError:
    TrainReturnT = Dict[str, Any]  # type:ignore

# Current status is considered as initial support, many features are not properly
# supported yet.
#
# TODOs:
#   - CV
#   - Ranking
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
#       value.  This is caused by Client.sync can return both types depending on context.
#       Right now there's no good way to silent:
#
#         await train(...)
#
#       if train returns an Union type.


LOGGER = logging.getLogger('[xgboost.dask]')


def _start_tracker(n_workers: int) -> Dict[str, Any]:
    """Start Rabit tracker """
    env = {'DMLC_NUM_WORKER': n_workers}
    host = get_host_ip('auto')
    rabit_context = RabitTracker(hostIP=host, nslave=n_workers)
    env.update(rabit_context.slave_envs())

    rabit_context.start(n_workers)
    thread = Thread(target=rabit_context.join)
    thread.daemon = True
    thread.start()
    return env


def _assert_dask_support() -> None:
    try:
        import dask             # pylint: disable=W0621,W0611
    except ImportError as e:
        raise ImportError(
            'Dask needs to be installed in order to use this module') from e

    if platform.system() == 'Windows':
        msg = 'Windows is not officially supported for dask/xgboost,'
        msg += ' contribution are welcomed.'
        LOGGER.warning(msg)


class RabitContext:
    '''A context controling rabit initialization and finalization.'''
    def __init__(self, args: List[bytes]) -> None:
        self.args = args
        worker = distributed.get_worker()
        self.args.append(
            ('DMLC_TASK_ID=[xgboost.dask]:' + str(worker.address)).encode())

    def __enter__(self) -> None:
        rabit.init(self.args)
        LOGGER.debug('-------------- rabit say hello ------------------')

    def __exit__(self, *args: List) -> None:
        rabit.finalize()
        LOGGER.debug('--------------- rabit say bye ------------------')


def concat(value: Any) -> Any:  # pylint: disable=too-many-return-statements
    '''To be replaced with dask builtin.'''
    if isinstance(value[0], numpy.ndarray):
        return numpy.concatenate(value, axis=0)
    if scipy_sparse and isinstance(value[0], scipy_sparse.spmatrix):
        return scipy_sparse.vstack(value, format='csr')
    if sparse and isinstance(value[0], sparse.SparseArray):
        return sparse.concatenate(value, axis=0)
    if PANDAS_INSTALLED and isinstance(value[0], (DataFrame, Series)):
        return pandas_concat(value, axis=0)
    if lazy_isinstance(value[0], 'cudf.core.dataframe', 'DataFrame') or \
       lazy_isinstance(value[0], 'cudf.core.series', 'Series'):
        from cudf import concat as CUDF_concat  # pylint: disable=import-error
        return CUDF_concat(value, axis=0)
    if lazy_isinstance(value[0], 'cupy.core.core', 'ndarray'):
        import cupy
        # pylint: disable=c-extension-no-member,no-member
        d = cupy.cuda.runtime.getDevice()
        for v in value:
            d_v = v.device.id
            assert d_v == d, 'Concatenating arrays on different devices.'
        return cupy.concatenate(value, axis=0)
    return dd.multi.concat(list(value), axis=0)


def _xgb_get_client(client: Optional["distributed.Client"]) -> "distributed.Client":
    '''Simple wrapper around testing None.'''
    if not isinstance(client, (type(distributed.get_client()), type(None))):
        raise TypeError(
            _expect([type(distributed.get_client()), type(None)], type(client)))
    ret = distributed.get_client() if client is None else client
    return ret

# From the implementation point of view, DaskDMatrix complicates a lots of
# things.  A large portion of the code base is about syncing and extracting
# stuffs from DaskDMatrix.  But having an independent data structure gives us a
# chance to perform some specialized optimizations, like building histogram
# index directly.


class DaskDMatrix:
    # pylint: disable=missing-docstring, too-many-instance-attributes
    '''DMatrix holding on references to Dask DataFrame or Dask Array.  Constructing
    a `DaskDMatrix` forces all lazy computation to be carried out.  Wait for
    the input data explicitly if you want to see actual computation of
    constructing `DaskDMatrix`.

    .. note::

        DaskDMatrix does not repartition or move data between workers.  It's
        the caller's responsibility to balance the data.

    .. versionadded:: 1.0.0

    Parameters
    ----------
    client :
        Specify the dask client used for training.  Use default client returned from dask
        if it's set to None.
    data :
        data source of DMatrix.
    label :
        label used for trainin.
    missing :
        Value in the input data (e.g. `numpy.ndarray`) which needs to be present as a
        missing value. If None, defaults to np.nan.
    weight :
        Weight for each instance.
    base_margin :
        Global bias for each instance.
    qid :
        Query ID for ranking.
    label_lower_bound :
        Upper bound for survival training.
    label_upper_bound :
        Lower bound for survival training.
    feature_weights :
        Weight for features used in column sampling.
    feature_names :
        Set names for features.
    feature_types :
        Set types for features

    '''

    @_deprecate_positional_args
    def __init__(
        self,
        client: "distributed.Client",
        data: _DaskCollection,
        label: Optional[_DaskCollection] = None,
        *,
        missing: float = None,
        weight: Optional[_DaskCollection] = None,
        base_margin: Optional[_DaskCollection] = None,
        qid: Optional[_DaskCollection] = None,
        label_lower_bound: Optional[_DaskCollection] = None,
        label_upper_bound: Optional[_DaskCollection] = None,
        feature_weights: Optional[_DaskCollection] = None,
        feature_names: Optional[Union[str, List[str]]] = None,
        feature_types: Optional[Union[Any, List[Any]]] = None
    ) -> None:
        _assert_dask_support()
        client = _xgb_get_client(client)

        self.feature_names = feature_names
        self.feature_types = feature_types
        self.missing = missing

        if qid is not None and weight is not None:
            raise NotImplementedError('per-group weight is not implemented.')

        if len(data.shape) != 2:
            raise ValueError(
                'Expecting 2 dimensional input, got: {shape}'.format(
                    shape=data.shape))

        if not isinstance(data, (dd.DataFrame, da.Array)):
            raise TypeError(_expect((dd.DataFrame, da.Array), type(data)))
        if not isinstance(label, (dd.DataFrame, da.Array, dd.Series,
                                  type(None))):
            raise TypeError(
                _expect((dd.DataFrame, da.Array, dd.Series), type(label)))

        self.worker_map: Dict[str, "distributed.Future"] = defaultdict(list)
        self.is_quantile: bool = False

        self._init = client.sync(self.map_local_data,
                                 client, data, label=label, weights=weight,
                                 base_margin=base_margin,
                                 qid=qid,
                                 feature_weights=feature_weights,
                                 label_lower_bound=label_lower_bound,
                                 label_upper_bound=label_upper_bound)

    def __await__(self) -> Generator:
        return self._init.__await__()

    async def map_local_data(
        self,
        client: "distributed.Client",
        data: _DaskCollection,
        label: Optional[_DaskCollection] = None,
        weights: Optional[_DaskCollection] = None,
        base_margin: Optional[_DaskCollection] = None,
        qid: Optional[_DaskCollection] = None,
        feature_weights: Optional[_DaskCollection] = None,
        label_lower_bound: Optional[_DaskCollection] = None,
        label_upper_bound: Optional[_DaskCollection] = None
    ) -> "DaskDMatrix":
        '''Obtain references to local data.'''

        def inconsistent(
            left: List[Any], left_name: str, right: List[Any], right_name: str
        ) -> str:
            msg = 'Partitions between {a_name} and {b_name} are not ' \
                'consistent: {a_len} != {b_len}.  ' \
                'Please try to repartition/rechunk your data.'.format(
                    a_name=left_name, b_name=right_name, a_len=len(left),
                    b_len=len(right)
                )
            return msg

        def check_columns(parts: Any) -> None:
            # x is required to be 2 dim in __init__
            assert parts.ndim == 1 or parts.shape[1], 'Data should be' \
                ' partitioned by row. To avoid this specify the number' \
                ' of columns for your dask Array explicitly. e.g.' \
                ' chunks=(partition_size, X.shape[1])'

        data = data.persist()
        for meta in [label, weights, base_margin, label_lower_bound,
                     label_upper_bound]:
            if meta is not None:
                meta = meta.persist()
        # Breaking data into partitions, a trick borrowed from dask_xgboost.

        # `to_delayed` downgrades high-level objects into numpy or pandas
        # equivalents.
        X_parts = data.to_delayed()
        if isinstance(X_parts, numpy.ndarray):
            check_columns(X_parts)
            X_parts = X_parts.flatten().tolist()

        def flatten_meta(
            meta: Optional[_DaskCollection]
        ) -> "Optional[List[dask.delayed.Delayed]]":
            if meta is not None:
                meta_parts = meta.to_delayed()
                if isinstance(meta_parts, numpy.ndarray):
                    check_columns(meta_parts)
                    meta_parts = meta_parts.flatten().tolist()
                return meta_parts
            return None

        y_parts = flatten_meta(label)
        w_parts = flatten_meta(weights)
        margin_parts = flatten_meta(base_margin)
        qid_parts = flatten_meta(qid)
        ll_parts = flatten_meta(label_lower_bound)
        lu_parts = flatten_meta(label_upper_bound)

        parts = [X_parts]
        meta_names = []

        def append_meta(
            m_parts: Optional[List["dask.delayed.delayed"]], name: str
        ) -> None:
            if m_parts is not None:
                assert len(X_parts) == len(
                    m_parts), inconsistent(X_parts, 'X', m_parts, name)
                parts.append(m_parts)
                meta_names.append(name)

        append_meta(y_parts, 'labels')
        append_meta(w_parts, 'weights')
        append_meta(margin_parts, 'base_margin')
        append_meta(qid_parts, 'qid')
        append_meta(ll_parts, 'label_lower_bound')
        append_meta(lu_parts, 'label_upper_bound')
        # At this point, `parts` looks like:
        # [(x0, x1, ..), (y0, y1, ..), ..] in delayed form

        # delay the zipped result
        parts = list(map(dask.delayed, zip(*parts)))  # pylint: disable=no-member
        # At this point, the mental model should look like:
        # [(x0, y0, ..), (x1, y1, ..), ..] in delayed form

        parts = client.compute(parts)
        await distributed.wait(parts)  # async wait for parts to be computed

        for part in parts:
            assert part.status == 'finished', part.status

        # Preserving the partition order for prediction.
        self.partition_order = {}
        for i, part in enumerate(parts):
            self.partition_order[part.key] = i

        key_to_partition = {part.key: part for part in parts}
        who_has = await client.scheduler.who_has(keys=[part.key for part in parts])

        worker_map: Dict[str, "distributed.Future"] = defaultdict(list)

        for key, workers in who_has.items():
            worker_map[next(iter(workers))].append(key_to_partition[key])

        self.worker_map = worker_map
        self.meta_names = meta_names

        if feature_weights is None:
            self.feature_weights = None
        else:
            self.feature_weights = await client.compute(feature_weights).result()

        return self

    def create_fn_args(self, worker_addr: str) -> Dict[str, Any]:
        '''Create a dictionary of objects that can be pickled for function
        arguments.

        '''
        return {'feature_names': self.feature_names,
                'feature_types': self.feature_types,
                'feature_weights': self.feature_weights,
                'meta_names': self.meta_names,
                'missing': self.missing,
                'parts': self.worker_map.get(worker_addr, None),
                'is_quantile': self.is_quantile}


_DataParts = List[Tuple[Any, Optional[Any], Optional[Any], Optional[Any], Optional[Any],
                        Optional[Any], Optional[Any]]]


def _get_worker_parts_ordered(
    meta_names: List[str], list_of_parts: _DataParts
) -> _DataParts:
    # List of partitions like: [(x3, y3, w3, m3, ..), ..], order is not preserved.
    assert isinstance(list_of_parts, list)

    result = []

    for i, _ in enumerate(list_of_parts):
        data = list_of_parts[i][0]
        labels = None
        weights = None
        base_margin = None
        qid = None
        label_lower_bound = None
        label_upper_bound = None
        # Iterate through all possible meta info, brings small overhead as in xgboost
        # there are constant number of meta info available.
        for j, blob in enumerate(list_of_parts[i][1:]):
            if meta_names[j] == 'labels':
                labels = blob
            elif meta_names[j] == 'weights':
                weights = blob
            elif meta_names[j] == 'base_margin':
                base_margin = blob
            elif meta_names[j] == 'qid':
                qid = blob
            elif meta_names[j] == 'label_lower_bound':
                label_lower_bound = blob
            elif meta_names[j] == 'label_upper_bound':
                label_upper_bound = blob
            else:
                raise ValueError('Unknown metainfo:', meta_names[j])
        result.append((data, labels, weights, base_margin, qid, label_lower_bound,
                       label_upper_bound))
    return result


def _unzip(list_of_parts: _DataParts) -> List[Tuple[Any, ...]]:
    return list(zip(*list_of_parts))


def _get_worker_parts(
    list_of_parts: _DataParts, meta_names: List[str]
) -> List[Tuple[Any, ...]]:
    partitions = _get_worker_parts_ordered(meta_names, list_of_parts)
    partitions_unzipped = _unzip(partitions)
    return partitions_unzipped


class DaskPartitionIter(DataIter):  # pylint: disable=R0902
    """A data iterator for `DaskDeviceQuantileDMatrix`."""

    def __init__(
        self,
        data: Tuple[Any, ...],
        label: Optional[Tuple[Any, ...]] = None,
        weight: Optional[Tuple[Any, ...]] = None,
        base_margin: Optional[Tuple[Any, ...]] = None,
        qid: Optional[Tuple[Any, ...]] = None,
        label_lower_bound: Optional[Tuple[Any, ...]] = None,
        label_upper_bound: Optional[Tuple[Any, ...]] = None,
        feature_names: Optional[Union[str, List[str]]] = None,
        feature_types: Optional[Union[Any, List[Any]]] = None
    ) -> None:
        self._data = data
        self._labels = label
        self._weights = weight
        self._base_margin = base_margin
        self._qid = qid
        self._label_lower_bound = label_lower_bound
        self._label_upper_bound = label_upper_bound
        self._feature_names = feature_names
        self._feature_types = feature_types

        assert isinstance(self._data, Sequence)

        types = (Sequence, type(None))
        assert isinstance(self._labels, types)
        assert isinstance(self._weights, types)
        assert isinstance(self._base_margin, types)
        assert isinstance(self._label_lower_bound, types)
        assert isinstance(self._label_upper_bound, types)

        self._iter = 0             # set iterator to 0
        super().__init__()

    def data(self) -> Any:
        '''Utility function for obtaining current batch of data.'''
        return self._data[self._iter]

    def labels(self) -> Any:
        '''Utility function for obtaining current batch of label.'''
        if self._labels is not None:
            return self._labels[self._iter]
        return None

    def weights(self) -> Any:
        '''Utility function for obtaining current batch of label.'''
        if self._weights is not None:
            return self._weights[self._iter]
        return None

    def qids(self) -> Any:
        '''Utility function for obtaining current batch of query id.'''
        if self._qid is not None:
            return self._qid[self._iter]
        return None

    def base_margins(self) -> Any:
        '''Utility function for obtaining current batch of base_margin.'''
        if self._base_margin is not None:
            return self._base_margin[self._iter]
        return None

    def label_lower_bounds(self) -> Any:
        '''Utility function for obtaining current batch of label_lower_bound.
        '''
        if self._label_lower_bound is not None:
            return self._label_lower_bound[self._iter]
        return None

    def label_upper_bounds(self) -> Any:
        '''Utility function for obtaining current batch of label_upper_bound.
        '''
        if self._label_upper_bound is not None:
            return self._label_upper_bound[self._iter]
        return None

    def reset(self) -> None:
        '''Reset the iterator'''
        self._iter = 0

    def next(self, input_data: Callable) -> int:
        '''Yield next batch of data'''
        if self._iter == len(self._data):
            # Return 0 when there's no more batch.
            return 0
        feature_names: Optional[Union[List[str], str]] = None
        if self._feature_names:
            feature_names = self._feature_names
        else:
            if hasattr(self.data(), 'columns'):
                feature_names = self.data().columns.format()
            else:
                feature_names = None
        input_data(data=self.data(), label=self.labels(),
                   weight=self.weights(), group=None,
                   qid=self.qids(),
                   label_lower_bound=self.label_lower_bounds(),
                   label_upper_bound=self.label_upper_bounds(),
                   feature_names=feature_names,
                   feature_types=self._feature_types)
        self._iter += 1
        return 1


class DaskDeviceQuantileDMatrix(DaskDMatrix):
    '''Specialized data type for `gpu_hist` tree method.  This class is used to
    reduce the memory usage by eliminating data copies.  Internally the all
    partitions/chunks of data are merged by weighted GK sketching.  So the
    number of partitions from dask may affect training accuracy as GK generates
    bounded error for each merge.

    .. versionadded:: 1.2.0

    Parameters
    ----------
    max_bin : Number of bins for histogram construction.

    '''
    def __init__(
        self,
        client: "distributed.Client",
        data: _DaskCollection,
        label: Optional[_DaskCollection] = None,
        missing: float = None,
        weight: Optional[_DaskCollection] = None,
        base_margin: Optional[_DaskCollection] = None,
        qid: Optional[_DaskCollection] = None,
        label_lower_bound: Optional[_DaskCollection] = None,
        label_upper_bound: Optional[_DaskCollection] = None,
        feature_weights: Optional[_DaskCollection] = None,
        feature_names: Optional[Union[str, List[str]]] = None,
        feature_types: Optional[Union[Any, List[Any]]] = None,
        max_bin: int = 256
    ) -> None:
        super().__init__(
            client=client,
            data=data,
            label=label,
            missing=missing,
            feature_weights=feature_weights,
            weight=weight,
            base_margin=base_margin,
            qid=qid,
            label_lower_bound=label_lower_bound,
            label_upper_bound=label_upper_bound,
            feature_names=feature_names,
            feature_types=feature_types
        )
        self.max_bin = max_bin
        self.is_quantile = True

    def create_fn_args(self, worker_addr: str) -> Dict[str, Any]:
        args = super().create_fn_args(worker_addr)
        args['max_bin'] = self.max_bin
        return args


def _create_device_quantile_dmatrix(
    feature_names: Optional[Union[str, List[str]]],
    feature_types: Optional[Union[Any, List[Any]]],
    feature_weights: Optional[Any],
    meta_names: List[str],
    missing: float,
    parts: Optional[_DataParts],
    max_bin: int
) -> DeviceQuantileDMatrix:
    worker = distributed.get_worker()
    if parts is None:
        msg = 'worker {address} has an empty DMatrix.  '.format(
            address=worker.address)
        LOGGER.warning(msg)
        import cupy
        d = DeviceQuantileDMatrix(cupy.zeros((0, 0)),
                                  feature_names=feature_names,
                                  feature_types=feature_types,
                                  max_bin=max_bin)
        return d

    (data, labels, weights, base_margin, qid,
     label_lower_bound, label_upper_bound) = _get_worker_parts(
         parts, meta_names)
    it = DaskPartitionIter(data=data, label=labels, weight=weights,
                           base_margin=base_margin,
                           qid=qid,
                           label_lower_bound=label_lower_bound,
                           label_upper_bound=label_upper_bound)

    dmatrix = DeviceQuantileDMatrix(it,
                                    missing=missing,
                                    feature_names=feature_names,
                                    feature_types=feature_types,
                                    nthread=worker.nthreads,
                                    max_bin=max_bin)
    dmatrix.set_info(feature_weights=feature_weights)
    return dmatrix


def _create_dmatrix(
    feature_names: Optional[Union[str, List[str]]],
    feature_types: Optional[Union[Any, List[Any]]],
    feature_weights: Optional[Any],
    meta_names: List[str],
    missing: float,
    parts: Optional[_DataParts]
) -> DMatrix:
    '''Get data that local to worker from DaskDMatrix.

      Returns
      -------
      A DMatrix object.

    '''
    worker = distributed.get_worker()
    list_of_parts = parts
    if list_of_parts is None:
        msg = 'worker {address} has an empty DMatrix.  '.format(address=worker.address)
        LOGGER.warning(msg)
        d = DMatrix(numpy.empty((0, 0)),
                    feature_names=feature_names,
                    feature_types=feature_types)
        return d

    T = TypeVar('T')

    def concat_or_none(data: Tuple[Optional[T], ...]) -> Optional[T]:
        if any([part is None for part in data]):
            return None
        return concat(data)

    (data, labels, weights, base_margin, qid,
     label_lower_bound, label_upper_bound) = _get_worker_parts(list_of_parts, meta_names)

    _labels = concat_or_none(labels)
    _weights = concat_or_none(weights)
    _base_margin = concat_or_none(base_margin)
    _qid = concat_or_none(qid)
    _label_lower_bound = concat_or_none(label_lower_bound)
    _label_upper_bound = concat_or_none(label_upper_bound)

    _data = concat(data)
    dmatrix = DMatrix(
        _data,
        _labels,
        missing=missing,
        feature_names=feature_names,
        feature_types=feature_types,
        nthread=worker.nthreads
    )
    dmatrix.set_info(
        base_margin=_base_margin, qid=_qid, weight=_weights,
        label_lower_bound=_label_lower_bound,
        label_upper_bound=_label_upper_bound,
        feature_weights=feature_weights
    )
    return dmatrix


def _dmatrix_from_list_of_parts(
    is_quantile: bool, **kwargs: Any
) -> Union[DMatrix, DeviceQuantileDMatrix]:
    if is_quantile:
        return _create_device_quantile_dmatrix(**kwargs)
    return _create_dmatrix(**kwargs)


async def _get_rabit_args(n_workers: int, client: "distributed.Client") -> List[bytes]:
    '''Get rabit context arguments from data distribution in DaskDMatrix.'''
    env = await client.run_on_scheduler(_start_tracker, n_workers)
    rabit_args = [('%s=%s' % item).encode() for item in env.items()]
    return rabit_args

# train and predict methods are supposed to be "functional", which meets the
# dask paradigm.  But as a side effect, the `evals_result` in single-node API
# is no longer supported since it mutates the input parameter, and it's not
# intuitive to sync the mutation result.  Therefore, a dictionary containing
# evaluation history is instead returned.


def _get_workers_from_data(
    dtrain: DaskDMatrix,
    evals: Optional[List[Tuple[DaskDMatrix, str]]]
) -> Set[str]:
    X_worker_map: Set[str] = set(dtrain.worker_map.keys())
    if evals:
        for e in evals:
            assert len(e) == 2
            assert isinstance(e[0], DaskDMatrix) and isinstance(e[1], str)
            worker_map = set(e[0].worker_map.keys())
            X_worker_map = X_worker_map.union(worker_map)
    return X_worker_map


async def _train_async(
    client: "distributed.Client",
    global_config: Dict[str, Any],
    params: Dict[str, Any],
    dtrain: DaskDMatrix,
    num_boost_round: int,
    evals: Optional[List[Tuple[DaskDMatrix, str]]],
    obj: Optional[Objective],
    feval: Optional[Metric],
    early_stopping_rounds: Optional[int],
    verbose_eval: Union[int, bool],
    xgb_model: Optional[Booster],
    callbacks: Optional[List[TrainingCallback]]
) -> Optional[TrainReturnT]:
    workers = list(_get_workers_from_data(dtrain, evals))
    _rabit_args = await _get_rabit_args(len(workers), client)

    def dispatched_train(
        worker_addr: str,
        rabit_args: List[bytes],
        dtrain_ref: Dict,
        dtrain_idt: int,
        evals_ref: Dict
    ) -> Optional[Dict[str, Union[Booster, Dict]]]:
        '''Perform training on a single worker.  A local function prevents pickling.

        '''
        LOGGER.debug('Training on %s', str(worker_addr))
        worker = distributed.get_worker()
        with RabitContext(rabit_args), config.config_context(**global_config):
            local_dtrain = _dmatrix_from_list_of_parts(**dtrain_ref)
            local_evals = []
            if evals_ref:
                for ref, name, idt in evals_ref:
                    if idt == dtrain_idt:
                        local_evals.append((local_dtrain, name))
                        continue
                    local_evals.append((_dmatrix_from_list_of_parts(**ref), name))

            local_history: Dict = {}
            local_param = params.copy()  # just to be consistent
            msg = 'Overriding `nthreads` defined in dask worker.'
            override = ['nthread', 'n_jobs']
            for p in override:
                val = local_param.get(p, None)
                if val is not None and val != worker.nthreads:
                    LOGGER.info(msg)
                else:
                    local_param[p] = worker.nthreads
            bst = worker_train(params=local_param,
                               dtrain=local_dtrain,
                               num_boost_round=num_boost_round,
                               evals_result=local_history,
                               evals=local_evals,
                               obj=obj,
                               feval=feval,
                               early_stopping_rounds=early_stopping_rounds,
                               verbose_eval=verbose_eval,
                               xgb_model=xgb_model,
                               callbacks=callbacks)
            ret: Optional[Dict[str, Union[Booster, Dict]]] = {
                'booster': bst, 'history': local_history}
            if local_dtrain.num_row() == 0:
                ret = None
            return ret

    # Note for function purity:
    # XGBoost is deterministic in most of the cases, which means train function is
    # supposed to be idempotent.  One known exception is gblinear with shotgun updater.
    # We haven't been able to do a full verification so here we keep pure to be False.
    futures = []
    for i, worker_addr in enumerate(workers):
        if evals:
            evals_per_worker = [(e.create_fn_args(worker_addr), name, id(e))
                                for e, name in evals]
        else:
            evals_per_worker = []
        f = client.submit(dispatched_train,
                          worker_addr,
                          _rabit_args,
                          dtrain.create_fn_args(workers[i]),
                          id(dtrain),
                          evals_per_worker,
                          pure=False,
                          workers=[worker_addr])
        futures.append(f)

    results = await client.gather(futures)
    return list(filter(lambda ret: ret is not None, results))[0]


def train(
    client: "distributed.Client",
    params: Dict[str, Any],
    dtrain: DaskDMatrix,
    num_boost_round: int = 10,
    evals: Optional[List[Tuple[DaskDMatrix, str]]] = None,
    obj: Optional[Objective] = None,
    feval: Optional[Metric] = None,
    early_stopping_rounds: Optional[int] = None,
    xgb_model: Optional[Booster] = None,
    verbose_eval: Union[int, bool] = True,
    callbacks: Optional[List[TrainingCallback]] = None
) -> Any:
    '''Train XGBoost model.

    .. versionadded:: 1.0.0

    .. note::

        Other parameters are the same as `xgboost.train` except for `evals_result`, which
        is returned as part of function return value instead of argument.

    Parameters
    ----------
    client :
        Specify the dask client used for training.  Use default client returned from dask
        if it's set to None.

    Returns
    -------
    results: dict
        A dictionary containing trained booster and evaluation history.  `history` field
        is the same as `eval_result` from `xgboost.train`.

        .. code-block:: python

            {'booster': xgboost.Booster,
             'history': {'train': {'logloss': ['0.48253', '0.35953']},
                         'eval': {'logloss': ['0.480385', '0.357756']}}}
    '''
    _assert_dask_support()
    client = _xgb_get_client(client)
    # Get global configuration before transferring computation to another thread or
    # process.
    global_config = config.get_config()
    return client.sync(_train_async,
                       client=client,
                       global_config=global_config,
                       num_boost_round=num_boost_round,
                       obj=obj,
                       feval=feval,
                       params=params,
                       dtrain=dtrain,
                       evals=evals,
                       early_stopping_rounds=early_stopping_rounds,
                       verbose_eval=verbose_eval,
                       xgb_model=xgb_model,
                       callbacks=callbacks)


async def _direct_predict_impl(
    client: "distributed.Client",
    data: _DaskCollection,
    predict_fn: Callable
) -> _DaskCollection:
    if isinstance(data, da.Array):
        predictions = await client.submit(
            da.map_blocks,
            predict_fn, data, False, drop_axis=1,
            dtype=numpy.float32
        ).result()
        return predictions
    if isinstance(data, dd.DataFrame):
        predictions = await client.submit(
            dd.map_partitions,
            predict_fn, data, True,
            meta=dd.utils.make_meta({'prediction': 'f4'})
        ).result()
        return predictions.iloc[:, 0]
    raise TypeError('data of type: ' + str(type(data)) +
                    ' is not supported by direct prediction')


# pylint: disable=too-many-statements
async def _predict_async(
    client: "distributed.Client",
    global_config: Dict[str, Any],
    model: Union[Booster, Dict],
    data: _DaskCollection,
    output_margin: bool,
    missing: float,
    pred_leaf: bool,
    pred_contribs: bool,
    approx_contribs: bool,
    pred_interactions: bool,
    validate_features: bool
) -> _DaskCollection:
    if isinstance(model, Booster):
        booster = model
    elif isinstance(model, dict):
        booster = model['booster']
    else:
        raise TypeError(_expect([Booster, dict], type(model)))
    if not isinstance(data, (DaskDMatrix, da.Array, dd.DataFrame)):
        raise TypeError(_expect([DaskDMatrix, da.Array, dd.DataFrame],
                                type(data)))

    def mapped_predict(partition: Any, is_df: bool) -> Any:
        worker = distributed.get_worker()
        with config.config_context(**global_config):
            booster.set_param({'nthread': worker.nthreads})
            m = DMatrix(partition, missing=missing, nthread=worker.nthreads)
            predt = booster.predict(
                data=m,
                output_margin=output_margin,
                pred_leaf=pred_leaf,
                pred_contribs=pred_contribs,
                approx_contribs=approx_contribs,
                pred_interactions=pred_interactions,
                validate_features=validate_features
            )
            if is_df:
                if lazy_isinstance(partition, 'cudf', 'core.dataframe.DataFrame'):
                    import cudf
                    predt = cudf.DataFrame(predt, columns=['prediction'])
                else:
                    predt = DataFrame(predt, columns=['prediction'])
            return predt
    # Predict on dask collection directly.
    if isinstance(data, (da.Array, dd.DataFrame)):
        return await _direct_predict_impl(client, data, mapped_predict)

    # Prediction on dask DMatrix.
    worker_map = data.worker_map
    partition_order = data.partition_order
    feature_names = data.feature_names
    feature_types = data.feature_types
    missing = data.missing
    meta_names = data.meta_names

    def dispatched_predict(
            worker_id: int, list_of_orders: List[int], list_of_parts: _DataParts
    ) -> List[Tuple[Tuple["dask.delayed.Delayed", int], int]]:
        '''Perform prediction on each worker.'''
        LOGGER.debug('Predicting on %d', worker_id)
        with config.config_context(**global_config):
            worker = distributed.get_worker()
            list_of_parts = _get_worker_parts_ordered(meta_names, list_of_parts)
            predictions = []

            booster.set_param({'nthread': worker.nthreads})
            for i, parts in enumerate(list_of_parts):
                (data, _, _, base_margin, _, _, _) = parts
                order = list_of_orders[i]
                local_part = DMatrix(
                    data,
                    base_margin=base_margin,
                    feature_names=feature_names,
                    feature_types=feature_types,
                    missing=missing,
                    nthread=worker.nthreads
                )
                predt = booster.predict(
                    data=local_part,
                    output_margin=output_margin,
                    pred_leaf=pred_leaf,
                    pred_contribs=pred_contribs,
                    approx_contribs=approx_contribs,
                    pred_interactions=pred_interactions,
                    validate_features=validate_features
                )
                columns = 1 if len(predt.shape) == 1 else predt.shape[1]
                ret = ((dask.delayed(predt), columns), order)  # pylint: disable=no-member
                predictions.append(ret)

            return predictions

    def dispatched_get_shape(
        worker_id: int, list_of_orders: List[int], list_of_parts: _DataParts
    ) -> List[Tuple[int, int]]:
        '''Get shape of data in each worker.'''
        LOGGER.debug('Get shape on %d', worker_id)
        list_of_parts = _get_worker_parts_ordered(meta_names, list_of_parts)
        shapes = []
        for i, parts in enumerate(list_of_parts):
            (data, _, _, _, _, _, _) = parts
            shapes.append((data.shape, list_of_orders[i]))
        return shapes

    async def map_function(
        func: Callable[[int, List[int], _DataParts], Any]
    ) -> List[Any]:
        '''Run function for each part of the data.'''
        futures = []
        workers_address = list(worker_map.keys())
        for wid, worker_addr in enumerate(workers_address):
            worker_addr = workers_address[wid]
            list_of_parts = worker_map[worker_addr]
            list_of_orders = [partition_order[part.key] for part in list_of_parts]

            f = client.submit(func, worker_id=wid,
                              list_of_orders=list_of_orders,
                              list_of_parts=list_of_parts,
                              pure=True, workers=[worker_addr])
            assert isinstance(f, distributed.client.Future)
            futures.append(f)
        # Get delayed objects
        results = await client.gather(futures)
        # flatten into 1 dim list
        results = [t for list_per_worker in results for t in list_per_worker]
        # sort by order, l[0] is the delayed object, l[1] is its order
        results = sorted(results, key=lambda l: l[1])
        results = [predt for predt, order in results]  # remove order
        return results

    results = await map_function(dispatched_predict)
    shapes = await map_function(dispatched_get_shape)

    # Constructing a dask array from list of numpy arrays
    # See https://docs.dask.org/en/latest/array-creation.html
    arrays = []
    for i, shape in enumerate(shapes):
        arrays.append(da.from_delayed(
            results[i][0], shape=(shape[0],)
            if results[i][1] == 1 else (shape[0], results[i][1]),
            dtype=numpy.float32))
    predictions = await da.concatenate(arrays, axis=0)
    return predictions


def predict(
    client: "distributed.Client",
    model: Union[TrainReturnT, Booster],
    data: Union[DaskDMatrix, _DaskCollection],
    output_margin: bool = False,
    missing: float = numpy.nan,
    pred_leaf: bool = False,
    pred_contribs: bool = False,
    approx_contribs: bool = False,
    pred_interactions: bool = False,
    validate_features: bool = True
) -> Any:
    '''Run prediction with a trained booster.

    .. note::

        Only default prediction mode is supported right now.

    .. versionadded:: 1.0.0

    Parameters
    ----------
    client:
        Specify the dask client used for training.  Use default client
        returned from dask if it's set to None.
    model:
        The trained model.
    data:
        Input data used for prediction.  When input is a dataframe object,
        prediction output is a series.
    missing:
        Used when input data is not DaskDMatrix.  Specify the value
        considered as missing.

    Returns
    -------
    prediction: dask.array.Array/dask.dataframe.Series

    '''
    _assert_dask_support()
    client = _xgb_get_client(client)
    global_config = config.get_config()
    return client.sync(
        _predict_async, client, global_config, model, data,
        output_margin=output_margin,
        missing=missing,
        pred_leaf=pred_leaf,
        pred_contribs=pred_contribs,
        approx_contribs=approx_contribs,
        pred_interactions=pred_interactions,
        validate_features=validate_features
    )


async def _inplace_predict_async(
    client: "distributed.Client",
    global_config: Dict[str, Any],
    model: Union[Booster, Dict],
    data: _DaskCollection,
    iteration_range: Tuple[int, int] = (0, 0),
    predict_type: str = 'value',
    missing: float = numpy.nan
) -> _DaskCollection:
    client = _xgb_get_client(client)
    if isinstance(model, Booster):
        booster = model
    elif isinstance(model, dict):
        booster = model['booster']
    else:
        raise TypeError(_expect([Booster, dict], type(model)))
    if not isinstance(data, (da.Array, dd.DataFrame)):
        raise TypeError(_expect([da.Array, dd.DataFrame], type(data)))

    def mapped_predict(data: Any, is_df: bool) -> Any:
        worker = distributed.get_worker()
        config.set_config(**global_config)
        booster.set_param({'nthread': worker.nthreads})
        prediction = booster.inplace_predict(
            data,
            iteration_range=iteration_range,
            predict_type=predict_type,
            missing=missing)
        if is_df:
            if lazy_isinstance(data, 'cudf.core.dataframe', 'DataFrame'):
                import cudf
                prediction = cudf.DataFrame({'prediction': prediction},
                                            dtype=numpy.float32)
            else:
                # If it's  from pandas, the partition is a numpy array
                prediction = DataFrame(prediction, columns=['prediction'],
                                       dtype=numpy.float32)
        return prediction

    return await _direct_predict_impl(client, data, mapped_predict)


def inplace_predict(
    client: "distributed.Client",
    model: Union[TrainReturnT, Booster],
    data: _DaskCollection,
    iteration_range: Tuple[int, int] = (0, 0),
    predict_type: str = 'value',
    missing: float = numpy.nan
) -> Any:
    '''Inplace prediction.

    .. versionadded:: 1.1.0

    Parameters
    ----------
    client:
        Specify the dask client used for training.  Use default client
        returned from dask if it's set to None.
    model:
        The trained model.
    iteration_range:
        Specify the range of trees used for prediction.
    predict_type:
        * 'value': Normal prediction result.
        * 'margin': Output the raw untransformed margin value.
    missing:
        Value in the input data which needs to be present as a missing
        value. If None, defaults to np.nan.

    Returns
    -------
    prediction
    '''
    _assert_dask_support()
    client = _xgb_get_client(client)
    global_config = config.get_config()
    return client.sync(_inplace_predict_async, client, global_config, model=model,
                       data=data,
                       iteration_range=iteration_range,
                       predict_type=predict_type,
                       missing=missing)


async def _evaluation_matrices(
    client: "distributed.Client",
    validation_set: Optional[List[Tuple[_DaskCollection, _DaskCollection]]],
    sample_weight: Optional[List[_DaskCollection]],
    sample_qid: Optional[List[_DaskCollection]],
    missing: float
) -> Optional[List[Tuple[DaskDMatrix, str]]]:
    '''
    Parameters
    ----------
    validation_set: list of tuples
        Each tuple contains a validation dataset including input X and label y.
        E.g.:

        .. code-block:: python

          [(X_0, y_0), (X_1, y_1), ... ]

    sample_weights: list of arrays
        The weight vector for validation data.

    Returns
    -------
    evals: list of validation DMatrix
    '''
    evals: Optional[List[Tuple[DaskDMatrix, str]]] = []
    if validation_set is not None:
        assert isinstance(validation_set, list)
        for i, e in enumerate(validation_set):
            w = sample_weight[i] if sample_weight is not None else None
            qid = sample_qid[i] if sample_qid is not None else None
            dmat = await DaskDMatrix(client=client, data=e[0], label=e[1],
                                     weight=w, missing=missing, qid=qid)
            assert isinstance(evals, list)
            evals.append((dmat, 'validation_{}'.format(i)))
    else:
        evals = None
    return evals


class DaskScikitLearnBase(XGBModel):
    '''Base class for implementing scikit-learn interface with Dask'''

    _client = None

    @_deprecate_positional_args
    async def _predict_async(
        self, data: _DaskCollection,
        output_margin: bool = False,
        validate_features: bool = True,
        base_margin: Optional[_DaskCollection] = None
    ) -> Any:
        test_dmatrix = await DaskDMatrix(
            client=self.client, data=data, base_margin=base_margin,
            missing=self.missing
        )
        pred_probs = await predict(client=self.client,
                                   model=self.get_booster(), data=test_dmatrix,
                                   output_margin=output_margin,
                                   validate_features=validate_features)
        return pred_probs

    def predict(
        self,
        X: _DaskCollection,
        output_margin: bool = False,
        ntree_limit: Optional[int] = None,
        validate_features: bool = True,
        base_margin: Optional[_DaskCollection] = None
    ) -> Any:
        _assert_dask_support()
        msg = '`ntree_limit` is not supported on dask, use model slicing instead.'
        assert ntree_limit is None, msg
        return self.client.sync(
            self._predict_async,
            X,
            output_margin=output_margin,
            validate_features=validate_features,
            base_margin=base_margin
        )

    def __await__(self) -> Awaitable[Any]:
        # Generate a coroutine wrapper to make this class awaitable.
        async def _() -> Awaitable[Any]:
            return self
        return self.client.sync(_).__await__()

    @property
    def client(self) -> "distributed.Client":
        '''The dask client used in this model.'''
        client = _xgb_get_client(self._client)
        return client

    @client.setter
    def client(self, clt: "distributed.Client") -> None:
        self._client = clt


@xgboost_model_doc(
    """Implementation of the Scikit-Learn API for XGBoost.""", ["estimators", "model"]
)
class DaskXGBRegressor(DaskScikitLearnBase, XGBRegressorBase):
    # pylint: disable=missing-class-docstring
    async def _fit_async(
        self,
        X: _DaskCollection,
        y: _DaskCollection,
        sample_weight: Optional[_DaskCollection],
        base_margin: Optional[_DaskCollection],
        eval_set: Optional[List[Tuple[_DaskCollection, _DaskCollection]]],
        eval_metric: Optional[Union[str, List[str], Metric]],
        sample_weight_eval_set: Optional[List[_DaskCollection]],
        early_stopping_rounds: int,
        verbose: bool,
        xgb_model: Optional[Union[Booster, XGBModel]],
        feature_weights: Optional[_DaskCollection],
        callbacks: Optional[List[TrainingCallback]],
    ) -> _DaskCollection:
        dtrain = await DaskDMatrix(
            client=self.client,
            data=X,
            label=y,
            weight=sample_weight,
            base_margin=base_margin,
            feature_weights=feature_weights,
            missing=self.missing,
        )
        params = self.get_xgb_params()
        evals = await _evaluation_matrices(
            self.client, eval_set, sample_weight_eval_set, None, self.missing
        )

        if callable(self.objective):
            obj = _objective_decorator(self.objective)
        else:
            obj = None
        model, metric, params = self._configure_fit(
            booster=xgb_model, eval_metric=eval_metric, params=params
        )
        results = await train(
            client=self.client,
            params=params,
            dtrain=dtrain,
            num_boost_round=self.get_num_boosting_rounds(),
            evals=evals,
            feval=metric,
            obj=obj,
            verbose_eval=verbose,
            early_stopping_rounds=early_stopping_rounds,
            callbacks=callbacks,
            xgb_model=model,
        )
        self._Booster = results["booster"]
        # pylint: disable=attribute-defined-outside-init
        self.evals_result_ = results["history"]
        return self

    # pylint: disable=missing-docstring
    @_deprecate_positional_args
    def fit(
        self,
        X: _DaskCollection,
        y: _DaskCollection,
        *,
        sample_weight: Optional[_DaskCollection] = None,
        base_margin: Optional[_DaskCollection] = None,
        eval_set: Optional[List[Tuple[_DaskCollection, _DaskCollection]]] = None,
        eval_metric: Optional[Union[str, List[str], Metric]] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: bool = True,
        xgb_model: Optional[Union[Booster, XGBModel]] = None,
        sample_weight_eval_set: Optional[List[_DaskCollection]] = None,
        feature_weights: Optional[_DaskCollection] = None,
        callbacks: Optional[List[TrainingCallback]] = None
    ) -> "DaskXGBRegressor":
        _assert_dask_support()
        return self.client.sync(
            self._fit_async,
            X=X,
            y=y,
            sample_weight=sample_weight,
            base_margin=base_margin,
            eval_set=eval_set,
            eval_metric=eval_metric,
            sample_weight_eval_set=sample_weight_eval_set,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
            xgb_model=xgb_model,
            feature_weights=feature_weights,
            callbacks=callbacks,
        )


@xgboost_model_doc(
    'Implementation of the scikit-learn API for XGBoost classification.',
    ['estimators', 'model'])
class DaskXGBClassifier(DaskScikitLearnBase, XGBClassifierBase):
    # pylint: disable=missing-class-docstring
    async def _fit_async(
        self, X: _DaskCollection, y: _DaskCollection,
        sample_weight: Optional[_DaskCollection],
        base_margin: Optional[_DaskCollection],
        eval_set: Optional[List[Tuple[_DaskCollection, _DaskCollection]]],
        eval_metric: Optional[Union[str, List[str], Metric]],
        sample_weight_eval_set: Optional[List[_DaskCollection]],
        early_stopping_rounds: int,
        verbose: bool,
        xgb_model: Optional[Union[Booster, XGBModel]],
        feature_weights: Optional[_DaskCollection],
        callbacks: Optional[List[TrainingCallback]]
    ) -> "DaskXGBClassifier":
        dtrain = await DaskDMatrix(client=self.client,
                                   data=X,
                                   label=y,
                                   weight=sample_weight,
                                   base_margin=base_margin,
                                   feature_weights=feature_weights,
                                   missing=self.missing)
        params = self.get_xgb_params()

        # pylint: disable=attribute-defined-outside-init
        if isinstance(y, (da.Array)):
            self.classes_ = await self.client.compute(da.unique(y))
        else:
            self.classes_ = await self.client.compute(y.drop_duplicates())
        self.n_classes_ = len(self.classes_)

        if self.n_classes_ > 2:
            params["objective"] = "multi:softprob"
            params['num_class'] = self.n_classes_
        else:
            params["objective"] = "binary:logistic"

        evals = await _evaluation_matrices(self.client, eval_set,
                                           sample_weight_eval_set,
                                           None,
                                           self.missing)

        if callable(self.objective):
            obj = _objective_decorator(self.objective)
        else:
            obj = None
        model, metric, params = self._configure_fit(
            booster=xgb_model,
            eval_metric=eval_metric,
            params=params
        )
        results = await train(
            client=self.client,
            params=params,
            dtrain=dtrain,
            num_boost_round=self.get_num_boosting_rounds(),
            evals=evals,
            obj=obj,
            feval=metric,
            verbose_eval=verbose,
            early_stopping_rounds=early_stopping_rounds,
            callbacks=callbacks,
            xgb_model=model,
        )
        self._Booster = results['booster']

        if not callable(self.objective):
            self.objective = params["objective"]

        # pylint: disable=attribute-defined-outside-init
        self.evals_result_ = results['history']
        return self

    @_deprecate_positional_args
    def fit(
        self,
        X: _DaskCollection,
        y: _DaskCollection,
        *,
        sample_weight: Optional[_DaskCollection] = None,
        base_margin: Optional[_DaskCollection] = None,
        eval_set: Optional[List[Tuple[_DaskCollection, _DaskCollection]]] = None,
        eval_metric: Optional[Union[str, List[str], Metric]] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: bool = True,
        xgb_model: Optional[Union[Booster, XGBModel]] = None,
        sample_weight_eval_set: Optional[List[_DaskCollection]] = None,
        feature_weights: Optional[_DaskCollection] = None,
        callbacks: Optional[List[TrainingCallback]] = None
    ) -> "DaskXGBClassifier":
        _assert_dask_support()
        return self.client.sync(
            self._fit_async,
            X=X,
            y=y,
            sample_weight=sample_weight,
            base_margin=base_margin,
            eval_set=eval_set,
            eval_metric=eval_metric,
            sample_weight_eval_set=sample_weight_eval_set,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
            xgb_model=xgb_model,
            feature_weights=feature_weights,
            callbacks=callbacks,
        )

    async def _predict_proba_async(
        self,
        X: _DaskCollection,
        validate_features: bool,
        output_margin: bool,
        base_margin: Optional[_DaskCollection]
    ) -> _DaskCollection:
        test_dmatrix = await DaskDMatrix(
            client=self.client, data=X, base_margin=base_margin,
            missing=self.missing
        )
        pred_probs = await predict(client=self.client,
                                   model=self.get_booster(),
                                   data=test_dmatrix,
                                   validate_features=validate_features,
                                   output_margin=output_margin)
        return _cls_predict_proba(self.objective, pred_probs, da.vstack)

    # pylint: disable=missing-docstring
    def predict_proba(
        self,
        X: _DaskCollection,
        ntree_limit: Optional[int] = None,
        validate_features: bool = True,
        output_margin: bool = False,
        base_margin: Optional[_DaskCollection] = None
    ) -> Any:
        _assert_dask_support()
        msg = '`ntree_limit` is not supported on dask, use model slicing instead.'
        assert ntree_limit is None, msg
        return self.client.sync(
            self._predict_proba_async,
            X=X,
            validate_features=validate_features,
            output_margin=output_margin,
            base_margin=base_margin
        )

    async def _predict_async(
        self, data: _DaskCollection,
        output_margin: bool = False,
        validate_features: bool = True,
        base_margin: Optional[_DaskCollection] = None
    ) -> _DaskCollection:
        pred_probs = await super()._predict_async(
            data, output_margin, validate_features, base_margin
        )
        if output_margin:
            return pred_probs

        if self.n_classes_ == 2:
            preds = (pred_probs > 0.5).astype(int)
        else:
            preds = da.argmax(pred_probs, axis=1)

        return preds


@xgboost_model_doc(
    "Implementation of the Scikit-Learn API for XGBoost Ranking.",
    ["estimators", "model"],
    end_note="""
        Note
        ----
        For dask implementation, group is not supported, use qid instead.
""",
)
class DaskXGBRanker(DaskScikitLearnBase):
    @_deprecate_positional_args
    def __init__(self, *, objective: str = "rank:pairwise", **kwargs: Any):
        if callable(objective):
            raise ValueError("Custom objective function not supported by XGBRanker.")
        super().__init__(objective=objective, kwargs=kwargs)

    async def _fit_async(
        self,
        X: _DaskCollection,
        y: _DaskCollection,
        qid: Optional[_DaskCollection],
        sample_weight: Optional[_DaskCollection],
        base_margin: Optional[_DaskCollection],
        eval_set: Optional[List[Tuple[_DaskCollection, _DaskCollection]]],
        sample_weight_eval_set: Optional[List[_DaskCollection]],
        eval_qid: Optional[List[_DaskCollection]],
        eval_metric: Optional[Union[str, List[str], Metric]],
        early_stopping_rounds: int,
        verbose: bool,
        xgb_model: Optional[Union[XGBModel, Booster]],
        feature_weights: Optional[_DaskCollection],
        callbacks: Optional[List[TrainingCallback]],
    ) -> "DaskXGBRanker":
        dtrain = await DaskDMatrix(
            client=self.client,
            data=X,
            label=y,
            qid=qid,
            weight=sample_weight,
            base_margin=base_margin,
            feature_weights=feature_weights,
            missing=self.missing,
        )
        params = self.get_xgb_params()
        evals = await _evaluation_matrices(
            self.client,
            eval_set,
            sample_weight_eval_set,
            sample_qid=eval_qid,
            missing=self.missing,
        )
        if eval_metric is not None:
            if callable(eval_metric):
                raise ValueError(
                    'Custom evaluation metric is not yet supported for XGBRanker.')
        model, metric, params = self._configure_fit(
            booster=xgb_model,
            eval_metric=eval_metric,
            params=params
        )
        results = await train(
            client=self.client,
            params=params,
            dtrain=dtrain,
            num_boost_round=self.get_num_boosting_rounds(),
            evals=evals,
            feval=metric,
            verbose_eval=verbose,
            early_stopping_rounds=early_stopping_rounds,
            callbacks=callbacks,
            xgb_model=model,
        )
        self._Booster = results["booster"]
        self.evals_result_ = results["history"]
        return self

    @_deprecate_positional_args
    def fit(  # pylint: disable=arguments-differ
        self,
        X: _DaskCollection,
        y: _DaskCollection,
        *,
        group: Optional[_DaskCollection] = None,
        qid: Optional[_DaskCollection] = None,
        sample_weight: Optional[_DaskCollection] = None,
        base_margin: Optional[_DaskCollection] = None,
        eval_set: Optional[List[Tuple[_DaskCollection, _DaskCollection]]] = None,
        sample_weight_eval_set: Optional[List[_DaskCollection]] = None,
        eval_group: Optional[List[_DaskCollection]] = None,
        eval_qid: Optional[List[_DaskCollection]] = None,
        eval_metric: Optional[Union[str, List[str], Metric]] = None,
        early_stopping_rounds: int = None,
        verbose: bool = False,
        xgb_model: Optional[Union[XGBModel, Booster]] = None,
        feature_weights: Optional[_DaskCollection] = None,
        callbacks: Optional[List[TrainingCallback]] = None
    ) -> "DaskXGBRanker":
        _assert_dask_support()
        msg = "Use `qid` instead of `group` on dask interface."
        if not (group is None and eval_group is None):
            raise ValueError(msg)
        if qid is None:
            raise ValueError("`qid` is required for ranking.")
        return self.client.sync(
            self._fit_async,
            X=X,
            y=y,
            qid=qid,
            sample_weight=sample_weight,
            base_margin=base_margin,
            eval_set=eval_set,
            sample_weight_eval_set=sample_weight_eval_set,
            eval_qid=eval_qid,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
            xgb_model=xgb_model,
            feature_weights=feature_weights,
            callbacks=callbacks,
        )

    # FIXME(trivialfis): arguments differ due to additional parameters like group and qid.
    fit.__doc__ = XGBRanker.fit.__doc__


@xgboost_model_doc(
    "Implementation of the Scikit-Learn API for XGBoost Random Forest Regressor.",
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
        **kwargs: Any
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bynode=colsample_bynode,
            reg_lambda=reg_lambda,
            **kwargs
        )

    def get_xgb_params(self) -> Dict[str, Any]:
        params = super().get_xgb_params()
        params["num_parallel_tree"] = self.n_estimators
        return params

    def get_num_boosting_rounds(self) -> int:
        return 1


@xgboost_model_doc(
    "Implementation of the Scikit-Learn API for XGBoost Random Forest Classifier.",
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
        **kwargs: Any
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bynode=colsample_bynode,
            reg_lambda=reg_lambda,
            **kwargs
        )

    def get_xgb_params(self) -> Dict[str, Any]:
        params = super().get_xgb_params()
        params["num_parallel_tree"] = self.n_estimators
        return params

    def get_num_boosting_rounds(self) -> int:
        return 1
