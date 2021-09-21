# pylint: disable=too-many-arguments, too-many-locals, no-name-in-module
# pylint: disable=missing-class-docstring, invalid-name
# pylint: disable=too-many-lines, fixme
# pylint: disable=too-few-public-methods
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
from contextlib import contextmanager
from collections import defaultdict
from collections.abc import Sequence
from threading import Thread
from functools import partial, update_wrapper
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
from .sklearn import XGBModel, XGBClassifier, XGBRegressorBase, XGBClassifierBase
from .sklearn import _wrap_evaluation_matrices, _objective_decorator, _check_rf_callback
from .sklearn import XGBRankerMixIn
from .sklearn import xgboost_model_doc
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

__all__ = [
    "RabitContext",
    "DaskDMatrix",
    "DaskDeviceQuantileDMatrix",
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
#       value.  This is caused by Client.sync can return both types depending on context.
#       Right now there's no good way to silent:
#
#         await train(...)
#
#       if train returns an Union type.


LOGGER = logging.getLogger('[xgboost.dask]')


def _multi_lock() -> Any:
    """MultiLock is only available on latest distributed.  See:

    https://github.com/dask/distributed/pull/4503

"""
    try:
        from distributed import MultiLock
    except ImportError:
        class MultiLock:        # type:ignore
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def __enter__(self) -> "MultiLock":
                return self

            def __exit__(self, *args: Any, **kwargs: Any) -> None:
                return

            async def __aenter__(self) -> "MultiLock":
                return self

            async def __aexit__(self, *args: Any, **kwargs: Any) -> None:
                return

    return MultiLock


def _start_tracker(n_workers: int) -> Dict[str, Any]:
    """Start Rabit tracker """
    env = {'DMLC_NUM_WORKER': n_workers}
    host = get_host_ip('auto')
    rabit_context = RabitTracker(hostIP=host, nslave=n_workers, use_logger=False)
    env.update(rabit_context.slave_envs())

    rabit_context.start(n_workers)
    thread = Thread(target=rabit_context.join)
    thread.daemon = True
    thread.start()
    return env


def _assert_dask_support() -> None:
    try:
        import dask  # pylint: disable=W0621,W0611
    except ImportError as e:
        raise ImportError(
            "Dask needs to be installed in order to use this module"
        ) from e

    if platform.system() == "Windows":
        msg = "Windows is not officially supported for dask/xgboost,"
        msg += " contribution are welcomed."
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
    if lazy_isinstance(value[0], 'cupy._core.core', 'ndarray'):
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
    '''DMatrix holding on references to Dask DataFrame or Dask Array.  Constructing a
    `DaskDMatrix` forces all lazy computation to be carried out.  Wait for the input data
    explicitly if you want to see actual computation of constructing `DaskDMatrix`.

    See doc for :py:obj:`xgboost.DMatrix` constructor for other parameters.  DaskDMatrix
    accepts only dask collection.

    .. note::

        DaskDMatrix does not repartition or move data between workers.  It's
        the caller's responsibility to balance the data.

    .. versionadded:: 1.0.0

    Parameters
    ----------
    client :
        Specify the dask client used for training.  Use default client returned from dask
        if it's set to None.

    '''

    @_deprecate_positional_args
    def __init__(
        self,
        client: "distributed.Client",
        data: _DaskCollection,
        label: Optional[_DaskCollection] = None,
        *,
        weight: Optional[_DaskCollection] = None,
        base_margin: Optional[_DaskCollection] = None,
        missing: float = None,
        silent: bool = False,   # pylint: disable=unused-argument
        feature_names: Optional[Union[str, List[str]]] = None,
        feature_types: Optional[Union[Any, List[Any]]] = None,
        group: Optional[_DaskCollection] = None,
        qid: Optional[_DaskCollection] = None,
        label_lower_bound: Optional[_DaskCollection] = None,
        label_upper_bound: Optional[_DaskCollection] = None,
        feature_weights: Optional[_DaskCollection] = None,
        enable_categorical: bool = False
    ) -> None:
        _assert_dask_support()
        client = _xgb_get_client(client)

        self.feature_names = feature_names
        self.feature_types = feature_types
        self.missing = missing
        self.enable_categorical = enable_categorical

        if qid is not None and weight is not None:
            raise NotImplementedError("per-group weight is not implemented.")
        if group is not None:
            raise NotImplementedError(
                "group structure is not implemented, use qid instead."
            )

        if len(data.shape) != 2:
            raise ValueError(
                f"Expecting 2 dimensional input, got: {data.shape}"
            )

        if not isinstance(data, (dd.DataFrame, da.Array)):
            raise TypeError(_expect((dd.DataFrame, da.Array), type(data)))
        if not isinstance(label, (dd.DataFrame, da.Array, dd.Series, type(None))):
            raise TypeError(_expect((dd.DataFrame, da.Array, dd.Series), type(label)))

        self._n_cols = data.shape[1]
        assert isinstance(self._n_cols, int)
        self.worker_map: Dict[str, "distributed.Future"] = defaultdict(list)
        self.is_quantile: bool = False

        self._init = client.sync(
            self._map_local_data,
            client,
            data,
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
        client: "distributed.Client",
        data: _DaskCollection,
        label: Optional[_DaskCollection] = None,
        weights: Optional[_DaskCollection] = None,
        base_margin: Optional[_DaskCollection] = None,
        qid: Optional[_DaskCollection] = None,
        feature_weights: Optional[_DaskCollection] = None,
        label_lower_bound: Optional[_DaskCollection] = None,
        label_upper_bound: Optional[_DaskCollection] = None,
    ) -> "DaskDMatrix":
        '''Obtain references to local data.'''

        def inconsistent(
            left: List[Any], left_name: str, right: List[Any], right_name: str
        ) -> str:
            msg = (f"Partitions between {left_name} and {right_name} are not "
                   f"consistent: {len(left)} != {len(right)}.  "
                   f"Please try to repartition/rechunk your data.")
            return msg

        def check_columns(parts: Any) -> None:
            # x is required to be 2 dim in __init__
            assert parts.ndim == 1 or parts.shape[1], 'Data should be' \
                ' partitioned by row. To avoid this specify the number' \
                ' of columns for your dask Array explicitly. e.g.' \
                ' chunks=(partition_size, X.shape[1])'

        data = client.persist(data)
        for meta in [label, weights, base_margin, label_lower_bound,
                     label_upper_bound]:
            if meta is not None:
                meta = client.persist(meta)
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

    def _create_fn_args(self, worker_addr: str) -> Dict[str, Any]:
        '''Create a dictionary of objects that can be pickled for function
        arguments.

        '''
        return {'feature_names': self.feature_names,
                'feature_types': self.feature_types,
                'feature_weights': self.feature_weights,
                'meta_names': self.meta_names,
                'missing': self.missing,
                'enable_categorical': self.enable_categorical,
                'parts': self.worker_map.get(worker_addr, None),
                'is_quantile': self.is_quantile}

    def num_col(self) -> int:
        return self._n_cols


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
    '''Specialized data type for `gpu_hist` tree method.  This class is used to reduce the
    memory usage by eliminating data copies.  Internally the all partitions/chunks of data
    are merged by weighted GK sketching.  So the number of partitions from dask may affect
    training accuracy as GK generates bounded error for each merge.  See doc string for
    :py:obj:`xgboost.DeviceQuantileDMatrix` and :py:obj:`xgboost.DMatrix` for other
    parameters.

    .. versionadded:: 1.2.0

    Parameters
    ----------
    max_bin : Number of bins for histogram construction.

    '''
    @_deprecate_positional_args
    def __init__(
        self,
        client: "distributed.Client",
        data: _DaskCollection,
        label: Optional[_DaskCollection] = None,
        *,
        weight: Optional[_DaskCollection] = None,
        base_margin: Optional[_DaskCollection] = None,
        missing: float = None,
        silent: bool = False,   # disable=unused-argument
        feature_names: Optional[Union[str, List[str]]] = None,
        feature_types: Optional[Union[Any, List[Any]]] = None,
        max_bin: int = 256,
        group: Optional[_DaskCollection] = None,
        qid: Optional[_DaskCollection] = None,
        label_lower_bound: Optional[_DaskCollection] = None,
        label_upper_bound: Optional[_DaskCollection] = None,
        feature_weights: Optional[_DaskCollection] = None,
        enable_categorical: bool = False,
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
        self.is_quantile = True

    def _create_fn_args(self, worker_addr: str) -> Dict[str, Any]:
        args = super()._create_fn_args(worker_addr)
        args["max_bin"] = self.max_bin
        return args


def _create_device_quantile_dmatrix(
    feature_names: Optional[Union[str, List[str]]],
    feature_types: Optional[Union[Any, List[Any]]],
    feature_weights: Optional[Any],
    meta_names: List[str],
    missing: float,
    parts: Optional[_DataParts],
    max_bin: int,
    enable_categorical: bool,
) -> DeviceQuantileDMatrix:
    worker = distributed.get_worker()
    if parts is None:
        msg = f"worker {worker.address} has an empty DMatrix."
        LOGGER.warning(msg)
        import cupy

        d = DeviceQuantileDMatrix(
            cupy.zeros((0, 0)),
            feature_names=feature_names,
            feature_types=feature_types,
            max_bin=max_bin,
            enable_categorical=enable_categorical,
        )
        return d

    (
        data,
        labels,
        weights,
        base_margin,
        qid,
        label_lower_bound,
        label_upper_bound,
    ) = _get_worker_parts(parts, meta_names)
    it = DaskPartitionIter(
        data=data,
        label=labels,
        weight=weights,
        base_margin=base_margin,
        qid=qid,
        label_lower_bound=label_lower_bound,
        label_upper_bound=label_upper_bound,
    )

    dmatrix = DeviceQuantileDMatrix(
        it,
        missing=missing,
        feature_names=feature_names,
        feature_types=feature_types,
        nthread=worker.nthreads,
        max_bin=max_bin,
        enable_categorical=enable_categorical,
    )
    dmatrix.set_info(feature_weights=feature_weights)
    return dmatrix


def _create_dmatrix(
    feature_names: Optional[Union[str, List[str]]],
    feature_types: Optional[Union[Any, List[Any]]],
    feature_weights: Optional[Any],
    meta_names: List[str],
    missing: float,
    enable_categorical: bool,
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
        msg = f"worker {worker.address} has an empty DMatrix."
        LOGGER.warning(msg)
        d = DMatrix(
            numpy.empty((0, 0)),
            feature_names=feature_names,
            feature_types=feature_types,
            enable_categorical=enable_categorical,
        )
        return d

    T = TypeVar('T')

    def concat_or_none(data: Tuple[Optional[T], ...]) -> Optional[T]:
        if any(part is None for part in data):
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
        nthread=worker.nthreads,
        enable_categorical=enable_categorical,
    )
    dmatrix.set_info(
        base_margin=_base_margin,
        qid=_qid,
        weight=_weights,
        label_lower_bound=_label_lower_bound,
        label_upper_bound=_label_upper_bound,
        feature_weights=feature_weights,
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
    rabit_args = [f"{k}={v}".encode() for k, v in env.items()]
    return rabit_args

# train and predict methods are supposed to be "functional", which meets the
# dask paradigm.  But as a side effect, the `evals_result` in single-node API
# is no longer supported since it mutates the input parameter, and it's not
# intuitive to sync the mutation result.  Therefore, a dictionary containing
# evaluation history is instead returned.


def _get_workers_from_data(
    dtrain: DaskDMatrix,
    evals: Optional[List[Tuple[DaskDMatrix, str]]]
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
    callbacks: Optional[List[TrainingCallback]],
) -> Optional[TrainReturnT]:
    workers = _get_workers_from_data(dtrain, evals)
    _rabit_args = await _get_rabit_args(len(workers), client)

    if params.get("booster", None) == "gblinear":
        raise NotImplementedError(
            f"booster `{params['booster']}` is not yet supported for dask."
        )

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
    async with _multi_lock()(workers, client):
        futures = []
        for worker_addr in workers:
            if evals:
                # pylint: disable=protected-access
                evals_per_worker = [
                    (e._create_fn_args(worker_addr), name, id(e)) for e, name in evals
                ]
            else:
                evals_per_worker = []
            f = client.submit(
                dispatched_train,
                worker_addr,
                _rabit_args,
                # pylint: disable=protected-access
                dtrain._create_fn_args(worker_addr),
                id(dtrain),
                evals_per_worker,
                pure=False,
                workers=[worker_addr],
                allow_other_workers=False
            )
            futures.append(f)

        results = await client.gather(futures, asynchronous=True)

        return list(filter(lambda ret: ret is not None, results))[0]


def train(                      # pylint: disable=unused-argument
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
    callbacks: Optional[List[TrainingCallback]] = None,
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

    """
    _assert_dask_support()
    client = _xgb_get_client(client)
    args = locals()
    return client.sync(_train_async, global_config=config.get_config(), **args)


def _can_output_df(is_df: bool, output_shape: Tuple) -> bool:
    return is_df and len(output_shape) <= 2


def _maybe_dataframe(
    data: Any, prediction: Any, columns: List[int], is_df: bool
) -> Any:
    """Return dataframe for prediction when applicable."""
    if _can_output_df(is_df, prediction.shape):
        # Need to preserve the index for dataframe.
        # See issue: https://github.com/dmlc/xgboost/issues/6939
        # In older versions of dask, the partition is actually a numpy array when input is
        # dataframe.
        index = getattr(data, "index", None)
        if lazy_isinstance(data, "cudf.core.dataframe", "DataFrame"):
            import cudf

            prediction = cudf.DataFrame(
                prediction, columns=columns, dtype=numpy.float32, index=index
            )
        else:
            prediction = DataFrame(
                prediction, columns=columns, dtype=numpy.float32, index=index
            )
    return prediction


async def _direct_predict_impl(  # pylint: disable=too-many-branches
    mapped_predict: Callable,
    booster: "distributed.Future",
    data: _DaskCollection,
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
            base_margin_df: Optional[dd.DataFrame] = base_margin.to_dask_dataframe()
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
            chunks: Optional[List[Tuple]] = list(data.chunks)
            assert isinstance(chunks, list)
            chunks[1] = (output_shape[1], )
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
    m = DMatrix(test_sample)
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
    if isinstance(model, Booster):
        booster = await client.scatter(model, broadcast=True)
    elif isinstance(model, dict):
        booster = await client.scatter(model["booster"], broadcast=True)
    elif isinstance(model, distributed.Future):
        booster = model
        if booster.type is not Booster:
            raise TypeError(
                f"Underlying type of model future should be `Booster`, got {booster.type}"
            )
    else:
        raise TypeError(_expect([Booster, dict, distributed.Future], type(model)))
    return booster


# pylint: disable=too-many-statements
async def _predict_async(
    client: "distributed.Client",
    global_config: Dict[str, Any],
    model: Union[Booster, Dict, "distributed.Future"],
    data: _DaskCollection,
    output_margin: bool,
    missing: float,
    pred_leaf: bool,
    pred_contribs: bool,
    approx_contribs: bool,
    pred_interactions: bool,
    validate_features: bool,
    iteration_range: Tuple[int, int],
    strict_shape: bool,
) -> _DaskCollection:
    _booster = await _get_model_future(client, model)
    if not isinstance(data, (DaskDMatrix, da.Array, dd.DataFrame)):
        raise TypeError(_expect([DaskDMatrix, da.Array, dd.DataFrame], type(data)))

    def mapped_predict(
        booster: Booster, partition: Any, is_df: bool, columns: List[int], _: Any
    ) -> Any:
        with config.config_context(**global_config):
            m = DMatrix(data=partition, missing=missing)
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
            mapped_predict, _booster, data, None, _output_shape, meta
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
    meta_names = data.meta_names

    def dispatched_predict(booster: Booster, part: Tuple) -> numpy.ndarray:
        data = part[0]
        assert isinstance(part, tuple), type(part)
        base_margin = None
        for i, blob in enumerate(part[1:]):
            if meta_names[i] == "base_margin":
                base_margin = blob
        with config.config_context(**global_config):
            m = DMatrix(
                data,
                missing=missing,
                base_margin=base_margin,
                feature_names=feature_names,
                feature_types=feature_types,
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
        s = client.submit(lambda part: part[0].shape[0], part, workers=[w])
        all_shapes.append(s)
    all_shapes = await client.gather(all_shapes)

    parts_with_order = list(zip(all_parts, all_shapes, all_orders))
    parts_with_order = sorted(parts_with_order, key=lambda p: p[2])
    all_parts = [part for part, shape, order in parts_with_order]
    all_shapes = [shape for part, shape, order in parts_with_order]

    futures = []
    for w, part in zip(all_workers, all_parts):
        f = client.submit(dispatched_predict, _booster, part, workers=[w])
        futures.append(f)

    # Constructing a dask array from list of numpy arrays
    # See https://docs.dask.org/en/latest/array-creation.html
    arrays = []
    for i, rows in enumerate(all_shapes):
        arrays.append(
            da.from_delayed(
                futures[i], shape=(rows,) + output_shape[1:], dtype=numpy.float32
            )
        )
    predictions = da.concatenate(arrays, axis=0)
    return predictions


def predict(                    # pylint: disable=unused-argument
    client: "distributed.Client",
    model: Union[TrainReturnT, Booster, "distributed.Future"],
    data: Union[DaskDMatrix, _DaskCollection],
    output_margin: bool = False,
    missing: float = numpy.nan,
    pred_leaf: bool = False,
    pred_contribs: bool = False,
    approx_contribs: bool = False,
    pred_interactions: bool = False,
    validate_features: bool = True,
    iteration_range: Tuple[int, int] = (0, 0),
    strict_shape: bool = False,
) -> Any:
    '''Run prediction with a trained booster.

    .. note::

        Using ``inplace_predict`` might be faster when some features are not needed.  See
        :py:meth:`xgboost.Booster.predict` for details on various parameters.  When output
        has more than 2 dimensions (shap value, leaf with strict_shape), input should be
        ``da.Array`` or ``DaskDMatrix``.

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
        When input data is ``dask.array.Array`` or ``DaskDMatrix``, the return value is an
        array, when input data is ``dask.dataframe.DataFrame``, return value can be
        ``dask.dataframe.Series``, ``dask.dataframe.DataFrame``, depending on the output
        shape.

    '''
    _assert_dask_support()
    client = _xgb_get_client(client)
    return client.sync(_predict_async, global_config=config.get_config(), **locals())


async def _inplace_predict_async(  # pylint: disable=too-many-branches
    client: "distributed.Client",
    global_config: Dict[str, Any],
    model: Union[Booster, Dict, "distributed.Future"],
    data: _DaskCollection,
    iteration_range: Tuple[int, int],
    predict_type: str,
    missing: float,
    validate_features: bool,
    base_margin: Optional[_DaskCollection],
    strict_shape: bool,
) -> _DaskCollection:
    client = _xgb_get_client(client)
    booster = await _get_model_future(client, model)
    if not isinstance(data, (da.Array, dd.DataFrame)):
        raise TypeError(_expect([da.Array, dd.DataFrame], type(data)))
    if base_margin is not None and not isinstance(
        data, (da.Array, dd.DataFrame, dd.Series)
    ):
        raise TypeError(_expect([da.Array, dd.DataFrame, dd.Series], type(base_margin)))

    def mapped_predict(
        booster: Booster, partition: Any, is_df: bool, columns: List[int], base_margin: Any
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
        mapped_predict, booster, data, base_margin, shape, meta
    )


def inplace_predict(  # pylint: disable=unused-argument
    client: "distributed.Client",
    model: Union[TrainReturnT, Booster, "distributed.Future"],
    data: _DaskCollection,
    iteration_range: Tuple[int, int] = (0, 0),
    predict_type: str = "value",
    missing: float = numpy.nan,
    validate_features: bool = True,
    base_margin: Optional[_DaskCollection] = None,
    strict_shape: bool = False,
) -> Any:
    """Inplace prediction. See doc in :py:meth:`xgboost.Booster.inplace_predict` for details.

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
        See :py:obj:`xgboost.DMatrix` for details. Right now classifier is not well
        supported with base_margin as it requires the size of base margin to be `n_classes
        * n_samples`.

        .. versionadded:: 1.4.0

    strict_shape:
        See :py:meth:`xgboost.Booster.predict` for details.

        .. versionadded:: 1.4.0

    Returns
    -------
    prediction :
        When input data is ``dask.array.Array``, the return value is an array, when input
        data is ``dask.dataframe.DataFrame``, return value can be
        ``dask.dataframe.Series``, ``dask.dataframe.DataFrame``, depending on the output
        shape.

    """
    _assert_dask_support()
    client = _xgb_get_client(client)
    # When used in asynchronous environment, the `client` object should have
    # `asynchronous` attribute as True.  When invoked by the skl interface, it's
    # responsible for setting up the client.
    return client.sync(
        _inplace_predict_async, global_config=config.get_config(), **locals()
    )


async def _async_wrap_evaluation_matrices(
    client: "distributed.Client", **kwargs: Any
) -> Tuple[DaskDMatrix, Optional[List[Tuple[DaskDMatrix, str]]]]:
    """A switch function for async environment."""

    def _inner(**kwargs: Any) -> DaskDMatrix:
        m = DaskDMatrix(client=client, **kwargs)
        return m

    train_dmatrix, evals = _wrap_evaluation_matrices(create_dmatrix=_inner, **kwargs)
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
        model.client = None


class DaskScikitLearnBase(XGBModel):
    """Base class for implementing scikit-learn interface with Dask"""

    _client = None

    async def _predict_async(
        self,
        data: _DaskCollection,
        output_margin: bool,
        validate_features: bool,
        base_margin: Optional[_DaskCollection],
        iteration_range: Optional[Tuple[int, int]],
    ) -> Any:
        iteration_range = self._get_iteration_range(iteration_range)
        if self._can_use_inplace_predict():
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
        else:
            test_dmatrix = await DaskDMatrix(
                self.client, data=data, base_margin=base_margin, missing=self.missing
            )
            predts = await predict(
                self.client,
                model=self.get_booster(),
                data=test_dmatrix,
                output_margin=output_margin,
                validate_features=validate_features,
                iteration_range=iteration_range,
            )
        return predts

    def predict(
        self,
        X: _DaskCollection,
        output_margin: bool = False,
        ntree_limit: Optional[int] = None,
        validate_features: bool = True,
        base_margin: Optional[_DaskCollection] = None,
        iteration_range: Optional[Tuple[int, int]] = None,
    ) -> Any:
        _assert_dask_support()
        msg = "`ntree_limit` is not supported on dask, use `iteration_range` instead."
        assert ntree_limit is None, msg
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
        X: _DaskCollection,
        iteration_range: Optional[Tuple[int, int]] = None,
    ) -> Any:
        iteration_range = self._get_iteration_range(iteration_range)
        test_dmatrix = await DaskDMatrix(self.client, data=X, missing=self.missing)
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
        X: _DaskCollection,
        ntree_limit: Optional[int] = None,
        iteration_range: Optional[Tuple[int, int]] = None,
    ) -> Any:
        _assert_dask_support()
        msg = "`ntree_limit` is not supported on dask, use `iteration_range` instead."
        assert ntree_limit is None, msg
        return self.client.sync(self._apply_async, X, iteration_range=iteration_range)

    def __await__(self) -> Awaitable[Any]:
        # Generate a coroutine wrapper to make this class awaitable.
        async def _() -> Awaitable[Any]:
            return self

        return self._client_sync(_).__await__()

    def __getstate__(self) -> Dict:
        this = self.__dict__.copy()
        if "_client" in this.keys():
            del this["_client"]
        return this

    @property
    def client(self) -> "distributed.Client":
        """The dask client used in this model.  The `Client` object can not be serialized for
        transmission, so if task is launched from a worker instead of directly from the
        client process, this attribute needs to be set at that worker.

        """

        client = _xgb_get_client(self._client)
        return client

    @client.setter
    def client(self, clt: "distributed.Client") -> None:
        # calling `worker_client' doesn't return the correct `asynchronous` attribute, so
        # we have to pass it ourselves.
        self._asynchronous = clt.asynchronous if clt is not None else False
        self._client = clt

    def _client_sync(self, func: Callable, **kwargs: Any) -> Any:
        """Get the correct client, when method is invoked inside a worker we
        should use `worker_client' instead of default client.

        """
        asynchronous = getattr(self, "_asynchronous", False)
        if self._client is None:
            try:
                distributed.get_worker()
                in_worker = True
            except ValueError:
                in_worker = False
            if in_worker:
                with distributed.worker_client() as client:
                    with _set_worker_client(self, client) as this:
                        ret = this.client.sync(func, **kwargs, asynchronous=asynchronous)
                        return ret
                    return ret

        return self.client.sync(func, **kwargs, asynchronous=asynchronous)


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
        base_margin_eval_set: Optional[List[_DaskCollection]],
        early_stopping_rounds: int,
        verbose: bool,
        xgb_model: Optional[Union[Booster, XGBModel]],
        feature_weights: Optional[_DaskCollection],
        callbacks: Optional[List[TrainingCallback]],
    ) -> _DaskCollection:
        params = self.get_xgb_params()
        dtrain, evals = await _async_wrap_evaluation_matrices(
            client=self.client,
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
        )

        if callable(self.objective):
            obj: Optional[Callable] = _objective_decorator(self.objective)
        else:
            obj = None
        model, metric, params = self._configure_fit(
            booster=xgb_model, eval_metric=eval_metric, params=params
        )
        results = await self.client.sync(
            _train_async,
            asynchronous=True,
            client=self.client,
            global_config=config.get_config(),
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
        self._Booster = results["booster"]
        self._set_evaluation_result(results["history"])
        return self

    # pylint: disable=missing-docstring, disable=unused-argument
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
        base_margin_eval_set: Optional[List[_DaskCollection]] = None,
        feature_weights: Optional[_DaskCollection] = None,
        callbacks: Optional[List[TrainingCallback]] = None,
    ) -> "DaskXGBRegressor":
        _assert_dask_support()
        args = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        return self._client_sync(self._fit_async, **args)


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
        base_margin_eval_set: Optional[List[_DaskCollection]],
        early_stopping_rounds: int,
        verbose: bool,
        xgb_model: Optional[Union[Booster, XGBModel]],
        feature_weights: Optional[_DaskCollection],
        callbacks: Optional[List[TrainingCallback]]
    ) -> "DaskXGBClassifier":
        params = self.get_xgb_params()
        dtrain, evals = await _async_wrap_evaluation_matrices(
            self.client,
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
        )

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

        if callable(self.objective):
            obj: Optional[Callable] = _objective_decorator(self.objective)
        else:
            obj = None
        model, metric, params = self._configure_fit(
            booster=xgb_model, eval_metric=eval_metric, params=params
        )
        results = await self.client.sync(
            _train_async,
            asynchronous=True,
            client=self.client,
            global_config=config.get_config(),
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
        self._set_evaluation_result(results["history"])
        return self

    # pylint: disable=unused-argument
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
        base_margin_eval_set: Optional[List[_DaskCollection]] = None,
        feature_weights: Optional[_DaskCollection] = None,
        callbacks: Optional[List[TrainingCallback]] = None
    ) -> "DaskXGBClassifier":
        _assert_dask_support()
        args = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        return self._client_sync(self._fit_async, **args)

    async def _predict_proba_async(
        self,
        X: _DaskCollection,
        validate_features: bool,
        base_margin: Optional[_DaskCollection],
        iteration_range: Optional[Tuple[int, int]],
    ) -> _DaskCollection:
        predts = await super()._predict_async(
            data=X,
            output_margin=self.objective == "multi:softmax",
            validate_features=validate_features,
            base_margin=base_margin,
            iteration_range=iteration_range,
        )
        vstack = update_wrapper(
            partial(da.vstack, allow_unknown_chunksizes=True), da.vstack
        )
        return _cls_predict_proba(getattr(self, "n_classes_", None), predts, vstack)

    # pylint: disable=missing-function-docstring
    def predict_proba(
        self,
        X: _DaskCollection,
        ntree_limit: Optional[int] = None,
        validate_features: bool = True,
        base_margin: Optional[_DaskCollection] = None,
        iteration_range: Optional[Tuple[int, int]] = None,
    ) -> Any:
        _assert_dask_support()
        msg = "`ntree_limit` is not supported on dask, use `iteration_range` instead."
        assert ntree_limit is None, msg
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
        data: _DaskCollection,
        output_margin: bool,
        validate_features: bool,
        base_margin: Optional[_DaskCollection],
        iteration_range: Optional[Tuple[int, int]],
    ) -> _DaskCollection:
        pred_probs = await super()._predict_async(
            data, output_margin, validate_features, base_margin, iteration_range
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
    end_note="""
        Note
        ----
        For dask implementation, group is not supported, use qid instead.
""",
)
class DaskXGBRanker(DaskScikitLearnBase, XGBRankerMixIn):
    @_deprecate_positional_args
    def __init__(self, *, objective: str = "rank:pairwise", **kwargs: Any):
        if callable(objective):
            raise ValueError("Custom objective function not supported by XGBRanker.")
        super().__init__(objective=objective, kwargs=kwargs)

    async def _fit_async(
        self,
        X: _DaskCollection,
        y: _DaskCollection,
        group: Optional[_DaskCollection],
        qid: Optional[_DaskCollection],
        sample_weight: Optional[_DaskCollection],
        base_margin: Optional[_DaskCollection],
        eval_set: Optional[List[Tuple[_DaskCollection, _DaskCollection]]],
        sample_weight_eval_set: Optional[List[_DaskCollection]],
        base_margin_eval_set: Optional[List[_DaskCollection]],
        eval_group: Optional[List[_DaskCollection]],
        eval_qid: Optional[List[_DaskCollection]],
        eval_metric: Optional[Union[str, List[str], Metric]],
        early_stopping_rounds: int,
        verbose: bool,
        xgb_model: Optional[Union[XGBModel, Booster]],
        feature_weights: Optional[_DaskCollection],
        callbacks: Optional[List[TrainingCallback]],
    ) -> "DaskXGBRanker":
        msg = "Use `qid` instead of `group` on dask interface."
        if not (group is None and eval_group is None):
            raise ValueError(msg)
        if qid is None:
            raise ValueError("`qid` is required for ranking.")
        params = self.get_xgb_params()
        dtrain, evals = await _async_wrap_evaluation_matrices(
            self.client,
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
        )
        if eval_metric is not None:
            if callable(eval_metric):
                raise ValueError(
                    "Custom evaluation metric is not yet supported for XGBRanker."
                )
        model, metric, params = self._configure_fit(
            booster=xgb_model, eval_metric=eval_metric, params=params
        )
        results = await self.client.sync(
            _train_async,
            asynchronous=True,
            client=self.client,
            global_config=config.get_config(),
            params=params,
            dtrain=dtrain,
            num_boost_round=self.get_num_boosting_rounds(),
            evals=evals,
            obj=None,
            feval=metric,
            verbose_eval=verbose,
            early_stopping_rounds=early_stopping_rounds,
            callbacks=callbacks,
            xgb_model=model,
        )
        self._Booster = results["booster"]
        self.evals_result_ = results["history"]
        return self

    # pylint: disable=unused-argument, arguments-differ
    @_deprecate_positional_args
    def fit(
        self,
        X: _DaskCollection,
        y: _DaskCollection,
        *,
        group: Optional[_DaskCollection] = None,
        qid: Optional[_DaskCollection] = None,
        sample_weight: Optional[_DaskCollection] = None,
        base_margin: Optional[_DaskCollection] = None,
        eval_set: Optional[List[Tuple[_DaskCollection, _DaskCollection]]] = None,
        eval_group: Optional[List[_DaskCollection]] = None,
        eval_qid: Optional[List[_DaskCollection]] = None,
        eval_metric: Optional[Union[str, List[str], Metric]] = None,
        early_stopping_rounds: int = None,
        verbose: bool = False,
        xgb_model: Optional[Union[XGBModel, Booster]] = None,
        sample_weight_eval_set: Optional[List[_DaskCollection]] = None,
        base_margin_eval_set: Optional[List[_DaskCollection]] = None,
        feature_weights: Optional[_DaskCollection] = None,
        callbacks: Optional[List[TrainingCallback]] = None
    ) -> "DaskXGBRanker":
        _assert_dask_support()
        args = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        return self._client_sync(self._fit_async, **args)

    # FIXME(trivialfis): arguments differ due to additional parameters like group and qid.
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

    # pylint: disable=unused-argument
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
        base_margin_eval_set: Optional[List[_DaskCollection]] = None,
        feature_weights: Optional[_DaskCollection] = None,
        callbacks: Optional[List[TrainingCallback]] = None
    ) -> "DaskXGBRFRegressor":
        _assert_dask_support()
        args = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        _check_rf_callback(early_stopping_rounds, callbacks)
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

    # pylint: disable=unused-argument
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
        base_margin_eval_set: Optional[List[_DaskCollection]] = None,
        feature_weights: Optional[_DaskCollection] = None,
        callbacks: Optional[List[TrainingCallback]] = None
    ) -> "DaskXGBRFClassifier":
        _assert_dask_support()
        args = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        _check_rf_callback(early_stopping_rounds, callbacks)
        super().fit(**args)
        return self
