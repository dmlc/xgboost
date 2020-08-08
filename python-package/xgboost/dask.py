# pylint: disable=too-many-arguments, too-many-locals
# pylint: disable=missing-class-docstring, invalid-name
# pylint: disable=too-many-lines
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

import numpy

from . import rabit

from .compat import DASK_INSTALLED
from .compat import distributed_get_worker, distributed_wait, distributed_comm
from .compat import da, dd, delayed, get_client
from .compat import sparse, scipy_sparse
from .compat import PANDAS_INSTALLED, DataFrame, Series, pandas_concat
from .compat import CUDF_concat
from .compat import lazy_isinstance

from .core import DMatrix, DeviceQuantileDMatrix, Booster, _expect, DataIter
from .training import train as worker_train
from .tracker import RabitTracker
from .sklearn import XGBModel, XGBRegressorBase, XGBClassifierBase
from .sklearn import xgboost_model_doc

try:
    from distributed import Client
except ImportError:
    Client = None

# Current status is considered as initial support, many features are
# not properly supported yet.
#
# TODOs:
#   - Callback.
#   - Label encoding.
#   - CV
#   - Ranking
#
# Note for developers:

#   As of writing asyncio is still a new feature of Python and in depth
#   documentation is rare.  Best examples of various asyncio tricks are in dask
#   (luckily).  Classes like Client, Worker are awaitable.  Some general rules
#   for the implementation here:
#     - Synchronous world is different from asynchronous one, and they don't
#       mix well.
#     - Write everything with async, then use distributed Client sync function
#       to do the switch.


LOGGER = logging.getLogger('[xgboost.dask]')


def _start_tracker(host, n_workers):
    """Start Rabit tracker """
    env = {'DMLC_NUM_WORKER': n_workers}
    rabit_context = RabitTracker(hostIP=host, nslave=n_workers)
    env.update(rabit_context.slave_envs())

    rabit_context.start(n_workers)
    thread = Thread(target=rabit_context.join)
    thread.daemon = True
    thread.start()
    return env


def _assert_dask_support():
    if not DASK_INSTALLED:
        raise ImportError(
            'Dask needs to be installed in order to use this module')
    if platform.system() == 'Windows':
        msg = 'Windows is not officially supported for dask/xgboost,'
        msg += ' contribution are welcomed.'
        LOGGER.warning(msg)


class RabitContext:
    '''A context controling rabit initialization and finalization.'''
    def __init__(self, args):
        self.args = args
        worker = distributed_get_worker()
        self.args.append(
            ('DMLC_TASK_ID=[xgboost.dask]:' + str(worker.address)).encode())

    def __enter__(self):
        rabit.init(self.args)
        LOGGER.debug('-------------- rabit say hello ------------------')

    def __exit__(self, *args):
        rabit.finalize()
        LOGGER.debug('--------------- rabit say bye ------------------')


def concat(value):              # pylint: disable=too-many-return-statements
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
        return CUDF_concat(value, axis=0)
    if lazy_isinstance(value[0], 'cupy.core.core', 'ndarray'):
        import cupy             # pylint: disable=import-error
        # pylint: disable=c-extension-no-member,no-member
        d = cupy.cuda.runtime.getDevice()
        for v in value:
            d_v = v.device.id
            assert d_v == d, 'Concatenating arrays on different devices.'
        return cupy.concatenate(value, axis=0)
    return dd.multi.concat(list(value), axis=0)


def _xgb_get_client(client):
    '''Simple wrapper around testing None.'''
    if not isinstance(client, (type(get_client()), type(None))):
        raise TypeError(
            _expect([type(get_client()), type(None)], type(client)))
    ret = get_client() if client is None else client
    return ret


def _get_client_workers(client):
    workers = client.scheduler_info()['workers']
    return workers

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
    client: dask.distributed.Client
        Specify the dask client used for training.  Use default client
        returned from dask if it's set to None.
    data : dask.array.Array/dask.dataframe.DataFrame
        data source of DMatrix.
    label: dask.array.Array/dask.dataframe.DataFrame
        label used for trainin.
    missing : float, optional
        Value in the  input data (e.g. `numpy.ndarray`) which needs
        to be present as a missing value. If None, defaults to np.nan.
    weight : dask.array.Array/dask.dataframe.DataFrame
        Weight for each instance.
    feature_names : list, optional
        Set names for features.
    feature_types : list, optional
        Set types for features

    '''

    def __init__(self,
                 client,
                 data,
                 label=None,
                 missing=None,
                 weight=None,
                 feature_names=None,
                 feature_types=None):
        _assert_dask_support()
        client: Client = _xgb_get_client(client)

        self.feature_names = feature_names
        self.feature_types = feature_types
        self.missing = missing

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

        self.worker_map = None
        self.has_label = label is not None
        self.has_weights = weight is not None

        self.is_quantile = False

        self._init = client.sync(self.map_local_data,
                                 client, data, label, weight)

    def __await__(self):
        return self._init.__await__()

    async def map_local_data(self, client, data, label=None, weights=None):
        '''Obtain references to local data.'''

        def inconsistent(left, left_name, right, right_name):
            msg = 'Partitions between {a_name} and {b_name} are not ' \
                'consistent: {a_len} != {b_len}.  ' \
                'Please try to repartition/rechunk your data.'.format(
                    a_name=left_name, b_name=right_name, a_len=len(left),
                    b_len=len(right)
                )
            return msg

        def check_columns(parts):
            # x is required to be 2 dim in __init__
            assert parts.ndim == 1 or parts.shape[1], 'Data should be' \
                ' partitioned by row. To avoid this specify the number' \
                ' of columns for your dask Array explicitly. e.g.' \
                ' chunks=(partition_size, X.shape[1])'

        data = data.persist()
        if label is not None:
            label = label.persist()
        if weights is not None:
            weights = weights.persist()
        # Breaking data into partitions, a trick borrowed from dask_xgboost.

        # `to_delayed` downgrades high-level objects into numpy or pandas
        # equivalents.
        X_parts = data.to_delayed()
        if isinstance(X_parts, numpy.ndarray):
            check_columns(X_parts)
            X_parts = X_parts.flatten().tolist()

        if label is not None:
            y_parts = label.to_delayed()
            if isinstance(y_parts, numpy.ndarray):
                check_columns(y_parts)
                y_parts = y_parts.flatten().tolist()
        if weights is not None:
            w_parts = weights.to_delayed()
            if isinstance(w_parts, numpy.ndarray):
                check_columns(w_parts)
                w_parts = w_parts.flatten().tolist()

        parts = [X_parts]
        if label is not None:
            assert len(X_parts) == len(
                y_parts), inconsistent(X_parts, 'X', y_parts, 'labels')
            parts.append(y_parts)
        if weights is not None:
            assert len(X_parts) == len(
                w_parts), inconsistent(X_parts, 'X', w_parts, 'weights')
            parts.append(w_parts)
        parts = list(map(delayed, zip(*parts)))

        parts = client.compute(parts)
        await distributed_wait(parts)  # async wait for parts to be computed

        for part in parts:
            assert part.status == 'finished'

        self.partition_order = {}
        for i, part in enumerate(parts):
            self.partition_order[part.key] = i

        key_to_partition = {part.key: part for part in parts}
        who_has = await client.scheduler.who_has(
            keys=[part.key for part in parts])

        worker_map = defaultdict(list)
        for key, workers in who_has.items():
            worker_map[next(iter(workers))].append(key_to_partition[key])

        self.worker_map = worker_map

        return self

    def create_fn_args(self):
        '''Create a dictionary of objects that can be pickled for function
        arguments.

        '''
        return {'feature_names': self.feature_names,
                'feature_types': self.feature_types,
                'has_label': self.has_label,
                'has_weights': self.has_weights,
                'missing': self.missing,
                'worker_map': self.worker_map,
                'is_quantile': self.is_quantile}


def _get_worker_x_ordered(worker_map, partition_order, worker):
    list_of_parts = worker_map[worker.address]
    client = get_client()
    list_of_parts_value = client.gather(list_of_parts)
    result = []
    for i, part in enumerate(list_of_parts):
        result.append((list_of_parts_value[i][0],
                       partition_order[part.key]))
    return result


def _get_worker_parts(has_label, has_weights, worker_map, worker):
    '''Get mapped parts of data in each worker from DaskDMatrix.'''
    list_of_parts = worker_map[worker.address]
    assert list_of_parts, 'data in ' + worker.address + ' was moved.'
    assert isinstance(list_of_parts, list)

    # `_get_worker_parts` is launched inside worker.  In dask side
    # this should be equal to `worker._get_client`.
    client = get_client()
    list_of_parts = client.gather(list_of_parts)

    if has_label:
        if has_weights:
            data, labels, weights = zip(*list_of_parts)
        else:
            data, labels = zip(*list_of_parts)
            weights = None
    else:
        data = [d[0] for d in list_of_parts]
        labels = None
        weights = None
    return data, labels, weights


class DaskPartitionIter(DataIter):  # pylint: disable=R0902
    '''A data iterator for `DaskDeviceQuantileDMatrix`.
    '''
    def __init__(self, data, label=None, weight=None, base_margin=None,
                 label_lower_bound=None, label_upper_bound=None,
                 feature_names=None, feature_types=None):
        self._data = data
        self._labels = label
        self._weights = weight
        self._base_margin = base_margin
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

    def data(self):
        '''Utility function for obtaining current batch of data.'''
        return self._data[self._iter]

    def labels(self):
        '''Utility function for obtaining current batch of label.'''
        if self._labels is not None:
            return self._labels[self._iter]
        return None

    def weights(self):
        '''Utility function for obtaining current batch of label.'''
        if self._weights is not None:
            return self._weights[self._iter]
        return None

    def base_margins(self):
        '''Utility function for obtaining current batch of base_margin.'''
        if self._base_margin is not None:
            return self._base_margin[self._iter]
        return None

    def label_lower_bounds(self):
        '''Utility function for obtaining current batch of label_lower_bound.
        '''
        if self._label_lower_bound is not None:
            return self._label_lower_bound[self._iter]
        return None

    def label_upper_bounds(self):
        '''Utility function for obtaining current batch of label_upper_bound.
        '''
        if self._label_upper_bound is not None:
            return self._label_upper_bound[self._iter]
        return None

    def reset(self):
        '''Reset the iterator'''
        self._iter = 0

    def next(self, input_data):
        '''Yield next batch of data'''
        if self._iter == len(self._data):
            # Return 0 when there's no more batch.
            return 0
        if self._feature_names:
            feature_names = self._feature_names
        else:
            if hasattr(self.data(), 'columns'):
                feature_names = self.data().columns.format()
            else:
                feature_names = None
        input_data(data=self.data(), label=self.labels(),
                   weight=self.weights(), group=None,
                   label_lower_bound=self.label_lower_bounds(),
                   label_upper_bound=self.label_upper_bounds(),
                   feature_names=feature_names,
                   feature_types=self._feature_types)
        self._iter += 1
        return 1


class DaskDeviceQuantileDMatrix(DaskDMatrix):
    '''Specialized data type for `gpu_hist` tree method.  This class is
    used to reduce the memory usage by eliminating data copies.
    Internally the data is merged by weighted GK sketching.  So the
    number of partitions from dask may affect training accuracy as GK
    generates error for each merge.

    .. versionadded:: 1.2.0

    Parameters
    ----------
    max_bin: Number of bins for histogram construction.


    '''
    def __init__(self, client, data, label=None, weight=None,
                 missing=None,
                 feature_names=None,
                 feature_types=None,
                 max_bin=256):
        super().__init__(client=client, data=data, label=label, weight=weight,
                         missing=missing,
                         feature_names=feature_names,
                         feature_types=feature_types)
        self.max_bin = max_bin
        self.is_quantile = True

    def create_fn_args(self):
        args = super().create_fn_args()
        args['max_bin'] = self.max_bin
        return args


def _create_device_quantile_dmatrix(feature_names, feature_types,
                                    has_label,
                                    has_weights, missing, worker_map,
                                    max_bin):
    worker = distributed_get_worker()
    if worker.address not in set(worker_map.keys()):
        msg = 'worker {address} has an empty DMatrix.  ' \
            'All workers associated with this DMatrix: {workers}'.format(
                address=worker.address,
                workers=set(worker_map.keys()))
        LOGGER.warning(msg)
        import cupy         # pylint: disable=import-error
        d = DeviceQuantileDMatrix(cupy.zeros((0, 0)),
                                  feature_names=feature_names,
                                  feature_types=feature_types,
                                  max_bin=max_bin)
        return d

    data, labels, weights = _get_worker_parts(has_label, has_weights,
                                              worker_map, worker)
    it = DaskPartitionIter(data=data, label=labels, weight=weights)

    dmatrix = DeviceQuantileDMatrix(it,
                                    missing=missing,
                                    feature_names=feature_names,
                                    feature_types=feature_types,
                                    nthread=worker.nthreads,
                                    max_bin=max_bin)
    return dmatrix


def _create_dmatrix(feature_names, feature_types, has_label,
                    has_weights, missing, worker_map):
    '''Get data that local to worker from DaskDMatrix.

      Returns
      -------
      A DMatrix object.

    '''
    worker = distributed_get_worker()
    if worker.address not in set(worker_map.keys()):
        msg = 'worker {address} has an empty DMatrix.  ' \
            'All workers associated with this DMatrix: {workers}'.format(
                address=worker.address,
                workers=set(worker_map.keys()))
        LOGGER.warning(msg)
        d = DMatrix(numpy.empty((0, 0)),
                    feature_names=feature_names,
                    feature_types=feature_types)
        return d

    data, labels, weights = _get_worker_parts(has_label, has_weights,
                                              worker_map, worker)
    data = concat(data)

    if has_label:
        labels = concat(labels)
    else:
        labels = None
    if has_weights:
        weights = concat(weights)
    else:
        weights = None
    dmatrix = DMatrix(data,
                      labels,
                      weight=weights,
                      missing=missing,
                      feature_names=feature_names,
                      feature_types=feature_types,
                      nthread=worker.nthreads)
    return dmatrix


def _dmatrix_from_worker_map(is_quantile, **kwargs):
    if is_quantile:
        return _create_device_quantile_dmatrix(**kwargs)
    return _create_dmatrix(**kwargs)


async def _get_rabit_args(worker_map, client: Client):
    '''Get rabit context arguments from data distribution in DaskDMatrix.'''
    host = distributed_comm.get_address_host(client.scheduler.address)
    env = await client.run_on_scheduler(
        _start_tracker, host.strip('/:'), len(worker_map))
    rabit_args = [('%s=%s' % item).encode() for item in env.items()]
    return rabit_args

# train and predict methods are supposed to be "functional", which meets the
# dask paradigm.  But as a side effect, the `evals_result` in single-node API
# is no longer supported since it mutates the input parameter, and it's not
# intuitive to sync the mutation result.  Therefore, a dictionary containing
# evaluation history is instead returned.


async def _train_async(client, params, dtrain: DaskDMatrix,
                       *args, evals=(), **kwargs):
    _assert_dask_support()
    client: Client = _xgb_get_client(client)
    if 'evals_result' in kwargs.keys():
        raise ValueError(
            'evals_result is not supported in dask interface.',
            'The evaluation history is returned as result of training.')

    workers = list(_get_client_workers(client).keys())
    rabit_args = await _get_rabit_args(workers, client)

    def dispatched_train(worker_addr, dtrain_ref, evals_ref):
        '''Perform training on a single worker.  A local function prevents pickling.

        '''
        LOGGER.info('Training on %s', str(worker_addr))
        worker = distributed_get_worker()
        with RabitContext(rabit_args):
            local_dtrain = _dmatrix_from_worker_map(**dtrain_ref)
            local_evals = []
            if evals_ref:
                for ref, name in evals_ref:
                    if ref['worker_map'] == dtrain_ref['worker_map']:
                        local_evals.append((local_dtrain, name))
                        continue
                    local_evals.append((_dmatrix_from_worker_map(**ref), name))

            local_history = {}
            local_param = params.copy()  # just to be consistent
            msg = 'Overriding `nthreads` defined in dask worker.'
            if 'nthread' in local_param.keys() and \
               local_param['nthread'] is not None and \
               local_param['nthread'] != worker.nthreads:
                msg += '`nthread` is specified.  ' + msg
                LOGGER.warning(msg)
            elif 'n_jobs' in local_param.keys() and \
                 local_param['n_jobs'] is not None and \
                 local_param['n_jobs'] != worker.nthreads:
                msg = '`n_jobs` is specified.  ' + msg
                LOGGER.warning(msg)
            else:
                local_param['nthread'] = worker.nthreads
            bst = worker_train(params=local_param,
                               dtrain=local_dtrain,
                               *args,
                               evals_result=local_history,
                               evals=local_evals,
                               **kwargs)
            ret = {'booster': bst, 'history': local_history}
            if local_dtrain.num_row() == 0:
                ret = None
            return ret

    if evals:
        evals = [(e.create_fn_args(), name) for e, name in evals]

    futures = client.map(dispatched_train,
                         workers,
                         [dtrain.create_fn_args()] * len(workers),
                         [evals] * len(workers),
                         pure=False,
                         workers=workers)
    results = await client.gather(futures)
    return list(filter(lambda ret: ret is not None, results))[0]


def train(client, params, dtrain, *args, evals=(), **kwargs):
    '''Train XGBoost model.

    .. versionadded:: 1.0.0

    Parameters
    ----------
    client: dask.distributed.Client
        Specify the dask client used for training.  Use default client
        returned from dask if it's set to None.
    \\*\\*kwargs:
        Other parameters are the same as `xgboost.train` except for
        `evals_result`, which is returned as part of function return value
        instead of argument.

    Returns
    -------
    results: dict
        A dictionary containing trained booster and evaluation history.
        `history` field is the same as `eval_result` from `xgboost.train`.

        .. code-block:: python

            {'booster': xgboost.Booster,
             'history': {'train': {'logloss': ['0.48253', '0.35953']},
                         'eval': {'logloss': ['0.480385', '0.357756']}}}

    '''
    _assert_dask_support()
    client = _xgb_get_client(client)
    return client.sync(_train_async, client, params,
                       dtrain=dtrain, *args, evals=evals, **kwargs)


async def _direct_predict_impl(client, data, predict_fn):
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
async def _predict_async(client: Client, model, data, *args,
                         missing=numpy.nan):
    if isinstance(model, Booster):
        booster = model
    elif isinstance(model, dict):
        booster = model['booster']
    else:
        raise TypeError(_expect([Booster, dict], type(model)))
    if not isinstance(data, (DaskDMatrix, da.Array, dd.DataFrame)):
        raise TypeError(_expect([DaskDMatrix, da.Array, dd.DataFrame],
                                type(data)))

    def mapped_predict(partition, is_df):
        worker = distributed_get_worker()
        booster.set_param({'nthread': worker.nthreads})
        m = DMatrix(partition, missing=missing, nthread=worker.nthreads)
        predt = booster.predict(m, *args, validate_features=False)
        if is_df:
            if lazy_isinstance(partition, 'cudf', 'core.dataframe.DataFrame'):
                import cudf     # pylint: disable=import-error
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

    def dispatched_predict(worker_id):
        '''Perform prediction on each worker.'''
        LOGGER.info('Predicting on %d', worker_id)
        worker = distributed_get_worker()
        list_of_parts = _get_worker_x_ordered(worker_map, partition_order,
                                              worker)
        predictions = []
        booster.set_param({'nthread': worker.nthreads})
        for part, order in list_of_parts:
            local_x = DMatrix(part, feature_names=feature_names,
                              feature_types=feature_types,
                              missing=missing, nthread=worker.nthreads)
            predt = booster.predict(data=local_x,
                                    validate_features=local_x.num_row() != 0,
                                    *args)
            columns = 1 if len(predt.shape) == 1 else predt.shape[1]
            ret = ((delayed(predt), columns), order)
            predictions.append(ret)
        return predictions

    def dispatched_get_shape(worker_id):
        '''Get shape of data in each worker.'''
        LOGGER.info('Get shape on %d', worker_id)
        worker = distributed_get_worker()
        list_of_parts = _get_worker_x_ordered(worker_map,
                                              partition_order, worker)
        shapes = [(part.shape, order) for part, order in list_of_parts]
        return shapes

    async def map_function(func):
        '''Run function for each part of the data.'''
        futures = []
        for wid in range(len(worker_map)):
            list_of_workers = [list(worker_map.keys())[wid]]
            f = await client.submit(func, wid,
                                    pure=False,
                                    workers=list_of_workers)
            futures.append(f)
        # Get delayed objects
        results = await client.gather(futures)
        results = [t for l in results for t in l]     # flatten into 1 dim list
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


def predict(client, model, data, *args, missing=numpy.nan):
    '''Run prediction with a trained booster.

    .. note::

        Only default prediction mode is supported right now.

    .. versionadded:: 1.0.0

    Parameters
    ----------
    client: dask.distributed.Client
        Specify the dask client used for training.  Use default client
        returned from dask if it's set to None.
    model: A Booster or a dictionary returned by `xgboost.dask.train`.
        The trained model.
    data: DaskDMatrix/dask.dataframe.DataFrame/dask.array.Array
        Input data used for prediction.
    missing: float
        Used when input data is not DaskDMatrix.  Specify the value
        considered as missing.

    Returns
    -------
    prediction: dask.array.Array/dask.dataframe.Series

    '''
    _assert_dask_support()
    client = _xgb_get_client(client)
    return client.sync(_predict_async, client, model, data, *args,
                       missing=missing)


async def _inplace_predict_async(client, model, data,
                                 iteration_range=(0, 0),
                                 predict_type='value',
                                 missing=numpy.nan):
    client = _xgb_get_client(client)
    if isinstance(model, Booster):
        booster = model
    elif isinstance(model, dict):
        booster = model['booster']
    else:
        raise TypeError(_expect([Booster, dict], type(model)))
    if not isinstance(data, (da.Array, dd.DataFrame)):
        raise TypeError(_expect([da.Array, dd.DataFrame], type(data)))

    def mapped_predict(data, is_df):
        worker = distributed_get_worker()
        booster.set_param({'nthread': worker.nthreads})
        prediction = booster.inplace_predict(
            data,
            iteration_range=iteration_range,
            predict_type=predict_type,
            missing=missing)
        if is_df:
            if lazy_isinstance(data, 'cudf.core.dataframe', 'DataFrame'):
                import cudf     # pylint: disable=import-error
                prediction = cudf.DataFrame({'prediction': prediction},
                                            dtype=numpy.float32)
            else:
                # If it's  from pandas, the partition is a numpy array
                prediction = DataFrame(prediction, columns=['prediction'],
                                       dtype=numpy.float32)
        return prediction

    return await _direct_predict_impl(client, data, mapped_predict)


def inplace_predict(client, model, data,
                    iteration_range=(0, 0),
                    predict_type='value',
                    missing=numpy.nan):
    '''Inplace prediction.

    .. versionadded:: 1.1.0

    Parameters
    ----------
    client: dask.distributed.Client
        Specify the dask client used for training.  Use default client
        returned from dask if it's set to None.
    model: Booster/dict
        The trained model.
    iteration_range: tuple
        Specify the range of trees used for prediction.
    predict_type: str
        * 'value': Normal prediction result.
        * 'margin': Output the raw untransformed margin value.
    missing: float
        Value in the input data which needs to be present as a missing
        value. If None, defaults to np.nan.
    Returns
    -------
    prediction: dask.array.Array
    '''
    _assert_dask_support()
    client = _xgb_get_client(client)
    return client.sync(_inplace_predict_async, client, model=model, data=data,
                       iteration_range=iteration_range,
                       predict_type=predict_type,
                       missing=missing)


async def _evaluation_matrices(client, validation_set,
                               sample_weights, missing):
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
    evals = []
    if validation_set is not None:
        assert isinstance(validation_set, list)
        for i, e in enumerate(validation_set):
            w = (sample_weights[i]
                 if sample_weights is not None else None)
            dmat = await DaskDMatrix(client=client, data=e[0], label=e[1],
                                     weight=w, missing=missing)
            evals.append((dmat, 'validation_{}'.format(i)))
    else:
        evals = None
    return evals


class DaskScikitLearnBase(XGBModel):
    '''Base class for implementing scikit-learn interface with Dask'''

    _client = None

    # pylint: disable=arguments-differ
    def fit(self, X, y,
            sample_weights=None,
            eval_set=None,
            sample_weight_eval_set=None,
            verbose=True):
        '''Fit the regressor.

        Parameters
        ----------
        X : array_like
            Feature matrix
        y : array_like
            Labels
        sample_weight : array_like
            instance weights
        eval_set : list, optional
            A list of (X, y) tuple pairs to use as validation sets, for which
            metrics will be computed.
            Validation metrics will help us track the performance of the model.
        sample_weight_eval_set : list, optional
            A list of the form [L_1, L_2, ..., L_n], where each L_i is a list
            of group weights on the i-th validation set.
        verbose : bool
            If `verbose` and an evaluation set is used, writes the evaluation
            metric measured on the validation set to stderr.'''
        raise NotImplementedError

    def predict(self, data):  # pylint: disable=arguments-differ
        '''Predict with `data`.
        Parameters
        ----------
          data: data that can be used to construct a DaskDMatrix
        Returns
        -------
        prediction : dask.array.Array'''
        raise NotImplementedError

    def __await__(self):
        # Generate a coroutine wrapper to make this class awaitable.
        async def _():
            return self
        return self.client.sync(_).__await__()

    @property
    def client(self):
        '''The dask client used in this model.'''
        client = _xgb_get_client(self._client)
        return client

    @client.setter
    def client(self, clt):
        self._client = clt


@xgboost_model_doc("""Implementation of the Scikit-Learn API for XGBoost.""",
                   ['estimators', 'model'])
class DaskXGBRegressor(DaskScikitLearnBase, XGBRegressorBase):
    # pylint: disable=missing-class-docstring
    async def _fit_async(self,
                         X,
                         y,
                         sample_weights=None,
                         eval_set=None,
                         sample_weight_eval_set=None,
                         verbose=True):
        dtrain = await DaskDMatrix(client=self.client,
                                   data=X, label=y, weight=sample_weights,
                                   missing=self.missing)
        params = self.get_xgb_params()
        evals = await _evaluation_matrices(self.client,
                                           eval_set, sample_weight_eval_set,
                                           self.missing)
        results = await train(client=self.client, params=params, dtrain=dtrain,
                              num_boost_round=self.get_num_boosting_rounds(),
                              evals=evals, verbose_eval=verbose)
        self._Booster = results['booster']
        # pylint: disable=attribute-defined-outside-init
        self.evals_result_ = results['history']
        return self

    # pylint: disable=missing-docstring
    def fit(self, X, y,
            sample_weights=None,
            eval_set=None,
            sample_weight_eval_set=None,
            verbose=True):
        _assert_dask_support()
        return self.client.sync(self._fit_async, X, y, sample_weights,
                                eval_set, sample_weight_eval_set,
                                verbose)

    async def _predict_async(self, data):  # pylint: disable=arguments-differ
        test_dmatrix = await DaskDMatrix(client=self.client, data=data,
                                         missing=self.missing)
        pred_probs = await predict(client=self.client,
                                   model=self.get_booster(), data=test_dmatrix)
        return pred_probs

    def predict(self, data):
        _assert_dask_support()
        return self.client.sync(self._predict_async, data)


@xgboost_model_doc(
    'Implementation of the scikit-learn API for XGBoost classification.',
    ['estimators', 'model']
)
class DaskXGBClassifier(DaskScikitLearnBase, XGBClassifierBase):
    async def _fit_async(self, X, y,
                         sample_weights=None,
                         eval_set=None,
                         sample_weight_eval_set=None,
                         verbose=True):
        dtrain = await DaskDMatrix(client=self.client,
                                   data=X, label=y, weight=sample_weights,
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

        evals = await _evaluation_matrices(self.client,
                                           eval_set, sample_weight_eval_set,
                                           self.missing)
        results = await train(client=self.client, params=params, dtrain=dtrain,
                              num_boost_round=self.get_num_boosting_rounds(),
                              evals=evals, verbose_eval=verbose)
        self._Booster = results['booster']
        # pylint: disable=attribute-defined-outside-init
        self.evals_result_ = results['history']
        return self

    def fit(self, X, y,
            sample_weights=None,
            eval_set=None,
            sample_weight_eval_set=None,
            verbose=True):
        _assert_dask_support()
        return self.client.sync(self._fit_async, X, y, sample_weights,
                                eval_set, sample_weight_eval_set, verbose)

    async def _predict_proba_async(self, data):
        _assert_dask_support()

        test_dmatrix = await DaskDMatrix(client=self.client, data=data,
                                         missing=self.missing)
        pred_probs = await predict(client=self.client,
                                   model=self.get_booster(), data=test_dmatrix)
        return pred_probs

    def predict_proba(self, data):  # pylint: disable=arguments-differ,missing-docstring
        _assert_dask_support()
        return self.client.sync(self._predict_proba_async, data)

    async def _predict_async(self, data):
        _assert_dask_support()

        test_dmatrix = await DaskDMatrix(client=self.client, data=data,
                                         missing=self.missing)
        pred_probs = await predict(client=self.client,
                                   model=self.get_booster(), data=test_dmatrix)

        if self.n_classes_ == 2:
            preds = (pred_probs > 0.5).astype(int)
        else:
            preds = da.argmax(pred_probs, axis=1)

        return preds

    def predict(self, data):  # pylint: disable=arguments-differ
        _assert_dask_support()
        return self.client.sync(self._predict_async, data)
