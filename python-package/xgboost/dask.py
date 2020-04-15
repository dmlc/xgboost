# pylint: disable=too-many-arguments, too-many-locals
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
from threading import Thread

import numpy

from . import rabit

from .compat import DASK_INSTALLED
from .compat import distributed_get_worker, distributed_wait, distributed_comm
from .compat import da, dd, delayed, get_client
from .compat import sparse, scipy_sparse
from .compat import PANDAS_INSTALLED, DataFrame, Series, pandas_concat
from .compat import CUDF_INSTALLED, CUDF_DataFrame, CUDF_Series, CUDF_concat
from .compat import lazy_isinstance

from .core import DMatrix, Booster, _expect
from .training import train as worker_train
from .tracker import RabitTracker
from .sklearn import XGBModel, XGBRegressorBase, XGBClassifierBase
from .sklearn import xgboost_model_doc

# Current status is considered as initial support, many features are
# not properly supported yet.
#
# TODOs:
#   - Callback.
#   - Label encoding.
#   - CV
#   - Ranking


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
    if CUDF_INSTALLED and isinstance(value[0], (CUDF_DataFrame, CUDF_Series)):
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


class DaskDMatrix:
    # pylint: disable=missing-docstring, too-many-instance-attributes
    '''DMatrix holding on references to Dask DataFrame or Dask Array.  Constructing
    a `DaskDMatrix` forces all lazy computation to be carried out.  Wait for
    the input data explicitly if you want to see actual computation of
    constructing `DaskDMatrix`.

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
        client = _xgb_get_client(client)

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

        client.sync(self.map_local_data, client, data, label, weight)

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

    def get_worker_x_ordered(self, worker):
        list_of_parts = self.worker_map[worker.address]
        client = get_client()
        list_of_parts_value = client.gather(list_of_parts)
        result = []
        for i, part in enumerate(list_of_parts):
            result.append((list_of_parts_value[i][0],
                           self.partition_order[part.key]))
        return result

    def get_worker_parts(self, worker):
        '''Get mapped parts of data in each worker.'''
        list_of_parts = self.worker_map[worker.address]
        assert list_of_parts, 'data in ' + worker.address + ' was moved.'
        assert isinstance(list_of_parts, list)

        # `get_worker_parts` is launched inside worker.  In dask side
        # this should be equal to `worker._get_client`.
        client = get_client()
        list_of_parts = client.gather(list_of_parts)

        if self.has_label:
            if self.has_weights:
                data, labels, weights = zip(*list_of_parts)
            else:
                data, labels = zip(*list_of_parts)
                weights = None
        else:
            data = [d[0] for d in list_of_parts]
            labels = None
            weights = None
        return data, labels, weights

    def get_worker_data(self, worker):
        '''Get data that local to worker.

          Parameters
          ----------
          worker: The worker used as key to data.

          Returns
          -------
          A DMatrix object.

        '''
        if worker.address not in set(self.worker_map.keys()):
            msg = 'worker {address} has an empty DMatrix.  ' \
                'All workers associated with this DMatrix: {workers}'.format(
                    address=worker.address,
                    workers=set(self.worker_map.keys()))
            LOGGER.warning(msg)
            d = DMatrix(numpy.empty((0, 0)),
                        feature_names=self.feature_names,
                        feature_types=self.feature_types)
            return d

        data, labels, weights = self.get_worker_parts(worker)

        data = concat(data)

        if self.has_label:
            labels = concat(labels)
        else:
            labels = None
        if self.has_weights:
            weights = concat(weights)
        else:
            weights = None
        dmatrix = DMatrix(data,
                          labels,
                          weight=weights,
                          missing=self.missing,
                          feature_names=self.feature_names,
                          feature_types=self.feature_types,
                          nthread=worker.nthreads)
        return dmatrix

    def get_worker_data_shape(self, worker):
        '''Get the shape of data X in each worker.'''
        data, _, _ = self.get_worker_parts(worker)

        shapes = [d.shape for d in data]
        rows = 0
        cols = 0
        for shape in shapes:
            rows += shape[0]

            c = shape[1]
            assert cols in (0, c), 'Shape between partitions are not the' \
                ' same. Got: {left} and {right}'.format(left=c, right=cols)
            cols = c
        return (rows, cols)


def _get_rabit_args(worker_map, client):
    '''Get rabit context arguments from data distribution in DaskDMatrix.'''
    host = distributed_comm.get_address_host(client.scheduler.address)

    env = client.run_on_scheduler(_start_tracker, host.strip('/:'),
                                  len(worker_map))
    rabit_args = [('%s=%s' % item).encode() for item in env.items()]
    return rabit_args

# train and predict methods are supposed to be "functional", which meets the
# dask paradigm.  But as a side effect, the `evals_result` in single-node API
# is no longer supported since it mutates the input parameter, and it's not
# intuitive to sync the mutation result.  Therefore, a dictionary containing
# evaluation history is instead returned.


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
    if 'evals_result' in kwargs.keys():
        raise ValueError(
            'evals_result is not supported in dask interface.',
            'The evaluation history is returned as result of training.')

    workers = list(_get_client_workers(client).keys())

    rabit_args = _get_rabit_args(workers, client)

    def dispatched_train(worker_addr):
        '''Perform training on a single worker.'''
        LOGGER.info('Training on %s', str(worker_addr))
        worker = distributed_get_worker()
        with RabitContext(rabit_args):
            local_dtrain = dtrain.get_worker_data(worker)

            local_evals = []
            if evals:
                for mat, name in evals:
                    if mat is dtrain:
                        local_evals.append((local_dtrain, name))
                        continue
                    local_mat = mat.get_worker_data(worker)
                    local_evals.append((local_mat, name))

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

    futures = client.map(dispatched_train,
                         workers,
                         pure=False,
                         workers=workers)
    results = client.gather(futures)
    return list(filter(lambda ret: ret is not None, results))[0]


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
        m = DMatrix(partition, missing=missing, nthread=worker.nthreads)
        predt = booster.predict(m, *args, validate_features=False)
        if is_df:
            predt = DataFrame(predt, columns=['prediction'])
        return predt

    if isinstance(data, da.Array):
        predictions = client.submit(
            da.map_blocks,
            mapped_predict, data, False, drop_axis=1,
            dtype=numpy.float32
        ).result()
        return predictions
    if isinstance(data, dd.DataFrame):
        predictions = client.submit(
            dd.map_partitions,
            mapped_predict, data, True,
            meta=dd.utils.make_meta({'prediction': 'f4'})
        ).result()
        return predictions.iloc[:, 0]

    # Prediction on dask DMatrix.
    worker_map = data.worker_map

    def dispatched_predict(worker_id):
        '''Perform prediction on each worker.'''
        LOGGER.info('Predicting on %d', worker_id)
        worker = distributed_get_worker()
        list_of_parts = data.get_worker_x_ordered(worker)
        predictions = []
        booster.set_param({'nthread': worker.nthreads})
        for part, order in list_of_parts:
            local_x = DMatrix(part,
                              feature_names=data.feature_names,
                              feature_types=data.feature_types,
                              missing=data.missing,
                              nthread=worker.nthreads)
            predt = booster.predict(data=local_x,
                                    validate_features=local_x.num_row() != 0,
                                    *args)
            ret = (delayed(predt), order)
            predictions.append(ret)
        return predictions

    def dispatched_get_shape(worker_id):
        '''Get shape of data in each worker.'''
        LOGGER.info('Trying to get data shape on %d', worker_id)
        worker = distributed_get_worker()
        list_of_parts = data.get_worker_x_ordered(worker)
        shapes = []
        for part, order in list_of_parts:
            shapes.append((part.shape, order))
        return shapes

    def map_function(func):
        '''Run function for each part of the data.'''
        futures = []
        for wid in range(len(worker_map)):
            list_of_workers = [list(worker_map.keys())[wid]]
            f = client.submit(func, wid,
                              pure=False,
                              workers=list_of_workers)
            futures.append(f)

        # Get delayed objects
        results = client.gather(futures)
        results = [t for l in results for t in l]     # flatten into 1 dim list
        # sort by order, l[0] is the delayed object, l[1] is its order
        results = sorted(results, key=lambda l: l[1])
        results = [predt for predt, order in results]  # remove order
        return results

    results = map_function(dispatched_predict)
    shapes = map_function(dispatched_get_shape)

    # Constructing a dask array from list of numpy arrays
    # See https://docs.dask.org/en/latest/array-creation.html
    arrays = []
    for i, shape in enumerate(shapes):
        arrays.append(da.from_delayed(results[i], shape=(shape[0], ),
                                      dtype=numpy.float32))
    predictions = da.concatenate(arrays, axis=0)
    return predictions


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

    if isinstance(data, da.Array):
        predictions = client.submit(
            da.map_blocks,
            mapped_predict, data, False, drop_axis=1,
            dtype=numpy.float32
        ).result()
        return predictions
    if isinstance(data, dd.DataFrame):
        predictions = client.submit(
            dd.map_partitions,
            mapped_predict, data, True,
            meta=dd.utils.make_meta({'prediction': 'f4'})
        ).result()
        return predictions.iloc[:, 0]


def _evaluation_matrices(client, validation_set, sample_weights, missing):
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
            dmat = DaskDMatrix(client=client, data=e[0], label=e[1], weight=w,
                               missing=missing)
            evals.append((dmat, 'validation_{}'.format(i)))
    else:
        evals = None
    return evals


class DaskScikitLearnBase(XGBModel):
    '''Base class for implementing scikit-learn interface with Dask'''

    _client = None

    # pylint: disable=arguments-differ
    def fit(self,
            X,
            y,
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
    # pylint: disable=missing-docstring
    def fit(self,
            X,
            y,
            sample_weights=None,
            eval_set=None,
            sample_weight_eval_set=None,
            verbose=True):
        _assert_dask_support()
        dtrain = DaskDMatrix(client=self.client,
                             data=X, label=y, weight=sample_weights,
                             missing=self.missing)
        params = self.get_xgb_params()
        evals = _evaluation_matrices(self.client,
                                     eval_set, sample_weight_eval_set,
                                     self.missing)

        results = train(self.client, params, dtrain,
                        num_boost_round=self.get_num_boosting_rounds(),
                        evals=evals, verbose_eval=verbose)
        # pylint: disable=attribute-defined-outside-init
        self._Booster = results['booster']
        # pylint: disable=attribute-defined-outside-init
        self.evals_result_ = results['history']
        return self

    def predict(self, data):  # pylint: disable=arguments-differ
        _assert_dask_support()
        test_dmatrix = DaskDMatrix(client=self.client, data=data,
                                   missing=self.missing)
        pred_probs = predict(client=self.client,
                             model=self.get_booster(), data=test_dmatrix)
        return pred_probs


@xgboost_model_doc(
    'Implementation of the scikit-learn API for XGBoost classification.',
    ['estimators', 'model']
)
class DaskXGBClassifier(DaskScikitLearnBase, XGBClassifierBase):
    # pylint: disable=missing-docstring
    _client = None

    def fit(self,
            X,
            y,
            sample_weights=None,
            eval_set=None,
            sample_weight_eval_set=None,
            verbose=True):
        _assert_dask_support()
        dtrain = DaskDMatrix(client=self.client,
                             data=X, label=y, weight=sample_weights,
                             missing=self.missing)
        params = self.get_xgb_params()

        # pylint: disable=attribute-defined-outside-init
        if isinstance(y, (da.Array)):
            self.classes_ = da.unique(y).compute()
        else:
            self.classes_ = y.drop_duplicates().compute()
        self.n_classes_ = len(self.classes_)

        if self.n_classes_ > 2:
            params["objective"] = "multi:softprob"
            params['num_class'] = self.n_classes_
        else:
            params["objective"] = "binary:logistic"

        evals = _evaluation_matrices(self.client,
                                     eval_set, sample_weight_eval_set,
                                     self.missing)
        results = train(self.client, params, dtrain,
                        num_boost_round=self.get_num_boosting_rounds(),
                        evals=evals, verbose_eval=verbose)
        self._Booster = results['booster']
        # pylint: disable=attribute-defined-outside-init
        self.evals_result_ = results['history']
        return self

    def predict(self, data):  # pylint: disable=arguments-differ
        _assert_dask_support()
        test_dmatrix = DaskDMatrix(client=self.client, data=data,
                                   missing=self.missing)
        pred_probs = predict(client=self.client,
                             model=self.get_booster(), data=test_dmatrix)
        return pred_probs
