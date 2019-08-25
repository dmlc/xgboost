# pylint: disable=too-many-arguments, too-many-locals
"""Dask extensions for distributed training. See xgboost/demo/dask for
examples.

The implementation is heavily influenced by dask_xgboost:
https://github.com/dask/dask-xgboost
"""
import platform
import logging
from collections import defaultdict
from threading import Thread
from toolz import first

import numpy
import pandas

from . import rabit
from .compat import DASK_INSTALLED
from .compat import distributed_get_worker, sparse, scipy_sparse, delayed
from .compat import da, dd, distributed_wait, get_client, distributed_comm
from .core import DMatrix, Booster, _expect
from .training import train as worker_train
from .tracker import RabitTracker
from .sklearn import XGBModel, XGBClassifierBase

# Current status is considered as initial support, many features are
# not properly supported yet.
#
# TODOs:
#   - Callback.
#   - Label encoding.
#   - Prediction for leaf, out_margin, probability etc.
#   - CV


def _start_tracker(host, n_workers):
    """ Start Rabit tracker """
    env = {'DMLC_NUM_WORKER': n_workers}
    rabit_context = RabitTracker(hostIP=host, nslave=n_workers)
    env.update(rabit_context.slave_envs())

    rabit_context.start(n_workers)
    thread = Thread(target=rabit_context.join)
    thread.daemon = True
    thread.start()
    return env


def _assert_dask_installed():
    if not DASK_INSTALLED:
        raise ImportError(
            'Dask needs to be installed in order to use this module')


class RabitContext:
    '''A context controling rabit initialization and finalization.'''
    def __init__(self, args):
        self.args = args

    def __enter__(self):
        rabit.init(self.args)
        logging.debug('-------------- rabit say hello ------------------')

    def __exit__(self, *args):
        rabit.finalize()
        logging.debug('--------------- rabit say bye ------------------')


def concat(value):
    '''To be replaced with dask builtin.'''
    if isinstance(value[0], numpy.ndarray):
        return numpy.concatenate(value, axis=0)
    if isinstance(value[0], (pandas.DataFrame, pandas.Series)):
        return pandas.concat(value, axis=0)
    if scipy_sparse and isinstance(value[0], scipy_sparse.spmatrix):
        return scipy_sparse.vstack(value, format='csr')
    if sparse and isinstance(value[0], sparse.SparseArray):
        return sparse.concatenate(value, axis=0)
    raise TypeError(
        _expect(['numpy arrays', 'pandas dataframes'], type(value[0])))


class DaskDMatrix:
    # pylint: disable=missing-docstring, too-many-instance-attributes
    __doc__ = '''DMatrix holding on references to Dask DataFrame or ''' + \
        '''Dask Array.\n\n'''.join(DMatrix.__doc__)

    _feature_names = None  # for previous version's pickle
    _feature_types = None

    def __init__(self,
                 data,
                 label=None,
                 missing=None,
                 weight=None,
                 feature_names=None,
                 feature_types=None,
                 client=None):
        _assert_dask_installed()

        self._feature_names = feature_names
        self._feature_types = feature_types
        self._missing = missing

        if len(data.shape) != 2:
            _expect('2 dimensions input', data.shape)
        self.n_rows = data.shape[0]
        self.n_cols = data.shape[1]

        if not any(isinstance(data, t) for t in (dd.DataFrame, da.Array)):
            raise TypeError(_expect((dd.DataFrame, da.Array), type(data)))
        if not any(
                isinstance(label, t)
                for t in (dd.DataFrame, da.Array, dd.Series, type(None))):
            raise TypeError(
                _expect((dd.DataFrame, da.Array, dd.Series), type(label)))

        if client is None:
            client = get_client()

        self.worker_map = None
        self.has_label = label is not None
        self.has_weights = weight is not None

        client.sync(self.map_local_data, client, data, label, weight)

    async def map_local_data(self, client, data, label=None, weights=None):
        '''Obtain references to local data.'''
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
            assert X_parts.shape[1] == 1
            X_parts = X_parts.flatten().tolist()

        if label is not None:
            y_parts = label.to_delayed()
            if isinstance(y_parts, numpy.ndarray):
                assert y_parts.ndim == 1 or y_parts.shape[1] == 1
                y_parts = y_parts.flatten().tolist()
        if weights is not None:
            w_parts = weights.to_delayed()
            if isinstance(w_parts, numpy.ndarray):
                assert w_parts.ndim == 1 or w_parts.shape[1] == 1
                w_parts = w_parts.flatten().tolist()

        parts = [X_parts]
        if label is not None:
            assert len(X_parts) == len(
                y_parts), 'Partitions between X and y are not consistent'
            parts.append(y_parts)
        if weights is not None:
            assert len(X_parts) == len(
                w_parts), 'Partitions between X and weight are not consistent.'
            parts.append(w_parts)
        parts = list(map(delayed, zip(*parts)))

        parts = client.compute(parts)
        await distributed_wait(parts)  # async wait for parts to be computed

        for part in parts:
            assert part.status == 'finished'

        key_to_partition = {part.key: part for part in parts}
        who_has = await client.scheduler.who_has(
            keys=[part.key for part in parts])

        worker_map = defaultdict(list)
        for key, workers in who_has.items():
            worker_map[first(workers)].append(key_to_partition[key])

        self.worker_map = worker_map

    def get_worker_data(self, worker):
        '''Get data that local to worker.
        worker: The worker used as key to data.

        Returns
        -------
        A DMatrix object.
        '''
        client = get_client()
        list_of_parts = self.worker_map[worker.address]
        assert list_of_parts, 'data in ' + worker.address + ' was moved.'

        list_of_parts = client.gather(list_of_parts)

        if self.has_label:
            if self.has_weights:
                data, labels, weights = zip(*list_of_parts)
            else:
                data, labels = zip(*list_of_parts)
        else:
            data = zip(*list_of_parts)

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
                          missing=self._missing,
                          feature_names=self._feature_names,
                          feature_types=self._feature_types)
        return dmatrix

    def num_row(self):
        return self.n_rows

    def num_col(self):
        return self.n_cols


# train and predict methods are supposed to be "functional", which meets the
# dask paradigm.  But as a side effect, the `evals_result` in single-node API
# is no longer supported since it mutates the input parameter, and it's not
# intuitive to sync the mutation result.  Therefore, a dictionary containing
# evaluation history is instead returned.


def train(client, params, dtrain, *args, evals=(), **kwargs):
    '''Train XGBoost model.'''
    _assert_dask_installed()
    if platform.system() == 'Windows':
        msg = 'Windows is not officially supported for dask/xgboost,'
        msg += ' contribution are welcomed.'
        logging.warning(msg)

    if 'evals_result' in kwargs.keys():
        raise ValueError(
            'evals_result is not supported in dask interface.',
            'The evaluation history is returned as result of training.')

    if client is None:
        client = get_client()

    host = distributed_comm.get_address_host(client.scheduler.address)
    worker_map = dtrain.worker_map

    env = client.run_on_scheduler(_start_tracker, host.strip('/:'),
                                  len(worker_map))
    rabit_args = [('%s=%s' % item).encode() for item in env.items()]

    def dispatched_train(worker_id):
        '''Perform training on worker.'''
        logging.info('Training on %d', worker_id)
        worker = distributed_get_worker()
        local_dtrain = dtrain.get_worker_data(worker)
        local_evals = []
        if evals:
            for mat, name in evals:
                local_mat = mat.get_worker_data(worker)
                local_evals.append((local_mat, name))

        with RabitContext(rabit_args):
            local_history = {}
            local_param = params.copy()  # just to be consistent
            bst = worker_train(params=local_param,
                               dtrain=local_dtrain,
                               *args,
                               evals_result=local_history,
                               evals=local_evals,
                               **kwargs)
            ret = {'booster': bst, 'history': local_history}
            if rabit.get_rank() != 0:
                ret = None
            return ret

    futures = client.map(dispatched_train,
                         range(len(worker_map)),
                         workers=list(worker_map.keys()))
    results = client.gather(futures)
    return list(filter(lambda ret: ret is not None, results))[0]


def predict(model, data, client=None):
    '''Run prediction with a trained booster.
    parameters:
    ----------
    model: A Booster or a dictionary returned by `xgboost.dask.train`.
        The trained model.
    data: DaskDMatrix
        Input data used for prediction.
    client (optional): dask.distributed.Client
       Specify the dask client.

    Returns
    -------
    prediction: dask.array.Array
    '''
    _assert_dask_installed()
    if isinstance(model, Booster):
        booster = model
    elif isinstance(model, dict):
        booster = model['booster']
    else:
        raise TypeError(_expect([Booster, dict], type(model)))

    if not isinstance(data, DaskDMatrix):
        raise TypeError(_expect([DaskDMatrix], type(data)))

    if client is None:
        client = get_client()
    worker_map = data.worker_map
    futures = client.map(predict,
                         range(len(worker_map)),
                         workers=list(worker_map.keys()),
                         booster=booster,
                         data=data)
    prediction = da.stack(futures, axis=0)
    return prediction


def _evaluation_matrices(validation_set, sample_weights):
    '''
    parameters:
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
            dmat = DaskDMatrix(data=e[0], label=e[1], weight=w)
            evals.append((dmat, 'validation_{}'.format(i)))
    else:
        evals = None
    return evals


class DaskXGBRegressor(XGBModel):
    # pylint: disable=missing-docstring
    __doc__ = XGBModel.__doc__.split('\n')[2:]
    _client = None

    @property
    def client(self):
        '''The dask client used in this regressor.'''
        return self._client

    @client.setter
    def client(self, clt):
        self._client = clt

    # pylint: disable=arguments-differ
    def fit(self,
            X,
            y,
            sample_weights=None,
            eval_set=None,
            sample_weight_eval_set=None):
        _assert_dask_installed()
        dtrain = DaskDMatrix(data=X, label=y, weight=sample_weights)
        params = self.get_xgb_params()
        evals = _evaluation_matrices(eval_set, sample_weight_eval_set)

        results = train(self.client, params, dtrain,
                        num_boost_round=self.get_num_boosting_rounds(),
                        evals=evals)
        self._Booster = results['booster']
        # pylint: disable=attribute-defined-outside-init
        self.evals_result_ = results['history']
        return self

    def predict(self, data):  # pylint: disable=arguments-differ
        _assert_dask_installed()
        test_dmatrix = DaskDMatrix(data)
        pred_probs = predict(self.get_booster(), test_dmatrix, self.client)
        return pred_probs


class DaskXGBClassifier(XGBModel, XGBClassifierBase):
    # pylint: disable=missing-docstring
    __doc__ = XGBModel.__doc__.split('\n')[2:]
    _client = None

    @property
    def client(self):
        '''The dask client used in this regressor.'''
        return self._client

    @client.setter
    def client(self, clt):
        self._client = clt

    # pylint: disable=arguments-differ
    def fit(self,
            X,
            y,
            sample_weights=None,
            eval_set=None,
            sample_weight_eval_set=None):
        _assert_dask_installed()
        dtrain = DaskDMatrix(data=X, label=y, weight=sample_weights)
        params = self.get_xgb_params()

        # pylint: disable=attribute-defined-outside-init
        self.classes_ = da.unique(y).compute()
        self.n_classes_ = len(self.classes_)

        if self.n_classes_ > 2:
            params["objective"] = "multi:softprob"
            params['num_class'] = self.n_classes_
        else:
            params["objective"] = "binary:logistic"
        params.setdefault('num_class', self.n_classes_)

        evals = _evaluation_matrices(eval_set, sample_weight_eval_set)
        results = train(self.client, params, dtrain,
                        num_boost_round=self.get_num_boosting_rounds(),
                        evals=evals)
        self._Booster = results['booster']
        # pylint: disable=attribute-defined-outside-init
        self.evals_result_ = results['history']
        return self

    def predict(self, data):  # pylint: disable=arguments-differ
        _assert_dask_installed()
        test_dmatrix = DaskDMatrix(data)
        pred_probs = predict(self.get_booster(), test_dmatrix, self.client)
        return pred_probs
