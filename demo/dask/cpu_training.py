import xgboost as xgb
from xgboost.dask import DaskDMatrix, EarlyStopping
from dask.distributed import Client
from dask.distributed import LocalCluster
from dask import array as da
import logging


def main(client):
    # generate some random data for demonstration
    m = 100000
    n = 100
    X = da.random.random(size=(m, n), chunks=100)
    y = da.random.random(size=(m, ), chunks=100)

    # DaskDMatrix acts like normal DMatrix, works as a proxy for local
    # DMatrix scatter around workers.
    dtrain = DaskDMatrix(client, X, y)

    early_stop = EarlyStopping(dtrain, 'train', 'rmse', 3)
    # Use train method from xgboost.dask instead of xgboost.  This
    # distributed version of train returns a dictionary containing the
    # resulting booster and evaluation history obtained from
    # evaluation metrics.
    output = xgb.dask.train(client,
                            params={'verbosity': 1,
                                    'tree_method': 'hist'},
                            dtrain=dtrain,
                            num_boost_round=4,
                            evals=[(dtrain, 'train')],
                            callbacks=[early_stop])

    bst = output['booster']
    history = output['history']

    # you can pass output directly into `predict` too.
    prediction = xgb.dask.predict(client, bst, dtrain)
    print('Evaluation history:', history)
    return prediction


if __name__ == '__main__':
    # or use other clusters for scaling
    logging.basicConfig(level=logging.INFO)
    with LocalCluster(n_workers=7, threads_per_worker=4) as cluster:
        with Client(cluster) as client:
            print(client.dashboard_link)
            main(client)
