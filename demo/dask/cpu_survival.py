"""
Example of training survival model with Dask on CPU
===================================================

"""

import os

import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

import xgboost as xgb
from xgboost.dask import DaskDMatrix


def main(client):
    # Load an example survival data from CSV into a Dask data frame.
    # The Veterans' Administration Lung Cancer Trial
    # The Statistical Analysis of Failure Time Data by Kalbfleisch J. and Prentice R (1980)
    CURRENT_DIR = os.path.dirname(__file__)
    df = dd.read_csv(os.path.join(CURRENT_DIR, os.pardir, 'data', 'veterans_lung_cancer.csv'))

    # DaskDMatrix acts like normal DMatrix, works as a proxy for local
    # DMatrix scatter around workers.
    # For AFT survival, you'd need to extract the lower and upper bounds for the label
    # and pass them as arguments to DaskDMatrix.
    y_lower_bound = df['Survival_label_lower_bound']
    y_upper_bound = df['Survival_label_upper_bound']
    X = df.drop(['Survival_label_lower_bound',
                 'Survival_label_upper_bound'], axis=1)
    dtrain = DaskDMatrix(client, X, label_lower_bound=y_lower_bound,
                         label_upper_bound=y_upper_bound)

    # Use train method from xgboost.dask instead of xgboost.  This
    # distributed version of train returns a dictionary containing the
    # resulting booster and evaluation history obtained from
    # evaluation metrics.
    params = {'verbosity': 1,
              'objective': 'survival:aft',
              'eval_metric': 'aft-nloglik',
              'learning_rate': 0.05,
              'aft_loss_distribution_scale': 1.20,
              'aft_loss_distribution': 'normal',
              'max_depth': 6,
              'lambda': 0.01,
              'alpha': 0.02}
    output = xgb.dask.train(client,
                            params,
                            dtrain,
                            num_boost_round=100,
                            evals=[(dtrain, 'train')])
    bst = output['booster']
    history = output['history']

    # you can pass output directly into `predict` too.
    prediction = xgb.dask.predict(client, bst, dtrain)
    print('Evaluation history: ', history)

    # Uncomment the following line to save the model to the disk
    # bst.save_model('survival_model.json')

    return prediction


if __name__ == '__main__':
    # or use other clusters for scaling
    with LocalCluster(n_workers=7, threads_per_worker=4) as cluster:
        with Client(cluster) as client:
            main(client)
