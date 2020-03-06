'''The example is taken from:
https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression_multioutput.html#sphx-glr-auto-examples-tree-plot-tree-regression-multioutput-py

Multi-target tree may have lower accuracy due to smaller model capacity, but
provides better computation performance for prediction.

The current implementation supports only exact tree method and is considered as
highly experimental.  We do not recommend any real world usage.

There are 3 different ways to train a multi target model.

- Train 1 model for each target manually.  See `train_stacked_native` below.
- Train 1 stack of trees for each target by XGBoost.  This is the default
  implementation with `output_type` set to `single`.
- Train 1 stack of trees for all target variables, with the tree leaf being a
  vector.  This can be enabled by setting `output_type` to `multi`.

'''

import numpy as np
from matplotlib import pyplot as plt
import xgboost as xgb
from xgboost.experimental import XGBMultiRegressor
import argparse

# Generate some random data with y being a circle.
rng = np.random.RandomState(1994)
X = np.sort(200 * rng.rand(100, 1) - 100, axis=0)
y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
y[::5, :] += (0.5 - rng.rand(20, 2))

boosted_rounds = 32

y = y - y.min()
y: np.ndarray = y / y.max()
y = y.copy()


def plot_predt(y, y_predt, name):
    '''Plot the output prediction along with labels.
    Parameters
    ----------
    y : np.ndarray
        labels
    y_predt : np.ndarray
        prediction from XGBoost.
    name : str
        output file name for matplotlib.
    '''
    s = 25
    plt.scatter(y[:, 0], y[:, 1], c="navy", s=s,
                edgecolor="black", label="data")
    plt.scatter(y_predt[:, 0], y_predt[:, 1], c="cornflowerblue", s=s,
                edgecolor="black", label="max_depth=2")
    plt.xlim([-1, 2])
    plt.ylim([-1, 2])
    plt.savefig(name + '.png')
    plt.close()


def train_multi_skl():
    '''Train a multi-target regression with XGBoost's scikit-learn interface.  This
    method demos training multi-target trees with each vector as leaf value,
    also training a model that uses single target tree with one stack of trees
    for each target variable.

    '''
    # Train with vector leaf trees.
    reg = XGBMultiRegressor(output_type='multi',
                            num_targets=y.shape[1],
                            n_estimators=boosted_rounds)
    reg.fit(X, y, eval_set=[(X, y)])
    y_predt = reg.predict(X)
    plot_predt(y, y_predt, 'skl-multi')

    # Train 1 stack of trees for each target variable.
    reg = XGBMultiRegressor(output_type='single',
                            num_targets=y.shape[1],
                            n_estimators=boosted_rounds)
    reg.fit(X, y, eval_set=[(X, y)])
    y_predt = reg.predict(X)
    plot_predt(y, y_predt, 'skl-sinlge')


def train_multi_native():
    '''Train a multi-target regression with native XGBoost interface.  This method
    demos training multi-target trees with each vector as leaf value, also
    training a model that uses single target tree with one stack of trees for
    each target variable.

    '''
    d = xgb.DMatrix(X, y)
    # Train with vector leaf trees.
    booster = xgb.train({'tree_method': 'exact',
                         'nthread': 16,
                         'output_type': 'multi',
                         'num_targets': y.shape[1],
                         'objective': 'reg:squarederror'
                         }, d,
                        num_boost_round=boosted_rounds,
                        evals=[(d, 'Train')])
    y_predt = booster.predict(d)
    plot_predt(y, y_predt, 'native-multi')

    # Train 1 stack of trees for each target variable.
    booster = xgb.train({'tree_method': 'exact',
                         'nthread': 16,
                         'output_type': 'single',
                         'num_targets': y.shape[1],
                         'objective': 'reg:squarederror'
                         }, d,
                        num_boost_round=boosted_rounds,
                        evals=[(d, 'Train')])
    y_predt = booster.predict(d)
    plot_predt(y, y_predt, 'native-single')


def train_stacked_native():
    '''Train 2 XGBoost models, each one targeting a single output variable.'''
    # Extract the first target variable
    d = xgb.DMatrix(X, y[:, 0].copy())
    params = {'tree_method': 'exact',
              'objective': 'reg:squarederror'}
    booster = xgb.train(
        params, d, num_boost_round=boosted_rounds, evals=[(d, 'Train')])
    y_predt_0 = booster.predict(d)

    # Extract the second target variable
    d = xgb.DMatrix(X, y[:, 1].copy())
    booster = xgb.train(params, d, num_boost_round=boosted_rounds)
    y_predt_1 = booster.predict(d)
    y_predt = np.stack([y_predt_0, y_predt_1], axis=-1)
    plot_predt(y, y_predt, 'stacked')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    train_multi_native()
    train_multi_skl()
    train_stacked_native()
