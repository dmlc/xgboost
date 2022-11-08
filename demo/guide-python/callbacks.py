'''
Demo for using and defining callback functions
==============================================

    .. versionadded:: 1.3.0
'''
import argparse
import os
import tempfile

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import xgboost as xgb


class Plotting(xgb.callback.TrainingCallback):
    '''Plot evaluation result during training.  Only for demonstration purpose as it's quite
    slow to draw.

    '''
    def __init__(self, rounds):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.rounds = rounds
        self.lines = {}
        self.fig.show()
        self.x = np.linspace(0, self.rounds, self.rounds)
        plt.ion()

    def _get_key(self, data, metric):
        return f'{data}-{metric}'

    def after_iteration(self, model, epoch, evals_log):
        '''Update the plot.'''
        if not self.lines:
            for data, metric in evals_log.items():
                for metric_name, log in metric.items():
                    key = self._get_key(data, metric_name)
                    expanded = log + [0] * (self.rounds - len(log))
                    self.lines[key],  = self.ax.plot(self.x, expanded, label=key)
                    self.ax.legend()
        else:
            # https://pythonspot.com/matplotlib-update-plot/
            for data, metric in evals_log.items():
                for metric_name, log in metric.items():
                    key = self._get_key(data, metric_name)
                    expanded = log + [0] * (self.rounds - len(log))
                    self.lines[key].set_ydata(expanded)
            self.fig.canvas.draw()
        # False to indicate training should not stop.
        return False


def custom_callback():
    '''Demo for defining a custom callback function that plots evaluation result during
    training.'''
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

    D_train = xgb.DMatrix(X_train, y_train)
    D_valid = xgb.DMatrix(X_valid, y_valid)

    num_boost_round = 100
    plotting = Plotting(num_boost_round)

    # Pass it to the `callbacks` parameter as a list.
    xgb.train(
        {
            'objective': 'binary:logistic',
            'eval_metric': ['error', 'rmse'],
            'tree_method': 'gpu_hist'
        },
        D_train,
        evals=[(D_train, 'Train'), (D_valid, 'Valid')],
        num_boost_round=num_boost_round,
        callbacks=[plotting])


def check_point_callback():
    # only for demo, set a larger value (like 100) in practice as checkpointing is quite
    # slow.
    rounds = 2

    def check(as_pickle):
        for i in range(0, 10, rounds):
            if i == 0:
                continue
            if as_pickle:
                path = os.path.join(tmpdir, 'model_' + str(i) + '.pkl')
            else:
                path = os.path.join(tmpdir, 'model_' + str(i) + '.json')
            assert(os.path.exists(path))

    X, y = load_breast_cancer(return_X_y=True)
    m = xgb.DMatrix(X, y)
    # Check point to a temporary directory for demo
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use callback class from xgboost.callback
        # Feel free to subclass/customize it to suit your need.
        check_point = xgb.callback.TrainingCheckPoint(directory=tmpdir,
                                                      iterations=rounds,
                                                      name='model')
        xgb.train({'objective': 'binary:logistic'}, m,
                  num_boost_round=10,
                  verbose_eval=False,
                  callbacks=[check_point])
        check(False)

        # This version of checkpoint saves everything including parameters and
        # model.  See: doc/tutorials/saving_model.rst
        check_point = xgb.callback.TrainingCheckPoint(directory=tmpdir,
                                                      iterations=rounds,
                                                      as_pickle=True,
                                                      name='model')
        xgb.train({'objective': 'binary:logistic'}, m,
                  num_boost_round=10,
                  verbose_eval=False,
                  callbacks=[check_point])
        check(True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', default=1, type=int)
    args = parser.parse_args()

    check_point_callback()

    if args.plot:
        custom_callback()
