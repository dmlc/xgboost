import xgboost as xgb
import tempfile
import os
from sklearn.datasets import load_breast_cancer


def check_point_callback():
    X, y = load_breast_cancer(return_X_y=True)
    m = xgb.DMatrix(X, y)
    # Check point to a temporary directory for demo
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use callback class from xgboost.callback
        # Feel free to subclass/customize it to suite your need.
        check_point = xgb.callback.TrainingCheckPoint(directory=tmpdir,
                                                      rounds=2,
                                                      name='model')
        xgb.train({'objective': 'binary:logistic'}, m,
                  num_boost_round=10,
                  verbose_eval=False,
                  callbacks=[check_point])
        for i in range(0, 10):
            assert os.path.exists(
                os.path.join(tmpdir, 'model_' + str(i) + '.json'))

        # This version of checkpoint saves everything including parameters and
        # model.  See: doc/tutorials/saving_model.rst
        check_point = xgb.callback.TrainingCheckPoint(directory=tmpdir,
                                                      rounds=1,
                                                      as_pickle=True,
                                                      name='model')
        xgb.train({'objective': 'binary:logistic'}, m,
                  num_boost_round=10,
                  verbose_eval=False,
                  callbacks=[check_point])
        for i in range(0, 10):
            assert os.path.exists(
                os.path.join(tmpdir, 'model_' + str(i) + '.pkl'))


if __name__ == '__main__':
    check_point_callback()
