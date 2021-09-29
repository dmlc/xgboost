# -*- coding: utf-8 -*-
import unittest
import pytest
import numpy as np
import testing as tm
import xgboost as xgb

try:
    import pyarrow as pa
    import pyarrow.csv as pc
    import pyarrow.dataset as ds
    import pandas as pd
except ImportError:
    pass

pytestmark = pytest.mark.skipif(
    tm.no_arrow()['condition'] or tm.no_pandas()['condition'],
    reason=tm.no_arrow()['reason'] + ' or ' + tm.no_pandas()['reason'])

dpath = 'demo/data/'

class TestArrowTable(unittest.TestCase):

    def test_arrow_table(self):
        df = pd.DataFrame([[0, 1, 2., 3.], [1, 2, 3., 4.]],
                          columns=['a', 'b', 'c', 'd'])
        table = pa.Table.from_pandas(df)
        dm = xgb.DMatrix(table)
        assert dm.num_row() == 2
        assert dm.num_col() == 4

    def test_arrow_table_with_label(self):
        df = pd.DataFrame([[1, 2., 3.], [2, 3., 4.]],
                          columns=['a', 'b', 'c'])
        table = pa.Table.from_pandas(df)
        label = np.array([0, 1])
        dm = xgb.DMatrix(table)
        dm.set_label(label)
        assert dm.num_row() == 2
        assert dm.num_col() == 3
        np.testing.assert_array_equal(dm.get_label(), np.array([0, 1]))

    def test_arrow_table_with_label_name(self):
        df = pd.DataFrame([[1, 2., 3., 0], [2, 3., 4., 1]],
                          columns=['a', 'b', 'c', 'd'])
        table = pa.Table.from_pandas(df)
        dm = xgb.DMatrix(table, label='d')
        assert dm.num_row() == 2
        assert dm.num_col() == 3
        np.testing.assert_array_equal(dm.get_label(), np.array([0, 1]))

    def test_arrow_table_from_np(self):
        coldata = np.array([[1., 1., 0., 0.],
                            [2., 0., 1., 0.],
                            [3., 0., 0., 1.]])
        cols = list(map(pa.array, coldata))
        table = pa.Table.from_arrays(cols, ['a', 'b', 'c'])
        dm = xgb.DMatrix(table)
        assert dm.num_row() == 4
        assert dm.num_col() == 3

    def test_arrow_table_from_csv(self):
        dfile = dpath + 'veterans_lung_cancer.csv'
        table = pc.read_csv(dfile)
        dm = xgb.DMatrix(table)
        assert dm.num_row() == 137
        assert dm.num_col() == 13

    def test_arrow_dataset_from_csv(self):
        dfile = dpath + 'veterans_lung_cancer.csv'
        data = ds.dataset(dfile, format='csv')
        dm = xgb.DMatrix(data)
        assert dm.num_row() == 137
        assert dm.num_col() == 13

    def test_arrow_train(self):
        import pandas as pd
        rows = 100
        X = pd.DataFrame(
            {"A": np.random.randint(0, 10, size=rows),
             "B": np.random.randn(rows),
             "C": np.random.permutation([1, 0] * (rows // 2))})
        y = pd.Series(np.random.randn(rows))
        table = pa.Table.from_pandas(X)
        dtrain1 = xgb.DMatrix(table)
        dtrain1.set_label(y)
        bst1 = xgb.train({}, dtrain1, num_boost_round=10)
        preds1 = bst1.predict(xgb.DMatrix(X))
        dtrain2 = xgb.DMatrix(X, y)
        bst2 = xgb.train({}, dtrain2, num_boost_round=10)
        preds2 = bst2.predict(xgb.DMatrix(X))
        np.testing.assert_allclose(preds1, preds2)

    def test_arrow_survival(self):
        dfile = dpath + 'veterans_lung_cancer.csv'
        table = pc.read_csv(dfile)
        dtrain = xgb.DMatrix(table,
                label_lower_bound='Survival_label_lower_bound',
                label_upper_bound='Survival_label_upper_bound')

        base_params = {'verbosity': 0,
                       'objective': 'survival:aft',
                       'eval_metric': 'aft-nloglik',
                       'tree_method': 'hist',
                       'learning_rate': 0.05,
                       'aft_loss_distribution_scale': 1.20,
                       'max_depth': 6,
                       'lambda': 0.01,
                       'alpha': 0.02}
        nloglik_rec = {}
        dists = ['normal', 'logistic', 'extreme']
        for dist in dists:
            params = base_params
            params.update({'aft_loss_distribution': dist})
            evals_result = {}
            bst = xgb.train(params, dtrain, num_boost_round=500, evals=[(dtrain, 'train')],
                            evals_result=evals_result)
            nloglik_rec[dist] = evals_result['train']['aft-nloglik']
            # AFT metric (negative log likelihood) improve monotonically
            assert all(p >= q for p, q in zip(nloglik_rec[dist], nloglik_rec[dist][:1]))
        # For this data, normal distribution works the best
        assert nloglik_rec['normal'][-1] < 4.9
        assert nloglik_rec['logistic'][-1] > 4.9
        assert nloglik_rec['extreme'][-1] > 4.9
