# -*- coding: utf-8 -*-
import unittest
import pytest
import numpy as np

import testing as tm
import xgboost as xgb

try:
    import pyarrow as pa
    import pyarrow.csv as pc
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
        dm = xgb.DMatrix(table, label=label)
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


