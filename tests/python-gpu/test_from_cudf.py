import json
import sys

import numpy as np
import pytest

import xgboost as xgb
from xgboost import testing as tm

sys.path.append("tests/python")
from test_dmatrix import set_base_margin_info


def dmatrix_from_cudf(input_type, DMatrixT, missing=np.NAN):
    '''Test constructing DMatrix from cudf'''
    import cudf
    import pandas as pd

    kRows = 80
    kCols = 3

    na = np.random.randn(kRows, kCols)
    na[:, 0:2] = na[:, 0:2].astype(input_type)

    na[5, 0] = missing
    na[3, 1] = missing

    pa = pd.DataFrame({'0': na[:, 0],
                       '1': na[:, 1],
                       '2': na[:, 2].astype(np.int32)})

    np_label = np.random.randn(kRows).astype(input_type)
    pa_label = pd.DataFrame(np_label)

    cd = cudf.from_pandas(pa)
    cd_label = cudf.from_pandas(pa_label).iloc[:, 0]

    dtrain = DMatrixT(cd, missing=missing, label=cd_label)
    assert dtrain.num_col() == kCols
    assert dtrain.num_row() == kRows


def _test_from_cudf(DMatrixT):
    '''Test constructing DMatrix from cudf'''
    import cudf
    dmatrix_from_cudf(np.float32, DMatrixT, np.NAN)
    dmatrix_from_cudf(np.float64, DMatrixT, np.NAN)

    dmatrix_from_cudf(np.int8, DMatrixT, 2)
    dmatrix_from_cudf(np.int32, DMatrixT, -2)
    dmatrix_from_cudf(np.int64, DMatrixT, -3)

    cd = cudf.DataFrame({'x': [1, 2, 3], 'y': [0.1, 0.2, 0.3]})
    dtrain = DMatrixT(cd)

    assert dtrain.feature_names == ['x', 'y']
    assert dtrain.feature_types == ['int', 'float']

    series = cudf.DataFrame({'x': [1, 2, 3]}).iloc[:, 0]
    assert isinstance(series, cudf.Series)
    dtrain = DMatrixT(series)

    assert dtrain.feature_names == ['x']
    assert dtrain.feature_types == ['int']

    with pytest.raises(ValueError, match=r".*multi.*"):
        dtrain = DMatrixT(cd, label=cd)
        xgb.train({"tree_method": "gpu_hist", "objective": "multi:softprob"}, dtrain)

    # Test when number of elements is less than 8
    X = cudf.DataFrame({'x': cudf.Series([0, 1, 2, np.NAN, 4],
                                         dtype=np.int32)})
    dtrain = DMatrixT(X)
    assert dtrain.num_col() == 1
    assert dtrain.num_row() == 5

    # Boolean is not supported.
    X_boolean = cudf.DataFrame({'x': cudf.Series([True, False])})
    with pytest.raises(Exception):
        dtrain = DMatrixT(X_boolean)

    y_boolean = cudf.DataFrame({
        'x': cudf.Series([True, False, True, True, True])})
    with pytest.raises(Exception):
        dtrain = DMatrixT(X_boolean, label=y_boolean)


def _test_cudf_training(DMatrixT):
    import pandas as pd
    from cudf import DataFrame as df
    np.random.seed(1)
    X = pd.DataFrame(np.random.randn(50, 10))
    y = pd.DataFrame(np.random.randn(50))
    weights = np.random.random(50) + 1.0
    cudf_weights = df.from_pandas(pd.DataFrame(weights))
    base_margin = np.random.random(50)
    cudf_base_margin = df.from_pandas(pd.DataFrame(base_margin))

    evals_result_cudf = {}
    dtrain_cudf = DMatrixT(df.from_pandas(X), df.from_pandas(y), weight=cudf_weights,
                           base_margin=cudf_base_margin)
    params = {'gpu_id': 0, 'tree_method': 'gpu_hist'}
    xgb.train(params, dtrain_cudf, evals=[(dtrain_cudf, "train")],
              evals_result=evals_result_cudf)
    evals_result_np = {}
    dtrain_np = xgb.DMatrix(X, y, weight=weights, base_margin=base_margin)
    xgb.train(params, dtrain_np, evals=[(dtrain_np, "train")],
              evals_result=evals_result_np)
    assert np.array_equal(evals_result_cudf["train"]["rmse"], evals_result_np["train"]["rmse"])


def _test_cudf_metainfo(DMatrixT):
    import pandas as pd
    from cudf import DataFrame as df
    n = 100
    X = np.random.random((n, 2))
    dmat_cudf = DMatrixT(df.from_pandas(pd.DataFrame(X)))
    dmat = xgb.DMatrix(X)
    floats = np.random.random(n)
    uints = np.array([4, 2, 8]).astype("uint32")
    cudf_floats = df.from_pandas(pd.DataFrame(floats))
    cudf_uints = df.from_pandas(pd.DataFrame(uints))
    dmat.set_float_info('weight', floats)
    dmat.set_float_info('label', floats)
    dmat.set_float_info('base_margin', floats)
    dmat.set_uint_info('group', uints)
    dmat_cudf.set_info(weight=cudf_floats)
    dmat_cudf.set_info(label=cudf_floats)
    dmat_cudf.set_info(base_margin=cudf_floats)
    dmat_cudf.set_info(group=cudf_uints)

    # Test setting info with cudf DataFrame
    assert np.array_equal(dmat.get_float_info('weight'), dmat_cudf.get_float_info('weight'))
    assert np.array_equal(dmat.get_float_info('label'), dmat_cudf.get_float_info('label'))
    assert np.array_equal(dmat.get_float_info('base_margin'),
                          dmat_cudf.get_float_info('base_margin'))
    assert np.array_equal(dmat.get_uint_info('group_ptr'), dmat_cudf.get_uint_info('group_ptr'))

    # Test setting info with cudf Series
    dmat_cudf.set_info(weight=cudf_floats[cudf_floats.columns[0]])
    dmat_cudf.set_info(label=cudf_floats[cudf_floats.columns[0]])
    dmat_cudf.set_info(base_margin=cudf_floats[cudf_floats.columns[0]])
    dmat_cudf.set_info(group=cudf_uints[cudf_uints.columns[0]])
    assert np.array_equal(dmat.get_float_info('weight'), dmat_cudf.get_float_info('weight'))
    assert np.array_equal(dmat.get_float_info('label'), dmat_cudf.get_float_info('label'))
    assert np.array_equal(dmat.get_float_info('base_margin'),
                          dmat_cudf.get_float_info('base_margin'))
    assert np.array_equal(dmat.get_uint_info('group_ptr'), dmat_cudf.get_uint_info('group_ptr'))

    set_base_margin_info(df, DMatrixT, "gpu_hist")


class TestFromColumnar:
    '''Tests for constructing DMatrix from data structure conforming Apache
Arrow specification.'''

    @pytest.mark.skipif(**tm.no_cudf())
    def test_simple_dmatrix_from_cudf(self):
        _test_from_cudf(xgb.DMatrix)

    @pytest.mark.skipif(**tm.no_cudf())
    def test_device_dmatrix_from_cudf(self):
        _test_from_cudf(xgb.QuantileDMatrix)

    @pytest.mark.skipif(**tm.no_cudf())
    def test_cudf_training_simple_dmatrix(self):
        _test_cudf_training(xgb.DMatrix)

    @pytest.mark.skipif(**tm.no_cudf())
    def test_cudf_training_device_dmatrix(self):
        _test_cudf_training(xgb.QuantileDMatrix)

    @pytest.mark.skipif(**tm.no_cudf())
    def test_cudf_metainfo_simple_dmatrix(self):
        _test_cudf_metainfo(xgb.DMatrix)

    @pytest.mark.skipif(**tm.no_cudf())
    def test_cudf_metainfo_device_dmatrix(self):
        _test_cudf_metainfo(xgb.QuantileDMatrix)

    @pytest.mark.skipif(**tm.no_cudf())
    def test_cudf_categorical(self) -> None:
        import cudf
        n_features = 30
        _X, _y = tm.make_categorical(100, n_features, 17, False)
        X = cudf.from_pandas(_X)
        y = cudf.from_pandas(_y)

        Xy = xgb.DMatrix(X, y, enable_categorical=True)
        assert Xy.feature_types is not None
        assert len(Xy.feature_types) == X.shape[1]
        assert all(t == "c" for t in Xy.feature_types)

        Xy = xgb.QuantileDMatrix(X, y, enable_categorical=True)
        assert Xy.feature_types is not None
        assert len(Xy.feature_types) == X.shape[1]
        assert all(t == "c" for t in Xy.feature_types)

        # mixed dtypes
        X["1"] = X["1"].astype(np.int64)
        X["3"] = X["3"].astype(np.int64)
        df, cat_codes, _, _ = xgb.data._transform_cudf_df(
            X, None, None, enable_categorical=True
        )
        assert X.shape[1] == n_features
        assert len(cat_codes) == X.shape[1]
        assert not cat_codes[0]
        assert not cat_codes[2]

        interfaces_str = xgb.data._cudf_array_interfaces(df, cat_codes)
        interfaces = json.loads(interfaces_str)
        assert len(interfaces) == X.shape[1]

        # test missing value
        X = cudf.DataFrame({"f0": ["a", "b", np.NaN]})
        X["f0"] = X["f0"].astype("category")
        df, cat_codes, _, _ = xgb.data._transform_cudf_df(
            X, None, None, enable_categorical=True
        )
        for col in cat_codes:
            assert col.has_nulls

        y = [0, 1, 2]
        with pytest.raises(ValueError):
            xgb.DMatrix(X, y)
        Xy = xgb.DMatrix(X, y, enable_categorical=True)
        assert Xy.num_row() == 3
        assert Xy.num_col() == 1

        with pytest.raises(ValueError, match="enable_categorical"):
            xgb.QuantileDMatrix(X, y)

        Xy = xgb.QuantileDMatrix(X, y, enable_categorical=True)
        assert Xy.num_row() == 3
        assert Xy.num_col() == 1

        X = X["f0"]
        with pytest.raises(ValueError):
            xgb.DMatrix(X, y)

        Xy = xgb.DMatrix(X, y, enable_categorical=True)
        assert Xy.num_row() == 3
        assert Xy.num_col() == 1


@pytest.mark.skipif(**tm.no_cudf())
@pytest.mark.skipif(**tm.no_cupy())
@pytest.mark.skipif(**tm.no_sklearn())
@pytest.mark.skipif(**tm.no_pandas())
def test_cudf_training_with_sklearn():
    import pandas as pd
    from cudf import DataFrame as df
    from cudf import Series as ss
    np.random.seed(1)
    X = pd.DataFrame(np.random.randn(50, 10))
    y = pd.DataFrame((np.random.randn(50) > 0).astype(np.int8))
    weights = np.random.random(50) + 1.0
    cudf_weights = df.from_pandas(pd.DataFrame(weights))
    base_margin = np.random.random(50)
    cudf_base_margin = df.from_pandas(pd.DataFrame(base_margin))

    X_cudf = df.from_pandas(X)
    y_cudf = df.from_pandas(y)
    y_cudf_series = ss(data=y.iloc[:, 0])

    for y_obj in [y_cudf, y_cudf_series]:
        clf = xgb.XGBClassifier(gpu_id=0, tree_method='gpu_hist')
        clf.fit(X_cudf, y_obj, sample_weight=cudf_weights, base_margin=cudf_base_margin,
                eval_set=[(X_cudf, y_obj)])
        pred = clf.predict(X_cudf)
        assert np.array_equal(np.unique(pred), np.array([0, 1]))


class IterForDMatrixTest(xgb.core.DataIter):
    '''A data iterator for XGBoost DMatrix.

    `reset` and `next` are required for any data iterator, other functions here
    are utilites for demonstration's purpose.

    '''
    ROWS_PER_BATCH = 100            # data is splited by rows
    BATCHES = 16

    def __init__(self, categorical):
        '''Generate some random data for demostration.

        Actual data can be anything that is currently supported by XGBoost.
        '''
        import cudf
        self.rows = self.ROWS_PER_BATCH

        if categorical:
            self._data = []
            self._labels = []
            for i in range(self.BATCHES):
                X, y = tm.make_categorical(self.ROWS_PER_BATCH, 4, 13, False)
                self._data.append(cudf.from_pandas(X))
                self._labels.append(y)
        else:
            rng = np.random.RandomState(1994)
            self._data = [
                cudf.DataFrame(
                    {'a': rng.randn(self.ROWS_PER_BATCH),
                     'b': rng.randn(self.ROWS_PER_BATCH)})] * self.BATCHES
            self._labels = [rng.randn(self.rows)] * self.BATCHES

        self.it = 0             # set iterator to 0
        super().__init__()

    def as_array(self):
        import cudf
        return cudf.concat(self._data)

    def as_array_labels(self):
        return np.concatenate(self._labels)

    def data(self):
        '''Utility function for obtaining current batch of data.'''
        return self._data[self.it]

    def labels(self):
        '''Utility function for obtaining current batch of label.'''
        return self._labels[self.it]

    def reset(self):
        '''Reset the iterator'''
        self.it = 0

    def next(self, input_data):
        '''Yield next batch of data'''
        if self.it == len(self._data):
            # Return 0 when there's no more batch.
            return 0
        input_data(data=self.data(), label=self.labels())
        self.it += 1
        return 1


@pytest.mark.skipif(**tm.no_cudf())
@pytest.mark.parametrize("enable_categorical", [True, False])
def test_from_cudf_iter(enable_categorical):
    rounds = 100
    it = IterForDMatrixTest(enable_categorical)
    params = {"tree_method": "gpu_hist"}

    # Use iterator
    m_it = xgb.QuantileDMatrix(it, enable_categorical=enable_categorical)
    reg_with_it = xgb.train(params, m_it, num_boost_round=rounds)

    X = it.as_array()
    y = it.as_array_labels()

    m = xgb.DMatrix(X, y, enable_categorical=enable_categorical)

    assert m_it.num_col() == m.num_col()
    assert m_it.num_row() == m.num_row()

    reg = xgb.train(params, m, num_boost_round=rounds)

    predict = reg.predict(m)
    predict_with_it = reg_with_it.predict(m_it)
    np.testing.assert_allclose(predict_with_it, predict)
