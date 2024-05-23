from typing import Type

import numpy as np
import pytest

import xgboost as xgb
from xgboost import testing as tm
from xgboost.core import DataSplitMode
from xgboost.testing.data import pd_arrow_dtypes, pd_dtypes, run_base_margin_info

try:
    import pandas as pd
except ImportError:
    pass


pytestmark = pytest.mark.skipif(**tm.no_pandas())


dpath = "demo/data/"
rng = np.random.RandomState(1994)


class TestPandas:
    def test_pandas(self, data_split_mode=DataSplitMode.ROW):
        world_size = xgb.collective.get_world_size()
        df = pd.DataFrame([[1, 2.0, True], [2, 3.0, False]], columns=["a", "b", "c"])
        dm = xgb.DMatrix(df, label=pd.Series([1, 2]), data_split_mode=data_split_mode)
        assert dm.num_row() == 2
        if data_split_mode == DataSplitMode.ROW:
            assert dm.feature_names == ["a", "b", "c"]
            assert dm.feature_types == ["int", "float", "i"]
            assert dm.num_col() == 3
        else:
            assert dm.feature_names == tm.column_split_feature_names(
                ["a", "b", "c"], world_size
            )
            assert dm.feature_types == ["int", "float", "i"] * world_size
            assert dm.num_col() == 3 * world_size
        np.testing.assert_array_equal(dm.get_label(), np.array([1, 2]))

        # overwrite feature_names and feature_types
        dm = xgb.DMatrix(
            df,
            label=pd.Series([1, 2]),
            feature_names=["x", "y", "z"],
            feature_types=["q", "q", "q"],
            data_split_mode=data_split_mode,
        )
        assert dm.num_row() == 2
        if data_split_mode == DataSplitMode.ROW:
            assert dm.feature_names == ["x", "y", "z"]
            assert dm.feature_types == ["q", "q", "q"]
            assert dm.num_col() == 3
        else:
            assert dm.feature_names == tm.column_split_feature_names(
                ["x", "y", "z"], world_size
            )
            assert dm.feature_types == ["q", "q", "q"] * world_size
            assert dm.num_col() == 3 * world_size

        # incorrect dtypes
        df = pd.DataFrame([[1, 2.0, "x"], [2, 3.0, "y"]], columns=["a", "b", "c"])
        with pytest.raises(ValueError):
            xgb.DMatrix(df, data_split_mode=data_split_mode)

        # numeric columns
        df = pd.DataFrame([[1, 2.0, True], [2, 3.0, False]])
        dm = xgb.DMatrix(df, label=pd.Series([1, 2]), data_split_mode=data_split_mode)
        assert dm.num_row() == 2
        if data_split_mode == DataSplitMode.ROW:
            assert dm.feature_names == ["0", "1", "2"]
            assert dm.feature_types == ["int", "float", "i"]
            assert dm.num_col() == 3
        else:
            assert dm.feature_names == tm.column_split_feature_names(
                ["0", "1", "2"], world_size
            )
            assert dm.feature_types == ["int", "float", "i"] * world_size
            assert dm.num_col() == 3 * world_size
        np.testing.assert_array_equal(dm.get_label(), np.array([1, 2]))

        df = pd.DataFrame([[1, 2.0, 1], [2, 3.0, 1]], columns=[4, 5, 6])
        dm = xgb.DMatrix(df, label=pd.Series([1, 2]), data_split_mode=data_split_mode)
        assert dm.num_row() == 2
        if data_split_mode == DataSplitMode.ROW:
            assert dm.feature_names == ["4", "5", "6"]
            assert dm.feature_types == ["int", "float", "int"]
            assert dm.num_col() == 3
        else:
            assert dm.feature_names == tm.column_split_feature_names(
                ["4", "5", "6"], world_size
            )
            assert dm.feature_types == ["int", "float", "int"] * world_size
            assert dm.num_col() == 3 * world_size

        df = pd.DataFrame({"A": ["X", "Y", "Z"], "B": [1, 2, 3]})
        dummies = pd.get_dummies(df)
        #    B  A_X  A_Y  A_Z
        # 0  1    1    0    0
        # 1  2    0    1    0
        # 2  3    0    0    1
        result, _, _ = xgb.data._transform_pandas_df(dummies, enable_categorical=False)
        exp = np.array(
            [[1.0, 1.0, 0.0, 0.0], [2.0, 0.0, 1.0, 0.0], [3.0, 0.0, 0.0, 1.0]]
        ).T
        np.testing.assert_array_equal(result.columns, exp)
        dm = xgb.DMatrix(dummies, data_split_mode=data_split_mode)
        assert dm.num_row() == 3
        if data_split_mode == DataSplitMode.ROW:
            assert dm.feature_names == ["B", "A_X", "A_Y", "A_Z"]
            if int(pd.__version__[0]) >= 2:
                assert dm.feature_types == ["int", "i", "i", "i"]
            else:
                assert dm.feature_types == ["int", "int", "int", "int"]
            assert dm.num_col() == 4
        else:
            assert dm.feature_names == tm.column_split_feature_names(
                ["B", "A_X", "A_Y", "A_Z"], world_size
            )
            if int(pd.__version__[0]) >= 2:
                assert dm.feature_types == ["int", "i", "i", "i"] * world_size
            else:
                assert dm.feature_types == ["int", "int", "int", "int"] * world_size
            assert dm.num_col() == 4 * world_size

        df = pd.DataFrame({"A=1": [1, 2, 3], "A=2": [4, 5, 6]})
        dm = xgb.DMatrix(df, data_split_mode=data_split_mode)
        assert dm.num_row() == 3
        if data_split_mode == DataSplitMode.ROW:
            assert dm.feature_names == ["A=1", "A=2"]
            assert dm.feature_types == ["int", "int"]
            assert dm.num_col() == 2
        else:
            assert dm.feature_names == tm.column_split_feature_names(
                ["A=1", "A=2"], world_size
            )
            assert dm.feature_types == ["int", "int"] * world_size
            assert dm.num_col() == 2 * world_size

        df_int = pd.DataFrame([[1, 1.1], [2, 2.2]], columns=[9, 10])
        dm_int = xgb.DMatrix(df_int, data_split_mode=data_split_mode)
        df_range = pd.DataFrame([[1, 1.1], [2, 2.2]], columns=range(9, 11, 1))
        dm_range = xgb.DMatrix(df_range, data_split_mode=data_split_mode)
        if data_split_mode == DataSplitMode.ROW:
            assert dm_int.feature_names == ["9", "10"]  # assert not "9 "
        else:
            assert dm_int.feature_names == tm.column_split_feature_names(
                ["9", "10"], world_size
            )
        assert dm_int.feature_names == dm_range.feature_names

        # test MultiIndex as columns
        df = pd.DataFrame(
            [(1, 2, 3, 4, 5, 6), (6, 5, 4, 3, 2, 1)],
            columns=pd.MultiIndex.from_tuples(
                (
                    ("a", 1),
                    ("a", 2),
                    ("a", 3),
                    ("b", 1),
                    ("b", 2),
                    ("b", 3),
                )
            ),
        )
        dm = xgb.DMatrix(df, data_split_mode=data_split_mode)
        assert dm.num_row() == 2
        if data_split_mode == DataSplitMode.ROW:
            assert dm.feature_names == ["a 1", "a 2", "a 3", "b 1", "b 2", "b 3"]
            assert dm.feature_types == ["int", "int", "int", "int", "int", "int"]
            assert dm.num_col() == 6
        else:
            assert dm.feature_names == tm.column_split_feature_names(
                ["a 1", "a 2", "a 3", "b 1", "b 2", "b 3"], world_size
            )
            assert (
                dm.feature_types
                == ["int", "int", "int", "int", "int", "int"] * world_size
            )
            assert dm.num_col() == 6 * world_size

        # test Index as columns
        df = pd.DataFrame([[1, 1.1], [2, 2.2]], columns=pd.Index([1, 2]))
        Xy = xgb.DMatrix(df, data_split_mode=data_split_mode)
        if data_split_mode == DataSplitMode.ROW:
            np.testing.assert_equal(np.array(Xy.feature_names), np.array(["1", "2"]))
        else:
            np.testing.assert_equal(
                np.array(Xy.feature_names),
                np.array(tm.column_split_feature_names(["1", "2"], world_size)),
            )

        # test pandas series
        data_series = pd.Series([1, 2, 3, 4, 5])
        dm = xgb.DMatrix(data_series, data_split_mode=data_split_mode)
        assert dm.num_row() == 5
        if data_split_mode == DataSplitMode.ROW:
            assert dm.num_col() == 1
        else:
            assert dm.num_col() == 1 * world_size

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_multi_target(self) -> None:
        from sklearn.datasets import make_regression

        X, y = make_regression(n_samples=1024, n_features=4, n_targets=3)
        ydf = pd.DataFrame({i: y[:, i] for i in range(y.shape[1])})

        Xy = xgb.DMatrix(X, ydf)
        assert Xy.num_row() == y.shape[0]
        assert Xy.get_label().size == y.shape[0] * y.shape[1]
        Xy = xgb.QuantileDMatrix(X, ydf)
        assert Xy.num_row() == y.shape[0]
        assert Xy.get_label().size == y.shape[0] * y.shape[1]

    def test_slice(self):
        rng = np.random.RandomState(1994)
        rows = 100
        X = rng.randint(3, 7, size=rows)
        X = pd.DataFrame({"f0": X})
        y = rng.randn(rows)
        ridxs = [1, 2, 3, 4, 5, 6]
        m = xgb.DMatrix(X, y)
        sliced = m.slice(ridxs)

        assert m.feature_types == sliced.feature_types

    def test_pandas_categorical(self, data_split_mode=DataSplitMode.ROW):
        world_size = xgb.collective.get_world_size()
        rng = np.random.RandomState(1994)
        rows = 100
        X = rng.randint(3, 7, size=rows)
        X = pd.Series(X, dtype="category")
        X = pd.DataFrame({"f0": X})
        y = rng.randn(rows)
        m = xgb.DMatrix(X, y, enable_categorical=True, data_split_mode=data_split_mode)
        assert m.feature_types[0] == "c"

        X_0 = ["f", "o", "o"]
        X_1 = [4, 3, 2]
        X = pd.DataFrame({"feat_0": X_0, "feat_1": X_1})
        X["feat_0"] = X["feat_0"].astype("category")
        transformed, _, feature_types = xgb.data._transform_pandas_df(
            X, enable_categorical=True
        )

        assert transformed.columns[0].min() == 0

        # test missing value
        X = pd.DataFrame({"f0": ["a", "b", np.nan]})
        X["f0"] = X["f0"].astype("category")
        arr, _, _ = xgb.data._transform_pandas_df(X, enable_categorical=True)
        for c in arr.columns:
            assert not np.any(c == -1.0)

        X = X["f0"]
        y = y[: X.shape[0]]
        with pytest.raises(ValueError, match=r".*enable_categorical.*"):
            xgb.DMatrix(X, y, data_split_mode=data_split_mode)

        Xy = xgb.DMatrix(X, y, enable_categorical=True, data_split_mode=data_split_mode)
        assert Xy.num_row() == 3
        if data_split_mode == DataSplitMode.ROW:
            assert Xy.num_col() == 1
        else:
            assert Xy.num_col() == 1 * world_size

    def test_pandas_sparse(self):
        import pandas as pd

        rows = 100
        X = pd.DataFrame(
            {
                "A": pd.arrays.SparseArray(np.random.randint(0, 10, size=rows)),
                "B": pd.arrays.SparseArray(np.random.randn(rows)),
                "C": pd.arrays.SparseArray(
                    np.random.permutation([True, False] * (rows // 2))
                ),
            }
        )
        y = pd.Series(pd.arrays.SparseArray(np.random.randn(rows)))
        with pytest.warns(UserWarning, match="Sparse arrays from pandas"):
            dtrain = xgb.DMatrix(X, y)
        booster = xgb.train({}, dtrain, num_boost_round=4)
        with pytest.warns(UserWarning, match="Sparse arrays from pandas"):
            predt_sparse = booster.predict(xgb.DMatrix(X))
            predt_dense = booster.predict(xgb.DMatrix(X.sparse.to_dense()))
        np.testing.assert_allclose(predt_sparse, predt_dense)

    def test_pandas_label(
        self, data_split_mode: DataSplitMode = DataSplitMode.ROW
    ) -> None:
        world_size = xgb.collective.get_world_size()
        # label must be a single column
        df = pd.DataFrame({"A": ["X", "Y", "Z"], "B": [1, 2, 3]})
        with pytest.raises(ValueError):
            xgb.data._transform_pandas_df(df, False, None, None, "label")

        # label must be supported dtype
        df = pd.DataFrame({"A": np.array(["a", "b", "c"], dtype=object)})
        with pytest.raises(ValueError):
            xgb.data._transform_pandas_df(df, False, None, None, "label")

        df = pd.DataFrame({"A": np.array([1, 2, 3], dtype=int)})
        result, _, _ = xgb.data._transform_pandas_df(df, False, None, None, "label")
        np.testing.assert_array_equal(
            np.stack(result.columns, axis=1),
            np.array([[1.0], [2.0], [3.0]], dtype=float),
        )
        dm = xgb.DMatrix(
            np.random.randn(3, 2), label=df, data_split_mode=data_split_mode
        )
        assert dm.num_row() == 3
        if data_split_mode == DataSplitMode.ROW:
            assert dm.num_col() == 2
        else:
            assert dm.num_col() == 2 * world_size

    def test_pandas_weight(self, data_split_mode=DataSplitMode.ROW):
        world_size = xgb.collective.get_world_size()
        kRows = 32
        kCols = 8

        X = np.random.randn(kRows, kCols)
        y = np.random.randn(kRows)
        w = np.random.uniform(size=kRows).astype(np.float32)
        w_pd = pd.DataFrame(w)
        data = xgb.DMatrix(X, y, weight=w_pd, data_split_mode=data_split_mode)

        assert data.num_row() == kRows
        if data_split_mode == DataSplitMode.ROW:
            assert data.num_col() == kCols
        else:
            assert data.num_col() == kCols * world_size
        np.testing.assert_array_equal(data.get_weight(), w)

    def test_base_margin(self):
        run_base_margin_info(pd.DataFrame, xgb.DMatrix, "cpu")

    def test_cv_as_pandas(self):
        dm, _ = tm.load_agaricus(__file__)
        params = {
            "max_depth": 2,
            "eta": 1,
            "objective": "binary:logistic",
            "eval_metric": "error",
        }

        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10)
        assert isinstance(cv, pd.DataFrame)
        exp = pd.Index(
            ["test-error-mean", "test-error-std", "train-error-mean", "train-error-std"]
        )
        assert len(cv.columns.intersection(exp)) == 4

        # show progress log (result is the same as above)
        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10, verbose_eval=True)
        assert isinstance(cv, pd.DataFrame)
        exp = pd.Index(
            ["test-error-mean", "test-error-std", "train-error-mean", "train-error-std"]
        )
        assert len(cv.columns.intersection(exp)) == 4
        cv = xgb.cv(
            params, dm, num_boost_round=10, nfold=10, verbose_eval=True, show_stdv=False
        )
        assert isinstance(cv, pd.DataFrame)
        exp = pd.Index(
            ["test-error-mean", "test-error-std", "train-error-mean", "train-error-std"]
        )
        assert len(cv.columns.intersection(exp)) == 4

        params = {
            "max_depth": 2,
            "eta": 1,
            "objective": "binary:logistic",
            "eval_metric": "auc",
        }
        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10, as_pandas=True)
        assert "eval_metric" in params
        assert "auc" in cv.columns[0]

        params = {
            "max_depth": 2,
            "eta": 1,
            "objective": "binary:logistic",
            "eval_metric": ["auc"],
        }
        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10, as_pandas=True)
        assert "eval_metric" in params
        assert "auc" in cv.columns[0]

        params = {
            "max_depth": 2,
            "eta": 1,
            "objective": "binary:logistic",
            "eval_metric": ["auc"],
        }
        cv = xgb.cv(
            params,
            dm,
            num_boost_round=10,
            nfold=10,
            as_pandas=True,
            early_stopping_rounds=1,
        )
        assert "eval_metric" in params
        assert "auc" in cv.columns[0]
        assert cv.shape[0] < 10

        params = {
            "max_depth": 2,
            "eta": 1,
            "objective": "binary:logistic",
        }
        cv = xgb.cv(
            params, dm, num_boost_round=10, nfold=10, as_pandas=True, metrics="auc"
        )
        assert "auc" in cv.columns[0]

        params = {
            "max_depth": 2,
            "eta": 1,
            "objective": "binary:logistic",
        }
        cv = xgb.cv(
            params, dm, num_boost_round=10, nfold=10, as_pandas=True, metrics=["auc"]
        )
        assert "auc" in cv.columns[0]

        params = {
            "max_depth": 2,
            "eta": 1,
            "objective": "binary:logistic",
            "eval_metric": ["auc"],
        }
        cv = xgb.cv(
            params, dm, num_boost_round=10, nfold=10, as_pandas=True, metrics="error"
        )
        assert "eval_metric" in params
        assert "auc" not in cv.columns[0]
        assert "error" in cv.columns[0]

        cv = xgb.cv(
            params, dm, num_boost_round=10, nfold=10, as_pandas=True, metrics=["error"]
        )
        assert "eval_metric" in params
        assert "auc" not in cv.columns[0]
        assert "error" in cv.columns[0]

        params = list(params.items())
        cv = xgb.cv(
            params, dm, num_boost_round=10, nfold=10, as_pandas=True, metrics=["error"]
        )
        assert isinstance(params, list)
        assert "auc" not in cv.columns[0]
        assert "error" in cv.columns[0]

    @pytest.mark.parametrize("DMatrixT", [xgb.DMatrix, xgb.QuantileDMatrix])
    def test_nullable_type(self, DMatrixT) -> None:
        from xgboost.data import is_pd_cat_dtype

        for orig, df in pd_dtypes():
            if hasattr(df.dtypes, "__iter__"):
                enable_categorical = any(is_pd_cat_dtype(dtype) for dtype in df.dtypes)
            else:
                # series
                enable_categorical = is_pd_cat_dtype(df.dtype)

            f0_orig = orig[orig.columns[0]] if isinstance(orig, pd.DataFrame) else orig
            f0 = df[df.columns[0]] if isinstance(df, pd.DataFrame) else df
            y_orig = f0_orig.astype(pd.Float32Dtype()).fillna(0)
            y = f0.astype(pd.Float32Dtype()).fillna(0)

            m_orig = DMatrixT(orig, enable_categorical=enable_categorical, label=y_orig)
            # extension types
            copy = df.copy()
            m_etype = DMatrixT(df, enable_categorical=enable_categorical, label=y)
            # no mutation
            assert df.equals(copy)
            # different from pd.BooleanDtype(), None is converted to False with bool
            if hasattr(orig.dtypes, "__iter__") and any(
                dtype == "bool" for dtype in orig.dtypes
            ):
                assert not tm.predictor_equal(m_orig, m_etype)
            else:
                assert tm.predictor_equal(m_orig, m_etype)

            np.testing.assert_allclose(m_orig.get_label(), m_etype.get_label())
            np.testing.assert_allclose(m_etype.get_label(), y.values.astype(np.float32))

            if isinstance(df, pd.DataFrame):
                f0 = df["f0"]
                with pytest.raises(ValueError, match="Label contains NaN"):
                    xgb.DMatrix(df, f0, enable_categorical=enable_categorical)

    @pytest.mark.skipif(**tm.no_arrow())
    @pytest.mark.parametrize("DMatrixT", [xgb.DMatrix, xgb.QuantileDMatrix])
    def test_pyarrow_type(self, DMatrixT: Type[xgb.DMatrix]) -> None:
        for orig, df in pd_arrow_dtypes():
            f0_orig: pd.Series = orig["f0"]
            f0 = df["f0"]

            if f0.dtype.name.startswith("bool"):
                y = None
                y_orig = None
            else:
                y_orig = f0_orig.fillna(0, inplace=False)
                y = f0.fillna(0, inplace=False)

            m_orig = DMatrixT(orig, enable_categorical=True, label=y_orig)
            m_etype = DMatrixT(df, enable_categorical=True, label=y)

            assert tm.predictor_equal(m_orig, m_etype)
            if y is not None:
                np.testing.assert_allclose(m_orig.get_label(), m_etype.get_label())
                np.testing.assert_allclose(m_etype.get_label(), y.values)

    @pytest.mark.parametrize("DMatrixT", [xgb.DMatrix, xgb.QuantileDMatrix])
    def test_mixed_type(self, DMatrixT: Type[xgb.DMatrix]) -> None:
        f0 = np.arange(0, 4)
        f1 = pd.Series(f0, dtype="int64[pyarrow]")
        f2l = list(f0)
        f2l[0] = pd.NA
        f2 = pd.Series(f2l, dtype=pd.Int64Dtype())

        df = pd.DataFrame({"f0": f0})
        df["f2"] = f2

        m = DMatrixT(df)
        assert m.num_col() == df.shape[1]

        df["f1"] = f1
        m = DMatrixT(df)
        assert m.num_col() == df.shape[1]
        assert m.num_row() == df.shape[0]
        assert m.num_nonmissing() == df.size - 1
        assert m.feature_names == list(map(str, df.columns))
        assert m.feature_types == ["int"] * df.shape[1]

        y = f0
        m.set_info(label=y)
        booster = xgb.train({}, m)
        p0 = booster.inplace_predict(df)
        p1 = booster.predict(m)
        np.testing.assert_allclose(p0, p1)

    @pytest.mark.skipif(tm.is_windows(), reason="Rabit does not run on windows")
    def test_pandas_column_split(self):
        tm.run_with_rabit(
            world_size=3, test_fn=self.test_pandas, data_split_mode=DataSplitMode.COL
        )

    @pytest.mark.skipif(tm.is_windows(), reason="Rabit does not run on windows")
    def test_pandas_categorical_column_split(self):
        tm.run_with_rabit(
            world_size=3,
            test_fn=self.test_pandas_categorical,
            data_split_mode=DataSplitMode.COL,
        )

    @pytest.mark.skipif(tm.is_windows(), reason="Rabit does not run on windows")
    def test_pandas_sparse_column_split(self):
        rows = 100
        X = pd.DataFrame(
            {
                "A": pd.arrays.SparseArray(np.random.randint(0, 10, size=rows)),
                "B": pd.arrays.SparseArray(np.random.randn(rows)),
                "C": pd.arrays.SparseArray(
                    np.random.permutation([True, False] * (rows // 2))
                ),
            }
        )
        y = pd.Series(pd.arrays.SparseArray(np.random.randn(rows)))

        def verify_pandas_sparse():
            with pytest.warns(UserWarning, match="Sparse arrays from pandas"):
                dtrain = xgb.DMatrix(X, y, data_split_mode=DataSplitMode.COL)
            booster = xgb.train({}, dtrain, num_boost_round=4)
            with pytest.warns(UserWarning, match="Sparse arrays from pandas"):
                predt_sparse = booster.predict(
                    xgb.DMatrix(X, data_split_mode=DataSplitMode.COL)
                )
                predt_dense = booster.predict(
                    xgb.DMatrix(X.sparse.to_dense(), data_split_mode=DataSplitMode.COL)
                )
            np.testing.assert_allclose(predt_sparse, predt_dense)

        tm.run_with_rabit(world_size=3, test_fn=verify_pandas_sparse)

    @pytest.mark.skipif(tm.is_windows(), reason="Rabit does not run on windows")
    def test_pandas_label_column_split(self):
        tm.run_with_rabit(
            world_size=3,
            test_fn=self.test_pandas_label,
            data_split_mode=DataSplitMode.COL,
        )

    @pytest.mark.skipif(tm.is_windows(), reason="Rabit does not run on windows")
    def test_pandas_weight_column_split(self):
        tm.run_with_rabit(
            world_size=3,
            test_fn=self.test_pandas_weight,
            data_split_mode=DataSplitMode.COL,
        )
