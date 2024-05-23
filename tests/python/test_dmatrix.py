import csv
import os
import tempfile
import warnings

import numpy as np
import pytest
import scipy.sparse
from hypothesis import given, settings, strategies
from scipy.sparse import csr_matrix, rand

import xgboost as xgb
from xgboost import testing as tm
from xgboost.core import DataSplitMode
from xgboost.testing.data import np_dtypes, run_base_margin_info

dpath = "demo/data/"
rng = np.random.RandomState(1994)


class TestDMatrix:
    def test_warn_missing(self):
        from xgboost import data

        with pytest.warns(UserWarning):
            data._warn_unused_missing("uri", 4)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            data._warn_unused_missing("uri", None)
            data._warn_unused_missing("uri", np.nan)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            x = rng.randn(10, 10)
            y = rng.randn(10)

            xgb.DMatrix(x, y, missing=4)

    def test_dmatrix_numpy_init(self):
        data = np.random.randn(5, 5)
        dm = xgb.DMatrix(data)
        assert dm.num_row() == 5
        assert dm.num_col() == 5

        data = np.array([[1, 2], [3, 4]])
        dm = xgb.DMatrix(data)
        assert dm.num_row() == 2
        assert dm.num_col() == 2

        # 0d array
        with pytest.raises(ValueError):
            xgb.DMatrix(np.array(1))
        # 1d array
        with pytest.raises(ValueError):
            xgb.DMatrix(np.array([1, 2, 3]))
        # 3d array
        data = np.random.randn(5, 5, 5)
        with pytest.raises(ValueError):
            xgb.DMatrix(data)
        # object dtype
        data = np.array([["a", "b"], ["c", "d"]])
        with pytest.raises(ValueError):
            xgb.DMatrix(data)

    def test_np_view(self):
        # Sliced Float32 array
        y = np.array([12, 34, 56], np.float32)[::2]
        from_view = xgb.DMatrix(np.array([[]]), label=y).get_label()
        from_array = xgb.DMatrix(np.array([[]]), label=y + 0).get_label()
        assert from_view.shape == from_array.shape
        assert (from_view == from_array).all()

        # Sliced UInt array
        z = np.array([12, 34, 56], np.uint32)[::2]
        dmat = xgb.DMatrix(np.array([[]]))
        dmat.set_uint_info("group", z)
        from_view = dmat.get_uint_info("group_ptr")
        dmat = xgb.DMatrix(np.array([[]]))
        dmat.set_uint_info("group", z + 0)
        from_array = dmat.get_uint_info("group_ptr")
        assert from_view.shape == from_array.shape
        assert (from_view == from_array).all()

    def test_slice(self):
        X = rng.randn(100, 100)
        y = rng.randint(low=0, high=3, size=100).astype(np.float32)
        d = xgb.DMatrix(X, y)
        np.testing.assert_equal(d.get_label(), y)

        fw = rng.uniform(size=100).astype(np.float32)
        d.set_info(feature_weights=fw)

        # base margin is per-class in multi-class classifier
        base_margin = rng.randn(100, 3).astype(np.float32)
        d.set_base_margin(base_margin)
        np.testing.assert_allclose(d.get_base_margin().reshape(100, 3), base_margin)

        ridxs = [1, 2, 3, 4, 5, 6]
        sliced = d.slice(ridxs)

        # Slicing works with label and other meta info fields
        np.testing.assert_equal(sliced.get_label(), y[1:7])
        np.testing.assert_equal(sliced.get_float_info("feature_weights"), fw)
        np.testing.assert_equal(sliced.get_base_margin(), base_margin[1:7, :].flatten())
        np.testing.assert_equal(
            sliced.get_base_margin(), sliced.get_float_info("base_margin")
        )

        # Slicing a DMatrix results into a DMatrix that's equivalent to a DMatrix that's
        # constructed from the corresponding NumPy slice
        d2 = xgb.DMatrix(X[1:7, :], y[1:7])
        d2.set_base_margin(base_margin[1:7, :])
        eval_res = {}
        _ = xgb.train(
            {"num_class": 3, "objective": "multi:softprob", "eval_metric": "mlogloss"},
            d,
            num_boost_round=2,
            evals=[(d2, "d2"), (sliced, "sliced")],
            evals_result=eval_res,
        )
        np.testing.assert_equal(
            eval_res["d2"]["mlogloss"], eval_res["sliced"]["mlogloss"]
        )

        ridxs_arr = np.array(ridxs)[1:]  # handles numpy slice correctly
        sliced = d.slice(ridxs_arr)
        np.testing.assert_equal(sliced.get_label(), y[2:7])

    def test_feature_names_slice(self):
        data = np.random.randn(5, 5)

        # different length
        with pytest.raises(ValueError):
            xgb.DMatrix(data, feature_names=list("abcdef"))
        # contains duplicates
        with pytest.raises(ValueError):
            xgb.DMatrix(data, feature_names=["a", "b", "c", "d", "d"])
        # contains symbol
        with pytest.raises(ValueError):
            xgb.DMatrix(data, feature_names=["a", "b", "c", "d", "e<1"])

        dm = xgb.DMatrix(data)
        dm.feature_names = list("abcde")
        assert dm.feature_names == list("abcde")

        assert dm.slice([0, 1]).num_col() == dm.num_col()
        assert dm.slice([0, 1]).feature_names == dm.feature_names

        with pytest.raises(ValueError, match=r"Duplicates found: \[.*'bar'.*\]"):
            dm.feature_names = ["bar"] * (data.shape[1] - 2) + ["a", "b"]

        dm.feature_types = list("qiqiq")
        assert dm.feature_types == list("qiqiq")

        with pytest.raises(ValueError):
            dm.feature_types = list("abcde")

        # reset
        dm.feature_names = None
        dm.feature_types = None
        assert dm.feature_names is None
        assert dm.feature_types is None

    def test_feature_names(self):
        data = np.random.randn(100, 5)
        target = np.array([0, 1] * 50)

        cases = [
            ["Feature1", "Feature2", "Feature3", "Feature4", "Feature5"],
            ["要因1", "要因2", "要因3", "要因4", "要因5"],
        ]

        for features in cases:
            dm = xgb.DMatrix(data, label=target, feature_names=features)
            assert dm.feature_names == features
            assert dm.num_row() == 100
            assert dm.num_col() == 5

            params = {
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "eta": 0.3,
                "num_class": 3,
            }

            bst = xgb.train(params, dm, num_boost_round=10)
            scores = bst.get_fscore()
            assert list(sorted(k for k in scores)) == features

            dummy = np.random.randn(5, 5)
            dm = xgb.DMatrix(dummy, feature_names=features)
            bst.predict(dm)

            # different feature name must raises error
            dm = xgb.DMatrix(dummy, feature_names=list("abcde"))
            with pytest.raises(ValueError):
                bst.predict(dm)

    @pytest.mark.skipif(**tm.no_pandas())
    def test_save_binary(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "m.dmatrix")
            data = pd.DataFrame({"a": [0, 1], "b": [2, 3], "c": [4, 5]})
            m0 = xgb.DMatrix(data.loc[:, ["a", "b"]], data["c"])
            assert m0.feature_names == ["a", "b"]
            m0.save_binary(path)
            m1 = xgb.DMatrix(path)
            assert m0.feature_names == m1.feature_names
            assert m0.feature_types == m1.feature_types

    def test_get_info(self):
        dtrain, _ = tm.load_agaricus(__file__)
        dtrain.get_float_info("label")
        dtrain.get_float_info("weight")
        dtrain.get_float_info("base_margin")
        dtrain.get_uint_info("group_ptr")

        group_len = np.array([2, 3, 4])
        dtrain.set_group(group_len)
        np.testing.assert_equal(group_len, dtrain.get_group())

    def test_qid(self):
        rows = 100
        cols = 10
        X, y = rng.randn(rows, cols), rng.randn(rows)
        qid = rng.randint(low=0, high=10, size=rows, dtype=np.uint32)
        qid = np.sort(qid)

        Xy = xgb.DMatrix(X, y)
        Xy.set_info(qid=qid)
        group_ptr = Xy.get_uint_info("group_ptr")
        assert group_ptr[0] == 0
        assert group_ptr[-1] == rows

    def test_feature_weights(self):
        kRows = 10
        kCols = 50
        rng = np.random.RandomState(1994)
        fw = rng.uniform(size=kCols)
        X = rng.randn(kRows, kCols)
        m = xgb.DMatrix(X)
        m.set_info(feature_weights=fw)
        np.testing.assert_allclose(fw, m.get_float_info("feature_weights"))
        # Handle empty
        m.set_info(feature_weights=np.empty((0,)))

        assert m.get_float_info("feature_weights").shape[0] == 0

        fw -= 1

        with pytest.raises(ValueError):
            m.set_info(feature_weights=fw)

    def test_sparse_dmatrix_csr(self):
        nrow = 100
        ncol = 1000
        x = rand(nrow, ncol, density=0.0005, format="csr", random_state=rng)
        assert x.indices.max() < ncol
        x.data[:] = 1
        dtrain = xgb.DMatrix(x, label=rng.binomial(1, 0.3, nrow))
        assert (dtrain.num_row(), dtrain.num_col()) == (nrow, ncol)
        watchlist = [(dtrain, "train")]
        param = {"max_depth": 3, "objective": "binary:logistic"}
        bst = xgb.train(param, dtrain, 5, evals=watchlist)
        bst.predict(dtrain)

        i32 = csr_matrix((x.data.astype(np.int32), x.indices, x.indptr), shape=x.shape)
        f32 = csr_matrix(
            (i32.data.astype(np.float32), x.indices, x.indptr), shape=x.shape
        )
        di32 = xgb.DMatrix(i32)
        df32 = xgb.DMatrix(f32)
        dense = xgb.DMatrix(f32.toarray(), missing=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "f32.dmatrix")
            df32.save_binary(path)
            with open(path, "rb") as fd:
                df32_buffer = np.array(fd.read())
            path = os.path.join(tmpdir, "f32.dmatrix")
            di32.save_binary(path)
            with open(path, "rb") as fd:
                di32_buffer = np.array(fd.read())

            path = os.path.join(tmpdir, "dense.dmatrix")
            dense.save_binary(path)
            with open(path, "rb") as fd:
                dense_buffer = np.array(fd.read())

            np.testing.assert_equal(df32_buffer, di32_buffer)
            np.testing.assert_equal(df32_buffer, dense_buffer)

    def test_sparse_dmatrix_csc(self):
        nrow = 1000
        ncol = 100
        x = rand(nrow, ncol, density=0.0005, format="csc", random_state=rng)
        assert x.indices.max() < nrow - 1
        x.data[:] = 1
        dtrain = xgb.DMatrix(x, label=rng.binomial(1, 0.3, nrow))
        assert (dtrain.num_row(), dtrain.num_col()) == (nrow, ncol)
        watchlist = [(dtrain, "train")]
        param = {"max_depth": 3, "objective": "binary:logistic"}
        bst = xgb.train(param, dtrain, 5, evals=watchlist)
        bst.predict(dtrain)

    def test_unknown_data(self):
        class Data:
            pass

        with pytest.raises(TypeError):
            with pytest.warns(UserWarning):
                d = Data()
                xgb.DMatrix(d)

        from scipy import sparse

        rng = np.random.RandomState(1994)
        X = rng.rand(10, 10)
        y = rng.rand(10)
        X = sparse.dok_matrix(X)
        with pytest.warns(UserWarning, match="dok_matrix"):
            Xy = xgb.DMatrix(X, y)
            assert Xy.num_row() == 10
            assert Xy.num_col() == 10

    @pytest.mark.skipif(**tm.no_pandas())
    def test_np_categorical(self):
        n_features = 10
        X, y = tm.make_categorical(10, n_features, n_categories=4, onehot=False)
        X = X.values.astype(np.float32)
        feature_types = ["c"] * n_features

        assert isinstance(X, np.ndarray)
        Xy = xgb.DMatrix(X, y, feature_types=feature_types)
        np.testing.assert_equal(np.array(Xy.feature_types), np.array(feature_types))

    def test_scipy_categorical(self):
        from scipy import sparse

        n_features = 10
        X, y = tm.make_categorical(10, n_features, n_categories=4, onehot=False)
        X = X.values.astype(np.float32)
        feature_types = ["c"] * n_features

        X[1, 3] = np.nan
        X[2, 4] = np.nan
        X = sparse.csr_matrix(X)

        Xy = xgb.DMatrix(X, y, feature_types=feature_types)
        np.testing.assert_equal(np.array(Xy.feature_types), np.array(feature_types))

        X = sparse.csc_matrix(X)

        Xy = xgb.DMatrix(X, y, feature_types=feature_types)
        np.testing.assert_equal(np.array(Xy.feature_types), np.array(feature_types))

        X = sparse.coo_matrix(X)

        Xy = xgb.DMatrix(X, y, feature_types=feature_types)
        np.testing.assert_equal(np.array(Xy.feature_types), np.array(feature_types))

    def test_uri_categorical(self):
        path = os.path.join(dpath, "agaricus.txt.train")
        feature_types = ["q"] * 5 + ["c"] + ["q"] * 120
        Xy = xgb.DMatrix(
            path + "?indexing_mode=1&format=libsvm", feature_types=feature_types
        )
        np.testing.assert_equal(np.array(Xy.feature_types), np.array(feature_types))

    def test_base_margin(self) -> None:
        run_base_margin_info(np.asarray, xgb.DMatrix, "cpu")

    @given(
        strategies.integers(0, 1000),
        strategies.integers(0, 100),
        strategies.fractions(0, 1),
    )
    @settings(deadline=None, print_blob=True)
    def test_to_csr(self, n_samples, n_features, sparsity) -> None:
        if n_samples == 0 or n_features == 0 or sparsity == 1.0:
            csr = scipy.sparse.csr_matrix(np.empty((0, 0)))
        else:
            csr = tm.make_sparse_regression(n_samples, n_features, sparsity, False)[
                0
            ].astype(np.float32)
        m = xgb.DMatrix(data=csr)
        ret = m.get_data()
        np.testing.assert_equal(csr.indptr, ret.indptr)
        np.testing.assert_equal(csr.data, ret.data)
        np.testing.assert_equal(csr.indices, ret.indices)

    def test_dtypes(self) -> None:
        n_samples = 128
        n_features = 16
        for orig, x in np_dtypes(n_samples, n_features):
            m0 = xgb.DMatrix(orig)
            m1 = xgb.DMatrix(x)
            assert tm.predictor_equal(m0, m1)


@pytest.mark.skipif(tm.is_windows(), reason="Rabit does not run on windows")
class TestDMatrixColumnSplit:
    def test_numpy(self):
        def verify_numpy():
            data = np.random.randn(5, 5)
            dm = xgb.DMatrix(data, data_split_mode=DataSplitMode.COL)
            assert dm.num_row() == 5
            assert dm.num_col() == 5 * xgb.collective.get_world_size()
            assert dm.feature_names is None
            assert dm.feature_types is None

        tm.run_with_rabit(world_size=3, test_fn=verify_numpy)

    def test_numpy_feature_names(self):
        def verify_numpy_feature_names():
            world_size = xgb.collective.get_world_size()
            data = np.random.randn(5, 5)
            feature_names = [f"feature{x}" for x in range(5)]
            feature_types = ["float"] * 5
            dm = xgb.DMatrix(
                data,
                feature_names=feature_names,
                feature_types=feature_types,
                data_split_mode=DataSplitMode.COL,
            )
            assert dm.num_row() == 5
            assert dm.num_col() == 5 * world_size
            assert len(dm.feature_names) == 5 * world_size
            assert dm.feature_names == tm.column_split_feature_names(
                feature_names, world_size
            )
            assert len(dm.feature_types) == 5 * world_size
            assert dm.feature_types == ["float"] * 5 * world_size

        tm.run_with_rabit(world_size=3, test_fn=verify_numpy_feature_names)

    def test_csr(self):
        def verify_csr():
            indptr = np.array([0, 2, 3, 6])
            indices = np.array([0, 2, 2, 0, 1, 2])
            data = np.array([1, 2, 3, 4, 5, 6])
            X = scipy.sparse.csr_matrix((data, indices, indptr), shape=(3, 3))
            dtrain = xgb.DMatrix(X, data_split_mode=DataSplitMode.COL)
            assert dtrain.num_row() == 3
            assert dtrain.num_col() == 3 * xgb.collective.get_world_size()

        tm.run_with_rabit(world_size=3, test_fn=verify_csr)

    def test_csc(self):
        def verify_csc():
            row = np.array([0, 2, 2, 0, 1, 2])
            col = np.array([0, 0, 1, 2, 2, 2])
            data = np.array([1, 2, 3, 4, 5, 6])
            X = scipy.sparse.csc_matrix((data, (row, col)), shape=(3, 3))
            dtrain = xgb.DMatrix(X, data_split_mode=DataSplitMode.COL)
            assert dtrain.num_row() == 3
            assert dtrain.num_col() == 3 * xgb.collective.get_world_size()

        tm.run_with_rabit(world_size=3, test_fn=verify_csc)

    def test_coo(self):
        def verify_coo():
            row = np.array([0, 2, 2, 0, 1, 2])
            col = np.array([0, 0, 1, 2, 2, 2])
            data = np.array([1, 2, 3, 4, 5, 6])
            X = scipy.sparse.coo_matrix((data, (row, col)), shape=(3, 3))
            dtrain = xgb.DMatrix(X, data_split_mode=DataSplitMode.COL)
            assert dtrain.num_row() == 3
            assert dtrain.num_col() == 3 * xgb.collective.get_world_size()

        tm.run_with_rabit(world_size=3, test_fn=verify_coo)

    def test_uri(self):
        def verify_uri():
            rank = xgb.collective.get_rank()
            with tempfile.TemporaryDirectory() as tmpdir:
                filename = os.path.join(tmpdir, f"test_data_{rank}.csv")

                data = np.random.rand(5, 5)
                with open(filename, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    for row in data:
                        writer.writerow(row)
                dtrain = xgb.DMatrix(
                    f"{filename}?format=csv", data_split_mode=DataSplitMode.COL
                )
                assert dtrain.num_row() == 5
                assert dtrain.num_col() == 5 * xgb.collective.get_world_size()

        tm.run_with_rabit(world_size=3, test_fn=verify_uri)

    def test_list(self):
        def verify_list():
            data = [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25],
            ]
            dm = xgb.DMatrix(data, data_split_mode=DataSplitMode.COL)
            assert dm.num_row() == 5
            assert dm.num_col() == 5 * xgb.collective.get_world_size()

        tm.run_with_rabit(world_size=3, test_fn=verify_list)

    def test_tuple(self):
        def verify_tuple():
            data = (
                (1, 2, 3, 4, 5),
                (6, 7, 8, 9, 10),
                (11, 12, 13, 14, 15),
                (16, 17, 18, 19, 20),
                (21, 22, 23, 24, 25),
            )
            dm = xgb.DMatrix(data, data_split_mode=DataSplitMode.COL)
            assert dm.num_row() == 5
            assert dm.num_col() == 5 * xgb.collective.get_world_size()

        tm.run_with_rabit(world_size=3, test_fn=verify_tuple)
