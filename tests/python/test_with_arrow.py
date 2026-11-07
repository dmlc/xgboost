import mmap
import os

import numpy as np
import pytest

import xgboost as xgb
from xgboost import testing as tm
from xgboost.compat import is_dataframe

pytestmark = pytest.mark.skipif(
    tm.no_arrow()["condition"] or tm.no_pandas()["condition"],
    reason=tm.no_arrow()["reason"] + " or " + tm.no_pandas()["reason"],
)

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pc


def test_type_check() -> None:
    df = pd.DataFrame(
        [[0, 1, 2.0, 3.0], [1, 2, 3.0, 4.0]], columns=["a", "b", "c", "d"]
    )
    table = pa.Table.from_pandas(df)
    assert is_dataframe(table)


class TestArrowTable:
    @pytest.mark.parametrize("DMatrixT", [xgb.DMatrix, xgb.QuantileDMatrix])
    @pytest.mark.parametrize("string_type", [pa.string(), pa.large_string()])
    def test_dictionary_all_valid_bitmap(self, DMatrixT, string_type):
        dictionary = pa.array(["α", "東京"], type=string_type)
        dictionary = pa.Array.from_buffers(
            string_type, 2, [pa.py_buffer(b"\x03"), *dictionary.buffers()[1:]]
        )
        assert dictionary.null_count == 0
        indices = pa.array([0, None, 1, 0, 1])
        table = pa.table({"cat": pa.DictionaryArray.from_arrays(indices, dictionary)})
        table.validate(full=True)
        reference_table = pa.table(
            {"cat": pa.DictionaryArray.from_arrays(indices, pa.array(["α", "東京"]))}
        )
        dmat = DMatrixT(table, enable_categorical=True, nthread=1)
        reference = DMatrixT(reference_table, enable_categorical=True, nthread=1)
        assert dmat.feature_types == ["c"]
        categories = dmat.get_categories(export_to_arrow=True).to_arrow()
        assert categories[0][1].to_pylist() == ["α", "東京"]
        np.testing.assert_array_equal(
            dmat.get_data().indptr, reference.get_data().indptr
        )
        np.testing.assert_array_equal(dmat.get_data().data, reference.get_data().data)

    @pytest.mark.parametrize("string_type", [pa.string(), pa.large_string()])
    def test_empty_dictionary_buffers(self, string_type):
        from xgboost._data_utils import _arrow_cat_names_inf

        names, buffers = _arrow_cat_names_inf(pa.array([], type=string_type))
        assert names["offsets"]["shape"] == (1,)
        assert names["offsets"]["typestr"] == "<i4"
        np.testing.assert_array_equal(np.frombuffer(buffers[1], dtype=np.int32), [0])

    @pytest.mark.parametrize("DMatrixT", [xgb.DMatrix, xgb.QuantileDMatrix])
    @pytest.mark.parametrize("string_type", [pa.string(), pa.large_string()])
    def test_dictionary_missing_category(self, DMatrixT, string_type):
        dictionary = pa.array(["a", None], type=string_type)
        array = pa.DictionaryArray.from_arrays(pa.array([0, 1, None]), dictionary)
        with pytest.raises(
            ValueError, match="Categories should not contain missing values"
        ):
            DMatrixT(pa.table({"cat": array}), enable_categorical=True, nthread=1)

    @pytest.mark.parametrize("array_offset", [0, 1])
    def test_large_string_dictionary_oversized_view(self, array_offset, monkeypatch):
        from xgboost._data_utils import _arrow_cat_names_inf

        if os.name != "posix" or np.dtype(np.intp).itemsize < 8:
            pytest.skip("Requires a 64-bit POSIX anonymous virtual mapping")
        start = mmap.PAGESIZE
        end = start + (1 << 31) + mmap.PAGESIZE
        try:
            backing = mmap.mmap(-1, end)
        except (OSError, OverflowError, MemoryError):
            pytest.skip("Insufficient virtual address space for the Arrow buffer")
        # Never touch or decode this genuinely oversized logical value.
        offsets = [0, start, end] if array_offset else [start, end]
        dictionary = pa.Array.from_buffers(
            pa.large_string(),
            len(offsets) - 1,
            [
                None,
                pa.py_buffer(np.array(offsets, dtype=np.int64)),
                pa.py_buffer(backing),
            ],
        )
        if array_offset:
            dictionary = dictionary.slice(1)
        del backing

        def no_materialization(_):
            pytest.fail(
                "An unsupported logical span must reject before copying", pytrace=False
            )

        # Make the regression safe even if normalization is moved before the guard.
        monkeypatch.setattr(pa, "concat_arrays", no_materialization)
        with pytest.raises(pa.ArrowInvalid, match="input array too large"):
            _arrow_cat_names_inf(dictionary)

    @pytest.mark.parametrize("DMatrixT", [xgb.DMatrix, xgb.QuantileDMatrix])
    @pytest.mark.parametrize("array_offset", [0, 1])
    def test_large_string_dictionary_backing_offset(self, DMatrixT, array_offset):
        if os.name != "posix" or np.dtype(np.intp).itemsize < 8:
            pytest.skip("Requires a 64-bit POSIX anonymous virtual mapping")
        start = (1 << 31) + mmap.PAGESIZE
        try:
            backing = mmap.mmap(-1, start + 2)
        except (OSError, OverflowError, MemoryError):
            pytest.skip("Insufficient virtual address space for the Arrow buffer")
        # Reserve virtual space, but touch only the two logical string bytes.
        backing[start : start + 2] = b"ab"
        offsets = (
            [0, start, start + 1, start + 2]
            if array_offset
            else [start, start + 1, start + 2]
        )
        dictionary = pa.Array.from_buffers(
            pa.large_string(),
            len(offsets) - 1,
            [
                None,
                pa.py_buffer(np.array(offsets, dtype=np.int64)),
                pa.py_buffer(backing),
            ],
        )
        if array_offset:
            dictionary = dictionary.slice(1)
        # The Arrow buffer keeps the mapping alive until all views are released.
        del backing
        assert dictionary.offset == array_offset
        assert dictionary.to_pylist() == ["a", "b"]
        codes = pa.array([0, 1, None, 0, 1])
        table = pa.table({"cat": pa.DictionaryArray.from_arrays(codes, dictionary)})
        reference_table = pa.table(
            {"cat": pa.DictionaryArray.from_arrays(codes, pa.array(["a", "b"]))}
        )
        dmat = DMatrixT(table, enable_categorical=True, nthread=1)
        reference = DMatrixT(reference_table, enable_categorical=True, nthread=1)
        categories = dmat.get_categories(export_to_arrow=True).to_arrow()
        assert categories[0][1].to_pylist() == ["a", "b"]
        for actual, expected in zip(
            (dmat.get_data().indptr, dmat.get_data().indices, dmat.get_data().data),
            (
                reference.get_data().indptr,
                reference.get_data().indices,
                reference.get_data().data,
            ),
        ):
            np.testing.assert_array_equal(actual, expected)

    @pytest.mark.parametrize("DMatrixT", [xgb.DMatrix, xgb.QuantileDMatrix])
    @pytest.mark.parametrize("string_type", [pa.string(), pa.large_string()])
    @pytest.mark.parametrize("start,length", [(0, 4), (0, 2), (1, 2), (1, 3)])
    def test_sliced_dictionary(self, DMatrixT, string_type, start, length):
        dictionary = pa.array(["discard", "α", "東京", "last"], type=string_type)
        dictionary = dictionary.slice(start, length)
        codes = np.tile(np.arange(length), 8)
        indices = pa.array(codes, mask=np.arange(codes.size) % 7 == 0)
        array = pa.DictionaryArray.from_arrays(indices, dictionary)
        table = pa.table({"cat": array})
        labels = np.arange(codes.size, dtype=np.float32) % 5

        dmat = DMatrixT(table, labels, enable_categorical=True, nthread=1)
        categories = dmat.get_categories(export_to_arrow=True).to_arrow()
        assert categories[0][1].to_pylist() == dictionary.to_pylist()

        # Pandas materializes the logical strings independently of Arrow's
        # dictionary offsets and supplies its own categorical conversion path.
        frame = pd.DataFrame(
            {
                "cat": pd.Categorical(
                    array.to_pylist(), categories=dictionary.to_pylist()
                )
            }
        )
        reference = DMatrixT(frame, labels, enable_categorical=True, nthread=1)
        np.testing.assert_array_equal(
            dmat.get_data().indptr, reference.get_data().indptr
        )
        np.testing.assert_array_equal(dmat.get_data().data, reference.get_data().data)

        params = {"tree_method": "hist", "max_depth": 2, "nthread": 1}
        model = xgb.train(params, dmat, num_boost_round=3)
        expected = xgb.train(params, reference, num_boost_round=3)
        np.testing.assert_allclose(model.predict(dmat), expected.predict(reference))
        np.testing.assert_allclose(
            model.inplace_predict(table), model.inplace_predict(frame)
        )

    def test_arrow_table(self):
        df = pd.DataFrame(
            [[0, 1, 2.0, 3.0], [1, 2, 3.0, 4.0]], columns=["a", "b", "c", "d"]
        )
        table = pa.Table.from_pandas(df)
        dm = xgb.DMatrix(table)
        assert dm.num_row() == 2
        assert dm.num_col() == 4

    @pytest.mark.parametrize("DMatrixT", [xgb.DMatrix, xgb.QuantileDMatrix])
    def test_arrow_table_with_custom_feature_names(self, DMatrixT):
        table = pa.table({"a": [1, 2], "b": [1.0, 2.0]})
        dm = DMatrixT(table)
        assert dm.feature_names == ["a", "b"]

        dm = DMatrixT(table, feature_names=["x", "y"])
        assert dm.feature_names == ["x", "y"]
        assert dm.feature_types == ["int", "float"]

    def test_arrow_table_with_label(self):
        df = pd.DataFrame([[1, 2.0, 3.0], [2, 3.0, 4.0]], columns=["a", "b", "c"])
        table = pa.Table.from_pandas(df)
        label = np.array([0, 1])
        dm = xgb.DMatrix(table)
        dm.set_label(label)
        assert dm.num_row() == 2
        assert dm.num_col() == 3
        np.testing.assert_array_equal(dm.get_label(), np.array([0, 1]))

    def test_arrow_table_from_np(self):
        coldata = np.array(
            [[1.0, 1.0, 0.0, 0.0], [2.0, 0.0, 1.0, 0.0], [3.0, 0.0, 0.0, 1.0]]
        )
        cols = list(map(pa.array, coldata))
        table = pa.Table.from_arrays(cols, ["a", "b", "c"])
        dm = xgb.DMatrix(table)
        assert dm.num_row() == 4
        assert dm.num_col() == 3

    @pytest.mark.parametrize("DMatrixT", [xgb.DMatrix, xgb.QuantileDMatrix])
    def test_arrow_train(self, DMatrixT):
        import pandas as pd

        rows = 100
        X = pd.DataFrame(
            {
                "A": np.random.randint(0, 10, size=rows),
                "B": np.random.randn(rows),
                "C": np.random.permutation([1, 0] * (rows // 2)),
            }
        )
        y = pd.Series(np.random.randn(rows))

        table = pa.Table.from_pandas(X)
        dtrain1 = DMatrixT(table)
        dtrain1.set_label(pa.Table.from_pandas(pd.DataFrame(y)))
        bst1 = xgb.train({}, dtrain1, num_boost_round=10)
        preds1 = bst1.predict(DMatrixT(X))

        dtrain2 = DMatrixT(X, y)
        bst2 = xgb.train({}, dtrain2, num_boost_round=10)
        preds2 = bst2.predict(DMatrixT(X))

        np.testing.assert_allclose(preds1, preds2)

        preds3 = bst2.inplace_predict(table)
        np.testing.assert_allclose(preds1, preds3)
        assert bst2.feature_names == ["A", "B", "C"]
        assert bst2.feature_types == ["int", "float", "int"]

        for invalid in [
            table.rename_columns(["x", "y", "z"]),
            table.select(["C", "B", "A"]),
        ]:
            with pytest.raises(ValueError, match="feature_names mismatch"):
                bst2.inplace_predict(invalid)
            np.testing.assert_allclose(
                bst2.inplace_predict(invalid, validate_features=False),
                bst2.inplace_predict(invalid.to_pandas().to_numpy()),
            )

    @pytest.mark.parametrize("DMatrixT", [xgb.DMatrix, xgb.QuantileDMatrix])
    def test_arrow_unsupported_dtype(self, DMatrixT):
        table = pa.table({"a": [1.0, 2.0], "b": ["x", "y"], "c": [1, 2]})
        with pytest.raises(ValueError, match="Invalid columns:b: string$"):
            DMatrixT(table)

        table = pa.table({"a": pa.array([1, 2], pa.date32())})
        with pytest.raises(ValueError, match="Invalid columns:a: date32"):
            DMatrixT(table)

        table = pa.table({"a": pa.array([1, 2]).dictionary_encode()})
        with pytest.raises(
            ValueError, match="Invalid columns:a: dictionary<values=int64"
        ):
            DMatrixT(table, enable_categorical=True)

    def test_arrow_survival(self):
        data = os.path.join(tm.data_dir(__file__), "veterans_lung_cancer.csv")
        table = pc.read_csv(data)
        y_lower_bound = table["Survival_label_lower_bound"]
        y_upper_bound = table["Survival_label_upper_bound"]
        X = table.drop(["Survival_label_lower_bound", "Survival_label_upper_bound"])

        dtrain = xgb.DMatrix(
            X, label_lower_bound=y_lower_bound, label_upper_bound=y_upper_bound
        )
        y_np_up = dtrain.get_float_info("label_upper_bound")
        y_np_low = dtrain.get_float_info("label_lower_bound")
        np.testing.assert_equal(y_np_up, y_upper_bound.to_pandas().values)
        np.testing.assert_equal(y_np_low, y_lower_bound.to_pandas().values)
