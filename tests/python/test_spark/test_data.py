import sys
import tempfile
import shutil

import pytest
import numpy as np
import pandas as pd

import testing as tm

if tm.no_spark()["condition"]:
    pytest.skip(msg=tm.no_spark()["reason"], allow_module_level=True)
if sys.platform.startswith("win") or sys.platform.startswith("darwin"):
    pytest.skip("Skipping PySpark tests on Windows", allow_module_level=True)

from xgboost.spark.data import (
    _row_tuple_list_to_feature_matrix_y_w,
    _convert_partition_data_to_dmatrix,
)

from xgboost import DMatrix, XGBClassifier
from xgboost.training import train as worker_train
from .utils import SparkTestCase
import logging

logging.getLogger("py4j").setLevel(logging.INFO)


class DataTest(SparkTestCase):
    def test_sparse_dense_vector(self):
        def row_tup_iter(data):
            pdf = pd.DataFrame(data)
            yield pdf

        expected_ndarray = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 5.5]])
        data = {"values": [[1.0, 2.0, 3.0], [0.0, 1.0, 5.5]]}
        feature_matrix, y, w, _ = _row_tuple_list_to_feature_matrix_y_w(
            list(row_tup_iter(data)),
            train=False,
            has_weight=False,
            has_fit_base_margin=False,
            has_predict_base_margin=False,
        )
        self.assertIsNone(y)
        self.assertIsNone(w)
        self.assertTrue(np.allclose(feature_matrix, expected_ndarray))

        data["label"] = [1, 0]
        feature_matrix, y, w, _ = _row_tuple_list_to_feature_matrix_y_w(
            row_tup_iter(data),
            train=True,
            has_weight=False,
            has_fit_base_margin=False,
            has_predict_base_margin=False,
        )
        self.assertIsNone(w)
        self.assertTrue(np.allclose(feature_matrix, expected_ndarray))
        self.assertTrue(np.array_equal(y, np.array(data["label"])))

        data["weight"] = [0.2, 0.8]
        feature_matrix, y, w, _ = _row_tuple_list_to_feature_matrix_y_w(
            list(row_tup_iter(data)),
            train=True,
            has_weight=True,
            has_fit_base_margin=False,
            has_predict_base_margin=False,
        )
        self.assertTrue(np.allclose(feature_matrix, expected_ndarray))
        self.assertTrue(np.array_equal(y, np.array(data["label"])))
        self.assertTrue(np.array_equal(w, np.array(data["weight"])))

    def test_dmatrix_creator(self):

        # This function acts as a pseudo-itertools.chain()
        def row_tup_iter(data):
            pdf = pd.DataFrame(data)
            yield pdf

        # Standard testing DMatrix creation
        expected_features = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 5.5]] * 100)
        expected_labels = np.array([1, 0] * 100)
        expected_dmatrix = DMatrix(data=expected_features, label=expected_labels)

        data = {
            "values": [[1.0, 2.0, 3.0], [0.0, 1.0, 5.5]] * 100,
            "label": [1, 0] * 100,
        }
        output_dmatrix = _convert_partition_data_to_dmatrix(
            [pd.DataFrame(data)],
            has_weight=False,
            has_validation=False,
            has_base_margin=False,
        )
        # You can't compare DMatrix outputs, so the only way is to predict on the two seperate DMatrices using
        # the same classifier and making sure the outputs are equal
        model = XGBClassifier()
        model.fit(expected_features, expected_labels)
        expected_preds = model.get_booster().predict(expected_dmatrix)
        output_preds = model.get_booster().predict(output_dmatrix)
        self.assertTrue(np.allclose(expected_preds, output_preds, atol=1e-3))

        # DMatrix creation with weights
        expected_weight = np.array([0.2, 0.8] * 100)
        expected_dmatrix = DMatrix(
            data=expected_features, label=expected_labels, weight=expected_weight
        )

        data["weight"] = [0.2, 0.8] * 100
        output_dmatrix = _convert_partition_data_to_dmatrix(
            [pd.DataFrame(data)],
            has_weight=True,
            has_validation=False,
            has_base_margin=False,
        )

        model.fit(expected_features, expected_labels, sample_weight=expected_weight)
        expected_preds = model.get_booster().predict(expected_dmatrix)
        output_preds = model.get_booster().predict(output_dmatrix)
        self.assertTrue(np.allclose(expected_preds, output_preds, atol=1e-3))

    def test_external_storage(self):
        # Instantiating base data (features, labels)
        features = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 5.5]] * 100)
        labels = np.array([1, 0] * 100)
        normal_dmatrix = DMatrix(features, labels)
        test_dmatrix = DMatrix(features)

        data = {
            "values": [[1.0, 2.0, 3.0], [0.0, 1.0, 5.5]] * 100,
            "label": [1, 0] * 100,
        }

        # Creating the dmatrix based on storage
        temporary_path = tempfile.mkdtemp()
        storage_dmatrix = _convert_partition_data_to_dmatrix(
            [pd.DataFrame(data)],
            has_weight=False,
            has_validation=False,
            has_base_margin=False,
        )

        # Testing without weights
        normal_booster = worker_train({}, normal_dmatrix)
        storage_booster = worker_train({}, storage_dmatrix)
        normal_preds = normal_booster.predict(test_dmatrix)
        storage_preds = storage_booster.predict(test_dmatrix)
        self.assertTrue(np.allclose(normal_preds, storage_preds, atol=1e-3))
        shutil.rmtree(temporary_path)

        # Testing weights
        weights = np.array([0.2, 0.8] * 100)
        normal_dmatrix = DMatrix(data=features, label=labels, weight=weights)
        data["weight"] = [0.2, 0.8] * 100

        temporary_path = tempfile.mkdtemp()
        storage_dmatrix = _convert_partition_data_to_dmatrix(
            [pd.DataFrame(data)],
            has_weight=True,
            has_validation=False,
            has_base_margin=False,
        )

        normal_booster = worker_train({}, normal_dmatrix)
        storage_booster = worker_train({}, storage_dmatrix)
        normal_preds = normal_booster.predict(test_dmatrix)
        storage_preds = storage_booster.predict(test_dmatrix)
        self.assertTrue(np.allclose(normal_preds, storage_preds, atol=1e-3))
        shutil.rmtree(temporary_path)
