import tempfile
import shutil
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from xgboost.spark.data import (
    _row_tuple_list_to_feature_matrix_y_w,
    convert_partition_data_to_dmatrix,
    _dump_libsvm,
)

from xgboost import DMatrix, XGBClassifier
from xgboost.training import train as worker_train
from .utils_test import SparkTestCase
import logging

logging.getLogger("py4j").setLevel(logging.INFO)


class DataTest(SparkTestCase):
    def test_sparse_dense_vector(self):
        def row_tup_iter(data):
            pdf = pd.DataFrame(data)
            yield pdf

        # row1 = Vectors.dense(1.0, 2.0, 3.0),),
        # row2 = Vectors.sparse(3, {1: 1.0, 2: 5.5})
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
        # self.assertTrue(isinstance(feature_matrix, csr_matrix))
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
        # self.assertTrue(isinstance(feature_matrix, csr_matrix))
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
        # self.assertTrue(isinstance(feature_matrix, csr_matrix))
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
        output_dmatrix = convert_partition_data_to_dmatrix(
            [pd.DataFrame(data)], has_weight=False, has_validation=False
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
        output_dmatrix = convert_partition_data_to_dmatrix(
            [pd.DataFrame(data)], has_weight=True, has_validation=False
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
        storage_dmatrix = convert_partition_data_to_dmatrix(
            [pd.DataFrame(data)], has_weight=False, has_validation=False
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
        storage_dmatrix = convert_partition_data_to_dmatrix(
            [pd.DataFrame(data)], has_weight=True, has_validation=False
        )

        normal_booster = worker_train({}, normal_dmatrix)
        storage_booster = worker_train({}, storage_dmatrix)
        normal_preds = normal_booster.predict(test_dmatrix)
        storage_preds = storage_booster.predict(test_dmatrix)
        self.assertTrue(np.allclose(normal_preds, storage_preds, atol=1e-3))
        shutil.rmtree(temporary_path)

    def test_dump_libsvm(self):
        num_features = 3
        features_test_list = [
            [[1, 2, 3], [0, 1, 5.5]],
            csr_matrix(([1, 2, 3], [0, 2, 2], [0, 2, 3]), shape=(2, 3)),
        ]
        labels = [0, 1]

        for features in features_test_list:
            if isinstance(features, csr_matrix):
                features_array = features.toarray()
            else:
                features_array = features
            # testing without weights
            # The format should be label index:feature_value index:feature_value...
            # Note: from initial testing, it seems all of the indices must be listed regardless of whether
            # they exist or not
            output = _dump_libsvm(features, labels)
            for i, line in enumerate(output):
                split_line = line.split(" ")
                self.assertEqual(float(split_line[0]), labels[i])
                split_line = [elem.split(":") for elem in split_line[1:]]
                loaded_feature = [0.0] * num_features
                for split in split_line:
                    loaded_feature[int(split[0])] = float(split[1])
                self.assertListEqual(loaded_feature, list(features_array[i]))

            weights = [0.2, 0.8]
            # testing with weights
            # The format should be label:weight index:feature_value index:feature_value...
            output = _dump_libsvm(features, labels, weights)
            for i, line in enumerate(output):
                split_line = line.split(" ")
                split_line = [elem.split(":") for elem in split_line]
                self.assertEqual(float(split_line[0][0]), labels[i])
                self.assertEqual(float(split_line[0][1]), weights[i])

                split_line = split_line[1:]
                loaded_feature = [0.0] * num_features
                for split in split_line:
                    loaded_feature[int(split[0])] = float(split[1])
                self.assertListEqual(loaded_feature, list(features_array[i]))

        features = [
            [1.34234, 2.342321, 3.34322],
            [0.344234, 1.123123, 5.534322],
            [3.553423e10, 3.5632e10, 0.00000000000012345],
        ]
        features_prec = [
            [1.34, 2.34, 3.34],
            [0.344, 1.12, 5.53],
            [3.55e10, 3.56e10, 1.23e-13],
        ]
        labels = [0, 1]
        output = _dump_libsvm(features, labels, external_storage_precision=3)
        for i, line in enumerate(output):
            split_line = line.split(" ")
            self.assertEqual(float(split_line[0]), labels[i])
            split_line = [elem.split(":") for elem in split_line[1:]]
            self.assertListEqual([float(v[1]) for v in split_line], features_prec[i])
