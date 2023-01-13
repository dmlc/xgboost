import glob
import logging
import random
import sys
import uuid

import numpy as np
import pytest
import testing as tm

import xgboost as xgb
from xgboost import testing

if tm.no_spark()["condition"]:
    pytest.skip(msg=tm.no_spark()["reason"], allow_module_level=True)
if sys.platform.startswith("win") or sys.platform.startswith("darwin"):
    pytest.skip("Skipping PySpark tests on Windows", allow_module_level=True)

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import Vectors
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import functions as spark_sql_func
from xgboost.spark import (
    SparkXGBClassifier,
    SparkXGBClassifierModel,
    SparkXGBRanker,
    SparkXGBRegressor,
    SparkXGBRegressorModel,
)
from xgboost.spark.core import _non_booster_params

from xgboost import XGBClassifier, XGBModel, XGBRegressor

from .utils import SparkTestCase

logging.getLogger("py4j").setLevel(logging.INFO)

pytestmark = testing.timeout(60)


def no_sparse_unwrap():
    try:
        from pyspark.sql.functions import unwrap_udt

    except ImportError:
        return {"reason": "PySpark<3.4", "condition": True}

    return {"reason": "PySpark<3.4", "condition": False}


class XgboostLocalTest(SparkTestCase):
    def setUp(self):
        logging.getLogger().setLevel("INFO")
        random.seed(2020)

        # The following code use xgboost python library to train xgb model and predict.
        #
        # >>> import numpy as np
        # >>> import xgboost
        # >>> X = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 5.5]])
        # >>> y = np.array([0, 1])
        # >>> reg1 = xgboost.XGBRegressor()
        # >>> reg1.fit(X, y)
        # >>> reg1.predict(X)
        # array([8.8375784e-04, 9.9911624e-01], dtype=float32)
        # >>> def custom_lr(boosting_round):
        # ...     return 1.0 / (boosting_round + 1)
        # ...
        # >>> reg1.fit(X, y, callbacks=[xgboost.callback.LearningRateScheduler(custom_lr)])
        # >>> reg1.predict(X)
        # array([0.02406844, 0.9759315 ], dtype=float32)
        # >>> reg2 = xgboost.XGBRegressor(max_depth=5, n_estimators=10)
        # >>> reg2.fit(X, y)
        # >>> reg2.predict(X, ntree_limit=5)
        # array([0.22185266, 0.77814734], dtype=float32)
        self.reg_params = {
            "max_depth": 5,
            "n_estimators": 10,
            "ntree_limit": 5,
            "max_bin": 9,
        }
        self.reg_df_train = self.session.createDataFrame(
            [
                (Vectors.dense(1.0, 2.0, 3.0), 0),
                (Vectors.sparse(3, {1: 1.0, 2: 5.5}), 1),
            ],
            ["features", "label"],
        )
        self.reg_df_test = self.session.createDataFrame(
            [
                (Vectors.dense(1.0, 2.0, 3.0), 0.0, 0.2219, 0.02406),
                (Vectors.sparse(3, {1: 1.0, 2: 5.5}), 1.0, 0.7781, 0.9759),
            ],
            [
                "features",
                "expected_prediction",
                "expected_prediction_with_params",
                "expected_prediction_with_callbacks",
            ],
        )

        # >>> X = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 5.5]])
        # >>> y = np.array([0, 1])
        # >>> cl1 = xgboost.XGBClassifier()
        # >>> cl1.fit(X, y)
        # >>> cl1.predict(X)
        # array([0, 0])
        # >>> cl1.predict_proba(X)
        # array([[0.5, 0.5],
        #        [0.5, 0.5]], dtype=float32)
        # >>> cl2 = xgboost.XGBClassifier(max_depth=5, n_estimators=10, scale_pos_weight=4)
        # >>> cl2.fit(X, y)
        # >>> cl2.predict(X)
        # array([1, 1])
        # >>> cl2.predict_proba(X)
        # array([[0.27574146, 0.72425854 ],
        #        [0.27574146, 0.72425854 ]], dtype=float32)
        self.cls_params = {"max_depth": 5, "n_estimators": 10, "scale_pos_weight": 4}

        cls_df_train_data = [
            (Vectors.dense(1.0, 2.0, 3.0), 0),
            (Vectors.sparse(3, {1: 1.0, 2: 5.5}), 1),
        ]
        self.cls_df_train = self.session.createDataFrame(
            cls_df_train_data, ["features", "label"]
        )
        self.cls_df_train_large = self.session.createDataFrame(
            cls_df_train_data * 100, ["features", "label"]
        )
        self.cls_df_test = self.session.createDataFrame(
            [
                (
                    Vectors.dense(1.0, 2.0, 3.0),
                    0,
                    [0.5, 0.5],
                    1,
                    [0.27574146, 0.72425854],
                ),
                (
                    Vectors.sparse(3, {1: 1.0, 2: 5.5}),
                    0,
                    [0.5, 0.5],
                    1,
                    [0.27574146, 0.72425854],
                ),
            ],
            [
                "features",
                "expected_prediction",
                "expected_probability",
                "expected_prediction_with_params",
                "expected_probability_with_params",
            ],
        )

        # kwargs test (using the above data, train, we get the same results)
        self.cls_params_kwargs = {"tree_method": "approx", "sketch_eps": 0.03}

        # >>> X = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 4.0], [0.0, 1.0, 5.5], [-1.0, -2.0, 1.0]])
        # >>> y = np.array([0, 0, 1, 2])
        # >>> cl = xgboost.XGBClassifier()
        # >>> cl.fit(X, y)
        # >>> cl.predict_proba(np.array([[1.0, 2.0, 3.0]]))
        # array([[0.5374299 , 0.23128504, 0.23128504]], dtype=float32)
        multi_cls_df_train_data = [
            (Vectors.dense(1.0, 2.0, 3.0), 0),
            (Vectors.dense(1.0, 2.0, 4.0), 0),
            (Vectors.sparse(3, {1: 1.0, 2: 5.5}), 1),
            (Vectors.dense(-1.0, -2.0, 1.0), 2),
        ]
        self.multi_cls_df_train = self.session.createDataFrame(
            multi_cls_df_train_data, ["features", "label"]
        )
        self.multi_cls_df_train_large = self.session.createDataFrame(
            multi_cls_df_train_data * 100, ["features", "label"]
        )
        self.multi_cls_df_test = self.session.createDataFrame(
            [
                (Vectors.dense(1.0, 2.0, 3.0), [0.5374, 0.2312, 0.2312]),
            ],
            ["features", "expected_probability"],
        )

        # Test regressor with weight and eval set
        # >>> import numpy as np
        # >>> import xgboost
        # >>> X = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 5.5], [4.0, 5.0, 6.0], [0.0, 6.0, 7.5]])
        # >>> w = np.array([1.0, 2.0, 1.0, 2.0])
        # >>> y = np.array([0, 1, 2, 3])
        # >>> reg1 = xgboost.XGBRegressor()
        # >>> reg1.fit(X, y, sample_weight=w)
        # >>> reg1.predict(X)
        # >>> array([1.0679445e-03, 1.0000550e+00, ...
        # >>> X_train = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 5.5]])
        # >>> X_val = np.array([[4.0, 5.0, 6.0], [0.0, 6.0, 7.5]])
        # >>> y_train = np.array([0, 1])
        # >>> y_val = np.array([2, 3])
        # >>> w_train = np.array([1.0, 2.0])
        # >>> w_val = np.array([1.0, 2.0])
        # >>> reg2 = xgboost.XGBRegressor()
        # >>> reg2.fit(X_train, y_train, eval_set=[(X_val, y_val)],
        # >>>          early_stopping_rounds=1, eval_metric='rmse')
        # >>> reg2.predict(X)
        # >>> array([8.8370638e-04, 9.9911624e-01, ...
        # >>> reg2.best_score
        # 2.0000002682208837
        # >>> reg3 = xgboost.XGBRegressor()
        # >>> reg3.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_val, y_val)],
        # >>>          sample_weight_eval_set=[w_val],
        # >>>          early_stopping_rounds=1, eval_metric='rmse')
        # >>> reg3.predict(X)
        # >>> array([0.03155671, 0.98874104,...
        # >>> reg3.best_score
        # 1.9970891552124017
        self.reg_df_train_with_eval_weight = self.session.createDataFrame(
            [
                (Vectors.dense(1.0, 2.0, 3.0), 0, False, 1.0),
                (Vectors.sparse(3, {1: 1.0, 2: 5.5}), 1, False, 2.0),
                (Vectors.dense(4.0, 5.0, 6.0), 2, True, 1.0),
                (Vectors.sparse(3, {1: 6.0, 2: 7.5}), 3, True, 2.0),
            ],
            ["features", "label", "isVal", "weight"],
        )
        self.reg_params_with_eval = {
            "validation_indicator_col": "isVal",
            "early_stopping_rounds": 1,
            "eval_metric": "rmse",
        }
        self.reg_df_test_with_eval_weight = self.session.createDataFrame(
            [
                (Vectors.dense(1.0, 2.0, 3.0), 0.001068, 0.00088, 0.03155),
                (Vectors.sparse(3, {1: 1.0, 2: 5.5}), 1.000055, 0.9991, 0.9887),
            ],
            [
                "features",
                "expected_prediction_with_weight",
                "expected_prediction_with_eval",
                "expected_prediction_with_weight_and_eval",
            ],
        )
        self.reg_with_eval_best_score = 2.0
        self.reg_with_eval_and_weight_best_score = 1.997

        # Test classifier with weight and eval set
        # >>> import numpy as np
        # >>> import xgboost
        # >>> X = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 5.5], [4.0, 5.0, 6.0], [0.0, 6.0, 7.5]])
        # >>> w = np.array([1.0, 2.0, 1.0, 2.0])
        # >>> y = np.array([0, 1, 0, 1])
        # >>> cls1 = xgboost.XGBClassifier()
        # >>> cls1.fit(X, y, sample_weight=w)
        # >>> cls1.predict_proba(X)
        # array([[0.3333333, 0.6666667],...
        # >>> X_train = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 5.5]])
        # >>> X_val = np.array([[4.0, 5.0, 6.0], [0.0, 6.0, 7.5]])
        # >>> y_train = np.array([0, 1])
        # >>> y_val = np.array([0, 1])
        # >>> w_train = np.array([1.0, 2.0])
        # >>> w_val = np.array([1.0, 2.0])
        # >>> cls2 = xgboost.XGBClassifier()
        # >>> cls2.fit(X_train, y_train, eval_set=[(X_val, y_val)],
        # >>>               early_stopping_rounds=1, eval_metric='logloss')
        # >>> cls2.predict_proba(X)
        # array([[0.5, 0.5],...
        # >>> cls2.best_score
        # 0.6931
        # >>> cls3 = xgboost.XGBClassifier()
        # >>> cls3.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_val, y_val)],
        # >>>               sample_weight_eval_set=[w_val],
        # >>>               early_stopping_rounds=1, eval_metric='logloss')
        # >>> cls3.predict_proba(X)
        # array([[0.3344962, 0.6655038],...
        # >>> cls3.best_score
        # 0.6365
        self.cls_df_train_with_eval_weight = self.session.createDataFrame(
            [
                (Vectors.dense(1.0, 2.0, 3.0), 0, False, 1.0),
                (Vectors.sparse(3, {1: 1.0, 2: 5.5}), 1, False, 2.0),
                (Vectors.dense(4.0, 5.0, 6.0), 0, True, 1.0),
                (Vectors.sparse(3, {1: 6.0, 2: 7.5}), 1, True, 2.0),
            ],
            ["features", "label", "isVal", "weight"],
        )
        self.cls_params_with_eval = {
            "validation_indicator_col": "isVal",
            "early_stopping_rounds": 1,
            "eval_metric": "logloss",
        }
        self.cls_df_test_with_eval_weight = self.session.createDataFrame(
            [
                (
                    Vectors.dense(1.0, 2.0, 3.0),
                    [0.3333, 0.6666],
                    [0.5, 0.5],
                    [0.3097, 0.6903],
                ),
            ],
            [
                "features",
                "expected_prob_with_weight",
                "expected_prob_with_eval",
                "expected_prob_with_weight_and_eval",
            ],
        )
        self.cls_with_eval_best_score = 0.6931
        self.cls_with_eval_and_weight_best_score = 0.6378

        # Test classifier with both base margin and without
        # >>> import numpy as np
        # >>> import xgboost
        # >>> X = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 5.5], [4.0, 5.0, 6.0], [0.0, 6.0, 7.5]])
        # >>> w = np.array([1.0, 2.0, 1.0, 2.0])
        # >>> y = np.array([0, 1, 0, 1])
        # >>> base_margin = np.array([1,0,0,1])
        #
        # This is without the base margin
        # >>> cls1 = xgboost.XGBClassifier()
        # >>> cls1.fit(X, y, sample_weight=w)
        # >>> cls1.predict_proba(np.array([[1.0, 2.0, 3.0]]))
        # array([[0.3333333, 0.6666667]], dtype=float32)
        # >>> cls1.predict(np.array([[1.0, 2.0, 3.0]]))
        # array([1])
        #
        # This is with the same base margin for predict
        # >>> cls2 = xgboost.XGBClassifier()
        # >>> cls2.fit(X, y, sample_weight=w, base_margin=base_margin)
        # >>> cls2.predict_proba(np.array([[1.0, 2.0, 3.0]]), base_margin=[0])
        # array([[0.44142532, 0.5585747 ]], dtype=float32)
        # >>> cls2.predict(np.array([[1.0, 2.0, 3.0]]), base_margin=[0])
        # array([1])
        #
        # This is with a different base margin for predict
        # # >>> cls2 = xgboost.XGBClassifier()
        # >>> cls2.fit(X, y, sample_weight=w, base_margin=base_margin)
        # >>> cls2.predict_proba(np.array([[1.0, 2.0, 3.0]]), base_margin=[1])
        # array([[0.2252, 0.7747 ]], dtype=float32)
        # >>> cls2.predict(np.array([[1.0, 2.0, 3.0]]), base_margin=[0])
        # array([1])
        self.cls_df_train_without_base_margin = self.session.createDataFrame(
            [
                (Vectors.dense(1.0, 2.0, 3.0), 0, 1.0),
                (Vectors.sparse(3, {1: 1.0, 2: 5.5}), 1, 2.0),
                (Vectors.dense(4.0, 5.0, 6.0), 0, 1.0),
                (Vectors.sparse(3, {1: 6.0, 2: 7.5}), 1, 2.0),
            ],
            ["features", "label", "weight"],
        )
        self.cls_df_test_without_base_margin = self.session.createDataFrame(
            [
                (Vectors.dense(1.0, 2.0, 3.0), [0.3333, 0.6666], 1),
            ],
            [
                "features",
                "expected_prob_without_base_margin",
                "expected_prediction_without_base_margin",
            ],
        )

        self.cls_df_train_with_same_base_margin = self.session.createDataFrame(
            [
                (Vectors.dense(1.0, 2.0, 3.0), 0, 1.0, 1),
                (Vectors.sparse(3, {1: 1.0, 2: 5.5}), 1, 2.0, 0),
                (Vectors.dense(4.0, 5.0, 6.0), 0, 1.0, 0),
                (Vectors.sparse(3, {1: 6.0, 2: 7.5}), 1, 2.0, 1),
            ],
            ["features", "label", "weight", "base_margin"],
        )
        self.cls_df_test_with_same_base_margin = self.session.createDataFrame(
            [
                (Vectors.dense(1.0, 2.0, 3.0), 0, [0.4415, 0.5585], 1),
            ],
            [
                "features",
                "base_margin",
                "expected_prob_with_base_margin",
                "expected_prediction_with_base_margin",
            ],
        )

        self.cls_df_train_with_different_base_margin = self.session.createDataFrame(
            [
                (Vectors.dense(1.0, 2.0, 3.0), 0, 1.0, 1),
                (Vectors.sparse(3, {1: 1.0, 2: 5.5}), 1, 2.0, 0),
                (Vectors.dense(4.0, 5.0, 6.0), 0, 1.0, 0),
                (Vectors.sparse(3, {1: 6.0, 2: 7.5}), 1, 2.0, 1),
            ],
            ["features", "label", "weight", "base_margin"],
        )
        self.cls_df_test_with_different_base_margin = self.session.createDataFrame(
            [
                (Vectors.dense(1.0, 2.0, 3.0), 1, [0.2252, 0.7747], 1),
            ],
            [
                "features",
                "base_margin",
                "expected_prob_with_base_margin",
                "expected_prediction_with_base_margin",
            ],
        )

        self.reg_df_sparse_train = self.session.createDataFrame(
            [
                (Vectors.dense(1.0, 0.0, 3.0, 0.0, 0.0), 0),
                (Vectors.sparse(5, {1: 1.0, 3: 5.5}), 1),
                (Vectors.sparse(5, {4: -3.0}), 2),
            ]
            * 10,
            ["features", "label"],
        )

        self.cls_df_sparse_train = self.session.createDataFrame(
            [
                (Vectors.dense(1.0, 0.0, 3.0, 0.0, 0.0), 0),
                (Vectors.sparse(5, {1: 1.0, 3: 5.5}), 1),
                (Vectors.sparse(5, {4: -3.0}), 0),
            ]
            * 10,
            ["features", "label"],
        )

    def get_local_tmp_dir(self):
        return self.tempdir + str(uuid.uuid4())

    def assert_model_compatible(self, model: XGBModel, model_path: str):
        bst = xgb.Booster()
        path = glob.glob(f"{model_path}/**/model/part-00000", recursive=True)[0]
        bst.load_model(path)
        self.assertEqual(model.get_booster().save_raw("json"), bst.save_raw("json"))

    def test_regressor_params_basic(self):
        py_reg = SparkXGBRegressor()
        self.assertTrue(hasattr(py_reg, "n_estimators"))
        self.assertEqual(py_reg.n_estimators.parent, py_reg.uid)
        self.assertFalse(hasattr(py_reg, "gpu_id"))
        self.assertEqual(py_reg.getOrDefault(py_reg.n_estimators), 100)
        self.assertEqual(py_reg.getOrDefault(py_reg.objective), "reg:squarederror")
        py_reg2 = SparkXGBRegressor(n_estimators=200)
        self.assertEqual(py_reg2.getOrDefault(py_reg2.n_estimators), 200)
        py_reg3 = py_reg2.copy({py_reg2.max_depth: 10})
        self.assertEqual(py_reg3.getOrDefault(py_reg3.n_estimators), 200)
        self.assertEqual(py_reg3.getOrDefault(py_reg3.max_depth), 10)

    def test_classifier_params_basic(self):
        py_cls = SparkXGBClassifier()
        self.assertTrue(hasattr(py_cls, "n_estimators"))
        self.assertEqual(py_cls.n_estimators.parent, py_cls.uid)
        self.assertFalse(hasattr(py_cls, "gpu_id"))
        self.assertEqual(py_cls.getOrDefault(py_cls.n_estimators), 100)
        self.assertEqual(py_cls.getOrDefault(py_cls.objective), None)
        py_cls2 = SparkXGBClassifier(n_estimators=200)
        self.assertEqual(py_cls2.getOrDefault(py_cls2.n_estimators), 200)
        py_cls3 = py_cls2.copy({py_cls2.max_depth: 10})
        self.assertEqual(py_cls3.getOrDefault(py_cls3.n_estimators), 200)
        self.assertEqual(py_cls3.getOrDefault(py_cls3.max_depth), 10)

    def test_classifier_kwargs_basic(self):
        py_cls = SparkXGBClassifier(**self.cls_params_kwargs)
        self.assertTrue(hasattr(py_cls, "n_estimators"))
        self.assertEqual(py_cls.n_estimators.parent, py_cls.uid)
        self.assertFalse(hasattr(py_cls, "gpu_id"))
        self.assertTrue(hasattr(py_cls, "arbitrary_params_dict"))
        expected_kwargs = {"sketch_eps": 0.03}
        self.assertEqual(
            py_cls.getOrDefault(py_cls.arbitrary_params_dict), expected_kwargs
        )

        # Testing overwritten params
        py_cls = SparkXGBClassifier()
        py_cls.setParams(x=1, y=2)
        py_cls.setParams(y=3, z=4)
        xgb_params = py_cls._gen_xgb_params_dict()
        assert xgb_params["x"] == 1
        assert xgb_params["y"] == 3
        assert xgb_params["z"] == 4

    def test_param_alias(self):
        py_cls = SparkXGBClassifier(features_col="f1", label_col="l1")
        self.assertEqual(py_cls.getOrDefault(py_cls.featuresCol), "f1")
        self.assertEqual(py_cls.getOrDefault(py_cls.labelCol), "l1")
        with pytest.raises(
            ValueError, match="Please use param name features_col instead"
        ):
            SparkXGBClassifier(featuresCol="f1")

    def test_gpu_param_setting(self):
        py_cls = SparkXGBClassifier(use_gpu=True)
        train_params = py_cls._get_distributed_train_params(self.cls_df_train)
        assert train_params["tree_method"] == "gpu_hist"

    @staticmethod
    def test_param_value_converter():
        py_cls = SparkXGBClassifier(missing=np.float64(1.0), sketch_eps=np.float64(0.3))
        # don't check by isintance(v, float) because for numpy scalar it will also return True
        assert py_cls.getOrDefault(py_cls.missing).__class__.__name__ == "float"
        assert (
            py_cls.getOrDefault(py_cls.arbitrary_params_dict)[
                "sketch_eps"
            ].__class__.__name__
            == "float64"
        )

    def test_regressor_basic(self):
        regressor = SparkXGBRegressor()
        model = regressor.fit(self.reg_df_train)
        pred_result = model.transform(self.reg_df_test).collect()
        for row in pred_result:
            self.assertTrue(
                np.isclose(row.prediction, row.expected_prediction, atol=1e-3)
            )

    def test_classifier_basic(self):
        classifier = SparkXGBClassifier()
        model = classifier.fit(self.cls_df_train)
        pred_result = model.transform(self.cls_df_test).collect()
        for row in pred_result:
            self.assertEqual(row.prediction, row.expected_prediction)
            self.assertTrue(
                np.allclose(row.probability, row.expected_probability, rtol=1e-3)
            )

    def test_multi_classifier(self):
        classifier = SparkXGBClassifier()
        model = classifier.fit(self.multi_cls_df_train)
        pred_result = model.transform(self.multi_cls_df_test).collect()
        for row in pred_result:
            self.assertTrue(
                np.allclose(row.probability, row.expected_probability, rtol=1e-3)
            )

    def _check_sub_dict_match(self, sub_dist, whole_dict, excluding_keys):
        for k in sub_dist:
            if k not in excluding_keys:
                self.assertTrue(k in whole_dict, f"check on {k} failed")
                self.assertEqual(sub_dist[k], whole_dict[k], f"check on {k} failed")

    def test_regressor_with_params(self):
        regressor = SparkXGBRegressor(**self.reg_params)
        all_params = dict(
            **(regressor._gen_xgb_params_dict()),
            **(regressor._gen_fit_params_dict()),
            **(regressor._gen_predict_params_dict()),
        )
        self._check_sub_dict_match(
            self.reg_params, all_params, excluding_keys=_non_booster_params
        )

        model = regressor.fit(self.reg_df_train)
        all_params = dict(
            **(model._gen_xgb_params_dict()),
            **(model._gen_fit_params_dict()),
            **(model._gen_predict_params_dict()),
        )
        self._check_sub_dict_match(
            self.reg_params, all_params, excluding_keys=_non_booster_params
        )
        pred_result = model.transform(self.reg_df_test).collect()
        for row in pred_result:
            self.assertTrue(
                np.isclose(
                    row.prediction, row.expected_prediction_with_params, atol=1e-3
                )
            )

    def test_classifier_with_params(self):
        classifier = SparkXGBClassifier(**self.cls_params)
        all_params = dict(
            **(classifier._gen_xgb_params_dict()),
            **(classifier._gen_fit_params_dict()),
            **(classifier._gen_predict_params_dict()),
        )
        self._check_sub_dict_match(
            self.cls_params, all_params, excluding_keys=_non_booster_params
        )

        model = classifier.fit(self.cls_df_train)
        all_params = dict(
            **(model._gen_xgb_params_dict()),
            **(model._gen_fit_params_dict()),
            **(model._gen_predict_params_dict()),
        )
        self._check_sub_dict_match(
            self.cls_params, all_params, excluding_keys=_non_booster_params
        )
        pred_result = model.transform(self.cls_df_test).collect()
        for row in pred_result:
            self.assertEqual(row.prediction, row.expected_prediction_with_params)
            self.assertTrue(
                np.allclose(
                    row.probability, row.expected_probability_with_params, rtol=1e-3
                )
            )

    def test_regressor_model_save_load(self):
        tmp_dir = self.get_local_tmp_dir()
        path = "file:" + tmp_dir
        regressor = SparkXGBRegressor(**self.reg_params)
        model = regressor.fit(self.reg_df_train)
        model.save(path)
        loaded_model = SparkXGBRegressorModel.load(path)
        self.assertEqual(model.uid, loaded_model.uid)
        for k, v in self.reg_params.items():
            self.assertEqual(loaded_model.getOrDefault(k), v)

        pred_result = loaded_model.transform(self.reg_df_test).collect()
        for row in pred_result:
            self.assertTrue(
                np.isclose(
                    row.prediction, row.expected_prediction_with_params, atol=1e-3
                )
            )

        with self.assertRaisesRegex(AssertionError, "Expected class name"):
            SparkXGBClassifierModel.load(path)

        self.assert_model_compatible(model, tmp_dir)

    def test_classifier_model_save_load(self):
        tmp_dir = self.get_local_tmp_dir()
        path = "file:" + tmp_dir
        regressor = SparkXGBClassifier(**self.cls_params)
        model = regressor.fit(self.cls_df_train)
        model.save(path)
        loaded_model = SparkXGBClassifierModel.load(path)
        self.assertEqual(model.uid, loaded_model.uid)
        for k, v in self.cls_params.items():
            self.assertEqual(loaded_model.getOrDefault(k), v)

        pred_result = loaded_model.transform(self.cls_df_test).collect()
        for row in pred_result:
            self.assertTrue(
                np.allclose(
                    row.probability, row.expected_probability_with_params, atol=1e-3
                )
            )

        with self.assertRaisesRegex(AssertionError, "Expected class name"):
            SparkXGBRegressorModel.load(path)

        self.assert_model_compatible(model, tmp_dir)

    @staticmethod
    def _get_params_map(params_kv, estimator):
        return {getattr(estimator, k): v for k, v in params_kv.items()}

    def test_regressor_model_pipeline_save_load(self):
        tmp_dir = self.get_local_tmp_dir()
        path = "file:" + tmp_dir
        regressor = SparkXGBRegressor()
        pipeline = Pipeline(stages=[regressor])
        pipeline = pipeline.copy(extra=self._get_params_map(self.reg_params, regressor))
        model = pipeline.fit(self.reg_df_train)
        model.save(path)

        loaded_model = PipelineModel.load(path)
        for k, v in self.reg_params.items():
            self.assertEqual(loaded_model.stages[0].getOrDefault(k), v)

        pred_result = loaded_model.transform(self.reg_df_test).collect()
        for row in pred_result:
            self.assertTrue(
                np.isclose(
                    row.prediction, row.expected_prediction_with_params, atol=1e-3
                )
            )
        self.assert_model_compatible(model.stages[0], tmp_dir)

    def test_classifier_model_pipeline_save_load(self):
        tmp_dir = self.get_local_tmp_dir()
        path = "file:" + tmp_dir
        classifier = SparkXGBClassifier()
        pipeline = Pipeline(stages=[classifier])
        pipeline = pipeline.copy(
            extra=self._get_params_map(self.cls_params, classifier)
        )
        model = pipeline.fit(self.cls_df_train)
        model.save(path)

        loaded_model = PipelineModel.load(path)
        for k, v in self.cls_params.items():
            self.assertEqual(loaded_model.stages[0].getOrDefault(k), v)

        pred_result = loaded_model.transform(self.cls_df_test).collect()
        for row in pred_result:
            self.assertTrue(
                np.allclose(
                    row.probability, row.expected_probability_with_params, atol=1e-3
                )
            )
        self.assert_model_compatible(model.stages[0], tmp_dir)

    def test_classifier_with_cross_validator(self):
        xgb_classifer = SparkXGBClassifier()
        paramMaps = ParamGridBuilder().addGrid(xgb_classifer.max_depth, [1, 2]).build()
        cvBin = CrossValidator(
            estimator=xgb_classifer,
            estimatorParamMaps=paramMaps,
            evaluator=BinaryClassificationEvaluator(),
            seed=1,
            numFolds=2,
        )
        cvBinModel = cvBin.fit(self.cls_df_train_large)
        cvBinModel.transform(self.cls_df_test)

    def test_callbacks(self):
        from xgboost.callback import LearningRateScheduler

        path = self.get_local_tmp_dir()

        def custom_learning_rate(boosting_round):
            return 1.0 / (boosting_round + 1)

        cb = [LearningRateScheduler(custom_learning_rate)]
        regressor = SparkXGBRegressor(callbacks=cb)

        # Test the save/load of the estimator instead of the model, since
        # the callbacks param only exists in the estimator but not in the model
        regressor.save(path)
        regressor = SparkXGBRegressor.load(path)

        model = regressor.fit(self.reg_df_train)
        pred_result = model.transform(self.reg_df_test).collect()
        for row in pred_result:
            self.assertTrue(
                np.isclose(
                    row.prediction, row.expected_prediction_with_callbacks, atol=1e-3
                )
            )

    def test_train_with_initial_model(self):
        path = self.get_local_tmp_dir()
        reg1 = SparkXGBRegressor(**self.reg_params)
        model = reg1.fit(self.reg_df_train)
        init_booster = model.get_booster()
        reg2 = SparkXGBRegressor(max_depth=2, n_estimators=2, xgb_model=init_booster)
        model21 = reg2.fit(self.reg_df_train)
        pred_res21 = model21.transform(self.reg_df_test).collect()
        reg2.save(path)
        reg2 = SparkXGBRegressor.load(path)
        self.assertTrue(reg2.getOrDefault(reg2.xgb_model) is not None)
        model22 = reg2.fit(self.reg_df_train)
        pred_res22 = model22.transform(self.reg_df_test).collect()
        # Test the transform result is the same for original and loaded model
        for row1, row2 in zip(pred_res21, pred_res22):
            self.assertTrue(np.isclose(row1.prediction, row2.prediction, atol=1e-3))

    def test_classifier_with_base_margin(self):
        cls_without_base_margin = SparkXGBClassifier(weight_col="weight")
        model_without_base_margin = cls_without_base_margin.fit(
            self.cls_df_train_without_base_margin
        )
        pred_result_without_base_margin = model_without_base_margin.transform(
            self.cls_df_test_without_base_margin
        ).collect()
        for row in pred_result_without_base_margin:
            self.assertTrue(
                np.isclose(
                    row.prediction,
                    row.expected_prediction_without_base_margin,
                    atol=1e-3,
                )
            )
            np.testing.assert_allclose(
                row.probability, row.expected_prob_without_base_margin, atol=1e-3
            )

        cls_with_same_base_margin = SparkXGBClassifier(
            weight_col="weight", base_margin_col="base_margin"
        )
        model_with_same_base_margin = cls_with_same_base_margin.fit(
            self.cls_df_train_with_same_base_margin
        )
        pred_result_with_same_base_margin = model_with_same_base_margin.transform(
            self.cls_df_test_with_same_base_margin
        ).collect()
        for row in pred_result_with_same_base_margin:
            self.assertTrue(
                np.isclose(
                    row.prediction, row.expected_prediction_with_base_margin, atol=1e-3
                )
            )
            np.testing.assert_allclose(
                row.probability, row.expected_prob_with_base_margin, atol=1e-3
            )

        cls_with_different_base_margin = SparkXGBClassifier(
            weight_col="weight", base_margin_col="base_margin"
        )
        model_with_different_base_margin = cls_with_different_base_margin.fit(
            self.cls_df_train_with_different_base_margin
        )
        pred_result_with_different_base_margin = (
            model_with_different_base_margin.transform(
                self.cls_df_test_with_different_base_margin
            ).collect()
        )
        for row in pred_result_with_different_base_margin:
            self.assertTrue(
                np.isclose(
                    row.prediction, row.expected_prediction_with_base_margin, atol=1e-3
                )
            )
            np.testing.assert_allclose(
                row.probability, row.expected_prob_with_base_margin, atol=1e-3
            )

    def test_regressor_with_weight_eval(self):
        # with weight
        regressor_with_weight = SparkXGBRegressor(weight_col="weight")
        model_with_weight = regressor_with_weight.fit(
            self.reg_df_train_with_eval_weight
        )
        pred_result_with_weight = model_with_weight.transform(
            self.reg_df_test_with_eval_weight
        ).collect()
        for row in pred_result_with_weight:
            assert np.isclose(
                row.prediction, row.expected_prediction_with_weight, atol=1e-3
            )

        # with eval
        regressor_with_eval = SparkXGBRegressor(**self.reg_params_with_eval)
        model_with_eval = regressor_with_eval.fit(self.reg_df_train_with_eval_weight)
        assert np.isclose(
            model_with_eval._xgb_sklearn_model.best_score,
            self.reg_with_eval_best_score,
            atol=1e-3,
        ), (
            f"Expected best score: {self.reg_with_eval_best_score}, but ",
            f"get {model_with_eval._xgb_sklearn_model.best_score}",
        )

        pred_result_with_eval = model_with_eval.transform(
            self.reg_df_test_with_eval_weight
        ).collect()
        for row in pred_result_with_eval:
            self.assertTrue(
                np.isclose(
                    row.prediction, row.expected_prediction_with_eval, atol=1e-3
                ),
                f"Expect prediction is {row.expected_prediction_with_eval},"
                f"but get {row.prediction}",
            )
        # with weight and eval
        regressor_with_weight_eval = SparkXGBRegressor(
            weight_col="weight", **self.reg_params_with_eval
        )
        model_with_weight_eval = regressor_with_weight_eval.fit(
            self.reg_df_train_with_eval_weight
        )
        pred_result_with_weight_eval = model_with_weight_eval.transform(
            self.reg_df_test_with_eval_weight
        ).collect()
        self.assertTrue(
            np.isclose(
                model_with_weight_eval._xgb_sklearn_model.best_score,
                self.reg_with_eval_and_weight_best_score,
                atol=1e-3,
            )
        )
        for row in pred_result_with_weight_eval:
            self.assertTrue(
                np.isclose(
                    row.prediction,
                    row.expected_prediction_with_weight_and_eval,
                    atol=1e-3,
                )
            )

    def test_classifier_with_weight_eval(self):
        # with weight and eval
        # Added scale_pos_weight because in 1.4.2, the original answer returns 0.5 which
        # doesn't really indicate this working correctly.
        classifier_with_weight_eval = SparkXGBClassifier(
            weight_col="weight", scale_pos_weight=4, **self.cls_params_with_eval
        )
        model_with_weight_eval = classifier_with_weight_eval.fit(
            self.cls_df_train_with_eval_weight
        )
        pred_result_with_weight_eval = model_with_weight_eval.transform(
            self.cls_df_test_with_eval_weight
        ).collect()
        self.assertTrue(
            np.isclose(
                model_with_weight_eval._xgb_sklearn_model.best_score,
                self.cls_with_eval_and_weight_best_score,
                atol=1e-3,
            )
        )
        for row in pred_result_with_weight_eval:
            self.assertTrue(
                np.allclose(
                    row.probability, row.expected_prob_with_weight_and_eval, atol=1e-3
                )
            )

    def test_num_workers_param(self):
        regressor = SparkXGBRegressor(num_workers=-1)
        self.assertRaises(ValueError, regressor._validate_params)
        classifier = SparkXGBClassifier(num_workers=0)
        self.assertRaises(ValueError, classifier._validate_params)

    def test_use_gpu_param(self):
        classifier = SparkXGBClassifier(use_gpu=True, tree_method="exact")
        self.assertRaises(ValueError, classifier._validate_params)
        regressor = SparkXGBRegressor(use_gpu=True, tree_method="exact")
        self.assertRaises(ValueError, regressor._validate_params)
        regressor = SparkXGBRegressor(use_gpu=True, tree_method="gpu_hist")
        regressor = SparkXGBRegressor(use_gpu=True)
        classifier = SparkXGBClassifier(use_gpu=True, tree_method="gpu_hist")
        classifier = SparkXGBClassifier(use_gpu=True)

    def test_convert_to_sklearn_model(self):
        classifier = SparkXGBClassifier(
            n_estimators=200, missing=2.0, max_depth=3, sketch_eps=0.5
        )
        clf_model = classifier.fit(self.cls_df_train)

        regressor = SparkXGBRegressor(
            n_estimators=200, missing=2.0, max_depth=3, sketch_eps=0.5
        )
        reg_model = regressor.fit(self.reg_df_train)

        # Check that regardless of what booster, _convert_to_model converts to the correct class type
        sklearn_classifier = classifier._convert_to_sklearn_model(
            clf_model.get_booster().save_raw("json"),
            clf_model.get_booster().save_config(),
        )
        assert isinstance(sklearn_classifier, XGBClassifier)
        assert sklearn_classifier.n_estimators == 200
        assert sklearn_classifier.missing == 2.0
        assert sklearn_classifier.max_depth == 3
        assert sklearn_classifier.get_params()["sketch_eps"] == 0.5

        sklearn_regressor = regressor._convert_to_sklearn_model(
            reg_model.get_booster().save_raw("json"),
            reg_model.get_booster().save_config(),
        )
        assert isinstance(sklearn_regressor, XGBRegressor)
        assert sklearn_regressor.n_estimators == 200
        assert sklearn_regressor.missing == 2.0
        assert sklearn_regressor.max_depth == 3
        assert sklearn_classifier.get_params()["sketch_eps"] == 0.5

    def test_feature_importances(self):
        reg1 = SparkXGBRegressor(**self.reg_params)
        model = reg1.fit(self.reg_df_train)
        booster = model.get_booster()
        self.assertEqual(model.get_feature_importances(), booster.get_score())
        self.assertEqual(
            model.get_feature_importances(importance_type="gain"),
            booster.get_score(importance_type="gain"),
        )

    def test_regressor_array_col_as_feature(self):
        train_dataset = self.reg_df_train.withColumn(
            "features", vector_to_array(spark_sql_func.col("features"))
        )
        test_dataset = self.reg_df_test.withColumn(
            "features", vector_to_array(spark_sql_func.col("features"))
        )
        regressor = SparkXGBRegressor()
        model = regressor.fit(train_dataset)
        pred_result = model.transform(test_dataset).collect()
        for row in pred_result:
            self.assertTrue(
                np.isclose(row.prediction, row.expected_prediction, atol=1e-3)
            )

    def test_classifier_array_col_as_feature(self):
        train_dataset = self.cls_df_train.withColumn(
            "features", vector_to_array(spark_sql_func.col("features"))
        )
        test_dataset = self.cls_df_test.withColumn(
            "features", vector_to_array(spark_sql_func.col("features"))
        )
        classifier = SparkXGBClassifier()
        model = classifier.fit(train_dataset)

        pred_result = model.transform(test_dataset).collect()
        for row in pred_result:
            self.assertEqual(row.prediction, row.expected_prediction)
            self.assertTrue(
                np.allclose(row.probability, row.expected_probability, rtol=1e-3)
            )

    def test_classifier_with_feature_names_types_weights(self):
        classifier = SparkXGBClassifier(
            feature_names=["a1", "a2", "a3"],
            feature_types=["i", "int", "float"],
            feature_weights=[2.0, 5.0, 3.0],
        )
        model = classifier.fit(self.cls_df_train)
        model.transform(self.cls_df_test).collect()

    @pytest.mark.skipif(**no_sparse_unwrap())
    def test_regressor_with_sparse_optim(self):
        regressor = SparkXGBRegressor(missing=0.0)
        model = regressor.fit(self.reg_df_sparse_train)
        assert model._xgb_sklearn_model.missing == 0.0
        pred_result = model.transform(self.reg_df_sparse_train).collect()

        # enable sparse optimiaztion
        regressor2 = SparkXGBRegressor(missing=0.0, enable_sparse_data_optim=True)
        model2 = regressor2.fit(self.reg_df_sparse_train)
        assert model2.getOrDefault(model2.enable_sparse_data_optim)
        assert model2._xgb_sklearn_model.missing == 0.0
        pred_result2 = model2.transform(self.reg_df_sparse_train).collect()

        for row1, row2 in zip(pred_result, pred_result2):
            self.assertTrue(np.isclose(row1.prediction, row2.prediction, atol=1e-3))

    @pytest.mark.skipif(**no_sparse_unwrap())
    def test_classifier_with_sparse_optim(self):
        cls = SparkXGBClassifier(missing=0.0)
        model = cls.fit(self.cls_df_sparse_train)
        assert model._xgb_sklearn_model.missing == 0.0
        pred_result = model.transform(self.cls_df_sparse_train).collect()

        # enable sparse optimiaztion
        cls2 = SparkXGBClassifier(missing=0.0, enable_sparse_data_optim=True)
        model2 = cls2.fit(self.cls_df_sparse_train)
        assert model2.getOrDefault(model2.enable_sparse_data_optim)
        assert model2._xgb_sklearn_model.missing == 0.0
        pred_result2 = model2.transform(self.cls_df_sparse_train).collect()

        for row1, row2 in zip(pred_result, pred_result2):
            self.assertTrue(np.allclose(row1.probability, row2.probability, rtol=1e-3))

    def test_empty_validation_data(self) -> None:
        for tree_method in [
            "hist",
            "approx",
        ]:  # pytest.mark conflict with python unittest
            df_train = self.session.createDataFrame(
                [
                    (Vectors.dense(10.1, 11.2, 11.3), 0, False),
                    (Vectors.dense(1, 1.2, 1.3), 1, False),
                    (Vectors.dense(14.0, 15.0, 16.0), 0, False),
                    (Vectors.dense(1.1, 1.2, 1.3), 1, True),
                ],
                ["features", "label", "val_col"],
            )
            classifier = SparkXGBClassifier(
                num_workers=2,
                tree_method=tree_method,
                min_child_weight=0.0,
                reg_alpha=0,
                reg_lambda=0,
                validation_indicator_col="val_col",
            )
            model = classifier.fit(df_train)
            pred_result = model.transform(df_train).collect()
            for row in pred_result:
                self.assertEqual(row.prediction, row.label)

    def test_empty_train_data(self) -> None:
        for tree_method in [
            "hist",
            "approx",
        ]:  # pytest.mark conflict with python unittest
            df_train = self.session.createDataFrame(
                [
                    (Vectors.dense(10.1, 11.2, 11.3), 0, True),
                    (Vectors.dense(1, 1.2, 1.3), 1, True),
                    (Vectors.dense(14.0, 15.0, 16.0), 0, True),
                    (Vectors.dense(1.1, 1.2, 1.3), 1, False),
                ],
                ["features", "label", "val_col"],
            )
            classifier = SparkXGBClassifier(
                num_workers=2,
                min_child_weight=0.0,
                reg_alpha=0,
                reg_lambda=0,
                tree_method=tree_method,
                validation_indicator_col="val_col",
            )
            model = classifier.fit(df_train)
            pred_result = model.transform(df_train).collect()
            for row in pred_result:
                assert row.prediction == 1.0

    def test_empty_partition(self):
        # raw_df.repartition(4) will result int severe data skew, actually,
        # there is no any data in reducer partition 1, reducer partition 2
        # see https://github.com/dmlc/xgboost/issues/8221
        for tree_method in [
            "hist",
            "approx",
        ]:  # pytest.mark conflict with python unittest
            raw_df = self.session.range(0, 100, 1, 50).withColumn(
                "label",
                spark_sql_func.when(spark_sql_func.rand(1) > 0.5, 1).otherwise(0),
            )
            vector_assembler = (
                VectorAssembler().setInputCols(["id"]).setOutputCol("features")
            )
            data_trans = vector_assembler.setHandleInvalid("keep").transform(raw_df)

            classifier = SparkXGBClassifier(num_workers=4, tree_method=tree_method)
            classifier.fit(data_trans)

    def test_early_stop_param_validation(self):
        classifier = SparkXGBClassifier(early_stopping_rounds=1)
        with pytest.raises(ValueError, match="early_stopping_rounds"):
            classifier.fit(self.cls_df_train)

    def test_unsupported_params(self):
        with pytest.raises(ValueError, match="evals_result"):
            SparkXGBClassifier(evals_result={})


class XgboostRankerLocalTest(SparkTestCase):
    def setUp(self):
        self.session.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "8")
        self.ranker_df_train = self.session.createDataFrame(
            [
                (Vectors.dense(1.0, 2.0, 3.0), 0, 0),
                (Vectors.dense(4.0, 5.0, 6.0), 1, 0),
                (Vectors.dense(9.0, 4.0, 8.0), 2, 0),
                (Vectors.sparse(3, {1: 1.0, 2: 5.5}), 0, 1),
                (Vectors.sparse(3, {1: 6.0, 2: 7.5}), 1, 1),
                (Vectors.sparse(3, {1: 8.0, 2: 9.5}), 2, 1),
            ],
            ["features", "label", "qid"],
        )
        self.ranker_df_test = self.session.createDataFrame(
            [
                (Vectors.dense(1.5, 2.0, 3.0), 0, -1.87988),
                (Vectors.dense(4.5, 5.0, 6.0), 0, 0.29556),
                (Vectors.dense(9.0, 4.5, 8.0), 0, 2.36570),
                (Vectors.sparse(3, {1: 1.0, 2: 6.0}), 1, -1.87988),
                (Vectors.sparse(3, {1: 6.0, 2: 7.0}), 1, -0.30612),
                (Vectors.sparse(3, {1: 8.0, 2: 10.5}), 1, 2.44826),
            ],
            ["features", "qid", "expected_prediction"],
        )
        self.ranker_df_train_1 = self.session.createDataFrame(
            [
                (Vectors.sparse(3, {1: 1.0, 2: 5.5}), 0, 9),
                (Vectors.sparse(3, {1: 6.0, 2: 7.5}), 1, 9),
                (Vectors.sparse(3, {1: 8.0, 2: 9.5}), 2, 9),
                (Vectors.dense(1.0, 2.0, 3.0), 0, 8),
                (Vectors.dense(4.0, 5.0, 6.0), 1, 8),
                (Vectors.dense(9.0, 4.0, 8.0), 2, 8),
                (Vectors.sparse(3, {1: 1.0, 2: 5.5}), 0, 7),
                (Vectors.sparse(3, {1: 6.0, 2: 7.5}), 1, 7),
                (Vectors.sparse(3, {1: 8.0, 2: 9.5}), 2, 7),
                (Vectors.dense(1.0, 2.0, 3.0), 0, 6),
                (Vectors.dense(4.0, 5.0, 6.0), 1, 6),
                (Vectors.dense(9.0, 4.0, 8.0), 2, 6),
            ]
            * 4,
            ["features", "label", "qid"],
        )

    def test_ranker(self):
        ranker = SparkXGBRanker(qid_col="qid")
        assert ranker.getOrDefault(ranker.objective) == "rank:pairwise"
        model = ranker.fit(self.ranker_df_train)
        pred_result = model.transform(self.ranker_df_test).collect()

        for row in pred_result:
            assert np.isclose(row.prediction, row.expected_prediction, rtol=1e-3)

    def test_ranker_qid_sorted(self):
        ranker = SparkXGBRanker(qid_col="qid", num_workers=4)
        assert ranker.getOrDefault(ranker.objective) == "rank:pairwise"
        model = ranker.fit(self.ranker_df_train_1)
        model.transform(self.ranker_df_test).collect()
