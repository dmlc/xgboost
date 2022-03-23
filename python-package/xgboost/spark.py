#
# Copyright (c) 2022 by Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from pyspark import keyword_only
from pyspark.ml.common import inherit_doc

from .param import _XGBoostClassifierBase, _XGBoostClassificationModelBase
from .param import _XGBoostRegressorBase, _XGBoostRegressionModelBase


@inherit_doc
class XGBoostClassifier(_XGBoostClassifierBase):
    """
    XGBoostClassifier is a PySpark ML estimator. It implements the XGBoost classification
    algorithm based on `ml.dmlc.xgboost4j.scala.pyspark.XGBoostClassifier` in XGBoost jvm packages,
    and it can be used in PySpark Pipeline and PySpark ML meta algorithms like CrossValidator.

    .. versionadded:: 1.6.0

    Examples
    --------

    >>> from pyspark.ml.feature import StringIndexer, VectorAssembler
    >>> from pyspark.sql import SparkSession
    >>> from pyspark.sql.types import *
    >>> from xgboost.spark import XGBoostClassifier, XGBoostClassificationModel
    >>> iris_data_path = 'iris.csv'
    >>> schema = StructType([
    ...     StructField("sepal length", DoubleType(), nullable=True),
    ...     StructField("sepal width", DoubleType(), nullable=True),
    ...     StructField("petal length", DoubleType(), nullable=True),
    ...     StructField("petal width", DoubleType(), nullable=True),
    ...     StructField("class", StringType(), nullable=True),
    ... ])
    >>> raw_df = spark.read.schema(schema).csv(iris_data_path)
    >>> stringIndexer = StringIndexer(inputCol="class", outputCol="classIndex").fit(raw_df)
    >>> labeled_input = stringIndexer.transform(raw_df).drop("class")
    >>> vector_assembler = VectorAssembler()\
    ...     .setInputCols(("sepal length", "sepal width", "petal length", "petal width"))\
    ...     .setOutputCol("features")
    >>> xgb_input = vector_assembler.transform(labeled_input).select("features", "classIndex")
    >>> params = {
    ...     'objective': 'multi:softprob',
    ...     'treeMethod': 'hist',
    ...     'numWorkers': 1,
    ...     'numRound': 5,
    ...     'numClass': 3,
    ...     'labelCol': 'classIndex',
    ...     'featuresCol': 'features'
    ... }
    >>> classifier = XGBoostClassifier(**params)
    >>> classifier.write().overwrite().save("/tmp/xgboost_classifier")
    >>> classifier1 = XGBoostClassifier.load("/tmp/xgboost_classifier")
    >>> model = classifier1.fit(xgb_input)
    22/03/11 13:25:06 WARN XGBoostSpark: train_test_ratio is deprecated since XGBoost 0.82,
    we recommend to explicitly pass a training and multiple evaluation datasets by passing 'eval_sets' and 'eval_set_names'
    Tracker started, with env={DMLC_NUM_SERVER=0, DMLC_TRACKER_URI=xx.xx.xx.xx, DMLC_TRACKER_PORT=38625, DMLC_NUM_WORKER=1}
    [13:25:06] task 0 got new rank 0
    >>> model.write().overwrite().save("/tmp/xgboost_classifier_model")
    >>> model1 = XGBoostClassificationModel.load("/tmp/xgboost_classifier_model")
    >>> df = model1.transform(xgb_input)
    >>> df.show(2)
    +-----------------+----------+--------------------+--------------------+----------+
    |         features|classIndex|       rawPrediction|         probability|prediction|
    +-----------------+----------+--------------------+--------------------+----------+
    |[5.1,3.5,1.4,0.2]|       0.0|[1.84931623935699...|[0.82763016223907...|       0.0|
    |[4.9,3.0,1.4,0.2]|       0.0|[1.84931623935699...|[0.82763016223907...|       0.0|
    +-----------------+----------+--------------------+--------------------+----------+
    only showing top 2 rows

    Besides passing dictionary parameters to XGBoostClassifier, users can call set APIs to set the parameters,

    xgb_classifier = XGBoostClassifier() \
        .setFeaturesCol("features") \
        .setLabelCol("classIndex") \
        .setNumRound(100) \
        .setNumClass(3) \
        .setObjective('multi:softprob') \
        .setTreeMethod('hist')

    """

    # _java_class_name will be used when loading pipeline.
    _java_class_name = 'ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier'

    @keyword_only
    def __init__(self, *,
                 featuresCol=None,
                 labelCol=None,
                 treeMethod=None,
                 objective=None,
                 numClass=None,
                 numRound=None,
                 numWorkers=None
                 ):
        super(XGBoostClassifier, self).__init__()
        self._java_obj = self._new_java_obj(self.__class__._java_class_name, self.uid)
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def _create_model(self, java_model):
        return XGBoostClassificationModel(java_model)


class XGBoostClassificationModel(_XGBoostClassificationModelBase):
    """
    The model returned by :func:`xgboost.spark.XGBoostClassifier.fit()`

    """

    # _java_class_name will be used when loading pipeline.
    _java_class_name = 'ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel'

    def __init__(self, java_model=None):
        super(XGBoostClassificationModel, self).__init__(java_model=java_model)
        if not java_model:
            self._java_obj = self._new_java_obj(self.__class__._java_class_name, self.uid)
            # transfer jvm default values to python
            self._transfer_params_from_java()


class XGBoostRegressor(_XGBoostRegressorBase):
    """
    XGBoostRegressor is a PySpark ML estimator. It implements the XGBoost regression
    algorithm based on `ml.dmlc.xgboost4j.scala.pyspark.XGBoostRegressor` in XGBoost jvm packages,
    and it can be used in PySpark Pipeline and PySpark ML meta algorithms like CrossValidator.

    .. versionadded:: 1.6.0

    Examples
    --------

    >>> from pyspark.ml.linalg import Vectors
    >>> from pyspark.sql import SparkSession
    >>> from xgboost.spark import XGBoostRegressor, XGBoostRegressionModel
    >>> input = spark.createDataFrame([
    ...     (1.0, Vectors.dense(1.0)),
    ...     (0.0, Vectors.dense(2.0))], ["label", "features"])
    >>> params = {
    ...     'objective': 'reg:squarederror',
    ...     'treeMethod': 'hist',
    ...     'numWorkers': 1,
    ...     'numRound': 100,
    ...     'labelCol': 'label',
    ...     'featuresCol': 'features'
    ... }
    >>> regressor = XGBoostRegressor(**params)
    >>> regressor.write().overwrite().save("/tmp/xgboost_regressor")
    >>> regressor1 = XGBoostRegressor.load("/tmp/xgboost_regressor")
    >>> model = regressor1.fit(input)
    22/03/11 13:47:57 WARN XGBoostSpark: train_test_ratio is deprecated since XGBoost 0.82,
    we recommend to explicitly pass a training and multiple evaluation datasets by passing 'eval_sets' and 'eval_set_names'
    Tracker started, with env={DMLC_NUM_SERVER=0, DMLC_TRACKER_URI=xx.xx.xx.xx, DMLC_TRACKER_PORT=43619, DMLC_NUM_WORKER=1}
    [13:47:57] task 0 got new rank 0
    >>> model.write().overwrite().save("/tmp/xgboost_regressor_model")
    >>> model1 = XGBoostRegressionModel.load("/tmp/xgboost_regressor_model")
    >>> df = model1.transform(input)
    >>> df.show()
    +-----+--------+--------------------+
    |label|features|          prediction|
    +-----+--------+--------------------+
    |  1.0|   [1.0]|  0.9991162419319153|
    |  0.0|   [2.0]|8.837578352540731E-4|
    +-----+--------+--------------------+

    Besides passing dictionary parameters to XGBoostClassifier, users can call set APIs to set the parameters,

    xgb_classifier = XGBoostRegressor() \
        .setFeaturesCol("features") \
        .setLabelCol("label") \
        .setNumRound(100) \
        .setObjective('reg:squarederror') \
        .setTreeMethod('hist')

    """

    # _java_class_name will be used when loading pipeline.
    _java_class_name = 'ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor'

    @keyword_only
    def __init__(self, *,
                 featuresCol=None,
                 labelCol=None,
                 treeMethod=None,
                 objective=None,
                 numRound=None,
                 numWorkers=None
                 ):
        super(XGBoostRegressor, self).__init__()
        self._java_obj = self._new_java_obj(self.__class__._java_class_name, self.uid)
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def _create_model(self, java_model):
        return XGBoostRegressionModel(java_model)


class XGBoostRegressionModel(_XGBoostRegressionModelBase):
    """
    The model returned by :func:`xgboost.spark.XGBoostRegressor.fit()`

    """

    # _java_class_name will be used when loading pipeline.
    _java_class_name = 'ml.dmlc.xgboost4j.scala.spark.XGBoostRegressionModel'

    def __init__(self, java_model=None):
        super(XGBoostRegressionModel, self).__init__(java_model=java_model)
        if not java_model:
            self._java_obj = self._new_java_obj(self.__class__._java_class_name, self.uid)
            # transfer jvm default values to python
            self._transfer_params_from_java()
