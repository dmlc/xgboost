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
from pyspark.ml.util import JavaMLWritable, JavaMLReadable

from xgboost.param.internal import _XGBJavaProbabilisticClassifier, _XGBJavaProbabilisticClassificationModel
from xgboost.param.shared import _XGBoostClassifierParams


@inherit_doc
class XGBoostClassifier(_XGBoostClassifierParams, _XGBJavaProbabilisticClassifier,
                        JavaMLWritable, JavaMLReadable):
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
    >>> from xgboost.spark import XGBoostClassifier
    >>> iris_data_path = 'raw-iris.data'
    >>> spark = SparkSession.builder.appName("xgboost iris").getOrCreate()
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
    ...     'numRound': 100,
    ...     'numClass': 3,
    ...     'labelCol': 'classIndex',
    ...     'featuresCol': 'features'
    ... }
    >>> xgb_classifier = XGBoostClassifier(**params)
    >>> model = xgb_classifier.fit(xgb_input)
    22/02/28 17:39:22 WARN XGBoostSpark: train_test_ratio is deprecated since XGBoost 0.82, we recommend to explicitly pass a training and multiple evaluation datasets by passing 'eval_sets' and 'eval_set_names'
    Tracker started, with env={DMLC_NUM_SERVER=0, DMLC_TRACKER_URI=10.18.132.76, DMLC_TRACKER_PORT=9091, DMLC_NUM_WORKER=1}
    [17:39:22] task 0 got new rank 0
    >>> df = model.transform(xgb_input)
    >>> df.show(2)
    +-----------------+----------+--------------------+--------------------+----------+
    |         features|classIndex|       rawPrediction|         probability|prediction|
    +-----------------+----------+--------------------+--------------------+----------+
    |[5.1,3.5,1.4,0.2]|       0.0|[3.08765506744384...|[0.99680268764495...|       0.0|
    |[4.9,3.0,1.4,0.2]|       0.0|[3.08765506744384...|[0.99636262655258...|       0.0|
    +-----------------+----------+--------------------+--------------------+----------+
    only showing top 2 rows

    Besides passing dictionary to XGBoostClassifier, users can call set APIs to set the parameters,

    xgb_classifier = XGBoostClassifier() \
        .setFeaturesCol("features") \
        .setLabelCol("classIndex") \
        .setNumRound(100) \
        .setNumClass(3) \
        .setObjective('multi:softprob') \
        .setTreeMethod('hist')

    """

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
        self._java_obj = self._new_java_obj(
            'ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier', self.uid)
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def _create_model(self, java_model):
        return XGBoostClassificationModel(java_model)

    def setNumRound(self, value):
        """
        Sets the value of :py:attr:`numRound`.
        """
        self._set(numRound=value)
        return self

    def setNumWorkers(self, value):
        """
        Sets the value of :py:attr:`numWorkers`.
        """
        self._set(numWorkers=value)
        return self

    def setNthread(self, value):
        """
        Sets the value of :py:attr:`nthread`.
        """
        self._set(nthread=value)
        return self

    def setNumClass(self, value):
        """
        Sets the value of :py:attr:`numClass`.
        """
        self._set(numClass=value)
        return self

    def setObjective(self, value):
        """
        Sets the value of :py:attr:`objective`.
        """
        self._set(objective=value)
        return self

    def setTreeMethod(self, value):
        """
        Sets the value of :py:attr:`treeMethod`.
        """
        self._set(treeMethod=value)
        return self


class XGBoostClassificationModel(_XGBJavaProbabilisticClassificationModel, JavaMLWritable, JavaMLReadable):
    """
    The model returned by :func:`xgboost.spark.XGBoostClassifier.fit()`

    """
    pass
