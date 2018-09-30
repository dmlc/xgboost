#
# Copyright (c) 2018 by Contributors
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

from pyspark import SparkContext, keyword_only
from pyspark.ml.util import JavaMLWritable, JavaPredictionModel
from pyspark.ml.wrapper import JavaEstimator, JavaModel
from sparkxgb.util import XGBoostReadable


class JavaParamsOverrides(object):
    """
    Mixin for overriding methods derived from JavaParams.
    """

    # Override the "_from_java" method, so we can read our objects.
    @classmethod
    def _from_java(cls, java_stage):
        """
        Given a Java object, create and return a Python wrapper of it.
        Used for ML persistence.
        """

        # Create a new instance of this stage.
        py_stage = cls()

        # Load information from java_stage to the instance.
        py_stage._java_obj = java_stage
        py_stage._create_params_from_java()
        py_stage._resetUid(java_stage.uid())
        py_stage._transfer_params_from_java()

        return py_stage


class XGBoostClassifier(JavaParamsOverrides, JavaEstimator, JavaMLWritable, XGBoostReadable):
    """
    A PySpark implementation of ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier.
    """

    @keyword_only
    def __init__(self, alpha=0.0, baseMarginCol=None, baseScore=0.5, checkpointInterval=-1, checkpointPath='',
                 colsampleBylevel=1.0, colsampleBytree=1.0, contribPredictionCol=None, customEval=None, customObj=None,
                 eta=0.3, evalMetric=None, featuresCol='features', gamm=0.0, growPolicy='depthwise', labelCol='label',
                 reg_lambda=1.0, lambdaBias=0.0, leafPredictionCol=None, maxBin=16, maxDeltaStep=0.0, maxDepth=2,
                 minChildWeight=1.0, missing=float('nan'), normalizeType='tree', nthread=1, numClass=None,
                 numEarlyStoppingRounds=0, numRound=1, numWorkers=1, objective='binary:logistic',
                 predictionCol='prediction', probabilityCol='probability', rateDrop=0.0,
                 rawPredictionCol='rawPrediction', sampleType='uniform', scalePosWeight=1.0, seed=0, silent=0,
                 sketchEps=0.03, skipDrop=0.0, subsample=1.0, thresholds=None, timeoutRequestWorkers=1800000,
                 trainTestRatio=1.0, treeLimit=0, treeMethod='auto', useExternalMemory=False, weightCol=None):

        super(XGBoostClassifier, self).__init__()
        self._java_obj = self._new_java_obj("ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier", self.uid)
        self._create_params_from_java()
        self._setDefault()  # We get our defaults from the embedded Scala object, so no need to specify them here.
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, alpha=0.0, baseMarginCol=None, baseScore=0.5, checkpointInterval=-1, checkpointPath='',
                 colsampleBylevel=1.0, colsampleBytree=1.0, contribPredictionCol=None, customEval=None, customObj=None,
                 eta=0.3, evalMetric=None, featuresCol='features', gamm=0.0, growPolicy='depthwise', labelCol='label',
                 reg_lambda=1.0, lambdaBias=0.0, leafPredictionCol=None, maxBin=16, maxDeltaStep=0.0, maxDepth=2,
                 minChildWeight=1.0, missing=float('nan'), normalizeType='tree', nthread=1, numClass=None,
                 numEarlyStoppingRounds=0, numRound=1, numWorkers=1, objective='binary:logistic',
                 predictionCol='prediction', probabilityCol='probability', rateDrop=0.0,
                 rawPredictionCol='rawPrediction', sampleType='uniform', scalePosWeight=1.0, seed=0, silent=0,
                 sketchEps=0.03, skipDrop=0.0, subsample=1.0, thresholds=None, timeoutRequestWorkers=1800000,
                 trainTestRatio=1.0, treeLimit=0, treeMethod='auto', useExternalMemory=False, weightCol=None):

        kwargs = self._input_kwargs_processed()
        return self._set(**kwargs)

    def _input_kwargs_processed(self):
        """
        Until consensus on parameter names can be achieved, we must rename kwargs which would break Python.
        """

        kwargs = self._input_kwargs
        if "reg_lambda" in kwargs:
            kwargs["lambda"] = kwargs.pop("reg_lambda")

        return kwargs

    def _create_model(self, java_model):
        return XGBoostClassificationModel(java_model)


class XGBoostClassificationModel(JavaParamsOverrides, JavaModel, JavaPredictionModel, JavaMLWritable, XGBoostReadable):
    """
    A PySpark implementation of ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel.
    """

    @property
    def nativeBooster(self):
        """
        Get the native booster instance of this model.
        This is used to call low-level APIs on native booster, such as "getFeatureScore".
        """
        return self._call_java("nativeBooster")

    @property
    def numClasses(self):
        """
        Number of classes (values which the label can take).
        """
        return self._call_java("numClasses")

    def setThresholds(self, value):
        """
        Sets the value of :py:attr:`thresholds`.
        """
        return self._set(thresholds=value)

    def getThresholds(self):
        """
        Gets the value of thresholds or its default value.
        """
        return self.getOrDefault(self.thresholds)

    def setRawPredictionCol(self, value):
        """
        Sets the value of :py:attr:`rawPredictionCol`.
        """
        return self._set(rawPredictionCol=value)

    def getRawPredictionCol(self):
        """
        Gets the value of rawPredictionCol or its default value.
        """
        return self.getOrDefault(self.rawPredictionCol)


class XGBoostRegressor(JavaParamsOverrides, JavaEstimator, JavaMLWritable, XGBoostReadable):
    """
    A PySpark implementation of ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor.
    """

    @keyword_only
    def __init__(self, alpha=0.0, baseMarginCol=None, baseScore=0.5, checkpointInterval=-1, checkpointPath='',
                 colsampleBylevel=1.0, colsampleBytree=1.0, contribPredictionCol=None, customEval=None, customObj=None,
                 eta=0.3, evalMetric=None, featuresCol='features', gamm=0.0, groupCol=None, growPolicy='depthwise',
                 labelCol='label', reg_lambda=1.0, lambdaBias=0.0, leafPredictionCol=None, maxBin=16, maxDeltaStep=0.0,
                 maxDepth=2, minChildWeight=1.0, missing=float('nan'), normalizeType='tree', nthread=1,
                 numEarlyStoppingRounds=0, numRound=1, numWorkers=1, objective='reg:linear', predictionCol='prediction',
                 rateDrop=0.0, sampleType='uniform', scalePosWeight=1.0, seed=0, silent=0, sketchEps=0.03, skipDrop=0.0,
                 subsample=1.0, timeoutRequestWorkers=1800000, trainTestRatio=1.0, treeLimit=0, treeMethod='auto',
                 useExternalMemory=False, weightCol=None):

        super(XGBoostRegressor, self).__init__()
        self._java_obj = self._new_java_obj("ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor", self.uid)
        self._create_params_from_java()
        self._setDefault()  # We get our defaults from the embedded Scala object, so no need to specify them here.
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, alpha=0.0, baseMarginCol=None, baseScore=0.5, checkpointInterval=-1, checkpointPath='',
                 colsampleBylevel=1.0, colsampleBytree=1.0, contribPredictionCol=None, customEval=None, customObj=None,
                 eta=0.3, evalMetric=None, featuresCol='features', gamm=0.0, groupCol=None, growPolicy='depthwise',
                 labelCol='label', reg_lambda=1.0, lambdaBias=0.0, leafPredictionCol=None, maxBin=16, maxDeltaStep=0.0,
                 maxDepth=2, minChildWeight=1.0, missing=float('nan'), normalizeType='tree', nthread=1,
                 numEarlyStoppingRounds=0, numRound=1, numWorkers=1, objective='reg:linear', predictionCol='prediction',
                 rateDrop=0.0, sampleType='uniform', scalePosWeight=1.0, seed=0, silent=0, sketchEps=0.03, skipDrop=0.0,
                 subsample=1.0, timeoutRequestWorkers=1800000, trainTestRatio=1.0, treeLimit=0, treeMethod='auto',
                 useExternalMemory=False, weightCol=None):

        kwargs = self._input_kwargs_processed()
        return self._set(**kwargs)

    def _input_kwargs_processed(self):
        """
        Until consensus on parameter names can be achieved, we must rename kwargs which would break Python.
        """

        kwargs = self._input_kwargs
        if "reg_lambda" in kwargs:
            kwargs["lambda"] = kwargs.pop("reg_lambda")

        return kwargs

    def _create_model(self, java_model):
        return XGBoostRegressionModel(java_model)


class XGBoostRegressionModel(JavaParamsOverrides, JavaModel, JavaPredictionModel, JavaMLWritable, XGBoostReadable):
    """
    A PySpark implementation of ml.dmlc.xgboost4j.scala.spark.XGBoostRegressionModel.
    """

    @property
    def nativeBooster(self):
        """
        Get the native booster instance of this model.
        This is used to call low-level APIs on native booster, such as "getFeatureScore".
        """
        return self._call_java("nativeBooster")
