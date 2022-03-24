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

# pylint: disable=invalid-name, too-many-ancestors, protected-access
"""
Interal mixins for pyspark module.
"""
from abc import ABCMeta

from pyspark.ml.classification import (
    Classifier,
    ProbabilisticClassifier,
    ProbabilisticClassificationModel,
    ClassificationModel,
)
from pyspark.ml.common import inherit_doc
from pyspark.ml.regression import Regressor, RegressionModel
from pyspark.ml.util import JavaMLReadable, JavaMLReader, JavaMLWritable
from pyspark.ml.wrapper import JavaPredictor, JavaPredictionModel

from .shared import _XGBoostRegressorParams, _XGBoostClassifierParams


class _XGBJavaMLReadable(JavaMLReadable):
    """
    Mixin a version of xgboost JavaMLReadable
    """

    @classmethod
    def read(cls):
        return _XGBJavaMLReader(cls)


class _XGBJavaMLReader(JavaMLReader):
    """
    Mixin a version of xgboost JavaMLReader
    """

    @classmethod
    def _java_loader_class(cls, clazz):
        if hasattr(clazz, "_java_class_name") and clazz._java_class_name is not None:
            return clazz._java_class_name
        return JavaMLReader._java_loader_class(clazz)


@inherit_doc
class _XGBJavaClassifier(Classifier, JavaPredictor, metaclass=ABCMeta):
    """
    Java Classifier for classification tasks.
    Classes are indexed {0, 1, ..., numClasses - 1}.
    """

    # Copied from _JavaClassifier
    def setRawPredictionCol(self, value):
        """
        Sets the value of :py:attr:`rawPredictionCol`.
        """
        return self._set(rawPredictionCol=value)


@inherit_doc
class _XGBJavaProbabilisticClassifier(
    ProbabilisticClassifier, _XGBJavaClassifier, metaclass=ABCMeta
):
    """
    Java Probabilistic Classifier for classification tasks.
    """


class _XGBoostClassifierBase(
    _XGBJavaProbabilisticClassifier,
    _XGBoostClassifierParams,
    JavaMLWritable,
    _XGBJavaMLReadable,
    metaclass=ABCMeta,
):
    """
    The base class of XGBoostClassifier
    """


@inherit_doc
class _XGBJavaClassificationModel(ClassificationModel, JavaPredictionModel):
    """
    Java Model produced by a ``Classifier``.
    Classes are indexed {0, 1, ..., numClasses - 1}.
    To be mixed in with :class:`pyspark.ml.JavaModel`
    """

    @property
    def numClasses(self):
        """
        Number of classes (values which the label can take).
        """
        return self._call_java("numClasses")

    def predictRaw(self, value):
        """
        Raw prediction for each possible label.
        """
        return self._call_java("predictRaw", value)


@inherit_doc
class _XGBJavaProbabilisticClassificationModel(
    ProbabilisticClassificationModel, _XGBJavaClassificationModel
):
    """
    Java Model produced by a ``ProbabilisticClassifier``.
    """

    def predictProbability(self, value):
        """
        Predict the probability of each class given the features.
        """
        return self._call_java("predictProbability", value)


class _XGBoostClassificationModelBase(
    _XGBJavaProbabilisticClassificationModel,
    _XGBoostClassifierParams,
    JavaMLWritable,
    _XGBJavaMLReadable,
):
    """
    The base class of XGBoostClassificationModel
    """


@inherit_doc
class _XGBJavaRegressor(Regressor, JavaPredictor, metaclass=ABCMeta):
    """
    Java Regressor for regression tasks.

    .. versionadded:: 3.0.0
    """


class _XGBoostRegressorBase(
    _XGBJavaRegressor,
    _XGBoostRegressorParams,
    JavaMLWritable,
    _XGBJavaMLReadable,
    metaclass=ABCMeta,
):
    """
    The base class of XGBoostRegressor
    """


@inherit_doc
class _XGBJavaRegressionModel(RegressionModel, JavaPredictionModel, metaclass=ABCMeta):
    """
    Java Model produced by a ``_JavaRegressor``.
    To be mixed in with :class:`pyspark.ml.JavaModel`

    .. versionadded:: 3.0.0
    """


class _XGBoostRegressionModelBase(
    _XGBJavaRegressionModel, JavaMLWritable, _XGBJavaMLReadable
):
    """
    The base class of XGBoostRegressionModel
    """
