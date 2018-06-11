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

from pyspark.ml.util import JavaMLReadable, JavaMLReader


class XGBoostReadable(JavaMLReadable):
    """
    Mixin for instances that provide XGBoostReader.
    """

    @classmethod
    def read(cls):
        """Returns an XGBoostReader instance for this class."""
        return XGBoostReader(cls)


class XGBoostReader(JavaMLReader):
    """
    A reader for XGBoost objects.
    """

    def __init__(self, clazz):
        self._clazz = clazz
        self._jread = self._load_java_obj(clazz).read()

    @classmethod
    def _java_loader_class(cls, clazz):
        """
        Returns the full class name of the Java XGBoost class. In this case,
        we prepend "ml.dmlc.xgboost4j.scala.spark" to the name of the class.
        """

        # Deal with Python locations for XGBoost objects.
        if clazz.__name__ in ("XGBoostEstimator", "XGBoostClassificationModel", "XGBoostRegressionModel"):
            java_package = clazz.__module__.replace("sparkxgb", "ml.dmlc.xgboost4j.scala.spark")

        # Our Pipeline object calls down to Spark's Pipeline object.
        elif clazz.__name__ == "XGBoostPipeline":
            return "org.apache.spark.ml.Pipeline"

        # Our PipelineModel object calls down to Spark's PipelineModel object.
        elif clazz.__name__ == "XGBoostPipelineModel":
            return "org.apache.spark.ml.PipelineModel"

        # Allow default PySpark objects to be loaded.
        else:
            java_package = clazz.__module__.replace("pyspark", "org.apache.spark")

        if clazz.__name__ in ("Pipeline", "PipelineModel", "XGBoostEstimator", "XGBoostClassificationModel",
                              "XGBoostRegressionModel"):
            # Remove the last dangling package name, e.g. "pipeline" or "xgboost".
            java_package = ".".join(java_package.split(".")[0:-1])

        return java_package + "." + clazz.__name__
