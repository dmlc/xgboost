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

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.base import Estimator, Transformer
from pyspark.ml.wrapper import JavaParams
from sparkxgb.util import XGBoostReader


class JavaParamsXGBoost(JavaParams):
    """
    A class derived from pyspark.ml.wrapper.JavaParams which is able to load XGBoost stages.
    """

    @staticmethod
    def _from_java(java_stage):
        """
        Given a Java object, create and return a Python wrapper of it.
        Used for ML persistence.
        """

        def __get_class(clazz):
            """
            Loads Python class from its name.
            """
            parts = clazz.split('.')
            module = ".".join(parts[:-1])
            m = __import__(module)
            for comp in parts[1:]:
                m = getattr(m, comp)
            return m

        stage_name = java_stage.getClass().getName()
        class_name = stage_name.split('.')[-1]

        if class_name in ("XGBoostClassifier", "XGBoostRegressor", "XGBoostClassificationModel",
                          "XGBoostRegressionModel"):
            stage_name = stage_name.replace("ml.dmlc.xgboost4j.scala.spark", "sparkxgb")

        # Our Pipeline's Java object is just the default Spark one. (Cast all Pipelines to our version)
        elif class_name == "Pipeline":
            stage_name = "sparkxgb.XGBoostPipeline"

        # Our PipelineModel's Java object is just the default Spark one. (Cast all PipelineModels to our version)
        elif class_name == "PipelineModel":
            stage_name = "sparkxgb.XGBoostPipelineModel"

        else:
            stage_name = stage_name.replace("org.apache.spark", "pyspark")

        # Generate a default new instance from the stage_name class.
        py_type = __get_class(stage_name)

        if hasattr(py_type, "_from_java"):
            py_stage = py_type._from_java(java_stage)

        else:
            raise NotImplementedError("This Java stage cannot be loaded into Python currently: %r"
                                      % stage_name)
        return py_stage


class XGBoostPipeline(Pipeline):
    """
    A class derived from Pipeline which is able to read/write XGBoost estimator stages.
    """

    @classmethod
    def read(cls):
        """Returns an XGBoostReader instance for this class."""
        return XGBoostReader(cls)

    @classmethod
    def _from_java(cls, java_stage):
        """
        Given a Java Pipeline, create and return a Python wrapper of it.
        Used for ML persistence.
        """
        # Create a new instance of this stage.
        py_stage = cls()
        # Load information from java_stage to the instance.
        py_stages = [JavaParamsXGBoost._from_java(s) for s in java_stage.getStages()]
        py_stage.setStages(py_stages)
        py_stage._resetUid(java_stage.uid())
        return py_stage

    def _fit(self, dataset):
        """
        This method mirrors the default Pipeline _fit, the only difference is we return an XGBoostPipelineModel object.
        """
        stages = self.getStages()
        for stage in stages:
            if not (isinstance(stage, Estimator) or isinstance(stage, Transformer)):
                raise TypeError(
                    "Cannot recognize a pipeline stage of type %s." % type(stage))
        indexOfLastEstimator = -1
        for i, stage in enumerate(stages):
            if isinstance(stage, Estimator):
                indexOfLastEstimator = i
        transformers = []
        for i, stage in enumerate(stages):
            if i <= indexOfLastEstimator:
                if isinstance(stage, Transformer):
                    transformers.append(stage)
                    dataset = stage.transform(dataset)
                else:  # must be an Estimator
                    model = stage.fit(dataset)
                    transformers.append(model)
                    if i < indexOfLastEstimator:
                        dataset = model.transform(dataset)
            else:
                transformers.append(stage)
        return XGBoostPipelineModel(transformers)


class XGBoostPipelineModel(PipelineModel):
    """
    A class derived from PipelineModel which is able to read/write XGBoost transformer stages.
    """

    @classmethod
    def read(cls):
        """Returns an XGBoostReader instance for this class."""
        return XGBoostReader(cls)

    @classmethod
    def _from_java(cls, java_stage):
        """
        Given a Java PipelineModel, create and return a Python wrapper of it.
        Used for ML persistence.
        """
        # Load information from java_stage to the instance.
        py_stages = [JavaParamsXGBoost._from_java(s) for s in java_stage.stages()]
        # Create a new instance of this stage.
        py_stage = cls(py_stages)
        py_stage._resetUid(java_stage.uid())
        return py_stage
