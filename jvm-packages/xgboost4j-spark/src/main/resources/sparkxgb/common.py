#
# Copyright (c) 2019 by Contributors
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
import re

from pyspark.ml.param import Params
from pyspark.ml.util import JavaMLWritable
from pyspark.ml.wrapper import JavaModel, JavaEstimator

from sparkxgb.util import XGBoostReadable


class ParamGettersSetters(Params):
    """
    Mixin class used to generate the setters/getters for all params.
    """

    def _create_param_getters_and_setters(self):
        for param in self.params:
            param_name = param.name
            fg_attr = "get" + re.sub(r"(?:^|_)(.)", lambda m: m.group(1).upper(), param_name)
            fs_attr = "set" + re.sub(r"(?:^|_)(.)", lambda m: m.group(1).upper(), param_name)
            # Generates getter and setter only if not exists
            try:
                getattr(self, fg_attr)
            except AttributeError:
                setattr(self, fg_attr, self._get_param_value(param_name))
            try:
                getattr(self, fs_attr)
            except AttributeError:
                setattr(self, fs_attr, self._set_param_value(param_name))

    def _get_param_value(self, param_name):
        def r():
            try:
                return self.getOrDefault(param_name)
            except KeyError:
                return None
        return r

    def _set_param_value(self, param_name):
        def r(v):
            self.set(self.getParam(param_name), v)
            return self
        return r


class XGboostEstimator(JavaEstimator, XGBoostReadable, JavaMLWritable, ParamGettersSetters):
    """
    Mixin class for XGBoost estimators, like XGBoostClassifier and XGBoostRegressor.
    """

    def __init__(self, classname):
        super(XGboostEstimator, self).__init__()
        self.__class__._java_class_name = classname
        self._java_obj = self._new_java_obj(classname, self.uid)
        self._create_params_from_java()
        self._create_param_getters_and_setters()


class XGboostModel(JavaModel, XGBoostReadable, JavaMLWritable, ParamGettersSetters):
    """
    Mixin class for XGBoost models, like XGBoostClassificationModel and XGBoostRegressionModel.
    """

    def __init__(self, classname, java_model=None):
        super(XGboostModel, self).__init__(java_model=java_model)
        if classname and not java_model:
            self.__class__._java_class_name = classname
            self._java_obj = self._new_java_obj(classname, self.uid)
        if java_model is not None:
            self._transfer_params_from_java()
        self._create_param_getters_and_setters()
