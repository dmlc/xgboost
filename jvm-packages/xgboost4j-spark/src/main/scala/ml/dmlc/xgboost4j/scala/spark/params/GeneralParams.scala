/*
 Copyright (c) 2014-2024 by Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

package ml.dmlc.xgboost4j.scala.spark.params

import org.apache.spark.ml.param._

/**
 * General xgboost parameters, more details can be found
 * at https://xgboost.readthedocs.io/en/stable/parameter.html#general-parameters
 */
private[spark] trait GeneralParams extends Params {

  final val booster = new Param[String](this, "booster", "Which booster to use. Can be gbtree, " +
    "gblinear or dart; gbtree and dart use tree based models while gblinear uses linear " +
    "functions.", ParamValidators.inArray(Array("gbtree", "dart")))

  final def getBooster: String = $(booster)

  final val device = new Param[String](this, "device", "Device for XGBoost to run. User can " +
    "set it to one of the following values: {cpu, cuda, gpu}",
    ParamValidators.inArray(Array("cpu", "cuda", "gpu")))

  final def getDevice: String = $(device)

  final val verbosity = new IntParam(this, "verbosity", "Verbosity of printing messages. Valid " +
    "values are 0 (silent), 1 (warning), 2 (info), 3 (debug). Sometimes XGBoost tries to change " +
    "configurations based on heuristics, which is displayed as warning message. If there's " +
    "unexpected behaviour, please try to increase value of verbosity.",
    ParamValidators.inRange(0, 3, true, true))

  final def getVerbosity: Int = $(verbosity)

  final val validateParameters = new BooleanParam(this, "validate_parameters", "When set to " +
    "True, XGBoost will perform validation of input parameters to check whether a parameter " +
    "is used or not. A warning is emitted when there's unknown parameter.")

  final def getValidateParameters: Boolean = $(validateParameters)

  final val nthread = new IntParam(this, "nthread", "Number of threads used by per worker",
    ParamValidators.gtEq(1))

  final def getNthread: Int = $(nthread)

  setDefault(booster -> "gbtree", device -> "cpu", verbosity -> 1, validateParameters -> false,
    nthread -> 1)
}
