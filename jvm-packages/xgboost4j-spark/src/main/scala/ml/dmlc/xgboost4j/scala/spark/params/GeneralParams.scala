/*
 Copyright (c) 2014 by Contributors

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

import ml.dmlc.xgboost4j.scala.{EvalTrait, ObjectiveTrait}
import org.apache.spark.ml.param._

private[spark] trait GeneralParams extends Params {

  val round = new IntParam(this, "num_round", "The number of rounds for boosting",
    ParamValidators.gtEq(1))

  val nWorkers = new IntParam(this, "nthread", "number of workers used to run xgboost",
    ParamValidators.gtEq(1))

  val useExternalMemory = new BooleanParam(this, "use_external_memory", "whether to use external" +
    "memory as cache")

  val silent = new IntParam(this, "silent",
    "0 means printing running messages, 1 means silent mode.",
    (value: Int) => value >= 0 && value <= 1)

  val customObj = new Param[ObjectiveTrait](this, "custom_obj", "customized objective function " +
    "provided by the user")

  val customEval = new Param[EvalTrait](this, "custom_obj", "customized evaluation function " +
    "provided by the user")

  val missing = new FloatParam(this, "missing", "the value treated as missing")

  setDefault(round -> 1, nWorkers -> 1, useExternalMemory -> false, silent -> 0,
    customObj -> null, customEval -> null, missing -> Float.NaN)
}
