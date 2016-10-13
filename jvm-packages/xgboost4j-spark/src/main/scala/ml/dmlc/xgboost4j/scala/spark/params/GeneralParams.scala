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

import org.apache.spark.ml.param._

private[spark] trait GeneralParams extends Params {

  val round = new IntParam(this, "num_round", "The number of rounds for boosting",
    ParamValidators.gtEq(1))

  setDefault(round, 1)

  val nWorkers = new IntParam(this, "nthread", "number of workers used to run xgboost",
    ParamValidators.gtEq(1))

  setDefault(nWorkers, 1)

  val useExternalMemory = new BooleanParam(this, "use_external_memory", "whether to use external" +
    "memory as cache")

  setDefault(useExternalMemory, false)

  val boosterType = new Param[String](this, "booster",
    s"Booster to use ${GeneralParams.supportedBoosters.mkString(", ")}.",
    (value: String) => GeneralParams.supportedBoosters.contains(value.toLowerCase))

  setDefault(boosterType, "gbtree")

  val silent = new Param[Int](this, "silent",
    "0 means printing running messages, 1 means silent mode.",
    (value: Int) => value >= 0 && value <= 1)

  setDefault(silent, 0)
}

private[spark] object GeneralParams {

  val supportedBoosters: Array[String] = Array("gbtree", "gblinear", "dart")
}
