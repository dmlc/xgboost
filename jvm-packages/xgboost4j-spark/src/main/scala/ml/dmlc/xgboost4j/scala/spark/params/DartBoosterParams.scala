/*
 Copyright (c) 2024 by Contributors

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

/** Dropout parameters for tree boosters. */
private[spark] trait DartBoosterParams extends Params {

  final val dropoutRate = new DoubleParam(this, "dropout_rate", "Probability of independently " +
    "dropping each existing tree before gradient computation",
    ParamValidators.inRange(0, 1, true, false))

  final def getDropoutRate: Double = $(dropoutRate)

  final val sampleType = new Param[String](this, "sample_type", "Deprecated and ignored; " +
    "scheduled for removal", ParamValidators.inArray(Array("uniform", "weighted")))

  final def getSampleType: String = $(sampleType)

  final val normalizeType = new Param[String](this, "normalize_type", "Deprecated and ignored; " +
    "scheduled for removal",
    ParamValidators.inArray(Array("tree", "forest")))

  final def getNormalizeType: String = $(normalizeType)

  final val rateDrop = new DoubleParam(this, "rate_drop", "Deprecated and ignored; scheduled " +
    "for removal",
    ParamValidators.inRange(0, 1, true, true))

  final def getRateDrop: Double = $(rateDrop)

  final val oneDrop = new BooleanParam(this, "one_drop", "Deprecated and ignored; scheduled " +
    "for removal")

  final def getOneDrop: Boolean = $(oneDrop)

  final val skipDrop = new DoubleParam(this, "skip_drop", "Deprecated alias for dropout_rate",
    ParamValidators.inRange(0, 1, true, false))

  final def getSkipDrop: Double = $(skipDrop)

  setDefault(dropoutRate -> 0, sampleType -> "uniform", normalizeType -> "tree", rateDrop -> 0,
    skipDrop -> 0)

}
