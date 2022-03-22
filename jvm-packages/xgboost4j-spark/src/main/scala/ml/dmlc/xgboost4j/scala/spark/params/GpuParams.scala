/*
 Copyright (c) 2021-2022 by Contributors

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

import org.apache.spark.ml.param.{Params, StringArrayParam}

trait GpuParams extends Params {
  /**
   * Param for the names of feature columns for GPU pipeline.
   * @group param
   */
  final val featuresCols: StringArrayParam = new StringArrayParam(this, "featuresCols",
    "an array of feature column names for GPU pipeline.")

  setDefault(featuresCols, Array.empty[String])

  /** @group getParam */
  final def getFeaturesCols: Array[String] = $(featuresCols)

}
