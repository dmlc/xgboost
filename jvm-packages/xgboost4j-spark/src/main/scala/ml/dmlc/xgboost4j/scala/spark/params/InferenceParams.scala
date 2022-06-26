/*
 Copyright (c) 2014-2022 by Contributors

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

import org.apache.spark.ml.param.{IntParam, Params}

private[spark] trait InferenceParams extends Params {

  /**
   * batch size of inference iteration
   */
  final val inferBatchSize = new IntParam(this, "batchSize", "batch size of inference iteration")

  /** @group getParam */
  final def getInferBatchSize: Int = $(inferBatchSize)

  setDefault(inferBatchSize, 32 << 10)
}
