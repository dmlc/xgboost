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

package ml.dmlc.xgboost4j.scala

import ml.dmlc.xgboost4j.java
import ml.dmlc.xgboost4j.java.IEvaluation

trait EvalTrait extends IEvaluation {

  /**
   * get evaluate metric
   *
   * @return evalMetric
   */
  def getMetric: String

  /**
   * evaluate with predicts and data
   *
   * @param predicts predictions as array
   * @param dmat     data matrix to evaluate
   * @return result of the metric
   */
  def eval(predicts: Array[Array[Float]], dmat: DMatrix): Float

  private[scala] def eval(predicts: Array[Array[Float]], jdmat: java.DMatrix): Float = {
    require(predicts.length == jdmat.getLabel.length, "predicts size and label size must match " +
      s" predicts size: ${predicts.length}, label size: ${jdmat.getLabel.length}")
    eval(predicts, new DMatrix(jdmat))
  }
}
