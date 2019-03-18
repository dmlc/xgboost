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

package ml.dmlc.xgboost4j.scala.spark

import ml.dmlc.xgboost4j.java.IEvaluationForDistributed
import ml.dmlc.xgboost4j.scala.{DMatrix, EvalTrait}

class DistributedEvalError extends EvalTrait with IEvaluationForDistributed {

  /**
   * get evaluate metric
   *
   * @return evalMetric
   */
  override def getMetric: String = "distributed_error"

  /**
   * evaluate with predicts and data
   *
   * @param predicts predictions as array
   * @param dmat     data matrix to evaluate
   * @return result of the metric
   */
  override def eval(predicts: Array[Array[Float]], dmat: DMatrix): Float = 0.0f

  /**
   * calculate the metrics for a single row given its label and prediction
   */
  override def evalRow(label: Float, pred: Float): Float = 0.0f

  /**
   * perform transformation with the sum of error and weights to get the final evaluation metrics
   */
  override def getFinal(errorSum: Float, weightSum: Float): Float = 0.0f
}
