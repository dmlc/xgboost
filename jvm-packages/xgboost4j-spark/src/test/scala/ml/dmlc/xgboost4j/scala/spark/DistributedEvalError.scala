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

import ml.dmlc.xgboost4j.java.{IEvalElementWiseDistributed, IEvalMultiClassesDistributed, IEvalRankListDistributed}

class DistributedEvalErrorElementWise extends IEvalElementWiseDistributed {

  /**
   * calculate the metrics for a single row given its label and prediction
   */
  def evalRow(label: Float, pred: Float): Float = 1.0f

  /**
   * perform transformation with the sum of error and weights to get the final evaluation metrics
   */
  def getFinal(errorSum: Float, weightSum: Float): Float = errorSum

  /**
   * get metrics' name
   */
  override def getMetric: String = "distributed_error_element_wise"
}

class DistributedEvalErrorMultiClasses extends IEvalMultiClassesDistributed {

  /**
   * calculate the metrics for a single row given its label and prediction
   */
  override def evalRow(label: Int, pred: Float, numClasses: Int): Float = {
    1.0f + numClasses
  }

  override def getFinal(errorSum: Float, weightSum: Float): Float = errorSum

  /**
   * get metrics' name
   */
  override def getMetric: String = "distributed_error_multi_classes"
}

class DistributedEvalErrorRankList extends IEvalRankListDistributed {

  override def evalMetric(preds: Array[Float], labels: Array[Int]): Float = {
    println(s"preds:${preds.mkString(",")}")
    println(s"labels:${preds.mkString(",")}")
    labels.sum
  }

  /**
   * get metrics' name
   */
  override def getMetric: String = "distributed_error_ranking"
}
