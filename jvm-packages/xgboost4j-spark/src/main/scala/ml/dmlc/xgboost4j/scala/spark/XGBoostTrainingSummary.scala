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

class XGBoostTrainingSummary private(
    val trainObjectiveHistory: Array[Float],
    val validationObjectiveHistory: (String, Array[Float])*) extends Serializable {

  override def toString: String = {
    val train = trainObjectiveHistory.mkString(",")
    val vaidationObjectiveHistoryString = {
      validationObjectiveHistory.map {
        case (name, metrics) =>
          s"${name}ObjectiveHistory=${metrics.mkString(",")}"
      }.mkString(";")
    }
    s"XGBoostTrainingSummary(trainObjectiveHistory=$train; $vaidationObjectiveHistoryString)"
  }
}

private[xgboost4j] object XGBoostTrainingSummary {
  def apply(metrics: Map[String, Array[Float]]): XGBoostTrainingSummary = {
    new XGBoostTrainingSummary(
      trainObjectiveHistory = metrics("train"),
      metrics.filter(_._1 != "train").toSeq: _*)
  }
}
