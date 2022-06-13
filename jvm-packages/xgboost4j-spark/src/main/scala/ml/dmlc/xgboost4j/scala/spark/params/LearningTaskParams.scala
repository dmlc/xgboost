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

import scala.collection.immutable.HashSet

import org.apache.spark.ml.param._

private[spark] trait LearningTaskParams extends Params {

  /**
   * Specify the learning task and the corresponding learning objective.
   * options: reg:squarederror, reg:squaredlogerror, reg:logistic, binary:logistic, binary:logitraw,
   * count:poisson, multi:softmax, multi:softprob, rank:pairwise, reg:gamma.
   * default: reg:squarederror
   */
  final val objective = new Param[String](this, "objective",
    "objective function used for training")

  final def getObjective: String = $(objective)

  /**
   * The learning objective type of the specified custom objective and eval.
   * Corresponding type will be assigned if custom objective is defined
   * options: regression, classification. default: null
   */
  final val objectiveType = new Param[String](this, "objectiveType", "objective type used for " +
    s"training, options: {${LearningTaskParams.supportedObjectiveType.mkString(",")}",
    (value: String) => LearningTaskParams.supportedObjectiveType.contains(value))

  final def getObjectiveType: String = $(objectiveType)


  /**
   * the initial prediction score of all instances, global bias. default=0.5
   */
  final val baseScore = new DoubleParam(this, "baseScore", "the initial prediction score of all" +
    " instances, global bias")

  final def getBaseScore: Double = $(baseScore)

  /**
   * evaluation metrics for validation data, a default metric will be assigned according to
   * objective(rmse for regression, and error for classification, mean average precision for
   * ranking). options: rmse, rmsle, mae, mape, logloss, error, merror, mlogloss, auc, aucpr, ndcg,
   * map, gamma-deviance
   */
  final val evalMetric = new Param[String](this, "evalMetric", "evaluation metrics for " +
    "validation data, a default metric will be assigned according to objective " +
    "(rmse for regression, and error for classification, mean average precision for ranking)")

  final def getEvalMetric: String = $(evalMetric)

  /**
   * Fraction of training points to use for testing.
   */
  final val trainTestRatio = new DoubleParam(this, "trainTestRatio",
    "fraction of training points to use for testing",
    ParamValidators.inRange(0, 1))
  setDefault(trainTestRatio, 1.0)

  final def getTrainTestRatio: Double = $(trainTestRatio)

  /**
   * whether caching training data
   */
  final val cacheTrainingSet = new BooleanParam(this, "cacheTrainingSet",
    "whether caching training data")

  /**
   * whether cleaning checkpoint, always cleaning by default, having this parameter majorly for
   * testing
   */
  final val skipCleanCheckpoint = new BooleanParam(this, "skipCleanCheckpoint",
    "whether cleaning checkpoint data")

  /**
   * If non-zero, the training will be stopped after a specified number
   * of consecutive increases in any evaluation metric.
   */
  final val numEarlyStoppingRounds = new IntParam(this, "numEarlyStoppingRounds",
    "number of rounds of decreasing eval metric to tolerate before " +
    "stopping the training",
    (value: Int) => value == 0 || value > 1)

  final def getNumEarlyStoppingRounds: Int = $(numEarlyStoppingRounds)


  final val maximizeEvaluationMetrics = new BooleanParam(this, "maximizeEvaluationMetrics",
    "define the expected optimization to the evaluation metrics, true to maximize otherwise" +
      " minimize it")

  final def getMaximizeEvaluationMetrics: Boolean = $(maximizeEvaluationMetrics)

}

private[spark] object LearningTaskParams {

  val supportedObjectiveType = HashSet("regression", "classification")

  val evalMetricsToMaximize = HashSet("auc", "aucpr", "ndcg", "map")

  val evalMetricsToMinimize = HashSet("rmse", "rmsle", "mae", "mape", "logloss", "error", "merror",
    "mlogloss", "gamma-deviance")
}
