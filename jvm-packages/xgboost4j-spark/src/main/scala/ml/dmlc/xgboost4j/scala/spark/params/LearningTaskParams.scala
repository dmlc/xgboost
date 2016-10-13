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

import scala.collection.immutable.HashSet

import org.apache.spark.ml.param.{DoubleParam, Param, Params}

private[spark] trait LearningTaskParams extends Params {

  val objective = new Param[String](this, "objective", "objective function used during training",
    (value: String) => LearningTaskParams.supportedObjective.contains(value))

  setDefault(objective, "reg:linear")

  val baseScore = new DoubleParam(this, "base_score", "the initial prediction score of all" +
    " instances, global bias")

  setDefault(baseScore, 0.5)

  val evalMetric = new Param[String](this, "eval_metric", "evaluation metrics for validation" +
    " data, a default metric will be assigned according to objective( rmse for regression, and" +
    " error for classification, mean average precision for ranking )",
    (value: String) => LearningTaskParams.supportedEvalMetrics.contains(value))

  // set according to objective
  if ($(objective).startsWith("reg:") || $(objective) == "count:poisson") {
    set(evalMetric, "rmse")
  } else if ($(objective) == "rank:pairwise") {
    set(evalMetric, "map")
  } else {
    set(evalMetric, "error")
  }
}

private[spark] object LearningTaskParams {
  val supportedObjective = HashSet("reg:linear", "reg:logistic", "binary:logistic",
    "binary:logitraw", "count:poisson", "multi:softmax", "multi:softprob", "rank:pairwise",
    "reg:gamma")

  val supportedEvalMetrics = HashSet("rmse", "mae", "logloss", "error", "merror", "mlogloss",
    "auc", "ndcg", "map", "gamma-deviance")
}
