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

import org.apache.spark.ml.param._

trait BoostingParams extends Params {

  /**
   * The number of rounds for boosting
   */
  val round = new IntParam(this, "num_round", "The number of rounds for boosting",
    ParamValidators.gtEq(1))

  /**
   * Specify the learning task and the corresponding learning objective.
   * options: reg:linear, reg:logistic, binary:logistic, binary:logitraw, count:poisson,
   * multi:softmax, multi:softprob, rank:pairwise, reg:gamma. default: reg:linear
   */
  val objective = new Param[String](this, "objective", "objective function used for training," +
    s" options: {${BoostingParams.supportedObjectives.mkString(",")}",
    (value: String) => BoostingParams.supportedObjectives.contains(value))

  setDefault(objective -> "reg:linear", round -> 1)

}

private[spark] object BoostingParams {
  val supportedObjectives = Set("reg:linear", "reg:logistic", "binary:logistic",
    "binary:logitraw", "count:poisson", "multi:softmax", "multi:softprob", "rank:pairwise",
    "reg:gamma")
}
