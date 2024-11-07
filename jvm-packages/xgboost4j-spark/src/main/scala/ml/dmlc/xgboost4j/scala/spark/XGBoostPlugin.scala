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
package ml.dmlc.xgboost4j.scala.spark

import java.io.Serializable

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset}

trait XGBoostPlugin extends Serializable {
  /**
   * Whether the plugin is enabled or not, if not enabled, fallback
   * to the regular CPU pipeline
   *
   * @param dataset the input dataset
   * @return Boolean
   */
  def isEnabled(dataset: Dataset[_]): Boolean

  /**
   * Convert Dataset to RDD[Watches] which will be fed into XGBoost
   *
   * @param estimator which estimator to be handled.
   * @param dataset   to be converted.
   * @return RDD[Watches]
   */
  def buildRddWatches[T <: XGBoostEstimator[T, M], M <: XGBoostModel[M]](
      estimator: XGBoostEstimator[T, M],
      dataset: Dataset[_]): RDD[Watches]

  /**
   * Transform the dataset
   */
  def transform[M <: XGBoostModel[M]](model: XGBoostModel[M], dataset: Dataset[_]): DataFrame

}
