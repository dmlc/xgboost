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

package ml.dmlc.xgboost4j.scala.spark

import ml.dmlc.xgboost4j.scala.spark.params.XGBoostEstimatorCommon

import org.apache.spark.ml.{Estimator, Model, PipelineStage}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

/**
 * PreXGBoost implementation provider
 */
private[scala] trait PreXGBoostProvider {

  /**
   * Whether the provider is enabled or not
   * @param dataset the input dataset
   * @return Boolean
   */
  def providerEnabled(dataset: Option[Dataset[_]]): Boolean = false

  /**
   * Transform schema
   * @param xgboostEstimator supporting XGBoostClassifier/XGBoostClassificationModel and
   *                 XGBoostRegressor/XGBoostRegressionModel
   * @param schema the input schema
   * @return the transformed schema
   */
  def transformSchema(xgboostEstimator: XGBoostEstimatorCommon, schema: StructType): StructType

  /**
   * Convert the Dataset[_] to RDD[() => Watches] which will be fed to XGBoost
   *
   * @param estimator supports XGBoostClassifier and XGBoostRegressor
   * @param dataset the training data
   * @param params all user defined and defaulted params
   * @return [[XGBoostExecutionParams]] => (RDD[[() => Watches]], Option[ RDD[_] ])
   *         RDD[() => Watches] will be used as the training input to build DMatrix
   *         Option[ RDD[_] ] is the optional cached RDD
   */
  def buildDatasetToRDD(
    estimator: Estimator[_],
    dataset: Dataset[_],
    params: Map[String, Any]):
  XGBoostExecutionParams => (RDD[() => Watches], Option[RDD[_]])

  /**
   * Transform Dataset
   *
   * @param model supporting [[XGBoostClassificationModel]] and [[XGBoostRegressionModel]]
   * @param dataset the input Dataset to transform
   * @return the transformed DataFrame
   */
  def transformDataset(model: Model[_], dataset: Dataset[_]): DataFrame

}
