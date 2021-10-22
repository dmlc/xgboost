/*
 Copyright (c) 2014,2021 by Contributors

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

import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasWeightCol}

private[scala] sealed trait XGBoostEstimatorCommon extends GeneralParams with LearningTaskParams
  with BoosterParams with RabitParams with ParamMapFuncs with NonParamVariables with HasWeightCol
  with HasBaseMarginCol with HasLeafPredictionCol with HasContribPredictionCol with HasFeaturesCol
  with HasLabelCol with GpuParams {

  def needDeterministicRepartitioning: Boolean = {
    getCheckpointPath != null && getCheckpointPath.nonEmpty && getCheckpointInterval > 0
  }
}

private[scala] trait XGBoostClassifierParams extends XGBoostEstimatorCommon with HasNumClass

private[scala] trait XGBoostRegressorParams extends XGBoostEstimatorCommon with HasGroupCol
