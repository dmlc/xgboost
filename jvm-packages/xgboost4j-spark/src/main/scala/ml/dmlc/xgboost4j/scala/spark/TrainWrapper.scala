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

import ml.dmlc.xgboost4j.scala.spark.XGBoost.{composeInputData, trainForNonRanking, trainForRanking}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions.{col, lit}
import ml.dmlc.xgboost4j.{LabeledPoint => XGBLabeledPoint}

object TrainWrapper {

  def getTrainImpl(est: XGBoostClassifier, dataset: Dataset[_])
                  (xgbExecParams: XGBoostExecutionParams): RDD[Watches] = {
    val weight = if (!est.isDefined(est.weightCol) || est.getWeightCol.isEmpty) lit(1.0)
      else col(est.getWeightCol)
    val baseMargin = if (!est.isDefined(est.baseMarginCol) || est.getBaseMarginCol.isEmpty) {
      lit(Float.NaN)
    } else {
      col(est.getBaseMarginCol)
    }
    val trainingSet: RDD[XGBLabeledPoint] = DataUtils.convertDataFrameToXGBLabeledPointRDDs(
      col(est.getLabelCol), col(est.getFeaturesCol), weight, baseMargin, None, est.getNumWorkers,
      est.needDeterministicRepartitioning, dataset.asInstanceOf[DataFrame]).head

    val evalRDDMap = est.getEvalSets(est.xgboostParams).map {
      case (name, dataFrame) => (name,
        DataUtils.convertDataFrameToXGBLabeledPointRDDs(col(est.getLabelCol),
          col(est.getFeaturesCol), weight, baseMargin, None, est.getNumWorkers,
          est.needDeterministicRepartitioning, dataFrame).head)
    }

    val transformedTrainingData = composeInputData(trainingSet, xgbExecParams.cacheTrainingSet,
      false, est.getNumWorkers)

    trainForNonRanking(transformedTrainingData.right.get, xgbExecParams, evalRDDMap)
  }

  def getTrainImpl(est: XGBoostRegressor, dataset: Dataset[_])
                  (xgbExecParams: XGBoostExecutionParams): RDD[Watches] = {

    val weight = if (!est.isDefined(est.weightCol) || est.getWeightCol.isEmpty) lit(1.0)
        else col(est.getWeightCol)
    val baseMargin = if (!est.isDefined(est.baseMarginCol) || est.getBaseMarginCol.isEmpty) {
      lit(Float.NaN)
    } else {
      col(est.getBaseMarginCol)
    }

    val group = if (!est.isDefined(est.groupCol) || est.getGroupCol.isEmpty) lit(-1)
      else col(est.getGroupCol)

    val trainingSet: RDD[XGBLabeledPoint] = DataUtils.convertDataFrameToXGBLabeledPointRDDs(
      col(est.getLabelCol), col(est.getFeaturesCol), weight, baseMargin, Some(group),
      est.getNumWorkers, est.needDeterministicRepartitioning, dataset.asInstanceOf[DataFrame]).head

    val evalRDDMap = est.getEvalSets(est.xgboostParams).map {
      case (name, dataFrame) => (name,
        DataUtils.convertDataFrameToXGBLabeledPointRDDs(col(est.getLabelCol),
          col(est.getFeaturesCol), weight, baseMargin, Some(group), est.getNumWorkers,
          est.needDeterministicRepartitioning, dataFrame).head)
    }

    val hasGroup = group != lit(-1)

    val transformedTrainingData = composeInputData(trainingSet, xgbExecParams.cacheTrainingSet,
      hasGroup, est.getNumWorkers)

    if (hasGroup) {
      trainForRanking(transformedTrainingData.left.get, xgbExecParams, evalRDDMap)
    } else {
      trainForNonRanking(transformedTrainingData.right.get, xgbExecParams, evalRDDMap)
    }
  }
}
