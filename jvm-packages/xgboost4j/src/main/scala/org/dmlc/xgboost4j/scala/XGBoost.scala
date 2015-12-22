package org.dmlc.xgboost4j.scala

import _root_.scala.collection.JavaConverters._

import org.dmlc.xgboost4j
import org.dmlc.xgboost4j.{XGBoost => JXGBoost, IEvaluation, IObjective}

object XGBoost {

  def train(params: Map[String, AnyRef], dtrain: xgboost4j.DMatrix, round: Int,
            watches: Map[String, xgboost4j.DMatrix], obj: IObjective, eval: IEvaluation): Booster = {
    val xgboostInJava = JXGBoost.train(params.asJava, dtrain, round, watches.asJava, obj, eval)
    new ScalaBoosterImpl(xgboostInJava)
  }

  def crossValiation(params: Map[String, AnyRef],
                     data: DMatrix,
                     round: Int,
                     nfold: Int,
                     metrics: Array[String],
                     obj: IObjective,
                     eval: IEvaluation): Array[String] = {
    JXGBoost.crossValiation(params.asJava, data.jDMatrix, round, nfold, metrics, obj,
      eval)
  }

  def initBoostModel(params: Map[String, AnyRef],  dMatrixs: Array[DMatrix]): Booster = {
    val xgboostInJava = JXGBoost.initBoostingModel(params.asJava, dMatrixs.map(_.jDMatrix))
    new ScalaBoosterImpl(xgboostInJava)
  }

  def loadBoostModel(params: Map[String, AnyRef], modelPath: String): Booster = {
    val xgboostInJava = JXGBoost.loadBoostModel(params.asJava, modelPath)
    new ScalaBoosterImpl(xgboostInJava)
  }
}
