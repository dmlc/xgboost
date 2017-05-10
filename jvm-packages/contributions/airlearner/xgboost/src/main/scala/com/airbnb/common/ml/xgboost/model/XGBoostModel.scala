package com.airbnb.common.ml.xgboost.model

import scala.collection.mutable

import ml.dmlc.xgboost4j.LabeledPoint
import ml.dmlc.xgboost4j.scala.{DMatrix, XGBoost}

import com.airbnb.common.ml.util.{PipelineUtil, ScalaLogging}
import com.airbnb.common.ml.search.MonteCarloSearch


class XGBoostModel(training: DMatrix, eval: DMatrix)
  extends MonteCarloSearch {

  override def eval(params: Map[String, Any]): Double = {
    val model = XGBoostModel.train(params, training)
    val prediction = model.predict(eval)
    val loss = eval.getLabel.zip(prediction).map(
      a => {
        math.abs(logLoss(a._2(0), a._1))
      }
    ).sum / prediction.length
    model.dispose
    loss
  }

  def logLoss(p: Double, y: Double): Double = {
    -y * math.log(p) - (1 - y) * math.log(1 - p)
  }

  override def dispose(): Unit = {
    training.delete()
    eval.delete()
  }
}

object XGBoostModel extends ScalaLogging {

  def getModelByLabeledPoint(
      training: Seq[LabeledPoint],
      eval: Seq[LabeledPoint]
  ): XGBoostModel = {
    val trainingDMatrix = new DMatrix(training.iterator, null)
    val evalDMatrix = new DMatrix(eval.iterator, null)
    new XGBoostModel(trainingDMatrix, evalDMatrix)
  }

  def getModelByFile(
      training: String,
      eval: String
  ): XGBoostModel = {
    val trainingDMatrix = new DMatrix(training)
    val evalDMatrix = new DMatrix(eval)
    new XGBoostModel(trainingDMatrix, evalDMatrix)
  }

  def train(params: Map[String, Any], training: DMatrix): ml.dmlc.xgboost4j.scala.Booster = {
    val round = params("round").asInstanceOf[Int]
    XGBoost.train(training, params, round)
  }

  def trainAndSave(
      training: DMatrix,
      output: String,
      round: Int,
      paramMap: Map[String, Any]
  ): Unit = {
    // log train-error
    val watches = new mutable.HashMap[String, DMatrix]
    watches += "train" -> training
    logger.info(s"xgbtraining start $output")
    val model = XGBoost.train(training, paramMap, round, watches.toMap)
    // save model to the file.
    val stream = PipelineUtil.getHDFSOutputStream(output)
    model.saveModel(stream)
    model.dispose
    training.delete()
  }
}
