package com.airbnb.common.ml.xgboost

import java.lang.{Double => JDouble}
import java.util.{Arrays => JArrays, Comparator => JComparator}

import scala.collection.mutable
import scala.language.postfixOps

import ml.dmlc.xgboost4j.scala.{DMatrix, XGBoost}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

import com.airbnb.common.ml.util.{PipelineUtil, ScalaLogging}
import com.airbnb.common.ml.xgboost.data.{ModelData, ScoringModelData}


object XGBoostEvalPipeline extends ScalaLogging {

  val evalFile = "eval"

  // modelData: RDD[(modelId, score, label)
  def perModelStats(modelData: RDD[(String, Double, Double)]) = {
    modelData.groupBy(_._1).map { case (modelId, scoreLabelData) => {
      val scoreLabel = scoreLabelData.map(item => (item._2, item._3)).toArray
      val evaluateResult = evaluate(scoreLabel)
      val auROC = evaluateResult._1
      val totalPositiveCount = evaluateResult._2
      val totalNegativeCount = evaluateResult._3
      val mse = scoreLabel.map(x => math.pow(x._1 - x._2, 2)).sum / scoreLabel.length
      val mae = scoreLabel.map(x => math.abs(x._1 - x._2)).sum / scoreLabel.length
      Seq(modelId, auROC, mse, mae, totalPositiveCount, totalNegativeCount).mkString("\t")
    }
    }
  }

  def aggregateModelStats(modelData: RDD[(String, Double, Double)]) = {
    val predictionAndLabels = modelData.map({
      case (modelId, score, label) => {
        (score, label)
      }
    })
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
    val auPRC = metrics.areaUnderPR
    val totalCounts = predictionAndLabels.count()
    val totalPositiveCount = predictionAndLabels.filter(_._2 > 0).count()
    val totalNegativeCount = totalCounts - totalPositiveCount
    val mse = predictionAndLabels.map(x => math.pow(x._1 - x._2, 2)).sum / totalCounts
    val mae = predictionAndLabels.map(x => math.abs(x._1 - x._2)).sum / totalCounts
    Seq("aggregate", auPRC, mse, mae, totalPositiveCount, totalNegativeCount).mkString("\t")
  }

  def getScoreFromHive(sc: SparkContext, query:String, modelId: Int, scoreIndex: Int, labelIndex: Int) = {
    val rawData = ModelData.getDataFrame(sc, query).map(
      row => {
        //id, label, score
        (row.getAs[String](modelId), row.getDouble(labelIndex), row.getDouble(scoreIndex))
      })
    rawData
  }

  def getScoreFromXgboost(train_model_path: String, id: String,
                          eval_data_path: String, evalFile: String) = {
      var localEvalData = ""
      try {
        val inputStreamModel = PipelineUtil.getHDFSInputStream(train_model_path + id)
        val model = XGBoost.loadModel(inputStreamModel)
        localEvalData = XGBoostPipeline.getDataFromHDFS(
          eval_data_path, id, evalFile,
          XGBoostPipeline.defaultTmpFolder)
        val scoring = new DMatrix(localEvalData)
        val predictionOutput = model.predict(scoring)
        val predictionArray = predictionOutput.flatten.map(x => x.toDouble)
        val labelArray = scoring.getLabel.map(x => x.toDouble)
        val predictionLabel = predictionArray.zip(labelArray)
        val rawData = predictionLabel.map(x => (id, x._1, x._2))
        rawData
      } finally {
        // del on exit is not enough since one executor may run multiple files in its life
        Utils.delFile(localEvalData)
      }
  }

  def getDataPointScoreFromXgboost(sc: SparkContext, query: String, scoringModelData: ScoringModelData,
                                   train_model_path: String, modelId: Int, scoreIndex: Int, labelIndex: Int) = {
    val rawData = ModelData.getLabeledPointsAndString(
      sc,
      query,
      scoringModelData)

    val output = rawData.flatMap {
      case (id, scoringData) => {
        val data = scoringData.map(x => x.labeledPoint)
        val prediction = XGBoostScoringPipeline.scoreWithPath(train_model_path + id, data)
        prediction.zip(scoringData.map(x => x.data)).map {
          case (score, index) => {
            s"$index\t${score(0)}"
          }
        }
      }
    }
    output.map { x => {
      val words = x.split("\t")
      val id = words(modelId)
      val label = words(labelIndex).toDouble
      val prediction = words(scoreIndex).toDouble
      (id, prediction, label)
      }
    }
  }

   def xgBoostFeatureImportance(train_model_path: String, id: String) = {
     val inputStreamModel = PipelineUtil.getHDFSInputStream(train_model_path + id)
     val model = XGBoost.loadModel(inputStreamModel)
     val featureScore: mutable.Map[String, Integer] = model.getFeatureScore()
     featureScore.toSeq.map {
       case (feature, score) => s"$id\t${feature}\t${score}"
      }.iterator
   }

  private def evaluate(scoreLabel: Array[(Double, Double)]) = {
    val comparator = new JComparator[(Double, Double)]() {
      override def compare(tuple1: (Double, Double), tuple2: (Double, Double)): Int = {
        JDouble.compare(tuple1._1, tuple2._1)
      }
    }
    JArrays.sort(scoreLabel, comparator.reversed())

    var rawAUC = 0.0
    var totalPositiveCount = 0.0
    var totalNegativeCount = 0.0
    var currentPositiveCount = 0.0
    var currentNegativeCount = 0.0
    var previousScore = scoreLabel.head._1

    scoreLabel.foreach { case (score, label) => {
      if (score != previousScore) {
        rawAUC += totalPositiveCount * currentNegativeCount + currentPositiveCount *
          currentNegativeCount / 2.0
        totalPositiveCount += currentPositiveCount
        totalNegativeCount += currentNegativeCount
        currentPositiveCount = 0.0
        currentNegativeCount = 0.0
      }
      if (label > 0.5) {
        currentPositiveCount += 1.0
      } else {
        currentNegativeCount += 1.0
      }
      previousScore = score
    }
    }
    rawAUC += totalPositiveCount * currentNegativeCount + currentPositiveCount *
      currentNegativeCount / 2.0
    totalPositiveCount += currentPositiveCount
    totalNegativeCount += currentNegativeCount
    //normalize AUC over the total area
    (rawAUC / (totalPositiveCount * totalNegativeCount), totalPositiveCount, totalNegativeCount)
  }
}
