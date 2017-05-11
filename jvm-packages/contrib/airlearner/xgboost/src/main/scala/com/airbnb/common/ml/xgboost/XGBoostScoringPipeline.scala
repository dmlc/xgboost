package com.airbnb.common.ml.xgboost

import com.typesafe.config.Config
import ml.dmlc.xgboost4j.LabeledPoint
import ml.dmlc.xgboost4j.scala.{DMatrix, XGBoost}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.hive.HiveContext

import com.airbnb.common.ml.util.{PipelineUtil, ScalaLogging}
import com.airbnb.common.ml.xgboost.config.XGBoostScoringConfig
import com.airbnb.common.ml.xgboost.data.{ModelData, ScoringLabeledPoint, ScoringModelData}


object XGBoostScoringPipeline extends ScalaLogging {
  def loadModels(path: String): Map[String, ml.dmlc.xgboost4j.scala.Booster] = {
    val streams = PipelineUtil.getHDFSInputStreams(path)
    streams.map {
      case (id, stream) => {
        (id, XGBoost.loadModel(stream))
      }
    }.toMap
  }

  def scorePartition(
      path: String,
      groupNumber: Int
  ) (iter: Iterator[(String, ScoringLabeledPoint)]): Iterator[String] = {
    iter.grouped(groupNumber).flatMap { seq =>
      seq.groupBy(_._1).flatMap {
        case (id, scoringData) => {
          val data = scoringData.map(x => x._2.labeledPoint)
          val model_path = path + id
          val prediction = scoreWithPath(model_path, data)
          prediction.zip(scoringData.map(x => x._2.data)).map {
            case (score, index) => {
              s"$index\t${score(0)}"
            }
          }
        }
      }.iterator
    }
  }

  def baseScore(
      modelData: ScoringModelData,
      sc: SparkContext,
      config: Config
  ): Unit = {
    // read training data, available at xgboost/demo/data
    val conf = XGBoostScoringConfig.loadConfig(sc, config)
    val data = ModelData.getScoringLabeledPoints(
      sc,
      conf.query,
      modelData)
    val output: RDD[String] = data.mapPartitions(scorePartition(conf.modelBasePath, conf
      .groupNumber))

    val hc = new HiveContext(sc)
    PipelineUtil.saveToHdfsAndUpdateHive(
      hc, conf.outputPath, conf.outputTable, conf.partitionSpec, output, conf.overwrite)
  }

  def scoreWithPath(
      path: String,
      data: Seq[LabeledPoint]
  ): Array[Array[Float]] = {
    logger.info(s"model_path: $path")
    val scoring = new DMatrix(data.iterator, null)
    val inputStream = PipelineUtil.getHDFSInputStream(path)
    val model = XGBoost.loadModel(inputStream)
    // output array[[score],[score],...]
    val output = model.predict(scoring)
    scoring.delete()
    model.dispose
    assert(data.length == output.length)
    output
  }

}
