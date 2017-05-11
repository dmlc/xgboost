package com.airbnb.common.ml.xgboost.data

import ml.dmlc.xgboost4j.LabeledPoint
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.{DataFrame, Row}

import com.airbnb.common.ml.util.HiveUtil

object ModelData {
  def parseRowToLabelAndVaue(row: Row, labelPos: Int, dropCount: Int): (Float, Array[Float]) = {
    // for row doesn't contains label, pass in negative labelPos
    val label: Number = if (labelPos >= 0) {
      row.getAs[Number](labelPos)
    } else {
      0
    }
    val seq = row.toSeq.drop(dropCount).map(x=>{
      if (x != null) {
        x.asInstanceOf[Number].floatValue()
      } else {
        ModelData.NULL_VALUE
      }
    }).toArray
    (label.floatValue(), seq)
  }

  def parseRowToRawXgboostLabeledPoint(row: Row, labelPos: Int, dropCount: Int): LabeledPoint = {
    val (label, seq) = parseRowToLabelAndVaue(row, labelPos, dropCount)
    LabeledPoint.fromDenseVector(label, seq)
  }

  def getLabeledPoints(
    sc: SparkContext,
    query: String,
    trainingLabeledPoint: TrainingModelData
  ): RDD[(String, Seq[LabeledPoint])] = {
    val df = ModelData.getDataFrame(sc, query)
    HiveUtil.loadDataFromDataFrameGroupByKey(
      df,
      ModelData.parseKeyFromHiveRow(ModelData.TRAINING_KEY_INDEX),
      trainingLabeledPoint.parseRowToXgboostLabeledPoint)
  }

  def getScoringLabeledPoints(sc: SparkContext,
                              query: String, scoringLabeledPoint: ScoringModelData): RDD[(String, ScoringLabeledPoint)] = {
    val df = ModelData.getDataFrame(sc, query)
    HiveUtil.loadDataFromDataFrame(
      df,
      // score_query_head of scoring.conf also defined S_node_10k_id same as TRAINING_KEY_INDEX
      ModelData.parseKeyFromHiveRow(ModelData.TRAINING_KEY_INDEX),
      scoringLabeledPoint.parseRowToXgboostLabeledPointAndData)
  }

  def getLabeledPointsAndString(sc: SparkContext,
                                query: String, scoringLabeledPoint: ScoringModelData): RDD[(String, Seq[ScoringLabeledPoint])] = {
    val df = ModelData.getDataFrame(sc, query)
    HiveUtil.loadDataFromDataFrameGroupByKey(
      df,
      // score_query_head of scoring.conf also defined S_node_10k_id same as TRAINING_KEY_INDEX
      ModelData.parseKeyFromHiveRow(ModelData.TRAINING_KEY_INDEX),
      scoringLabeledPoint.parseRowToXgboostLabeledPointAndData)
  }

  // parseRowToLabelAndVaue can't return null, so use -1 if input is null
  // this is same with train_with_prob.conf 's hql query.
  val NULL_VALUE: Int = -1
  // refer to query in xgboost/search.conf
  val TRAINING_KEY_INDEX: Int = 1
  // refer to pricing.grid_search
  val PARAMS_KEY_INDEX: Int = 0
  val PARAMS_INDEX: Int = 1

  // default use second field as node key
  def parseKeyFromHiveRow(keyIndex: Int)(row: Row): String= {
    row.getAs[Long](keyIndex).toString
  }

  def getParams(sc: SparkContext,
                query: String): RDD[(String, Seq[Array[Double]])] = {
    val df = getDataFrame(sc, query)
    HiveUtil.loadDataFromDataFrameGroupByKey(
      df,
      ModelData.parseKeyFromHiveRow(0),
      parseRowToParams)
  }

  def parseRowToParams(row: Row): Array[Double] = {
    row.getAs[scala.collection.mutable.WrappedArray[Double]](ModelData.PARAMS_INDEX).toArray
  }

  def getDataFrame(sc: SparkContext,
                   query: String): DataFrame = {
    val hc = new HiveContext(sc)
    hc.sql(query)
  }
}
