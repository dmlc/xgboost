package com.airbnb.common.ml.strategy.data

import java.io.Serializable

import scala.reflect.ClassTag

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.{DataFrame, Row}

import com.airbnb.common.ml.util.HiveUtil


trait TrainingData[+T] extends Serializable {
  def parseSampleFromHiveRow(row: Row): T

  // Long VS String: Long improve join performance
  // for both space efficiency
  // and comparator runtime efficiency.
  // String is more universal
  def parseKeyFromHiveRow(row: Row): String

  /**
    * Load data, grouping it by id key
    *
    * @param data DataFrame to group and process
    * @return (id, [sample])
    */
  def loadDataFromDataFrame[U >: T](
      data: DataFrame
  )(implicit c: ClassTag[U]): RDD[(String, Seq[U])] = {
    HiveUtil.loadDataFromDataFrameGroupByKey(
      data,
      parseKeyFromHiveRow,
      parseSampleFromHiveRow)
  }

  /**
    * Load data from hive
    *
    * @param hc        HiveContext
    * @param dataQuery HQL query for loading data
    * @return (id, Seq[sample])
    */
  def loadDataFromHive[U >: T](
      hc: HiveContext,
      dataQuery: String
  )(implicit c: ClassTag[U]): RDD[(String, Seq[U])] = {
    loadDataFromDataFrame(hc.sql(dataQuery))
  }

  def loadScoringDataFromHive[U >: T](
      hc: HiveContext,
      dataQuery: String
  )(implicit c: ClassTag[U]):  RDD[(String, U)] = {
    HiveUtil.loadDataFromHive(hc, dataQuery, parseKeyFromHiveRow, parseSampleFromHiveRow)
  }


  def selectData: String
}
