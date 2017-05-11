package com.airbnb.common.ml.util

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{Row, SaveMode}

trait HiveManageTable {
  def toRow(partition: String): Row
}

object HiveManageTable {
  def saveRDDToHive[T <: HiveManageTable](hiveContext: HiveContext,
                                          data: RDD[T],
                                          table: String,
                                          schema: StructType,
                                          mode: SaveMode,
                                          partition: String,
                                          partitionValue: String,
                                          hiveConfig: Map[String, String] = dynamicPartitions):Unit = {
    hiveConfig.foreach {
      case (key, value) =>
        hiveContext.setConf(key, value)
    }

    hiveContext.createDataFrame(data.map(_.toRow(partitionValue)), schema)
      .write
      .mode(mode)
      .partitionBy(partition)
      .insertInto(table)
  }

  lazy val dynamicPartitions = Map(
    "hive.exec.dynamic.partition" -> "true",
    "hive.exec.dynamic.partition.mode" -> "nonstrict"
  )
}
