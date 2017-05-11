package com.airbnb.common.ml.strategy.data

import org.apache.spark.sql.Row


case class BaseBinarySample(
    label: Boolean,
    x: Double,
    basePivot: Double,
    observedValue: Double,
    trueValue: Option[Double]
) extends BinaryTrainingSample {

  override def scoringPivot: Double = {
    basePivot
  }
}

object BaseBinarySample extends TrainingData[BaseBinarySample] {

  override def parseSampleFromHiveRow(row: Row): BaseBinarySample = ???

  override def selectData: String = ???

  override def parseKeyFromHiveRow(row: Row): String = {
    row.getAs[Long]("id_listing").toString
  }
}
