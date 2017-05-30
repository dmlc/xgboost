package com.airbnb.common.ml.strategy.trainer

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.hive.HiveContext

import com.airbnb.common.ml.strategy.config.TrainingOptions
import com.airbnb.common.ml.strategy.data.{BaseBinarySample, TrainingData}
import com.airbnb.common.ml.strategy.params.StrategyParams


case class BaseBinaryRegressionTrainer(
    strategyParams: StrategyParams[BaseBinarySample],
    trainingDataType: TrainingData[BaseBinarySample]
) extends BinaryRegressionTrainer[BaseBinarySample] {

  override def getLearningRate(
      r0: Double,
      r1: Double,
      example: BaseBinarySample,
      options: TrainingOptions
  ): Double = {
    val x = example.x
    val learningRate = if (example.label) {
      r1 * x
    } else {
      1 - x
    }
    r0 * learningRate
  }

  override def createDataFrameFromModelOutput(
      models: RDD[(String,
        StrategyParams[BaseBinarySample])], hc: HiveContext
  ): DataFrame = {
    ???
  }
}
