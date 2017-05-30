package com.airbnb.common.ml.strategy.config

import scala.util.Try

import com.typesafe.config.Config

import com.airbnb.common.ml.util.ScalaLogging


case class TrainingConfig(
    trainingDataQuery: String,
    partitions: Int,
    shuffle: Boolean
)

object TrainingConfig extends ScalaLogging {

  def loadConfig[T](
      config: Config
  ): TrainingConfig = {
    val trainingDataQuery = config.getString("training_data_query")
    val partitions = Try(config.getInt("partition_num")).getOrElse(5000)
    val shuffle = Try(config.getBoolean("shuffle")).getOrElse(false)
    logger.info(s"Training Data Query: $trainingDataQuery")
    TrainingConfig(trainingDataQuery, partitions, shuffle)
  }
}
