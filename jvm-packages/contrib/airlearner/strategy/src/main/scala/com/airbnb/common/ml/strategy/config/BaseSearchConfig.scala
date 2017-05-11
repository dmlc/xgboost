package com.airbnb.common.ml.strategy.config

import scala.util.Try

import com.typesafe.config.Config
import scala.collection.JavaConverters._

import com.airbnb.common.ml.util.ScalaLogging


case class BaseSearchConfig(
    searchParams: SearchParams[Double],
    table: String,
    partition: String,
    // ratio for training, eval and holdout.
    sampleRatio: List[Double],
    trainingOptions: TrainingOptions
) {

  def getTrainingOptions: Array[TrainingOptions] = {
    BaseSearchConfig.getTrainingOptions(searchParams, trainingOptions)
  }
}

object BaseSearchConfig extends ScalaLogging {

  def loadConfig(config: Config): BaseSearchConfig = {
    val taskConfig = config.getConfig("param_search")
    val searchParams = SearchParams.loadDoubleFromConfig(taskConfig)
    logger.info(s" params: ${searchParams.paramNames.mkString(",")}")

    val table = taskConfig.getString("table")
    val partition = taskConfig.getString("partition")
    val trainingOptions = TrainingOptions.loadBaseTrainingOptions(
      config.getConfig("training_options"))

    // use by paramSearchPerModel
    val sampleRatio: List[Double] =
      Try(config.getDoubleList("sampleRatio").asScala.map(_.doubleValue()).toList).
      getOrElse(List(0.85, 0.1, 0.05))

    BaseSearchConfig(
      searchParams,
      table,
      partition,
      sampleRatio,
      trainingOptions)
  }

  def getTrainingOptions(
      searchParams: SearchParams[Double],
      trainingOptions: TrainingOptions
  ): Array[TrainingOptions] = {
    searchParams.paramCombinations.map(currentParams =>
      trainingOptions.updateTrainingOptions(searchParams.paramNames, currentParams)
    ).toArray
  }
}
