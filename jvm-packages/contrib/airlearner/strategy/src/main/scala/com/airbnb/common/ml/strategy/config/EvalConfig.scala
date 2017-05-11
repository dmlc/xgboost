package com.airbnb.common.ml.strategy.config

import scala.language.existentials

import com.typesafe.config.Config

import com.airbnb.common.ml.strategy.data.TrainingData
import com.airbnb.common.ml.util.ScalaLogging


case class EvalConfig(
    trainingConfig: TrainingConfig,
    evalDataQuery: String,
    holdoutDataQuery: String
)

object DirectQueryEvalConfig extends ScalaLogging {

  def loadConfig[T](
      config: Config
  ): EvalConfig = {
    val evalDataQuery = config.getString("eval_data_query")
    val holdoutDataQuery = config.getString("holdout_data_query")

    logger.info(s"Eval Data Query: $evalDataQuery")

    EvalConfig(
      TrainingConfig.loadConfig(config),
      evalDataQuery,
      holdoutDataQuery)
  }
}

