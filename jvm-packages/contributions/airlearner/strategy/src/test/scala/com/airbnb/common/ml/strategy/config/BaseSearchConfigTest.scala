package com.airbnb.common.ml.strategy.config

import org.junit.Test

import com.airbnb.common.ml.util.ScalaLogging


class BaseSearchConfigTest extends ScalaLogging {
  @Test
  def testGetTrainingOptions(): Unit = {
    val trainingOptions = TrainingOptionsTest.getOption
    val p1 = SearchParamsTest.getDoubleParams
    val r = BaseSearchConfig.getTrainingOptions(p1, trainingOptions)
    logger.info(s"Training models with ${r.mkString(",")}")
  }
}
