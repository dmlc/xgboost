package com.airbnb.common.ml.xgboost

import org.junit.Test

import com.airbnb.common.ml.util.ScalaLogging


class XGBoostPipelineTest extends ScalaLogging {

  @Test
  def testParams(): Unit = {
    val params: Array[Double] = Array(0.1, 0.2, 10.0, 2.0, 0.3, 0.4, 5.0, 0.6, 0.7)
    val map = XGBoostPipeline.getParamMap(params)
    logger.info(map.toString())
  }
}
