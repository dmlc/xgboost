package com.airbnb.common.ml.strategy.trainer

import org.junit.Test

import com.airbnb.common.ml.strategy.data.BinarySampleMockUtil
import com.airbnb.common.ml.strategy.eval.BinaryMetrics
import com.airbnb.common.ml.strategy.params.BaseParam
import com.airbnb.common.ml.util.ScalaLogging


class BinaryRegressionTrainerTest
  extends ScalaLogging {

  @Test
  def evalExample(): Unit = {
    val params = BaseParam.getDefault
    val examples = BinarySampleMockUtil.getKnownSamples
    examples.foreach(e => {
      val score = params.score(e)
      val result = BinaryRegressionTrainer.evalExample(e, params, 0.5)
      logger.info(s"score $score value ${e.observedValue} result $result")
    })
  }

  @Test
  def getMetrics(): Unit = {
    val params = BaseParam.getDefault
    val examples = BinarySampleMockUtil.getKnownSamples
    val m = BinaryRegressionTrainer.getMetrics(examples, params)

    logger.debug(s"metrics ${BinaryMetrics.metricsHeader}")
    logger.debug(s"metrics ${m.toTSVRow}")
  }

}
