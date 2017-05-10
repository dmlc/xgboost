package com.airbnb.common.ml.strategy.eval

import org.junit.Assert.assertEquals
import org.junit.Test

import com.airbnb.common.ml.util.ScalaLogging


class BinaryMetricsTest extends ScalaLogging {

  @Test
  def testComputeEvalMetricFromCounts(): Unit = {
    val m: Map[(Boolean, Boolean), (Int, Double)] = Map((true, true) -> ((1, 0.1)))
    val b = BinaryMetrics.computeEvalMetricFromCounts(m, 0, 0)
    assertEquals(b.posCount, 1)
    assertEquals(b.posSugLower, 1)

    logger.info(b.toString)
  }
}
