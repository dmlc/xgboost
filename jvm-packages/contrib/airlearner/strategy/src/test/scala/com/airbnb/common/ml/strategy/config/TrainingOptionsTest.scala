package com.airbnb.common.ml.strategy.config

import org.junit.Assert.{assertEquals, assertTrue}
import org.junit.Test


class TrainingOptionsTest {

  @Test
  def testToArray(): Unit = {
    val opt1 = TrainingOptionsTest.getOption
    val array = opt1.toArray

    val opt2 = TrainingOptions.fromArray(array, "2016-08-31", "learning")
    assertEquals(opt1.trueLowerBound, opt2.trueLowerBound, 0.0001)
    assertEquals(opt1.falseUpperBound, opt2.falseUpperBound, 0.0001)
    assertEquals(opt1.falseLowerBound, opt2.falseLowerBound, 0.0001)
    assertEquals(opt1.trueUpperBound, opt2.trueUpperBound, 0.0001)
    assertEquals(opt1.r1, opt2.r1, 0.0001)
    assertEquals(opt1.r0, opt2.r0, 0.0001)
    assertEquals(opt1.numEpochs, opt2.numEpochs)
    assertEquals(opt1.miniBatchSize, opt2.miniBatchSize)
    assertEquals(opt1.rateDecay, opt2.rateDecay, 0.0001)
    assertEquals(opt1.evalRatio, opt2.evalRatio, 0.0001)
    assertEquals(opt1.maxAvgLossRatio, opt2.maxAvgLossRatio, 0.0001)
    assertEquals(opt1.minTrueLabelCount, opt2.minTrueLabelCount)

    assertTrue(s" ${opt1.min} ${opt2.min} ", opt1.min == opt2.min)
    assertTrue(opt1.max == opt2.max)
    assertTrue(opt1.default == opt2.default)
  }

  @Test
  def testDefaultOrNewValue(): Unit = {
    val pn = List("lowerBound", "upperBound")
    val pv = List(1.03, 3.0)
    val t = TrainingOptions.defaultOrNewValue(
      100.0, pn, pv, "lowerBound")
    assertEquals(t, 1.03, 0.01)

    val t1 = TrainingOptions.defaultOrNewValue(
      100.0, pn, pv, "lower_bound")
    assertEquals(t1, 100, 0.01)
  }
}

object TrainingOptionsTest {
  val getOption: TrainingOptions = {
    TrainingOptions(
      // trueLowerBound, falseUpperBound,
      0.98, 1.0,
      // falseLowerBound trueUpperBound
      0.65, 1.03,
      // r0
      0.1, 0.2, 0.95,
      //numEpochs: Int, miniBatchSize: Int,
      100, 50,
      0.5,
      2, 0.1,
      //min: Array[Double], max: Array[Double],
      List(0.1, 0.0), List(3.0, 3.0),
      List(2.0, 3.0),
      //dsEval: String, learningRateType: String
      "2016-08-31", "learning")
  }

}
