package com.airbnb.common.ml.strategy.config

import com.typesafe.config.{Config, ConfigFactory}
import org.junit.Assert.assertEquals
import org.junit.Test


class StrategyModelSearchConfigTest {

  @Test
  def testTrainingOptions(): Unit = {
    val paramNames = List("min","max")
    val paramValues = List(1,2)
    assertEquals(1, TrainingOptions.defaultOrNewValue(2, paramNames, paramValues, "min"))

    assertEquals(3, TrainingOptions.defaultOrNewValue(3, paramNames, paramValues, "xxx"))

    assertEquals(2, TrainingOptions.defaultOrNewValue(5, paramNames, paramValues, "max"))
  }

  def makeDoubleConfig: Config = {
    val config = """
                   |param_search {
                   |  params: ["r1"]
                   |  r1: [0.65, 0.7, 0.75]
                   |  upper_bound: [1.03, 1.05, 1.08]
                   |}
                 """.stripMargin
    ConfigFactory.parseString(config)
  }
}
