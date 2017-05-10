package com.airbnb.common.ml.search

import org.junit.Assert._
import org.junit.Test

import com.airbnb.common.ml.util.ScalaLogging


class MonteCarloSearchTest extends ScalaLogging {

  @Test
  def testParams(): Unit = {
    val randomizer = scala.util.Random
    val paramMap = List(
      "gamma" -> List(0.1, 0.2),
      "eta" -> List(0.4, 0.6),
      "max_depth" -> List(13, 18),
      "single" -> List(2),
      "multiple" -> List(2, 3, 4),
      "objective" -> List("binary:logistic"))

    val param = MonteCarloSearch.getParams(paramMap, randomizer)
    logger.info(s"param ${param.map(x => x._2).mkString(",")}")

    val map = param.toMap
    val gamma = MonteCarloParams.toDouble(map.get("gamma").get)
    assertTrue(gamma >= 0.1 && gamma <= 0.2)

    val single = MonteCarloParams.toInt(map.get("single").get)
    assertTrue(single == 2)

    val multiple = MonteCarloParams.toInt(map.get("multiple").get)
    assertTrue(multiple >= 2 && multiple <= 4)

    val str = map.get("objective").get
    assertTrue(str == "binary:logistic")

    val stableParamMap = List(
      "silent" -> 1,
      "round" -> 10,
      "objective" -> "binary:logistic").toMap
    val round = stableParamMap("round").asInstanceOf[Int]
    logger.info(s"round $round")
  }
}
