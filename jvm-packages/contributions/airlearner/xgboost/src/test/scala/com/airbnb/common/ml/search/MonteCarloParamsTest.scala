package com.airbnb.common.ml.search

import com.typesafe.config.ConfigFactory
import org.junit.Assert._
import org.junit.Test

import com.airbnb.common.ml.util.ScalaLogging


class MonteCarloParamsTest extends ScalaLogging {

  def makeConfig(): String = {
    """
      |params {
      |  double_params : ["eta", "gamma"]
      |  int_params : ["round", "max_depth"]
      |  eta: [0.0, 0.2]
      |  gamma: [0.0, 0.3]
      |  round: [50, 200]
      |  max_depth: [3, 5]
      |}
    """.stripMargin
  }

  @Test
  def testParams(): Unit = {
    val config = ConfigFactory.parseString(makeConfig()).getConfig("params")

    val p = MonteCarloParams.loadFromConfig(config)
    logger.info(s"MonteCarloParams $p")
    assertEquals("MonteCarloParams number ", 4, p.length)
  }

  @Test
  def testRange(): Unit = {
    val l1: List[Int] = List(3, 5)
    val x = MonteCarloParams.getRange(4, l1, 0.2)
    assertEquals(1, x.length)
    assertEquals(4, x.head)

    val y = MonteCarloParams.getRange(200, List(100, 600), 0.1)
    assertEquals(2, y.length)
    assertEquals(150, y.head)
    assertEquals(250, y(1))

    val z = MonteCarloParams.getRange(1.0, List(0.5d, 1.5d), 0.1)
    assertEquals(2, z.length)
    assertEquals(0.9, z.head)
    assertEquals(1.1, z(1))

    val a = MonteCarloParams.getRange(1.1, List(0.5d, 1.5d), 0.5).asInstanceOf[List[Double]]
    assertEquals(2, a.length)
    assertEquals(0.6d, a.head, 0.001d)
    assertEquals(1.5, a(1), 0.001d)

    val b = MonteCarloParams.getRange(0.9, List(0.5d, 1.5d), 0.5).asInstanceOf[List[Double]]
    assertEquals(2, b.length)
    assertEquals(0.5, b.head, 0.001d)
    assertEquals(1.4, b(1), 0.001d)
  }
}
