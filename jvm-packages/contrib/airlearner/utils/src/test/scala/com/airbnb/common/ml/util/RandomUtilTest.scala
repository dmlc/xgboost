package com.airbnb.common.ml.util

import org.junit.Assert.assertEquals
import org.junit.Test


class RandomUtilTest {

  @Test
  def evalSlice(): Unit = {
    val list = Seq.range(1, 101)
    val ratios = Seq(0.85, 0.1, 0.05)

    val r = RandomUtil.slice(list, ratios)

    assertEquals(85, r.head.last)
    assertEquals(86, r(1).head)
    assertEquals(96, r(2).head)
    assertEquals(100, r(2).last)
  }
}
