package com.airbnb.common.ml.util

import org.junit.Assert.assertEquals
import org.junit.Test


class DateTimeUtilTest {

  @Test
  def testQuickSelect(): Unit = {
    assertEquals(1, DateTimeUtil.daysRange("2015-01-01","2015-01-02"))
    assertEquals(366, DateTimeUtil.daysRange("2016-01-01","2017-01-01"))
  }

}
