package com.airbnb.common.ml.util

import com.typesafe.scalalogging.slf4j.Logger
import org.junit.Test


class ScalaLoggingTest {

  @Test def testLogger(): Unit = {
    // Create a dummy object that can check if it has a valid logger
    object Dummy extends ScalaLogging {
      def hasLogger: Boolean = {
        logger.isInstanceOf[Logger]
      }
    }

    assert(Dummy.hasLogger)
  }
}
