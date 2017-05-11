package com.airbnb.common.ml.util

import java.util

import com.google.common.base.Optional
import org.junit.Assert._
import org.junit.Test


/**
  * This class tests [[PipelineUtil]].
  */

class PipelineUtilTest {

  @Test
  def testCheckEmptyForAny(): Unit = {
    assertTrue(PipelineUtil.countAnyNonEmpty(null) == 0) // scalastyle:off null
    assertTrue(PipelineUtil.countAnyNonEmpty({}) == 0)
    assertTrue(PipelineUtil.countAnyNonEmpty(new util.HashMap[String, Any]())  == 0)
    assertTrue(PipelineUtil.countAnyNonEmpty(Optional.absent()) == 0)

    assertTrue(PipelineUtil.countAnyNonEmpty(Optional.of(1.0)) == 1)
    assertTrue(PipelineUtil.countAnyNonEmpty({1 -> 2}) == 1)
  }
}
