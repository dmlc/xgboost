package com.airbnb.common.ml.test

import org.apache.spark.sql.hive.test.TestHiveContext
import org.junit.{After, Before}


trait TestWithHiveContext
  extends TestWithSparkContext {

  @transient private var _hc: Option[TestHiveContext] = None

  // Can't be called before `initHiveContext()`
  def hc: TestHiveContext = _hc.get

  @Before
  override def initContexts(): Unit = {
    initSparkContext()
    initHiveContext()
  }

  @After
  override def cleanupContexts(): Unit = {
    cleanupSparkContext()
    cleanupHiveContext()
  }

  protected def initHiveContext(): Unit = {
    _hc = Some(TestHiveContextProvider.createContext(sc))
  }

  protected def cleanupHiveContext(): Unit = {
    TestHiveContextProvider.stopContext(hc)
    _hc = None
  }
}
