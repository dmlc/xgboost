package com.airbnb.common.ml.test

import org.apache.spark.SparkContext
import org.junit.{After, Before}


trait TestWithSparkContext {

  @transient private var _sc: Option[SparkContext] = None

  // Can't be called before `initSparkContext()`
  def sc: SparkContext = {
    _sc.get
  }

  @Before
  def initContexts(): Unit = {
    initSparkContext()
  }

  @After
  def cleanupContexts(): Unit = {
    cleanupSparkContext()
  }

  protected def initSparkContext(): Unit = {
    _sc = Some(TestSparkContextProvider.createContext())
  }

  protected def cleanupSparkContext(): Unit = {
    TestSparkContextProvider.stopContext(sc)
    _sc = None
  }
}
