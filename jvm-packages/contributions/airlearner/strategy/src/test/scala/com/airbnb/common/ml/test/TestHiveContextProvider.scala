package com.airbnb.common.ml.test

import org.apache.commons.io.FileUtils
import org.apache.spark.SparkContext
import org.apache.spark.sql.hive.test.TestHiveContext


object TestHiveContextProvider {

  /**
    * Create a new TestHiveContext
    *
    * @param sc SparkContext
    * @return TestHiveContext
    */
  def createContext(sc: SparkContext): TestHiveContext = {
    new TestHiveContext(sc)
  }

  /**
    * Thoroughly clear a TestHiveContext between tests.
    *
    * @param hc HiveContext
    */
  def stopContext(hc: TestHiveContext): Unit = {
    // Clean up the leftover temp directories
    Seq(
      hc.hiveFilesTemp,
      hc.scratchDirPath,
      hc.testTempDir,
      hc.warehousePath
    ).foreach(FileUtils.deleteQuietly)
  }
}
