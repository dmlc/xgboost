/*
 Copyright (c) 2021-2024 by Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

package ml.dmlc.xgboost4j.scala.rapids.spark

import java.nio.file.{Files, Path}
import java.util.{Locale, TimeZone}

import org.apache.spark.{GpuTestUtils, SparkConf}
import org.apache.spark.internal.Logging
import org.apache.spark.network.util.JavaUtils
import org.apache.spark.sql.SparkSession
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

trait GpuTestSuite extends AnyFunSuite with TmpFolderSuite {

  import SparkSessionHolder.withSparkSession

  protected def getResourcePath(resource: String): String = {
    require(resource.startsWith("/"), "resource must start with /")
    getClass.getResource(resource).getPath
  }

  def enableCsvConf(): SparkConf = {
    new SparkConf()
      .set("spark.rapids.sql.csv.read.float.enabled", "true")
      .set("spark.rapids.sql.csv.read.double.enabled", "true")
  }

  def withGpuSparkSession[U](conf: SparkConf = new SparkConf())(f: SparkSession => U): U = {
    // set "spark.rapids.sql.explain" to "ALL" to check if the operators
    // can be replaced by GPU
    val c = conf.clone()
      .set("spark.rapids.sql.enabled", "true")
    withSparkSession(c, f)
  }

  def withCpuSparkSession[U](conf: SparkConf = new SparkConf())(f: SparkSession => U): U = {
    val c = conf.clone()
      .set("spark.rapids.sql.enabled", "false") // Just to be sure
    withSparkSession(c, f)
  }
}

trait TmpFolderSuite extends BeforeAndAfterAll {
  self: AnyFunSuite =>
  protected var tempDir: Path = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    tempDir = Files.createTempDirectory(getClass.getName)
  }

  override def afterAll(): Unit = {
    JavaUtils.deleteRecursively(tempDir.toFile)
    super.afterAll()
  }

  protected def createTmpFolder(prefix: String): Path = {
    Files.createTempDirectory(tempDir, prefix)
  }
}

object SparkSessionHolder extends Logging {

  private var spark = createSparkSession()
  private var origConf = spark.conf.getAll
  private var origConfKeys = origConf.keys.toSet

  private def setAllConfs(confs: Array[(String, String)]): Unit = confs.foreach {
    case (key, value) if spark.conf.get(key, null) != value =>
      spark.conf.set(key, value)
    case _ => // No need to modify it
  }

  private def createSparkSession(): SparkSession = {
    GpuTestUtils.cleanupAnyExistingSession()

    // Timezone is fixed to UTC to allow timestamps to work by default
    TimeZone.setDefault(TimeZone.getTimeZone("UTC"))
    // Add Locale setting
    Locale.setDefault(Locale.US)

    val builder = SparkSession.builder()
      .master("local[2]")
      .config("spark.sql.adaptive.enabled", "false")
      .config("spark.rapids.sql.test.enabled", "false")
      .config("spark.stage.maxConsecutiveAttempts", "1")
      .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
      .config("spark.rapids.memory.gpu.pooling.enabled", "false") // Disable RMM for unit tests.
      .config("spark.sql.files.maxPartitionBytes", "1000")
      .appName("XGBoost4j-Spark-Gpu unit test")

    builder.getOrCreate()
  }

  private def reinitSession(): Unit = {
    spark = createSparkSession()
    origConf = spark.conf.getAll
    origConfKeys = origConf.keys.toSet
  }

  def sparkSession: SparkSession = {
    if (SparkSession.getActiveSession.isEmpty) {
      reinitSession()
    }
    spark
  }

  def resetSparkSessionConf(): Unit = {
    if (SparkSession.getActiveSession.isEmpty) {
      reinitSession()
    } else {
      setAllConfs(origConf.toArray)
      val currentKeys = spark.conf.getAll.keys.toSet
      val toRemove = currentKeys -- origConfKeys
      toRemove.foreach(spark.conf.unset)
    }
    logDebug(s"RESET CONF TO: ${spark.conf.getAll}")
  }

  def withSparkSession[U](conf: SparkConf, f: SparkSession => U): U = {
    resetSparkSessionConf
    logDebug(s"SETTING  CONF: ${conf.getAll.toMap}")
    setAllConfs(conf.getAll)
    logDebug(s"RUN WITH CONF: ${spark.conf.getAll}\n")
    spark.sparkContext.setLogLevel("WARN")
    f(spark)
  }
}
