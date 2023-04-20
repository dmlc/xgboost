/*
 Copyright (c) 2021-2023 by Contributors

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
import java.sql.{Date, Timestamp}
import java.util.{Locale, TimeZone}

import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.{GpuTestUtils, SparkConf}
import org.apache.spark.internal.Logging
import org.apache.spark.network.util.JavaUtils
import org.apache.spark.sql.{Row, SparkSession}

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

  def compareResults(
      sort: Boolean,
      floatEpsilon: Double,
      fromLeft: Array[Row],
      fromRight: Array[Row]): Boolean = {
    if (sort) {
      val left = fromLeft.map(_.toSeq).sortWith(seqLt)
      val right = fromRight.map(_.toSeq).sortWith(seqLt)
      compare(left, right, floatEpsilon)
    } else {
      compare(fromLeft, fromRight, floatEpsilon)
    }
  }

  // we guarantee that the types will be the same
  private def seqLt(a: Seq[Any], b: Seq[Any]): Boolean = {
    if (a.length < b.length) {
      return true
    }
    // lengths are the same
    for (i <- a.indices) {
      val v1 = a(i)
      val v2 = b(i)
      if (v1 != v2) {
        // null is always < anything but null
        if (v1 == null) {
          return true
        }

        if (v2 == null) {
          return false
        }

        (v1, v2) match {
          case (i1: Int, i2: Int) => if (i1 < i2) {
            return true
          } else if (i1 > i2) {
            return false
          }// else equal go on
          case (i1: Long, i2: Long) => if (i1 < i2) {
            return true
          } else if (i1 > i2) {
            return false
          } // else equal go on
          case (i1: Float, i2: Float) => if (i1.isNaN() && !i2.isNaN()) return false
          else if (!i1.isNaN() && i2.isNaN()) return true
          else if (i1 < i2) {
            return true
          } else if (i1 > i2) {
            return false
          } // else equal go on
          case (i1: Date, i2: Date) => if (i1.before(i2)) {
            return true
          } else if (i1.after(i2)) {
            return false
          } // else equal go on
          case (i1: Double, i2: Double) => if (i1.isNaN() && !i2.isNaN()) return false
          else if (!i1.isNaN() && i2.isNaN()) return true
          else if (i1 < i2) {
            return true
          } else if (i1 > i2) {
            return false
          } // else equal go on
          case (i1: Short, i2: Short) => if (i1 < i2) {
            return true
          } else if (i1 > i2) {
            return false
          } // else equal go on
          case (i1: Timestamp, i2: Timestamp) => if (i1.before(i2)) {
            return true
          } else if (i1.after(i2)) {
            return false
          } // else equal go on
          case (s1: String, s2: String) =>
            val cmp = s1.compareTo(s2)
            if (cmp < 0) {
              return true
            } else if (cmp > 0) {
              return false
            } // else equal go on
          case (o1, _) =>
            throw new UnsupportedOperationException(o1.getClass + " is not supported yet")
        }
      }
    }
    // They are equal...
    false
  }

  private def compare(expected: Any, actual: Any, epsilon: Double = 0.0): Boolean = {
    def doublesAreEqualWithinPercentage(expected: Double, actual: Double): (String, Boolean) = {
      if (!compare(expected, actual)) {
        if (expected != 0) {
          val v = Math.abs((expected - actual) / expected)
          (s"\n\nABS($expected - $actual) / ABS($actual) == $v is not <= $epsilon ", v <= epsilon)
        } else {
          val v = Math.abs(expected - actual)
          (s"\n\nABS($expected - $actual) == $v is not <= $epsilon ", v <= epsilon)
        }
      } else {
        ("SUCCESS", true)
      }
    }
    (expected, actual) match {
      case (a: Float, b: Float) if a.isNaN && b.isNaN => true
      case (a: Double, b: Double) if a.isNaN && b.isNaN => true
      case (null, null) => true
      case (null, _) => false
      case (_, null) => false
      case (a: Array[_], b: Array[_]) =>
        a.length == b.length && a.zip(b).forall { case (l, r) => compare(l, r, epsilon) }
      case (a: Map[_, _], b: Map[_, _]) =>
        a.size == b.size && a.keys.forall { aKey =>
          b.keys.find(bKey => compare(aKey, bKey))
            .exists(bKey => compare(a(aKey), b(bKey), epsilon))
        }
      case (a: Iterable[_], b: Iterable[_]) =>
        a.size == b.size && a.zip(b).forall { case (l, r) => compare(l, r, epsilon) }
      case (a: Product, b: Product) =>
        compare(a.productIterator.toSeq, b.productIterator.toSeq, epsilon)
      case (a: Row, b: Row) =>
        compare(a.toSeq, b.toSeq, epsilon)
      // 0.0 == -0.0, turn float/double to bits before comparison, to distinguish 0.0 and -0.0.
      case (a: Double, b: Double) if epsilon <= 0 =>
        java.lang.Double.doubleToRawLongBits(a) == java.lang.Double.doubleToRawLongBits(b)
      case (a: Double, b: Double) if epsilon > 0 =>
        val ret = doublesAreEqualWithinPercentage(a, b)
        if (!ret._2) {
          System.err.println(ret._1 + " (double)")
        }
        ret._2
      case (a: Float, b: Float) if epsilon <= 0 =>
        java.lang.Float.floatToRawIntBits(a) == java.lang.Float.floatToRawIntBits(b)
      case (a: Float, b: Float) if epsilon > 0 =>
        val ret = doublesAreEqualWithinPercentage(a, b)
        if (!ret._2) {
          System.err.println(ret._1 + " (float)")
        }
        ret._2
      case (a, b) => a == b
    }
  }

}

trait TmpFolderSuite extends BeforeAndAfterAll { self: AnyFunSuite =>
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
      .config("spark.rapids.sql.enabled", "false")
      .config("spark.rapids.sql.test.enabled", "false")
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
    f(spark)
  }

}
