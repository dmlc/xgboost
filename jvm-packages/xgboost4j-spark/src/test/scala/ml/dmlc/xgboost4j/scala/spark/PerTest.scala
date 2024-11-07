/*
 Copyright (c) 2014-2024 by Contributors

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

package ml.dmlc.xgboost4j.scala.spark

import java.io.{File, FileInputStream}

import org.apache.commons.io.IOUtils
import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql._
import org.scalatest.BeforeAndAfterEach
import org.scalatest.funsuite.AnyFunSuite

import ml.dmlc.xgboost4j.{LabeledPoint => XGBLabeledPoint}
import ml.dmlc.xgboost4j.scala.spark.Utils.{withResource, XGBLabeledPointFeatures}

trait PerTest extends BeforeAndAfterEach {
  self: AnyFunSuite =>

  protected val numWorkers: Int = 4

  @transient private var currentSession: SparkSession = _

  def ss: SparkSession = getOrCreateSession

  implicit def sc: SparkContext = ss.sparkContext

  protected def sparkSessionBuilder: SparkSession.Builder = SparkSession.builder()
    .master(s"local[${numWorkers}]")
    .appName("XGBoostSuite")
    .config("spark.ui.enabled", false)
    .config("spark.driver.memory", "512m")
    .config("spark.barrier.sync.timeout", 10)
    .config("spark.task.cpus", 1)
    .config("spark.stage.maxConsecutiveAttempts", 1)

  override def beforeEach(): Unit = getOrCreateSession

  override def afterEach(): Unit = {
    if (currentSession != null) {
      currentSession.stop()
      cleanExternalCache(currentSession.sparkContext.appName)
      currentSession = null
    }
  }

  private def getOrCreateSession = synchronized {
    if (currentSession == null) {
      currentSession = sparkSessionBuilder.getOrCreate()
      currentSession.sparkContext.setLogLevel("ERROR")
    }
    currentSession
  }

  private def cleanExternalCache(prefix: String): Unit = {
    val dir = new File(".")
    for (file <- dir.listFiles() if file.getName.startsWith(prefix)) {
      file.delete()
    }
  }

  protected def buildDataFrame(
      labeledPoints: Seq[XGBLabeledPoint],
      numPartitions: Int = numWorkers): DataFrame = {
    val it = labeledPoints.iterator.zipWithIndex
      .map { case (labeledPoint: XGBLabeledPoint, id: Int) =>
        (id, labeledPoint.label, labeledPoint.features, labeledPoint.weight)
      }
    ss.createDataFrame(sc.parallelize(it.toList, numPartitions))
      .toDF("id", "label", "features", "weight")
  }

  protected def buildDataFrameWithGroup(
      labeledPoints: Seq[XGBLabeledPoint],
      numPartitions: Int = numWorkers): DataFrame = {
    val it = labeledPoints.iterator.zipWithIndex
      .map { case (labeledPoint: XGBLabeledPoint, id: Int) =>
        (id, labeledPoint.label, labeledPoint.features, labeledPoint.group, labeledPoint.weight)
      }
    ss.createDataFrame(sc.parallelize(it.toList, numPartitions))
      .toDF("id", "label", "features", "group", "weight")
  }

  protected def compareTwoFiles(lhs: String, rhs: String): Boolean = {
    withResource(new FileInputStream(lhs)) { lfis =>
      withResource(new FileInputStream(rhs)) { rfis =>
        IOUtils.contentEquals(lfis, rfis)
      }
    }
  }

  def smallBinaryClassificationVector: DataFrame = ss.createDataFrame(sc.parallelize(Seq(
    (1.0, 0.5, 1.0, Vectors.dense(1.0, 2.0, 3.0)),
    (0.0, 0.4, -3.0, Vectors.dense(0.0, 0.0, 0.0)),
    (0.0, 0.3, 1.0, Vectors.dense(0.0, 3.0, 0.0)),
    (1.0, 1.2, 0.2, Vectors.dense(2.0, 0.0, 4.0)),
    (0.0, -0.5, 0.0, Vectors.dense(0.2, 1.2, 2.0)),
    (1.0, -0.4, -2.1, Vectors.dense(0.5, 2.2, 1.7))
  ))).toDF("label", "margin", "weight", "features")

  def smallBinaryClassificationArray: DataFrame = ss.createDataFrame(sc.parallelize(Seq(
    (1.0, 0.5, 1.0, Seq(1.0, 2.0, 3.0)),
    (0.0, 0.4, -3.0, Seq(0.0, 0.0, 0.0)),
    (0.0, 0.3, 1.0, Seq(0.0, 3.0, 0.0)),
    (1.0, 1.2, 0.2, Seq(2.0, 0.0, 4.0)),
    (0.0, -0.5, 0.0, Seq(0.2, 1.2, 2.0)),
    (1.0, -0.4, -2.1, Seq(0.5, 2.2, 1.7))
  ))).toDF("label", "margin", "weight", "features")

  def smallMultiClassificationVector: DataFrame = ss.createDataFrame(sc.parallelize(Seq(
    (1.0, 0.5, 1.0, Vectors.dense(1.0, 2.0, 3.0)),
    (0.0, 0.4, -3.0, Vectors.dense(0.0, 0.0, 0.0)),
    (2.0, 0.3, 1.0, Vectors.dense(0.0, 3.0, 0.0)),
    (1.0, 1.2, 0.2, Vectors.dense(2.0, 0.0, 4.0)),
    (0.0, -0.5, 0.0, Vectors.dense(0.2, 1.2, 2.0)),
    (2.0, -0.4, -2.1, Vectors.dense(0.5, 2.2, 1.7))
  ))).toDF("label", "margin", "weight", "features")

  def smallGroupVector: DataFrame = ss.createDataFrame(sc.parallelize(Seq(
    (1.0, 0, 0.5, 2.0, Vectors.dense(1.0, 2.0, 3.0)),
    (0.0, 1, 0.4, 1.0, Vectors.dense(0.0, 0.0, 0.0)),
    (0.0, 1, 0.3, 1.0, Vectors.dense(0.0, 3.0, 0.0)),
    (1.0, 0, 1.2, 2.0, Vectors.dense(2.0, 0.0, 4.0)),
    (1.0, 2, -0.5, 3.0, Vectors.dense(0.2, 1.2, 2.0)),
    (0.0, 2, -0.4, 3.0, Vectors.dense(0.5, 2.2, 1.7))
  ))).toDF("label", "group", "margin", "weight", "features")

}
