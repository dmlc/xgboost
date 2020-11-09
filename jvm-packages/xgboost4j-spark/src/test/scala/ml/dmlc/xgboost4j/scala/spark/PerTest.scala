/*
 Copyright (c) 2014 by Contributors

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

import java.io.File

import ml.dmlc.xgboost4j.{LabeledPoint => XGBLabeledPoint}
import org.apache.spark.{SparkContext, SparkContextUtils, TaskFailedListener}
import org.apache.spark.sql._
import org.scalatest.{BeforeAndAfterEach, FunSuite}

import scala.math.min
import scala.util.Random

trait PerTest extends BeforeAndAfterEach { self: FunSuite =>

  protected val numWorkers: Int = min(Runtime.getRuntime.availableProcessors(), 4)

  @transient private var currentSession: SparkSession = _

  def ss: SparkSession = getOrCreateSession
  implicit def sc: SparkContext = ss.sparkContext

  protected def sparkSessionBuilder: SparkSession.Builder = SparkSession.builder()
      .master(s"local[${numWorkers}]")
      .appName("XGBoostSuite")
      .config("spark.ui.enabled", false)
      .config("spark.driver.memory", "512m")
      .config("spark.task.cpus", 1)

  override def beforeEach(): Unit = getOrCreateSession

  override def afterEach() {
    synchronized {
      if (currentSession != null) {
        currentSession.stop()
        cleanExternalCache(currentSession.sparkContext.appName)
        currentSession = null
        waitSparkContextTotallyStopped
      }
      TaskFailedListener.killerStarted = false
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
    import DataUtils._
    val it = labeledPoints.iterator.zipWithIndex
      .map { case (labeledPoint: XGBLabeledPoint, id: Int) =>
        (id, labeledPoint.label, labeledPoint.features)
      }

    ss.createDataFrame(sc.parallelize(it.toList, numPartitions))
      .toDF("id", "label", "features")
  }

  protected def buildDataFrameWithRandSort(
      labeledPoints: Seq[XGBLabeledPoint],
      numPartitions: Int = numWorkers): DataFrame = {
    val df = buildDataFrame(labeledPoints, numPartitions)
    val rndSortedRDD = df.rdd.mapPartitions { iter =>
      iter.map(_ -> Random.nextDouble()).toList
        .sortBy(_._2)
        .map(_._1).iterator
    }
    ss.createDataFrame(rndSortedRDD, df.schema)
  }

  protected def buildDataFrameWithGroup(
      labeledPoints: Seq[XGBLabeledPoint],
      numPartitions: Int = numWorkers): DataFrame = {
    import DataUtils._
    val it = labeledPoints.iterator.zipWithIndex
      .map { case (labeledPoint: XGBLabeledPoint, id: Int) =>
        (id, labeledPoint.label, labeledPoint.features, labeledPoint.group)
      }

    ss.createDataFrame(sc.parallelize(it.toList, numPartitions))
      .toDF("id", "label", "features", "group")
  }

  // Background:
  // 1. XGBoost has enabled Stopping SparkContext by default while tasks run into exceptions,
  // 2. The Stopping SparkContext (SparkContext.getOrCreate().stop()) is called in a separate
  //    thread differring with thread running unit tests
  // 3. The unit tests are executed one by one in a same thread.
  //
  // Consider this sitution, TEST B follows TEST A (TEST A will throw exception in the worker)
  // a) TEST A will trigger SparkContext.getOrCreate().stop() which takes long time to be finished
  //        in a separate thread
  // b) TEST A calls afterEach which tries to stop SparkContext which is stopping because of a)
  // c) TEST B runs. first, calling beforeEach which creates SparkSession by wrapping a SparkContext
  //      called by SparkContext.getOrCreate(), Since SparkContext is a singleTon and kept by
  //      activeContext which was defined in SparkContext,
  //
  // SparkContext.stop will clear activeContext in the last of stop by
  //      SparkContext.clearActiveContext()
  // Here is the issue,
  // The newly created Sparksession of TEST B will wrap the SparkContext which is stopping by
  // TEST A, which will result sparkContext.assertNotStopped() throw exception and block the
  //  following unit tests in SparkSession.

  private def waitSparkContextTotallyStopped: Unit = {
    var totalWaitedTime = 0L
    while (!SparkContextUtils.getActiveSparkContext.isEmpty && totalWaitedTime <= 11000) {
      Thread.sleep(1000)
      totalWaitedTime += 1000
    }
    assert(SparkContextUtils.getActiveSparkContext.isEmpty === true)
  }
}
