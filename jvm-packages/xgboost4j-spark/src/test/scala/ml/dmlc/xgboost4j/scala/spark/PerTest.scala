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
import org.apache.spark.{SparkConf, SparkContext, TaskFailedListener}
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
    TaskFailedListener.sparkContextShutdownLock.synchronized {
      if (currentSession != null) {
        // this synchronization is mostly for the tests involving SparkContext shutdown
        // for unit test involving the sparkContext shutdown there are two different events sequence
        // 1. SparkContext killer is executed before afterEach, in this case, before SparkContext
        // is fully stopped, afterEach() will block at the following code block
        // 2. SparkContext killer is executed afterEach, in this case, currentSession.stop() in will
        // block to wait for all msgs in ListenerBus get processed. Because currentSession.stop()
        // has been called, SparkContext killer will not take effect
        while (TaskFailedListener.killerStarted) {
          TaskFailedListener.sparkContextShutdownLock.wait()
        }
        currentSession.stop()
        cleanExternalCache(currentSession.sparkContext.appName)
        currentSession = null
      }
      if (TaskFailedListener.sparkContextKiller != null) {
        TaskFailedListener.sparkContextKiller.interrupt()
        TaskFailedListener.sparkContextKiller = null
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
}
