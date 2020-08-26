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

package org.apache.spark

import org.scalatest.FunSuite
import _root_.ml.dmlc.xgboost4j.scala.spark.PerTest
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.math.min

class SparkParallelismTrackerSuite extends FunSuite with PerTest {

  val numParallelism: Int = min(Runtime.getRuntime.availableProcessors(), 4)

  override protected def sparkSessionBuilder: SparkSession.Builder = SparkSession.builder()
    .master(s"local[${numParallelism}]")
    .appName("XGBoostSuite")
    .config("spark.ui.enabled", true)
    .config("spark.driver.memory", "512m")
    .config("spark.task.cpus", 1)

  private def waitAndCheckSparkShutdown(waitMiliSec: Int): Boolean = {
    var totalWaitedTime = 0L
    while (!ss.sparkContext.isStopped && totalWaitedTime <= waitMiliSec) {
      Thread.sleep(100)
      totalWaitedTime += 100
    }
    ss.sparkContext.isStopped
  }

  test("tracker should not affect execution result when timeout is not larger than 0") {
    val nWorkers = numParallelism
    val rdd: RDD[Int] = sc.parallelize(1 to nWorkers)
    val tracker = new SparkParallelismTracker(sc, 10000, nWorkers)
    val disabledTracker = new SparkParallelismTracker(sc, 0, nWorkers)
    assert(tracker.execute(rdd.sum()) == rdd.sum())
    assert(disabledTracker.execute(rdd.sum()) == rdd.sum())
  }

  test("tracker should throw exception if parallelism is not sufficient") {
    val nWorkers = numParallelism * 3
    val rdd: RDD[Int] = sc.parallelize(1 to nWorkers)
    val tracker = new SparkParallelismTracker(sc, 1000, nWorkers)
    intercept[IllegalStateException] {
      tracker.execute {
        rdd.map { i =>
          // Test interruption
          Thread.sleep(Long.MaxValue)
          i
        }.sum()
      }
    }
  }

  test("tracker should throw exception if parallelism is not sufficient with" +
    " spark.task.cpus larger than 1") {
    sc.conf.set("spark.task.cpus", "2")
    val nWorkers = numParallelism
    val rdd: RDD[Int] = sc.parallelize(1 to nWorkers)
    val tracker = new SparkParallelismTracker(sc, 1000, nWorkers)
    intercept[IllegalStateException] {
      tracker.execute {
        rdd.map { i =>
          // Test interruption
          Thread.sleep(Long.MaxValue)
          i
        }.sum()
      }
    }
  }

  test("tracker should not kill SparkContext when killSparkContext=false") {
    val nWorkers = numParallelism
    val tracker = new SparkParallelismTracker(sc, 0, nWorkers, false)
    val rdd: RDD[Int] = sc.parallelize(1 to nWorkers, nWorkers)
    try {
      tracker.execute {
        rdd.map { i =>
          val partitionId = TaskContext.get().partitionId()
          if (partitionId == 0) {
            throw new RuntimeException("mocking task failing")
          }
          i
        }.sum()
      }
    } catch {
      case e: Exception => // catch the exception
    } finally {
      // wait 3s to check if SparkContext is killed
      assert(waitAndCheckSparkShutdown(3000) == false)
    }
  }

  test("tracker should cancel the correct job when killSparkContext=false") {
    val nWorkers = 2
    val tracker = new SparkParallelismTracker(sc, 0, nWorkers, false)
    val rdd: RDD[Int] = sc.parallelize(1 to 10, nWorkers)
    val thread = new MyThread(sc)
    thread.start()
    try {
      tracker.execute {
        rdd.map { i =>
          Thread.sleep(100)
          val partitionId = TaskContext.get().partitionId()
          if (partitionId == 0) {
            throw new RuntimeException("mocking task failing")
          }
          i
        }.sum()
      }
    } catch {
      case e: Exception => // catch the exception
    } finally {
      thread.join(8000)
      // wait 3s to check if SparkContext is killed
      assert(waitAndCheckSparkShutdown(3000) == false)
    }
  }

  private[this] class MyThread(sc: SparkContext) extends Thread {
    override def run(): Unit = {
      var sum: Double = 0.0f
      try {
        val rdd = sc.parallelize(1 to 4, 2)
        sum = rdd.mapPartitions(iter => {
          // sleep 2s to ensure task is alive when cancelling other jobs
          Thread.sleep(2000)
          iter
        }).sum()
      } finally {
        // get the correct result
        assert(sum.toInt == 10)
      }
    }
  }
}
