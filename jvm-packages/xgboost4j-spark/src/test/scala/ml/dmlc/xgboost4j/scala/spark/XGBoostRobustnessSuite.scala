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

import ml.dmlc.xgboost4j.java.{Rabit, RabitTracker => PyRabitTracker}
import ml.dmlc.xgboost4j.scala.rabit.{RabitTracker => ScalaRabitTracker}
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.FunSuite

class XGBoostRobustnessSuite extends FunSuite with Utils {
  test("test Java RabitTracker wrapper's exception handling: it should not hang forever.") {
    // Explicitly create new instances of SparkContext in each test to avoid reusing the same
    // thread pool, which corrupts the internal state of Rabit and causes crash.
    val sparkConf = new SparkConf().setMaster("local[*]")
      .setAppName("XGBoostSuite").set("spark.driver.memory", "512m")
    implicit val sparkContext = new SparkContext(sparkConf)
    sparkContext.setLogLevel("ERROR")

    val rdd = sparkContext.parallelize(1 to numWorkers, numWorkers).cache()

    val tracker = new PyRabitTracker(numWorkers)
    tracker.start(0)
    val trackerEnvs = tracker.getWorkerEnvs

    val dummyTasks = rdd.mapPartitions { iter =>
      Rabit.init(trackerEnvs)
      Thread.sleep(500)
      val index = iter.next()
      if (index == 1) {
        // kill the worker by throwing an exception
        throw new RuntimeException("Worker exception.")
      }
      Rabit.shutdown()
      Iterator(index)
    }.cache()

    val sparkThread = new Thread() {
      override def run(): Unit = {
        // forces a Spark job.
        dummyTasks.foreachPartition(() => _)
      }
    }

    sparkThread.setUncaughtExceptionHandler(tracker)
    sparkThread.start()
    assert(tracker.waitFor(0) != 0)
    sparkContext.stop()
  }

  test("test Scala RabitTracker's exception handling: it should not hang forever.") {
    val sparkConf = new SparkConf().setMaster("local[*]")
      .setAppName("XGBoostSuite").set("spark.driver.memory", "512m")
    implicit val sparkContext = new SparkContext(sparkConf)
    sparkContext.setLogLevel("ERROR")

    val rdd = sparkContext.parallelize(1 to numWorkers, numWorkers).cache()

    val tracker = new ScalaRabitTracker(numWorkers)
    tracker.start(0)
    val trackerEnvs = tracker.getWorkerEnvs

    val dummyTasks = rdd.mapPartitions { iter =>
      Rabit.init(trackerEnvs)
      Thread.sleep(500)
      val index = iter.next()
      if (index == 1) {
        // kill the worker by throwing an exception
        throw new RuntimeException("Worker exception.")
      }
      Rabit.shutdown()
      Iterator(index)
    }.cache()

    val sparkThread = new Thread() {
      override def run(): Unit = {
        // forces a Spark job.
        dummyTasks.foreachPartition(() => _)
      }
    }
    sparkThread.setUncaughtExceptionHandler(tracker)
    sparkThread.start()
    assert(tracker.waitFor() == ScalaRabitTracker.FAILURE.statusCode)
    sparkContext.stop()
  }

  test("test Scala RabitTracker's workerConnectionTimeout") {
    val sparkConf = new SparkConf().setMaster("local[*]")
      .setAppName("XGBoostSuite").set("spark.driver.memory", "512m")
    implicit val sparkContext = new SparkContext(sparkConf)
    sparkContext.setLogLevel("ERROR")

    val rdd = sparkContext.parallelize(1 to numWorkers, numWorkers).cache()

    val tracker = new ScalaRabitTracker(numWorkers)
    tracker.start(500)
    val trackerEnvs = tracker.getWorkerEnvs

    val dummyTasks = rdd.mapPartitions { iter =>
      val index = iter.next()
      // simulate that the first worker cannot connect to tracker due to network issues.
      if (index != 1) {
        Rabit.init(trackerEnvs)
        Thread.sleep(500)
        Rabit.shutdown()
      }

      Iterator(index)
    }.cache()

    val sparkThread = new Thread() {
      override def run(): Unit = {
        // forces a Spark job.
        dummyTasks.foreachPartition(() => _)
      }
    }
    sparkThread.setUncaughtExceptionHandler(tracker)
    sparkThread.start()
    // should fail due to connection timeout
    assert(tracker.waitFor() == ScalaRabitTracker.FAILURE.statusCode)
    sparkContext.stop()
  }
}
