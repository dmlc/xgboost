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

import ml.dmlc.xgboost4j.java.{IRabitTracker, Rabit, RabitTracker => PyRabitTracker}
import ml.dmlc.xgboost4j.scala.rabit.{RabitTracker => ScalaRabitTracker}
import ml.dmlc.xgboost4j.java.IRabitTracker.TrackerStatus
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.FunSuite


class RabitTrackerRobustnessSuite extends FunSuite with Utils {
  test("test Java RabitTracker wrapper's exception handling: it should not hang forever.") {
    /*
      Deliberately create new instances of SparkContext in each unit test to avoid reusing the
      same thread pool spawned by the local mode of Spark. As these tests simulate worker crashes
      by throwing exceptions, the crashed worker thread never calls Rabit.shutdown, and therefore
      corrupts the internal state of the native Rabit C++ code. Calling Rabit.init() in subsequent
      tests on a reentrant thread will crash the entire Spark application, an undesired side-effect
      that should be avoided.
     */
    val sparkConf = new SparkConf().setMaster("local[*]")
      .setAppName("XGBoostSuite").set("spark.driver.memory", "512m")
    implicit val sparkContext = new SparkContext(sparkConf)
    sparkContext.setLogLevel("ERROR")

    val rdd = sparkContext.parallelize(1 to numWorkers, numWorkers).cache()

    val tracker = new PyRabitTracker(numWorkers)
    tracker.start(0)
    val trackerEnvs = tracker.getWorkerEnvs

    val workerCount: Int = numWorkers
    /*
       Simulate worker crash events by creating dummy Rabit workers, and throw exceptions in the
       last created worker. A cascading event chain will be triggered once the RuntimeException is
       thrown: the thread running the dummy spark job (sparkThread) catches the exception and
       delegates it to the UnCaughtExceptionHandler, which is the Rabit tracker itself.

       The Java RabitTracker class reacts to exceptions by killing the spawned process running
       the Python tracker. If at least one Rabit worker has yet connected to the tracker before
       it is killed, the resulted connection failure will trigger the Rabit worker to call
       "exit(-1);" in the native C++ code, effectively ending the dummy Spark task.

       In cluster (standalone or YARN) mode of Spark, tasks are run in containers and thus are
       isolated from each other. That is, one task calling "exit(-1);" has no effect on other tasks
       running in separate containers. However, as unit tests are run in Spark local mode, in which
       tasks are executed by threads belonging to the same process, one thread calling "exit(-1);"
       ultimately kills the entire process, which also happens to host the Spark driver, causing
       the entire Spark application to crash.

       To prevent unit tests from crashing, deterministic delays were introduced to make sure that
       the exception is thrown at last, ideally after all worker connections have been established.
       For the same reason, the Java RabitTracker class delays the killing of the Python tracker
       process to ensure that pending worker connections are handled.
     */
    val dummyTasks = rdd.mapPartitions { iter =>
      Rabit.init(trackerEnvs)
      val index = iter.next()
      Thread.sleep(100 + index * 10)
      if (index == workerCount) {
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

    val workerCount: Int = numWorkers
    val dummyTasks = rdd.mapPartitions { iter =>
      Rabit.init(trackerEnvs)
      val index = iter.next()
      Thread.sleep(100 + index * 10)
      if (index == workerCount) {
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
    assert(tracker.waitFor(0L) == TrackerStatus.FAILURE.getStatusCode)
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
        Thread.sleep(1000)
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
    assert(tracker.waitFor(0L) == TrackerStatus.FAILURE.getStatusCode)
    sparkContext.stop()
  }
}
