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

import java.util.concurrent.LinkedBlockingDeque

import scala.util.Random

import ml.dmlc.xgboost4j.java.{Rabit, RabitTracker => PyRabitTracker}
import ml.dmlc.xgboost4j.scala.rabit.{RabitTracker => ScalaRabitTracker}
import ml.dmlc.xgboost4j.java.IRabitTracker.TrackerStatus
import ml.dmlc.xgboost4j.scala.DMatrix

import org.scalatest.{FunSuite, Ignore}

class RabitRobustnessSuite extends FunSuite with PerTest {

  test("training with Scala-implemented Rabit tracker") {
    val eval = new EvalError()
    val training = buildDataFrame(Classification.train)
    val testDM = new DMatrix(Classification.test.iterator)
    val paramMap = Map("eta" -> "1", "max_depth" -> "6",
      "objective" -> "binary:logistic", "num_round" -> 5, "num_workers" -> numWorkers,
      "tracker_conf" -> TrackerConf(60 * 60 * 1000, "scala"))
    val model = new XGBoostClassifier(paramMap).fit(training)
    assert(eval.eval(model._booster.predict(testDM, outPutMargin = true), testDM) < 0.1)
  }

  test("test Rabit allreduce to validate Scala-implemented Rabit tracker") {
    val vectorLength = 100
    val rdd = sc.parallelize(
      (1 to numWorkers * vectorLength).toArray.map { _ => Random.nextFloat() }, numWorkers).cache()

    val tracker = new ScalaRabitTracker(numWorkers)
    tracker.start(0)
    val trackerEnvs = tracker.getWorkerEnvs
    val collectedAllReduceResults = new LinkedBlockingDeque[Array[Float]]()

    val rawData = rdd.mapPartitions { iter =>
      Iterator(iter.toArray)
    }.collect()

    val maxVec = (0 until vectorLength).toArray.map { j =>
      (0 until numWorkers).toArray.map { i => rawData(i)(j) }.max
    }

    val allReduceResults = rdd.mapPartitions { iter =>
      Rabit.init(trackerEnvs)
      val arr = iter.toArray
      val results = Rabit.allReduce(arr, Rabit.OpType.MAX)
      Rabit.shutdown()
      Iterator(results)
    }.cache()

    val sparkThread = new Thread() {
      override def run(): Unit = {
        allReduceResults.foreachPartition(() => _)
        val byPartitionResults = allReduceResults.collect()
        assert(byPartitionResults(0).length == vectorLength)
        collectedAllReduceResults.put(byPartitionResults(0))
      }
    }
    sparkThread.start()
    assert(tracker.waitFor(0L) == 0)
    sparkThread.join()

    assert(collectedAllReduceResults.poll().sameElements(maxVec))
  }

  test("test Java RabitTracker wrapper's exception handling: it should not hang forever.") {
    /*
      Deliberately create new instances of SparkContext in each unit test to avoid reusing the
      same thread pool spawned by the local mode of Spark. As these tests simulate worker crashes
      by throwing exceptions, the crashed worker thread never calls Rabit.shutdown, and therefore
      corrupts the internal state of the native Rabit C++ code. Calling Rabit.init() in subsequent
      tests on a reentrant thread will crash the entire Spark application, an undesired side-effect
      that should be avoided.
     */
    val rdd = sc.parallelize(1 to numWorkers, numWorkers).cache()

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
  }

  test("test Scala RabitTracker's exception handling: it should not hang forever.") {
    val rdd = sc.parallelize(1 to numWorkers, numWorkers).cache()

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
  }

  test("test Scala RabitTracker's workerConnectionTimeout") {
    val rdd = sc.parallelize(1 to numWorkers, numWorkers).cache()

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
  }

  test("should allow the dataframe containing rabit calls to be partially evaluated for" +
    " multiple times (ISSUE-4406)") {
    val paramMap = Map(
      "eta" -> "1",
      "max_depth" -> "6",
      "silent" -> "1",
      "objective" -> "binary:logistic")
    val trainingDF = buildDataFrame(Classification.train)
    val model = new XGBoostClassifier(paramMap ++ Array("num_round" -> 10,
      "num_workers" -> numWorkers)).fit(trainingDF)
    val prediction = model.transform(trainingDF)
    // a partial evaluation of dataframe will cause rabit initialized but not shutdown in some
    // threads
    prediction.show()
    // a full evaluation here will re-run init and shutdown all rabit proxy
    // expecting no error
    prediction.collect()
  }
}
