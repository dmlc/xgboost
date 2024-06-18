/*
 Copyright (c) 2014-2022 by Contributors

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

import ml.dmlc.xgboost4j.java.{Communicator, RabitTracker => PyRabitTracker}
import ml.dmlc.xgboost4j.java.IRabitTracker.TrackerStatus
import ml.dmlc.xgboost4j.scala.DMatrix
import org.scalatest.funsuite.AnyFunSuite

class CommunicatorRobustnessSuite extends AnyFunSuite with PerTest {

  private def getXGBoostExecutionParams(paramMap: Map[String, Any]): XGBoostExecutionParams = {
    val classifier = new XGBoostClassifier(paramMap)
    val xgbParamsFactory = new XGBoostExecutionParamsFactory(classifier.MLlib2XGBoostParams, sc)
    xgbParamsFactory.buildXGBRuntimeParams
  }

  test("Customize host ip and python exec for Rabit tracker") {
    val hostIp = "192.168.22.111"
    val pythonExec = "/usr/bin/python3"

    val paramMap = Map(
      "num_workers" -> numWorkers,
      "tracker_conf" -> TrackerConf(0L, hostIp))
    val xgbExecParams = getXGBoostExecutionParams(paramMap)
    val tracker = XGBoost.getTracker(xgbExecParams.numWorkers, xgbExecParams.trackerConf)
    tracker match {
      case pyTracker: PyRabitTracker =>
        val cmd = pyTracker.getRabitTrackerCommand
        assert(cmd.contains(hostIp))
        assert(cmd.startsWith("python"))
      case _ => assert(false, "expected python tracker implementation")
    }

    val paramMap1 = Map(
      "num_workers" -> numWorkers,
      "tracker_conf" -> TrackerConf(0L, "", pythonExec))
    val xgbExecParams1 = getXGBoostExecutionParams(paramMap1)
    val tracker1 = XGBoost.getTracker(xgbExecParams1.numWorkers, xgbExecParams1.trackerConf)
    tracker1 match {
      case pyTracker: PyRabitTracker =>
        val cmd = pyTracker.getRabitTrackerCommand
        assert(cmd.startsWith(pythonExec))
        assert(!cmd.contains(hostIp))
      case _ => assert(false, "expected python tracker implementation")
    }

    val paramMap2 = Map(
      "num_workers" -> numWorkers,
      "tracker_conf" -> TrackerConf(0L, hostIp, pythonExec))
    val xgbExecParams2 = getXGBoostExecutionParams(paramMap2)
    val tracker2 = XGBoost.getTracker(xgbExecParams2.numWorkers, xgbExecParams2.trackerConf)
    tracker2 match {
      case pyTracker: PyRabitTracker =>
        val cmd = pyTracker.getRabitTrackerCommand
        assert(cmd.startsWith(pythonExec))
        assert(cmd.contains(s" --host-ip=${hostIp}"))
      case _ => assert(false, "expected python tracker implementation")
    }
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
      Communicator.init(trackerEnvs)
      val index = iter.next()
      Thread.sleep(100 + index * 10)
      if (index == workerCount) {
        // kill the worker by throwing an exception
        throw new RuntimeException("Worker exception.")
      }
      Communicator.shutdown()
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

  test("should allow the dataframe containing communicator calls to be partially evaluated for" +
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
