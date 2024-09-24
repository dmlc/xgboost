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

import org.scalatest.funsuite.AnyFunSuite

import ml.dmlc.xgboost4j.java.{Communicator, RabitTracker}

class CommunicatorRobustnessSuite extends AnyFunSuite with PerTest {

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

    val tracker = new RabitTracker(numWorkers)
    tracker.start()
    val trackerEnvs = tracker.getWorkerArgs

    val workerCount: Int = numWorkers
    /*
       Simulate worker crash events by creating dummy Rabit workers, and throw exceptions in the
       last created worker. A cascading event chain will be triggered once the RuntimeException is
       thrown: the thread running the dummy spark job (sparkThread) catches the exception and
       delegates it to the UnCaughtExceptionHandler, which is the Rabit tracker itself.

       To prevent unit tests from crashing, deterministic delays were introduced to make sure that
       the exception is thrown at last, ideally after all worker connections have been established.
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
  }

  test("Communicator allreduce works.") {
    val rdd = sc.parallelize(1 to numWorkers, numWorkers).cache()
    val tracker = new RabitTracker(numWorkers)
    tracker.start()
    val trackerEnvs = tracker.getWorkerArgs

    val workerCount: Int = numWorkers

    rdd.mapPartitions { iter =>
      val index = iter.next()
      Communicator.init(trackerEnvs)
      val a = Array(1.0f, 2.0f, 3.0f)
      System.out.println(a.mkString(", "))
      val b = Communicator.allReduce(a, Communicator.OpType.SUM)
      for (i <- 0 to 2) {
        assert(a(i) * workerCount == b(i))
      }
      val c = Communicator.allReduce(a, Communicator.OpType.MIN);
      for (i <- 0 to 2) {
        assert(a(i) == c(i))
      }
      Communicator.shutdown()
      Iterator(index)
    }.collect()
  }

  test("should allow the dataframe containing communicator calls to be partially evaluated for" +
    " multiple times (ISSUE-4406)") {
    val paramMap = Map(
      "eta" -> "1",
      "max_depth" -> "6",
      "silent" -> "1",
      "objective" -> "binary:logistic")
    val trainingDF = smallBinaryClassificationVector
    val model = new XGBoostClassifier(paramMap)
      .setNumWorkers(numWorkers)
      .setNumRound(10)
      .fit(trainingDF)
    val prediction = model.transform(trainingDF)
    // a partial evaluation of dataframe will cause rabit initialized but not shutdown in some
    // threads
    prediction.show()
    // a full evaluation here will re-run init and shutdown all rabit proxy
    // expecting no error
    prediction.collect()
  }
}
