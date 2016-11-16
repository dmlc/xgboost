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

import ml.dmlc.xgboost4j.java.Rabit
import ml.dmlc.xgboost4j.scala.rabit.{RabitTracker => ScalaRabitTracker}
import ml.dmlc.xgboost4j.java.{RabitTracker => PyRabitTracker}

import scala.util.Random

class XGBoostRobustnessSuite extends SharedSparkContext with Utils {
  test("test Java RabitTracker wrapper's exception handling: it should not hang forever.") {
    val rdd = sc.parallelize(1 to numWorkers, numWorkers).cache()

    val tracker = new PyRabitTracker(numWorkers)
    tracker.start(0)
    val trackerEnvs = tracker.getWorkerEnvs

    val allReduceResults = rdd.mapPartitions { iter =>
      Rabit.init(trackerEnvs)
      Thread.sleep(500)
      val index = iter.toArray
      if (index(0) == 1) {
        // kill the worker by throwing an exception
        throw new RuntimeException("Worker exception.")
      }
      Rabit.shutdown()
      Iterator(index(0))
    }.cache()

    val sparkThread = new Thread() {
      override def run(): Unit = {
        // forces a Spark job.
        allReduceResults.foreachPartition(() => _)
      }
    }
    sparkThread.setUncaughtExceptionHandler(tracker)
    sparkThread.start()
    assert(tracker.waitFor(0) != 0)
    sparkThread.join()
  }

  test("test Scala RabitTracker's exception handling: it should not hang forever.") {
    val rdd = sc.parallelize(1 to numWorkers, numWorkers).cache()

    val tracker = new ScalaRabitTracker(numWorkers)
    tracker.start(0)
    val trackerEnvs = tracker.getWorkerEnvs

    val allReduceResults = rdd.mapPartitions { iter =>
      Rabit.init(trackerEnvs)
      Thread.sleep(500)
      val index = iter.toArray
      if (index(0) == 1) {
        // kill the worker by throwing an exception
        throw new RuntimeException("Worker exception.")
      }
      Rabit.shutdown()
      Iterator(index(0))
    }.cache()

    val sparkThread = new Thread() {
      override def run(): Unit = {
        // forces a Spark job.
        allReduceResults.foreachPartition(() => _)
      }
    }
    sparkThread.setUncaughtExceptionHandler(tracker)
    sparkThread.start()
    assert(tracker.waitFor() == ScalaRabitTracker.FAILURE.statusCode)
    sparkThread.join()
  }
}
