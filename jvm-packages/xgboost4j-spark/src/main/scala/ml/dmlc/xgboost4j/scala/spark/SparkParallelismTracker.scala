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

import ml.dmlc.xgboost4j.java.XGBoostError
import org.apache.spark.SparkContext

import scala.concurrent.Future
import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.{Failure, Success}

/**
  * A tracker that periodically checks the number of alive Spark executor cores.
  * Throws an exception when the number of alive cores is less than nWorkers.
  *
  * @param sc The SparkContext object
  * @param checkInterval The interval to check executor status in milliseconds.
  * @param nWorkers nWorkers used in an XGBoost Job
  */
private[spark] class SparkParallelismTracker(
    sc: SparkContext,
    checkInterval: Long,
    nWorkers: Int) {

  private[this] def isRunning: Boolean = {
    val currentNumTasks = sc.statusTracker.getExecutorInfos.map(_.numRunningTasks()).sum
    if (currentNumTasks >= nWorkers) {
      true
    } else if (currentNumTasks == 0) {
      false
    } else {
      throw new XGBoostError(s"Requires numParallelism = $nWorkers but only " +
        s"$currentNumTasks tasks are alive. Please check logs in Spark History Server.")
    }
  }

  /**
    * Execute an async function call with periodic checks
    *
    * @param body An async function call
    * @tparam T Return type
    * @return The return of body
    */
  def execute[T](body: => T): T = {
    if (checkInterval <= 0) {
      body
    } else {
      val trackerThread = Thread.currentThread()
      // Start the body as a separate thread
      val bodyFuture = Future(body)
      bodyFuture.onComplete {
        _ => trackerThread.interrupt()
      }
      // Monitor the body thread
      try {
        do {
          Thread.sleep(checkInterval)
        } while (isRunning)
      } catch {
        case _: InterruptedException =>
      }
      // Get the result from the body thread
      bodyFuture.value.get match {
        case Success(t : T) => t
        case Failure(ex) => throw ex
      }
    }
  }
}
