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

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.concurrent.ExecutionContext.Implicits.global

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
  private[this] var isRunning = true
  private[this] val trackerThread = Thread.currentThread()

  private[this] def check(): Unit = {
    val currentNumExecutors = sc.getExecutorMemoryStatus.size - 1
    val numCoresPerExecutor = sc.getConf.getInt("spark.executor.cores", 1)
    val currentParallelism = currentNumExecutors * numCoresPerExecutor
    if (currentParallelism < nWorkers) {
      throw new XGBoostError(s"Requires numParallelism = $nWorkers but only " +
        s"$currentParallelism cores are alive. Please check logs in Spark History Server.")
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
      // Start the body as a separate thread
      val bodyFuture = Future {
        try {
          body
        } finally {
          isRunning = false
          trackerThread.interrupt()
        }
      }
      // Run periodic checks on the current thread
      try {
        while (isRunning) {
          Thread.sleep(checkInterval)
          check()
        }
      } catch {
        case _: InterruptedException =>
      }
      // Get the result from the body thread
      Await.result(bodyFuture, Duration.Inf)
    }
  }
}
