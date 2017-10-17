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

import java.net.URL

import org.apache.commons.logging.LogFactory
import org.apache.spark.scheduler.{SparkListener, SparkListenerTaskEnd}
import org.codehaus.jackson.map.ObjectMapper

import scala.collection.JavaConverters._
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._
import scala.concurrent.{Await, Future, TimeoutException}

/**
 * A tracker that ensures enough number of executor cores are alive.
 * Throws an exception when the number of alive cores is less than nWorkers.
 *
 * @param sc The SparkContext object
 * @param timeout The maximum time to wait for enough number of workers.
 * @param nWorkers nWorkers used in an XGBoost Job
 */
class SparkParallelismTracker(
    val sc: SparkContext,
    timeout: Long,
    nWorkers: Int) {

  private[this] val mapper = new ObjectMapper()
  private[this] val logger = LogFactory.getLog("XGBoostSpark")
  private[this] val url = sc.uiWebUrl match {
    case Some(baseUrl) => new URL(s"$baseUrl/api/v1/applications/${sc.applicationId}/executors")
    case _ => null
  }

  private[this] def numAliveCores: Int = {
    try {
      mapper.readTree(url).findValues("totalCores").asScala.map(_.asInt).sum
    } catch {
      case ex: Throwable =>
        logger.warn(s"Unable to read total number of alive cores from REST API." +
          s"Health Check will be ignored.")
        ex.printStackTrace()
        Int.MaxValue
    }
  }

  private[this] def waitForCondition(
      condition: => Boolean,
      timeout: Long,
      checkInterval: Long = 100L) = {
    val monitor = Future {
      while (!condition) {
        Thread.sleep(checkInterval)
      }
    }
    Await.ready(monitor, timeout.millis)
  }

  private[this] def safeExecute[T](body: => T): T = {
    sc.listenerBus.listeners.add(0, new TaskFailedListener)
    try {
      body
    } finally {
      sc.listenerBus.listeners.remove(0)
    }
  }

  /**
   * Execute a blocking function call with two checks on enough nWorkers:
   *  - Before the function starts, wait until there are enough executor cores.
   *  - During the execution, throws an exception if there is any executor lost.
   *
   * @param body A blocking function call
   * @tparam T Return type
   * @return The return of body
   */
  def execute[T](body: => T): T = {
    if (timeout <= 0) {
      body
    } else {
      try {
        waitForCondition(numAliveCores >= nWorkers, timeout)
      } catch {
        case _: TimeoutException =>
          throw new IllegalStateException(s"Unable to get $nWorkers workers for XGBoost training")
      }
      safeExecute(body)
    }
  }
}

private[spark] class TaskFailedListener extends SparkListener {
  override def onTaskEnd(taskEnd: SparkListenerTaskEnd): Unit = {
    taskEnd.reason match {
      case reason: TaskFailedReason =>
        throw new InterruptedException(s"ExecutorLost during XGBoost Training: " +
          s"${reason.toErrorString}")
      case _ =>
    }
  }
}
