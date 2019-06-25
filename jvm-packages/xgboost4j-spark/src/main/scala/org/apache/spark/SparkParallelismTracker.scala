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
import java.util.concurrent.atomic.AtomicBoolean

import org.apache.commons.logging.LogFactory

import org.apache.spark.scheduler._
import org.codehaus.jackson.map.ObjectMapper
import scala.collection.JavaConverters._
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._
import scala.concurrent.{Await, Future, TimeoutException}
import scala.util.control.ControlThrowable

/**
 * A tracker that ensures enough number of executor cores are alive.
 * Throws an exception when the number of alive cores is less than nWorkers.
 *
 * @param sc The SparkContext object
 * @param timeout The maximum time to wait for enough number of workers.
 * @param numWorkers nWorkers used in an XGBoost Job
 */
class SparkParallelismTracker(
    val sc: SparkContext,
    timeout: Long,
    numWorkers: Int) {

  private[this] val requestedCores = numWorkers * sc.conf.getInt("spark.task.cpus", 1)
  private[this] val mapper = new ObjectMapper()
  private[this] val logger = LogFactory.getLog("XGBoostSpark")
  private[this] val url = sc.uiWebUrl match {
    case Some(baseUrl) => new URL(s"$baseUrl/api/v1/applications/${sc.applicationId}/executors")
    case _ => null
  }

  private[this] def numAliveCores: Int = {
    try {
      if (url != null) {
        mapper.readTree(url).findValues("totalCores").asScala.map(_.asInt).sum
      } else {
        Int.MaxValue
      }
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
    val listener = new TaskFailedListener
    sc.addSparkListener(listener)
    try {
      body
    } finally {
      sc.removeSparkListener(listener)
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
      logger.info("starting training without setting timeout for waiting for resources")
      body
    } else {
      try {
        logger.info(s"starting training with timeout set as $timeout ms for waiting for resources")
        waitForCondition(numAliveCores >= requestedCores, timeout)
      } catch {
        case _: TimeoutException =>
          throw new IllegalStateException(s"Unable to get $requestedCores workers for" +
            s" XGBoost training")
      }
      safeExecute(body)
    }
  }
}

private[spark] class TaskFailedListener extends SparkListener {

  private[this] val logger = LogFactory.getLog("XGBoostTaskFailedListener")

  override def onTaskEnd(taskEnd: SparkListenerTaskEnd): Unit = {
    taskEnd.reason match {
      case taskEndReason: TaskFailedReason =>
        logger.error(s"Training Task Failed during XGBoost Training: " +
            s"$taskEndReason, stopping SparkContext")
        TaskFailedListener.startedSparkContextKiller()
      case _ =>
    }
  }
}

object TaskFailedListener {

  var killerStarted = false

  private def startedSparkContextKiller(): Unit = this.synchronized {
    if (!killerStarted) {
      // Spark does not allow ListenerThread to shutdown SparkContext so that we have to do it
      // in a separate thread
      val sparkContextKiller = new Thread() {
        override def run(): Unit = {
          LiveListenerBus.withinListenerThread.withValue(false) {
            SparkContext.getOrCreate().stop()
          }
        }
      }
      sparkContextKiller.setDaemon(true)
      sparkContextKiller.start()
      killerStarted = true
    }
  }
}
