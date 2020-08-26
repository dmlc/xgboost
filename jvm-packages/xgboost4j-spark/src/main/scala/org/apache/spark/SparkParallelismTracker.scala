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

import org.apache.commons.logging.LogFactory
import org.apache.spark.scheduler._

import scala.collection.mutable.{HashMap, HashSet}

/**
 * A tracker that ensures enough number of executor cores are alive.
 * Throws an exception when the number of alive cores is less than nWorkers.
 *
 * @param sc The SparkContext object
 * @param timeout The maximum time to wait for enough number of workers.
 * @param numWorkers nWorkers used in an XGBoost Job
 * @param killSparkContext kill SparkContext or not when task fails
 */
class SparkParallelismTracker(
    val sc: SparkContext,
    timeout: Long,
    numWorkers: Int,
    killSparkContext: Boolean = true) {

  private[this] val requestedCores = numWorkers * sc.conf.getInt("spark.task.cpus", 1)
  private[this] val logger = LogFactory.getLog("XGBoostSpark")

  private[this] def numAliveCores: Int = {
    sc.statusStore.executorList(true).map(_.totalCores).sum
  }

  private[this] def waitForCondition(
      condition: => Boolean,
      timeout: Long,
      checkInterval: Long = 100L) = {
    val waitImpl = new ((Long, Boolean) => Boolean) {
      override def apply(waitedTime: Long, status: Boolean): Boolean = status match {
        case s if s => true
        case _ => waitedTime match {
          case t if t < timeout =>
            Thread.sleep(checkInterval)
            apply(t + checkInterval, status = condition)
          case _ => false
        }
      }
    }
    waitImpl(0L, condition)
  }

  private[this] def safeExecute[T](body: => T): T = {
    val listener = new TaskFailedListener(killSparkContext)
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
      safeExecute(body)
    } else {
      logger.info(s"starting training with timeout set as $timeout ms for waiting for resources")
      if (!waitForCondition(numAliveCores >= requestedCores, timeout)) {
        throw new IllegalStateException(s"Unable to get $requestedCores cores for XGBoost training")
      }
      safeExecute(body)
    }
  }
}

class TaskFailedListener(killSparkContext: Boolean = true) extends SparkListener {

  private[this] val logger = LogFactory.getLog("XGBoostTaskFailedListener")

  // {jobId, [stageId0, stageId1, ...] }
  // keep track of the mapping of job id and stage ids
  // when a task failed, find the job id and stage Id the task belongs to, finally
  // cancel the jobs
  private val jobIdToStageIds: HashMap[Int, HashSet[Int]] = HashMap.empty

  override def onJobStart(jobStart: SparkListenerJobStart): Unit = {
    if (!killSparkContext) {
      jobStart.stageIds.foreach(stageId => {
        jobIdToStageIds.getOrElseUpdate(jobStart.jobId, new HashSet[Int]()) += stageId
      })
    }
  }

  override def onJobEnd(jobEnd: SparkListenerJobEnd): Unit = {
    if (!killSparkContext) {
      jobIdToStageIds.remove(jobEnd.jobId)
    }
  }

  override def onTaskEnd(taskEnd: SparkListenerTaskEnd): Unit = {
    taskEnd.reason match {
      case taskEndReason: TaskFailedReason =>
        logger.error(s"Training Task Failed during XGBoost Training: " +
            s"$taskEndReason")
        if (killSparkContext) {
          logger.error("killing SparkContext")
          TaskFailedListener.startedSparkContextKiller()
        } else {
          val stageId = taskEnd.stageId
          // find job ids according to stage id and then cancel the job
          jobIdToStageIds.foreach(t => {
            val jobId = t._1
            val stageIds = t._2

            if (stageIds.contains(stageId)) {
              logger.error("Cancelling jobId:" + jobId)
              jobIdToStageIds.remove(jobId)
              SparkContext.getOrCreate().cancelJob(jobId)
            }
          })
        }
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
