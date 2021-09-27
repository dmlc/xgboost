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

/**
 * A tracker that ensures enough number of executors are alive.
 * Throws an exception when the number of alive executors is less than nWorkers.
 *
 * @param sc The SparkContext object
 * @param timeout The maximum time to wait for enough number of workers.
 * @param numWorkers nWorkers used in an XGBoost Job
 */
class GpuParallelismTracker(override val sc: SparkContext, timeout: Long, numWorkers: Int)
    extends SparkParallelismTracker(sc, timeout, numWorkers) {

  private[this] val logger = LogFactory.getLog("GpuXGBoostSpark")

  protected[this] def numAliveWorkers: Int =
    sc.statusStore.executorList(true).size

  protected[this] def isActiveCoresEnough: Boolean = numAliveCores >= requestedCores

  /**
    * Execute a blocking function call with two checks on enough workers:
    *  - Before the function starts, wait until there are enough executors.
    *  - During the execution, throws an exception if there is any executor lost.
    *
    * @param body A blocking function call
    * @tparam T Return type
    * @return The return of body
    */
  def executeOnGpu[T](body: => T): T = {
    if (timeout <= 0) {
      logger.info("Starting training on GPU without setting timeout for waiting for resources")
      body
    } else {
      logger.info(s"Starting training on GPU with timeout ${timeout}ms for waiting for resources")
      if (!waitForCondition(isActiveCoresEnough && numAliveWorkers >= numWorkers, timeout)) {
        throw new IllegalStateException(s"Unable to get $numWorkers executors for GpuXGBoost" +
          s" training")
      }
      safeExecute(body)
    }
  }

}
