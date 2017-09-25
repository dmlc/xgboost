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

import java.net.URL

import ml.dmlc.xgboost4j.java.XGBoostError
import org.apache.commons.logging.LogFactory
import org.apache.spark.SparkContext
import org.codehaus.jackson.map.ObjectMapper

import scala.concurrent.Future
import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.{Failure, Success}
import scala.collection.JavaConverters._

/**
  * A tracker that ensures enough number of executor cores are alive.
  * Throws an exception when the number of alive cores is less than nWorkers.
  *
  * @param sc The SparkContext object
  * @param timeout The maximum time to wait for enough number of workers.
  * @param nWorkers nWorkers used in an XGBoost Job
  */
private[spark] class SparkParallelismTracker(
    sc: SparkContext,
    timeout: Long,
    nWorkers: Int) {

  private[this] val checkInterval = 100L // Check every 0.1 second.
  private[this] val mapper = new ObjectMapper()
  private[this] val logger = LogFactory.getLog("XGBoostSpark")
  private[this] val url = sc.uiWebUrl match {
    case Some(baseUrl) => new URL(s"$baseUrl/api/v1/applications/${sc.applicationId}/executors")
    case _ => null
  }


  private[this] def isHealthy: Boolean = {
    val numAliveCores = try {
      mapper.readTree(url).findValues("totalCores").asScala.map(_.asInt).sum
    } catch {
      case ex: Throwable =>
        logger.warn(s"Unable to read total number of alive cores from REST API." +
          s"Health Check will be ignored.")
        ex.printStackTrace()
        Int.MaxValue
    }
    numAliveCores >= nWorkers
  }

  /**
    * Execute a blocking function call with two checks on enough nWorkers:
    *  - Before the function starts, ensure there are enough executor cores.
    *  - During the execution, throws an exception if there is any executor lost.
    *
    * @param body A blocking function call
    * @tparam T Return type
    * @return The return of body
    */
  def execute[T](body: => T): T = {
    if (checkInterval <= 0 || url == null) {
      body
    } else {
      val trackerThread = Thread.currentThread()
      // Wait and start the body as a separate thread
      val bodyFuture = Future {
        synchronized {
          wait(timeout)
        }
        if (isHealthy) {
          body
        } else {
          throw new XGBoostError(s"Unable to get $nWorkers workers")
        }
      }
      bodyFuture.onComplete {
        _ => trackerThread.interrupt()
      }
      // Check until enough cores, notify the body thread, and check no core has lost
      try {
        while (!isHealthy) {
          sc.requestExecutors(1)
          Thread.sleep(checkInterval)
        }
        synchronized {
          notify()
        }
        while (isHealthy) {
          Thread.sleep(checkInterval)
        }
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
