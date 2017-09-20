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

  private[this] val mapper = new ObjectMapper()
  private[this] val logger = LogFactory.getLog("XGBoostSpark")
  private[this] val url = sc.uiWebUrl match {
    case Some(baseUrl) => new URL(s"$baseUrl/api/v1/applications/${sc.applicationId}/executors")
    case _ => null
  }


  private[this] def check(): Unit = {
    val numAliveCores = try {
      mapper.readTree(url).findValues("totalCores").asScala.map(_.asInt).sum
    } catch {
      case ex: Throwable =>
        logger.warn(s"Unable to read total number of alive cores from REST API." +
          s"Health Check will be ignored.")
        ex.printStackTrace()
        Int.MaxValue
    }
    if (numAliveCores < nWorkers) {
      throw new XGBoostError(s"Requires numParallelism = $nWorkers but only " +
        s"$numAliveCores cores are alive. Please check logs in Spark History Server.")
    }
  }

  /**
    * Execute a blocking function call with periodic checks on number of alive cores
    *
    * @param body A blocking function call
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
        while (true) {
          Thread.sleep(checkInterval)
          check()
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
