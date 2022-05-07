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

package ml.dmlc.xgboost4j.scala.rabit

import java.net.{InetAddress, InetSocketAddress}

import akka.actor.ActorSystem
import akka.pattern.ask
import ml.dmlc.xgboost4j.java.{IRabitTracker, TrackerProperties}
import ml.dmlc.xgboost4j.scala.rabit.handler.RabitTrackerHandler

import scala.concurrent.duration._
import scala.concurrent.{Await, Future}
import scala.util.{Failure, Success, Try}
import org.apache.commons.logging.LogFactory

/**
  * Scala implementation of the Rabit tracker interface without Python dependency.
  * The Scala Rabit tracker fully implements the timeout logic, effectively preventing the tracker
  * (and thus any distributed tasks) to hang indefinitely due to network issues or worker node
  * failures.
  *
  * Note that this implementation is currently experimental, and should be used at your own risk.
  *
  * Example usage:
  * {{{
  *   import scala.concurrent.duration._
  *
  *   val tracker = new RabitTracker(32)
  *   // allow up to 10 minutes for all workers to connect to the tracker.
  *   tracker.start(10 minutes)
  *
  *   /* ...
  *      launching workers in parallel
  *      ...
  *   */
  *
  *   // wait for worker execution up to 6 hours.
  *   // providing a finite timeout prevents a long-running task from hanging forever in
  *   // catastrophic events, like the loss of an executor during model training.
  *   tracker.waitFor(6 hours)
  * }}}
  *
  * @param numWorkers Number of distributed workers from which the tracker expects connections.
  * @param port The minimum port number that the tracker binds to.
  *             If port is omitted, or given as None, a random ephemeral port is chosen at runtime.
  * @param maxPortTrials The maximum number of trials of socket binding, by sequentially
  *                      increasing the port number.
  */
private[scala] class RabitTracker(numWorkers: Int, port: Option[Int] = None,
                                  maxPortTrials: Int = 1000)
  extends IRabitTracker {

  import scala.collection.JavaConverters._

  require(numWorkers >=1, "numWorkers must be greater than or equal to one (1).")

  val system = ActorSystem.create("RabitTracker")
  val handler = system.actorOf(RabitTrackerHandler.props(numWorkers), "Handler")
  implicit val askTimeout: akka.util.Timeout = akka.util.Timeout(30 seconds)
  private[this] val tcpBindingTimeout: Duration = 1 minute

  var workerEnvs: Map[String, String] = Map.empty
  private val logger = LogFactory.getLog("XGBoostSpark")
  logger.info("SUCCESS SUCCESS ERROR Enter Init  in RabitTracker.scala in xgboost4j")
  override def uncaughtException(t: Thread, e: Throwable): Unit = {
    handler ? RabitTrackerHandler.InterruptTracker(e)
  }

  /**
    * Start the Rabit tracker.
    *
    * @param timeout The timeout for awaiting connections from worker nodes.
    *      Note that when used in Spark applications, because all Spark transformations are
    *      lazily executed, the I/O time for loading RDDs/DataFrames from external sources
    *      (local dist, HDFS, S3 etc.) must be taken into account for the timeout value.
    *      If the timeout value is too small, the Rabit tracker will likely timeout before workers
    *      establishing connections to the tracker, due to the overhead of loading data.
    *      Using a finite timeout is encouraged, as it prevents the tracker (thus the Spark driver
    *      running it) from hanging indefinitely due to worker connection issues (e.g. firewall.)
    * @return Boolean flag indicating if the Rabit tracker starts successfully.
    */
  private def start(timeout: Duration): Boolean = {
    logger.info("SUCCESS SUCCESS ERROR Enter Start in RabitTracker.scala in xgboost4j")
    val hostAddress = Option(TrackerProperties.getInstance().getHostIp)
      .map(InetAddress.getByName).getOrElse(InetAddress.getLocalHost)

    handler ? RabitTrackerHandler.StartTracker(
      new InetSocketAddress(hostAddress, port.getOrElse(0)), maxPortTrials, timeout)

    // block by waiting for the actor to bind to a port
    Try(Await.result(handler ? RabitTrackerHandler.RequestBoundFuture, askTimeout.duration)
      .asInstanceOf[Future[Map[String, String]]]) match {
      case Success(futurePortBound) =>
        // The success of the Future is contingent on binding to an InetSocketAddress.
        val isBound = Try(Await.ready(futurePortBound, tcpBindingTimeout)).isSuccess
        if (isBound) {
          workerEnvs = Await.result(futurePortBound, 0 nano)
        }
        isBound
      case Failure(ex: Throwable) =>
        false
    }
  }

  /**
    * Start the Rabit tracker.
    *
    * @param connectionTimeoutMillis Timeout, in milliseconds, for the tracker to wait for worker
    *                                connections. If a non-positive value is provided, the tracker
    *                                waits for incoming worker connections indefinitely.
    * @return Boolean flag indicating if the Rabit tracker starts successfully.
    */
  def start(connectionTimeoutMillis: Long): Boolean = {
    if (connectionTimeoutMillis <= 0) {
      start(Duration.Inf)
    } else {
      start(Duration.fromNanos(connectionTimeoutMillis * 1e6))
    }
  }

  def stop(): Unit = {
    system.terminate()
  }

  /**
    * Get a Map of necessary environment variables to initiate Rabit workers.
    *
    * @return HashMap containing tracker information.
    */
  def getWorkerEnvs: java.util.Map[String, String] = {
    new java.util.HashMap((workerEnvs ++ Map(
        "DMLC_NUM_WORKER" -> numWorkers.toString,
        "DMLC_NUM_SERVER" -> "0"
    )).asJava)
  }

  /**
    * Await workers to complete assigned tasks for at most 'atMostMillis' milliseconds.
    * This method blocks until timeout or task completion.
    *
    * @param atMost the maximum execution time for the workers. By default,
    *     the tracker waits for the workers indefinitely.
    * @return 0 if the tasks complete successfully, and non-zero otherwise.
    */
  private def waitFor(atMost: Duration): Int = {
    // request the completion Future from the tracker actor
    logger.info("SUCCESS SUCCESS ERROR Enter waitFor in RabitTracker.scala in xgboost4j")
    Try(Await.result(handler ? RabitTrackerHandler.RequestCompletionFuture, askTimeout.duration)
      .asInstanceOf[Future[Int]]) match {
      case Success(futureCompleted) =>
        // wait for all workers to complete synchronously.
        val statusCode = Try(Await.result(futureCompleted, atMost)) match {
          case Success(n) if n == numWorkers =>
            IRabitTracker.TrackerStatus.SUCCESS.getStatusCode
          case Success(n) if n < numWorkers =>
            IRabitTracker.TrackerStatus.TIMEOUT.getStatusCode
          case Failure(e) =>
            IRabitTracker.TrackerStatus.FAILURE.getStatusCode
        }
        system.terminate()
        statusCode
      case Failure(ex: Throwable) =>
        system.terminate()
        IRabitTracker.TrackerStatus.FAILURE.getStatusCode
    }
  }

  /**
    * Await workers to complete assigned tasks for at most 'atMostMillis' milliseconds.
    * This method blocks until timeout or task completion.
    *
    * @param atMostMillis Number of milliseconds for the tracker to wait for workers. If a
    *                     non-positive number is given, the tracker waits indefinitely.
    * @return 0 if the tasks complete successfully, and non-zero otherwise
    */
  def waitFor(atMostMillis: Long): Int = {
    if (atMostMillis <= 0) {
      waitFor(Duration.Inf)
    } else {
      waitFor(Duration.fromNanos(atMostMillis * 1e6))
    }
  }
}

