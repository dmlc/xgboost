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
import ml.dmlc.xgboost4j.java.IRabitTracker
import ml.dmlc.xgboost4j.scala.rabit.handler.RabitTrackerHandler

import scala.concurrent.duration._
import scala.concurrent.{Await, Future, Promise}
import scala.util.{Failure, Random, Success, Try}

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
  * @param tcpBindTimeout Timeout for the tracker to bind to an InetSocketAddress.
  */
class RabitTracker private[scala](numWorkers: Int, port: Option[Int] = None,
                                  maxPortTrials: Int = 1000, tcpBindTimeout: Duration = 1 minute)
  extends IRabitTracker {

  import scala.collection.JavaConverters._

  require(numWorkers >=1, "numWorkers must be greater than or equal to one (1).")

  val system = ActorSystem.create("RabitTracker")
  val handler = system.actorOf(RabitTrackerHandler.props(numWorkers), "Handler")
  implicit val askTimeout: akka.util.Timeout = akka.util.Timeout(30 seconds)

  val futureWorkerEnvs: Future[Map[String, String]] = Try(
    Await.result(handler ? RabitTrackerHandler.RequestBoundFuture, askTimeout.duration)
      .asInstanceOf[Future[Map[String, String]]]
  ) match {
    case Success(fut) => fut
    case Failure(e) =>
      val delayedFailure = Promise[Map[String, String]]()
      delayedFailure.failure(e)
      delayedFailure.future
  }

  // a future for XGBoost worker execution
  val futureCompleted: Future[Int] = Try(
    Await.result(handler ? RabitTrackerHandler.RequestCompletionFuture, askTimeout.duration)
      .asInstanceOf[Future[Int]]
  ) match {
    case Success(fut) => fut
    case Failure(_) =>
      val delayedFailure = Promise[Int]()
      // mark 0 worker as completed, will trigger AssertionError
      // when call waitFor()
      delayedFailure.success(0)
      delayedFailure.future
  }

  /**
    * Start the Rabit tracker.
    *
    * @param timeout The timeout for awaiting connections from worker nodes.
    *        Note that when used in Spark applications, because all Spark transformations are
    *        lazily executed, the I/O time for loading RDDs/DataFrames from external sources
    *        (local dist, HDFS, S3 etc.) must be taken into account for the timeout value.
    *        If the timeout value is too small, the Rabit tracker will likely timeout before workers
    *        establishing connections to the tracker, due to the overhead of loading data.
    *        Using a finite timeout is encouraged, as it prevents the tracker (thus the Spark driver
    *        running it) from hanging indefinitely due to worker connection issues (e.g. firewall.)
    * @return Boolean flag indicating if the Rabit tracker starts successfully.
    */
  def start(timeout: Duration = 10 minutes): Boolean = {
    handler ? RabitTrackerHandler.StartTracker(
      new InetSocketAddress(InetAddress.getLocalHost, port.getOrElse(0)), maxPortTrials, timeout)

    // The success of the Future is contingent on binding to an InetSocketAddress.
    Try(Await.ready(futureWorkerEnvs, tcpBindTimeout)).isSuccess
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

  /**
    * Get a Map of necessary environment variables to initiate Rabit workers.
    *
    * @return HashMap containing tracker information.
    */
  def getWorkerEnvs: java.util.Map[String, String] = {
    new java.util.HashMap((Await.result(futureWorkerEnvs, 0 nano) ++ Map(
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
  def waitFor(atMost: Duration = Duration.Inf): Int = {
    // wait for all workers to complete synchronously.
    Try(Await.result(futureCompleted, atMost)) match {
      case Success(n) if n == numWorkers =>
        system.shutdown()
        RabitTracker.SUCCESS.statusCode
      case Success(n) if n < numWorkers =>
        RabitTracker.TIMEOUT.statusCode
      case Failure(e) =>
        RabitTracker.FAILURE.statusCode
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

private[scala] object RabitTracker {
  sealed abstract class TrackerStatus(val statusCode: Int)
  case object SUCCESS extends TrackerStatus(0)
  case object TIMEOUT extends TrackerStatus(1)
  case object FAILURE extends TrackerStatus(2)
}