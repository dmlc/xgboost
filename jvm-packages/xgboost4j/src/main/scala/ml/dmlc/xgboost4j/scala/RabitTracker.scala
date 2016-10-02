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

/**
  * Pure Scala implementation of the DMLC tracker using Akka, ported from the
  * original Python implementation. This implementation gets rid of the Python
  * dependency in the Java version (RabitTracker.java) and therefore is less
  * suspectible to Python-related issues.
  */

package ml.dmlc.xgboost4j.scala

import java.net.{InetAddress, InetSocketAddress}
import java.util.concurrent.TimeoutException

import scala.concurrent.duration._
import akka.actor.ActorSystem
import akka.pattern.ask
import ml.dmlc.xgboost4j.java.IRabitTracker

import scala.concurrent.{Await, Future, Promise}
import scala.util.{Failure, Random, Success, Try}
import ml.dmlc.xgboost4j.scala.handler.RabitTrackerHandler

object RabitTracker {
  def main(args: Array[String]): Unit = {
    val tracker = new RabitTracker(196, Some(10080))
    tracker.start()

    println(tracker.getWorkerEnvs)
    tracker.waitFor()
  }
}

/**
  * Synchronous tracker class that mimics the Java API.
  *
  * {{{
  *   import scala.concurrent.duration._
  *
  *   val tracker = new RabitTracker(32)
  *   // allow up to 1 minute for the tracker to bind to a socket address.
  *   tracker.start(1 minute)
  *   /* ...
  *      launching workers in parallel
  *      ...
  *   */
  *   // wait for worker execution up to 6 hours.
  *   tracker.waitFor(6 hours)
  * }}}
  *
  * @param numWorkers number of distributed workers, each corresponding to a Spark task.
  * @param port The minimum port that the tracker binds to.
  * @param maxPortTrials The maximum number of trials of socket binding, by sequentially
  *        increasing port.
  * @param workerConnectionTimeout The timeout for awaiting connections from worker nodes.
  *        When used in Spark applications, because all Spark transformations are lazily executed,
  *        the I/O time for loading RDDs from external sources (local disk, HDFS, S3 etc.) must be
  *        taken in account. If the timeout value is too small, the Rabit workers will likely time
  *        out before connecting, due to the overhead of loading RDDs.
  *        Using a finite timeout is encouraged, as it prevents the tracker (thus the Spark driver
  *        running it) from hanging indefinitely due to worker connection issues (e.g. firewall).
  */
class RabitTracker(numWorkers: Int, port: Option[Int] = None,
                   maxPortTrials: Int = 1000,
                   workerConnectionTimeout: Duration = 10 minutes) extends IRabitTracker {
  import scala.collection.JavaConverters._

  require(numWorkers >=1, "numWorkers must be greater than or equal to one (1).")

  val system = ActorSystem.create("RabitTracker")
  val handler = system.actorOf(RabitTrackerHandler.props(
    numWorkers, maxPortTrials, workerConnectionTimeout), "Handler")
  implicit val askTimeout: akka.util.Timeout = akka.util.Timeout(1 second)

  val futureWorkerEnvs: Future[Map[String, String]] = Try(
    Await.result(handler ? RabitTrackerHandler.RequestBoundFuture, 5 seconds)
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
    Await.result(handler ? RabitTrackerHandler.RequestCompletionFuture, 5 seconds)
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
    * @param timeout timeout for the tracker to bind to an address.
    * @return
    */
  def start(timeout: Duration = 30 seconds): Boolean = {
    handler ? RabitTrackerHandler.StartTracker(
      new InetSocketAddress(InetAddress.getLocalHost,
        // randomly allocate an ephemeral port if port is not specified
        port.getOrElse(new Random().nextInt(61000 - 32768) + 32768)
      ))

    Try(Await.ready(futureWorkerEnvs, timeout)).isSuccess
  }

  def start(): Boolean = start(30 seconds)
  def start(timeout: Long, unit: TimeUnit): Boolean = start(Duration(timeout, unit))

  def getWorkerEnvs: java.util.Map[String, String] = {
    (Await.result(futureWorkerEnvs, 0 nano) ++ Map(
        "DMLC_NUM_WORKER" -> numWorkers.toString,
        "DMLC_NUM_SERVER" -> "0"
    )).asJava
  }

  /**
    * Wait for all workers to complete assigned tasks in a blocking fashion.
    * @param atMost the maximum execution time for the workers. By default,
    *     the tracker waits for the workers indefinitely.
    * @return the number of completed workers.
    */
  @throws[TimeoutException]
  def waitFor(atMost: Duration = Duration.Inf): Int = {
    // wait for all workers to complete synchronously.
    Try(Await.result(futureCompleted, atMost)) match {
      case Success(n) if n == numWorkers =>
        system.shutdown()
        0
      case Success(n) if n < numWorkers =>
        throw new TimeoutException(s"Only $n out of $numWorkers workers have started.")
      case Failure(e) => throw e
    }
  }

  def waitFor(): Int = waitFor(Duration.Inf)
  def waitFor(atMost: Long, unit: TimeUnit): Int = waitFor(Duration(atMost, unit))
}
