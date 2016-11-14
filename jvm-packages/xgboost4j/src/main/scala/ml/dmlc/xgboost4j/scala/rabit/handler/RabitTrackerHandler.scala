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

package ml.dmlc.xgboost4j.scala.rabit.handler

import java.net.InetSocketAddress
import java.util.UUID

import scala.concurrent.duration._
import scala.collection.mutable
import scala.concurrent.{Promise, TimeoutException}
import akka.io.{IO, Tcp}
import akka.actor._
import ml.dmlc.xgboost4j.scala.rabit.util.{AssignedRank, LinkMap}

import scala.util.Random

/** The Akka actor for handling Rabit worker connections.
  *
  * @param numWorkers Number of workers to track.
  */
class RabitTrackerHandler private[scala](numWorkers: Int)
  extends Actor with ActorLogging {

  import context.system
  import RabitTrackerConnectionHandler._
  import RabitTrackerHandler._

  private[this] val promisedWorkerEnvs = Promise[Map[String, String]]()
  private[this] val promisedShutdownWorkers = Promise[Int]()
  private[this] val tcpManager = IO(Tcp)

  // resolves worker connection dependency.
  val resolver = context.actorOf(Props(classOf[WorkerDependencyResolver], self), "Resolver")

  // workers that have sent "shutdown" signal
  private[this] val shutdownWorkers = mutable.Set.empty[Int]
  private[this] val jobToRankMap = mutable.HashMap.empty[String, Int]
  private[this] val actorRefToHost = mutable.HashMap.empty[ActorRef, String]
  private[this] val ranksToAssign = mutable.ListBuffer(0 until numWorkers: _*)
  private[this] var maxPortTrials = 0
  private[this] var workerConnectionTimeout: Duration = Duration.Inf
  private[this] var portTrials = 0
  private[this] val startedWorkers = mutable.Set.empty[Int]

  val linkMap = new LinkMap(numWorkers)

  def decideRank(rank: Int, jobId: String = "NULL"): Option[Int] = {
    rank match {
      case r if r >= 0 => Some(r)
      case _ =>
        jobId match {
          case "NULL" => None
          case jid => jobToRankMap.get(jid)
        }
    }
  }

  /**
    * Handler for all Akka Tcp events.
    * @param event
    * @return
    */
  private def handleTcpEvents(event: Tcp.Event) = event match {
    case Tcp.Bound(local) =>
      // expect all workers to connect within timeout
      log.info(s"Tracker listening @ ${local.getAddress.getHostAddress}:${local.getPort}")
      log.info(s"Worker connection timeout is $workerConnectionTimeout.")

      context.setReceiveTimeout(workerConnectionTimeout)
      promisedWorkerEnvs.success(Map(
        "DMLC_TRACKER_URI" -> local.getAddress.getHostAddress,
        "DMLC_TRACKER_PORT" -> local.getPort.toString,
        // not required because the world size will be communicated to the
        // worker node after the rank is assigned.
        "rabit_world_size" -> numWorkers.toString
      ))

    case Tcp.CommandFailed(cmd: Tcp.Bind) =>
      if (portTrials < maxPortTrials) {
        portTrials += 1
        tcpManager ! Tcp.Bind(self,
          new InetSocketAddress(cmd.localAddress.getAddress, cmd.localAddress.getPort + 1),
          backlog = 256)
      }

    case Tcp.Connected(remote, local) =>
      log.debug(s"Incoming connection from worker @ ${remote.getAddress.getHostAddress}")
      // revoke timeout if all workers have connected.
      val connHandler = context.actorOf(RabitTrackerConnectionHandler.props(
        remote.getAddress.getHostAddress, numWorkers, self, sender()
      ), s"ConnectionHandler-${UUID.randomUUID().toString}")
      val connection = sender()
      connection ! Tcp.Register(connHandler, keepOpenOnPeerClosed = true)

      actorRefToHost.put(connHandler, remote.getAddress.getHostName)
  }

  def receive: Actor.Receive = {
    case tcpEvent: Tcp.Event => handleTcpEvents(tcpEvent)

    case StartTracker(addr, portTrials, connectionTimeout) =>
      maxPortTrials = portTrials
      workerConnectionTimeout = connectionTimeout

      // if the port number is missing, try binding to a random ephemeral port.
      if (addr.getPort == 0) {
        tcpManager ! Tcp.Bind(self,
          new InetSocketAddress(addr.getAddress, new Random().nextInt(61000 - 32768) + 32768),
          backlog = 256)
      } else {
        tcpManager ! Tcp.Bind(self, addr, backlog = 256)
      }
      sender() ! true

    case RequestBoundFuture =>
      sender() ! promisedWorkerEnvs.future

    case RequestCompletionFuture =>
      sender() ! promisedShutdownWorkers.future

    case req @ RequestAwaitConnWorkers(_, _) =>
      // since the requester may request to connect to other workers
      // that have not fully set up, delegate this request to the
      // dependency resolver that handles the dependencies properly.
      resolver forward req

    // process messages from worker
    case WorkerTrackerPrint(_, _, _, msg) =>
      log.info(msg.trim)

    case WorkerShutdown(rank, _, _) =>
      assert(rank >= 0, "Invalid rank.")
      assert(!shutdownWorkers.contains(rank))
      shutdownWorkers.add(rank)

      log.info(s"Received shutdown signal from $rank")

      if (shutdownWorkers.size == numWorkers) {
        promisedShutdownWorkers.success(shutdownWorkers.size)
        context.stop(self)
      }

    case WorkerRecover(prevRank, worldSize, jobId) =>
      assert(prevRank >= 0)
      sender() ! linkMap.assignRank(prevRank)

    case WorkerStart(rank, worldSize, jobId) =>
      assert(worldSize == numWorkers || worldSize == -1,
        s"Purported worldSize ($worldSize) does not match worker count ($numWorkers)."
      )
      val wkRank = decideRank(rank, jobId).getOrElse(ranksToAssign.remove(0))
      if (jobId != "NULL") {
        jobToRankMap.put(jobId, wkRank)
      }

      val assignedRank = linkMap.assignRank(wkRank)
      sender() ! assignedRank
      resolver ! assignedRank

      log.info("Received start signal from " +
        s"${actorRefToHost.getOrElse(sender(), "")} [rank: $wkRank]")

    case msg @ WorkerStarted(host, rank, awaitingAcceptance) =>
      log.info(s"Worker $host (rank: $rank) has started.")
      resolver forward msg

      startedWorkers.add(rank)
      if (startedWorkers.size == numWorkers) {
        log.info("All workers have started.")
      }

    case req @ DropFromWaitingList(_) =>
      // handled by resolver
      resolver forward req

    case akka.actor.ReceiveTimeout =>
      if (startedWorkers.size < numWorkers) {
        promisedShutdownWorkers.failure(
          new TimeoutException("Timed out waiting for workers to connect: " +
            s"${numWorkers - startedWorkers.size} of $numWorkers did not start/connect.")
        )
        context.stop(self)
      }

      context.setReceiveTimeout(Duration.Undefined)
  }
}

/**
  * Resolve the dependency between nodes as they connect to the tracker.
  * The dependency is enforced that a worker of rank K depends on its neighbors (from the treeMap
  * and ringMap) whose ranks are smaller than K. Since ranks are assigned in the order of
  * connections by workers, this dependency constraint assumes that a worker node connects first
  * is likely to finish setup first.
  */
class WorkerDependencyResolver(handler: ActorRef) extends Actor with ActorLogging {
  import RabitTrackerConnectionHandler._

  context.watch(handler)

  case class Fulfillment(toConnectSet: Set[Int], promise: Promise[AwaitingConnections])

  // worker nodes that have connected, but have not send WorkerStarted message.
  private val dependencyMap = mutable.Map.empty[Int, Set[Int]]
  private val startedWorkers = mutable.Set.empty[Int]
  // worker nodes that have started, and await for connections.
  private val awaitConnWorkers = mutable.Map.empty[Int, ActorRef]
  private val pendingFulfillment = mutable.Map.empty[Int, Fulfillment]

  def awaitingWorkers(linkSet: Set[Int]): AwaitingConnections = {
    val connSet = awaitConnWorkers.toMap
      .filterKeys(k => linkSet.contains(k))
    AwaitingConnections(connSet, linkSet.size - connSet.size)
  }

  def receive: Actor.Receive = {
    // a copy of the AssignedRank message that is also sent to the worker
    case AssignedRank(rank, tree_neighbors, ring, parent) =>
      // the workers that the worker of given `rank` depends on:
      // worker of rank K only depends on workers with rank smaller than K.
      val dependentWorkers = (tree_neighbors.toSet ++ Set(ring._1, ring._2))
        .filter{ r => r != -1 && r < rank}

      log.debug(s"Rank $rank connected, dependencies: $dependentWorkers")
      dependencyMap.put(rank, dependentWorkers)
    case RequestAwaitConnWorkers(rank, toConnectSet) =>
      val promise = Promise[AwaitingConnections]()

      assert(dependencyMap.contains(rank))

      val updatedDependency = dependencyMap(rank) diff startedWorkers
      if (updatedDependency.isEmpty) {
        // all dependencies are satisfied
        log.debug(s"Rank $rank has all dependencies satisfied.")
        promise.success(awaitingWorkers(toConnectSet))
      } else {
        log.debug(s"Rank $rank's request for AwaitConnWorkers is pending fulfillment.")
        // promise is pending fulfillment due to unresolved dependency
        pendingFulfillment.put(rank, Fulfillment(toConnectSet, promise))
      }

      sender() ! promise.future

    case WorkerStarted(_, started, awaitingAcceptance) =>
      startedWorkers.add(started)
      if (awaitingAcceptance > 0) {
        awaitConnWorkers.put(started, sender())
      }

      // remove the started rank from all dependencies.
      dependencyMap.remove(started)
      dependencyMap.foreach { case (r, dset) =>
        val updatedDependency = dset diff startedWorkers
        // fulfill the future if all dependencies are met (started.)
        if (updatedDependency.isEmpty) {
          log.debug(s"Rank $r has all dependencies satisfied.")
          pendingFulfillment.remove(r).map{
            case Fulfillment(toConnectSet, promise) =>
              promise.success(awaitingWorkers(toConnectSet))
          }
        }

        dependencyMap.update(r, updatedDependency)
      }

    case DropFromWaitingList(rank) =>
      assert(awaitConnWorkers.remove(rank).isDefined)
  }
}

object RabitTrackerHandler {
  // Messages sent by RabitTracker to this RabitTrackerHandler actor

  case object RequestCompletionFuture
  case object RequestBoundFuture
  // Start the Rabit tracker at given socket address awaiting worker connections.
  // All workers must connect to the tracker before connectionTimeout, otherwise the tracker will
  // shut down due to timeout.
  case class StartTracker(addr: InetSocketAddress,
                          maxPortTrials: Int,
                          connectionTimeout: Duration)

  def props(numWorkers: Int): Props =
    Props(new RabitTrackerHandler(numWorkers))
}
