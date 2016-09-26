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
  * dependency in the Java version (RabitTracker.java) and therefore is more
  * robust.
  */

package ml.dmlc.xgboost4j.scala

import java.net.{InetAddress, InetSocketAddress}
import java.nio.{ByteBuffer, ByteOrder}
import java.util.UUID
import java.util.concurrent.TimeoutException

import scala.concurrent.duration._
import akka.actor.{Actor, ActorLogging, ActorRef, ActorSystem, FSM, PoisonPill, Props, Stash}
import akka.pattern.ask
import akka.io.{IO, Tcp}
import akka.util.ByteString

import scala.concurrent.{Await, Future, Promise}
import scala.util.{Failure, Random, Success, Try}
import scala.collection.mutable

case class AssignedRank(rank: Int, neighbors: Seq[Int], ring: (Int, Int), parent: Int)

class LinkMap(numWorkers: Int) {
  private def getNeighbors(rank: Int): Seq[Int] = {
    val rank1 = rank + 1
    Vector(rank1 / 2 - 1, rank1 * 2 - 1, rank1 * 2).filter { r =>
      r >= 0 && r < numWorkers
    }
  }

  /**
    * Construct a ring structure that tends to share nodes with the tree.
    *
    * @param treeMap
    * @param parentMap
    * @param rank
    * @return Seq[Int] instance starting from rank.
    */
  private def constructShareRing(treeMap: Map[Int, Seq[Int]],
                                 parentMap: Map[Int, Int],
                                 rank: Int = 0): Seq[Int] = {
    treeMap(rank).toSet - parentMap(rank) match {
      case emptySet if emptySet.isEmpty =>
        List(rank)
      case connectionSet =>
          connectionSet.zipWithIndex.foldLeft(List(rank)) {
            case (ringSeq, (v, cnt)) =>
              val vConnSeq = constructShareRing(treeMap, parentMap, v)
              vConnSeq match {
                case vconn if vconn.size == cnt + 1 =>
                  ringSeq ++ vconn.reverse
                case vconn =>
                  ringSeq ++ vconn
              }
          }
    }
  }
  /**
    * Construct a ring connection used to recover local data
    *
    * @param treeMap
    * @param parentMap
    */
  private def constructRingMap(treeMap: Map[Int, Seq[Int]], parentMap: Map[Int, Int]) = {
    assert(parentMap(0) == -1)

    val sharedRing = constructShareRing(treeMap, parentMap, 0).toVector
    assert(sharedRing.length == treeMap.size)

    (0 until numWorkers).map { r =>
      val rPrev = (r + numWorkers - 1) % numWorkers
      val rNext = (r + 1) % numWorkers
      sharedRing(r) -> (sharedRing(rPrev), sharedRing(rNext))
    }.toMap
  }

  private[this] val treeMap_ = (0 until numWorkers).map { r => r -> getNeighbors(r) }.toMap
  private[this] val parentMap_ = (0 until numWorkers).map{ r => r -> ((r + 1) / 2 - 1) }.toMap
  private[this] val ringMap_ = constructRingMap(treeMap_, parentMap_)
  val rMap_ = (0 until (numWorkers - 1)).foldLeft((Map(0 -> 0), 0)) {
    case ((rmap, k), i) =>
      val kNext = ringMap_(k)._2
      (rmap ++ Map(kNext -> (i + 1)), kNext)
  }._1

  val ringMap = ringMap_.map {
    case (k, (v0, v1)) => rMap_(k) -> (rMap_(v0), rMap_(v1))
  }
  val treeMap = treeMap_.map {
    case (k, vSeq) => rMap_(k) -> vSeq.map{ v => rMap_(v) }
  }
  val parentMap = parentMap_.map {
    case (k, v) if k == 0 =>
      rMap_(k) -> -1
    case (k, v) =>
      rMap_(k) -> rMap_(v)
  }

  def assignRank(rank: Int): AssignedRank = {
    AssignedRank(rank, treeMap(rank), ringMap(rank), parentMap(rank))
  }
}

object RabitTrackerHelpers {
  implicit class ByteStringHelplers(bs: ByteString) {
    // Java by default uses big endian. Enforce native endian so that
    // the byte order is consistent with the workers.
    def asNativeOrderByteBuffer: ByteBuffer = {
      bs.asByteBuffer.order(ByteOrder.nativeOrder())
    }
  }

  implicit class ByteBufferHelpers(buf: ByteBuffer) {
    def getString(): String = {
      val len = buf.getInt()
      val stringBuffer = ByteBuffer.allocate(len)
        .order(ByteOrder.nativeOrder())
      buf.get(stringBuffer.array(), 0, len)

      new String(stringBuffer.array(), "utf-8")
    }
  }
}

object RabitTrackerConnectionHandler {
  val MAGIC_NUMBER = 0xff99

  // finite states
  sealed trait State
  case object AwaitingHandshake extends State
  case object AwaitingCommand extends State
  case object BuildingLinkMap extends State
  case object AwaitingErrorCount extends State
  case object SetupComplete extends State

  sealed trait DataField
  case object IntField extends DataField
  // an integer preceding the actual string
  case object StringField extends DataField
  case object IntSeqField extends DataField

  object DataStruct {
    def apply(): DataStruct = DataStruct(Seq.empty[DataField], 0)
  }

  case class DataStruct(fields: Seq[DataField], counter: Int) {
    /**
      * Validate whether the provided buffer is complete (i.e., contains
      * all data fields specified for this DataStruct.
      * @param buf
      */
    def verify(buf: ByteBuffer): Boolean = {
      if (fields.isEmpty) return true

      val dupBuf = buf.duplicate().order(ByteOrder.nativeOrder())
      dupBuf.flip()

      Try(fields.foldLeft(true) {
        case (complete, field) =>
          val remBytes = dupBuf.remaining()
          complete && (remBytes > 0) && (remBytes >= (field match {
            case IntField =>
              dupBuf.position(dupBuf.position() + 4)
              4
            case StringField =>
              val strLen = dupBuf.getInt
              dupBuf.position(dupBuf.position() + strLen)
              4 + strLen
            case IntSeqField =>
              val seqLen = dupBuf.getInt
              dupBuf.position(dupBuf.position() + seqLen * 4)
              4 + seqLen * 4
          }))
      }).getOrElse(false)
    }

    def increment(): DataStruct = DataStruct(fields, counter + 1)
    def decrement(): DataStruct = DataStruct(fields, counter - 1)
  }

  val StructNodes = DataStruct(List(IntSeqField), 0)
  val StructTrackerCommand = DataStruct(List(
    IntField, IntField, StringField, StringField
  ), 0)

  sealed trait TrackerCommand {
    def rank: Int
    def worldSize: Int
    def jobId: String
  }

  // packaged worker commands
  case class WorkerStart(rank: Int, worldSize: Int, jobId: String) extends TrackerCommand
  case class WorkerShutdown(rank: Int, worldSize: Int, jobId: String) extends TrackerCommand
  case class WorkerRecover(rank: Int, worldSize: Int, jobId: String) extends TrackerCommand
  case class WorkerTrackerPrint(rank: Int, worldSize: Int, jobId: String, msg: String)
    extends TrackerCommand

  // request host and port information from peer actors
  case object RequestHostPort
  // response to the above request
  case class DivulgedHostPort(rank: Int, host: String, port: Int)
  case class AcknowledgeAcceptance(peers: Map[Int, ActorRef], numBad: Int)
  case class ReduceWaitCount(count: Int)

  case class DropFromWaitingList(rank: Int)
  case class WorkerStarted(host: String, rank: Int, awaitingAcceptance: Int)
  // Request, from the tracker, the set of nodes to connect.
  case class RequestAwaitConnWorkers(rank: Int, toConnectSet: Set[Int])
  case class AwaitingConnections(workers: Map[Int, ActorRef], numBad: Int)

  def props(host: String, worldSize: Int, tracker: ActorRef, connection: ActorRef): Props = {
    Props(new RabitTrackerConnectionHandler(host, worldSize, tracker, connection))
  }
}

/**
  * Actor to handle socket communication from worker node.
  * To handle fragmentation in received data, this class acts like a FSM
  * (finite-state machine) to keep track of the internal states.
  *
  * @param host IP address of the remote worker
  * @param worldSize number of total workers
  * @param tracker the RabitTrackerHandler actor reference
  */
class RabitTrackerConnectionHandler(host: String, worldSize: Int, tracker: ActorRef,
                                    connection: ActorRef)
    extends FSM[RabitTrackerConnectionHandler.State, RabitTrackerConnectionHandler.DataStruct]
      with ActorLogging with Stash {

  import RabitTrackerConnectionHandler._
  import RabitTrackerHelpers._

  context.watch(tracker)

  private[this] var rank: Int = 0
  private[this] var port: Int = 0

  // indicate if the connection is transient (like "print" or "shutdown")
  private[this] var transient: Boolean = false

  // number of workers pending acceptance of current worker
  private[this] var awaitingAcceptance: Int = 0
  private[this] var neighboringWorkers = Set.empty[Int]

  // TODO: use a single memory allocation to host all buffers,
  // including the transient ones for writing.
  private[this] val readBuffer = ByteBuffer.allocate(4096)
    .order(ByteOrder.nativeOrder())
  // in case the received message is longer than needed,
  // stash the spilled over part in this buffer, and send
  // to self when transition occurs.
  private[this] val spillOverBuffer = ByteBuffer.allocate(4096)
    .order(ByteOrder.nativeOrder())
  // when setup is complete, need to notify peer handlers
  // to reduce the awaiting-connection counter.
  private[this] var pendingAcknowledgement: Option[AcknowledgeAcceptance] = None

  private def resetBuffers(): Unit = {
    readBuffer.clear()
    if (spillOverBuffer.position() > 0) {
      spillOverBuffer.flip()
      self ! Tcp.Received(ByteString.fromByteBuffer(spillOverBuffer))
      spillOverBuffer.clear()
    }
  }

  private def stashSpillOver(buf: ByteBuffer): Unit = {
    if (buf.remaining() > 0) spillOverBuffer.put(buf)
  }

  def decodeCommand(buffer: ByteBuffer): TrackerCommand = {
    val rank = buffer.getInt()
    val worldSize = buffer.getInt()
    val jobId = buffer.getString()

    val command = buffer.getString()
    command match {
      case "start" => WorkerStart(rank, worldSize, jobId)
      case "shutdown" =>
        transient = true
        WorkerShutdown(rank, worldSize, jobId)
      case "recover" =>
        require(rank >= 0, "Invalid rank for recovering worker.")
        WorkerRecover(rank, worldSize, jobId)
      case "print" =>
        transient = true
        WorkerTrackerPrint(rank, worldSize, jobId, buffer.getString())
    }
  }

  startWith(AwaitingHandshake, DataStruct())

  when(AwaitingHandshake) {
    case Event(Tcp.Received(magic), _) =>
      assert(magic.length == 4)
      val purportedMagic = magic.asNativeOrderByteBuffer.getInt
      assert(purportedMagic == MAGIC_NUMBER, s"invalid magic number $purportedMagic from $host")

      // echo back the magic number
      connection ! Tcp.Write(magic)
      /*
      val magicBuf = ByteBuffer.allocate(4)
        .order(ByteOrder.nativeOrder()).putInt(purportedMagic)
      magicBuf.flip()

      connection ! Tcp.Write(ByteString.fromByteBuffer(magicBuf))
      */

      goto(AwaitingCommand) using StructTrackerCommand
  }

  when(AwaitingCommand) {
    case Event(Tcp.Received(bytes), validator) =>
      bytes.asByteBuffers.foreach { buf => readBuffer.put(buf) }

      if (validator.verify(readBuffer)) {
        readBuffer.flip()
        tracker ! decodeCommand(readBuffer)
        stashSpillOver(readBuffer)
      }

      stay
    // when rank for a worker is assigned, send encoded rank information
    // back to worker over Tcp socket.
    case Event(AssignedRank(assignedRank, neighbors, ring, parent), _) =>
      log.debug(s"Assigned rank [$assignedRank] for $host, T: $neighbors, R: $ring, P: $parent")

      rank = assignedRank
      val buffer = ByteBuffer.allocate(4 * (neighbors.length + 6))
        .order(ByteOrder.nativeOrder())
      buffer.putInt(assignedRank).putInt(parent).putInt(worldSize).putInt(neighbors.length)
      // neighbors in tree structure
      neighbors.foreach { n => buffer.putInt(n) }
      // ranks from the ring
      val ringRanks = List(
        // ringPrev
        if (ring._1 != -1 && ring._1 != rank) ring._1 else -1,
        // ringNext
        if (ring._2 != -1 && ring._2 != rank) ring._2 else -1
      )
      ringRanks.foreach { r => buffer.putInt(r) }

      // update the set of all linked workers to current worker.
      neighboringWorkers = neighbors.toSet ++ ringRanks.filterNot(_ == -1).toSet

      buffer.flip()
      connection ! Tcp.Write(ByteString.fromByteBuffer(buffer))
      // to prevent reading before state transition
      connection ! Tcp.SuspendReading
      goto(BuildingLinkMap) using StructNodes
  }

  when(BuildingLinkMap) {
    case Event(Tcp.Received(bytes), validator) =>
      bytes.asByteBuffers.foreach { buf =>
        readBuffer.put(buf)
      }

      if (validator.verify(readBuffer)) {
        readBuffer.flip()
        // for a freshly started worker, numConnected should be 0.
        val numConnected = readBuffer.getInt()
        val toConnectSet = neighboringWorkers.diff(
          (0 until numConnected).map { index => readBuffer.getInt() }.toSet)

        // check which workers are currently awaiting connections
        tracker ! RequestAwaitConnWorkers(rank, toConnectSet)
      }
      stay

    // got a Future from the tracker (resolver) about workers that are
    // currently awaiting connections (particularly from this node.)
    case Event(future: Future[_], _) =>
      // blocks execution until all dependencies for current worker is resolved.
      Await.result(future, 1 minute).asInstanceOf[AwaitingConnections] match {
        // numNotReachable is the number of workers that currently
        // cannot be connected to (pending connection or setup). Instead, this worker will AWAIT
        // connections from those currently non-reachable nodes in the future.
        case AwaitingConnections(waitConnNodes, numNotReachable) =>
          log.debug(s"Rank $rank needs to connect to: $waitConnNodes, # bad: $numNotReachable")
          val buf = ByteBuffer.allocate(8).order(ByteOrder.nativeOrder())
          buf.putInt(waitConnNodes.size).putInt(numNotReachable)
          buf.flip()

          // cache this message until the final state (SetupComplete)
          pendingAcknowledgement = Some(AcknowledgeAcceptance(
            waitConnNodes, numNotReachable))

          connection ! Tcp.Write(ByteString.fromByteBuffer(buf))
          if (waitConnNodes.isEmpty) {
            connection ! Tcp.SuspendReading
            goto(AwaitingErrorCount)
          }
          else {
            waitConnNodes.foreach { case (peerRank, peerRef) =>
              peerRef ! RequestHostPort
            }

            // a countdown for DivulgedHostPort messages.
            stay using DataStruct(Seq.empty[DataField], waitConnNodes.size - 1)
          }
      }

    case Event(DivulgedHostPort(peerRank, peerHost, peerPort), data) =>
      val hostBytes = peerHost.getBytes()
      val buffer = ByteBuffer.allocate(4 * 3 + hostBytes.length)
        .order(ByteOrder.nativeOrder())
      buffer.putInt(peerHost.length).put(hostBytes)
        .putInt(peerPort).putInt(peerRank)

      buffer.flip()
      connection ! Tcp.Write(ByteString.fromByteBuffer(buffer))

      if (data.counter == 0) {
        // to prevent reading before state transition
        connection ! Tcp.SuspendReading
        goto(AwaitingErrorCount)
      }
      else {
        stay using data.decrement()
      }
  }

  when(AwaitingErrorCount) {
    case Event(Tcp.Received(numErrors), _) =>
      val buf = numErrors.asNativeOrderByteBuffer

      buf.getInt match {
        case 0 =>
          stashSpillOver(buf)
          goto(SetupComplete)
        case _ =>
          stashSpillOver(buf)
          goto(BuildingLinkMap) using StructNodes

      }
  }

  when(SetupComplete) {
    case Event(Tcp.Received(assignedPort), _) =>
      assert(assignedPort.length == 4)
      port = assignedPort.asNativeOrderByteBuffer.getInt
      log.debug(s"Rank $rank listening @ $host:$port")
      stay

    case Event(ReduceWaitCount(count: Int), _) =>
      awaitingAcceptance -= count
      if (awaitingAcceptance == 0) {
        tracker ! DropFromWaitingList(rank)
        // no long needed.
        context.stop(self)
      }
      stay

    case Event(AcknowledgeAcceptance(peers, numBad), _) =>
      awaitingAcceptance = numBad
      tracker ! WorkerStarted(host, rank, awaitingAcceptance)
      peers.values.foreach { peer =>
        peer ! ReduceWaitCount(1)
      }

      if (awaitingAcceptance == 0) self ! PoisonPill

      stay

    // can only divulge the complete host and port information
    // when this worker is declared fully connected (otherwise
    // port information is still missing.)
    case Event(RequestHostPort, _) =>
      sender() ! DivulgedHostPort(rank, host, port)
      stay
  }

  onTransition {
    // reset buffer when state transitions as data becomes stale
    case _ -> SetupComplete =>
      connection ! Tcp.ResumeReading
      resetBuffers()
      if (pendingAcknowledgement.isDefined) {
        self ! pendingAcknowledgement.get
      }
    case _ =>
      connection ! Tcp.ResumeReading
      resetBuffers()
  }

  // default message handler
  whenUnhandled {
    case Event(Tcp.PeerClosed, _) =>
      if (transient) context.stop(self)
      stay
  }
}

object RabitTracker {
  case object RequestCompletionFuture
  case object RequestBoundFuture
  case class StartTracker(addr: InetSocketAddress)
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
                   workerConnectionTimeout: Duration = 10 minutes) {
  require(numWorkers >=1, "numWorkers must be greater than or equal to one (1).")

  import RabitTracker._
  val system = ActorSystem.create("RabitTracker")
  val handler = system.actorOf(RabitTrackerHandler.props(
    numWorkers, maxPortTrials, workerConnectionTimeout), "Handler")
  implicit val askTimeout: akka.util.Timeout = akka.util.Timeout(1 second)

  val futureWorkerEnvs: Future[Map[String, String]] = Try(
    Await.result(handler ? RequestBoundFuture, 5 seconds)
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
    Await.result(handler ? RequestCompletionFuture, 5 seconds)
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
    handler ? StartTracker(
      new InetSocketAddress(InetAddress.getLocalHost,
        // randomly allocate an ephemeral port if port is not specified
        port.getOrElse(new Random().nextInt(61000 - 32768) + 32768)
      ))

    Try(Await.ready(futureWorkerEnvs, timeout)).isSuccess
  }

  def getWorkerEnvs: Map[String, String] = {
    Await.result(futureWorkerEnvs, 0 nano) ++ Map(
        "DMLC_NUM_WORKER" -> numWorkers.toString,
        "DMLC_NUM_SERVER" -> "0"
    )
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
}

object RabitTrackerHandler {
  def props(numWorkers: Int, maxPortTrials: Int = 1000,
            workerConnectionTimeout: Duration = 10 minutes): Props =
    Props(new RabitTrackerHandler(numWorkers, maxPortTrials, workerConnectionTimeout))
}

/**
  * Resolve the dependency between nodes as they connect to the tracker.
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
      }
      else {
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

/**
  *
  * @param numWorkers
  * @param workerConnectionTimeout Timeout for all workers to start and connect
  *                                to the tracker. Using this timeout prevents
  *                                the tracker and its owner from hanging indefinitely.
  */
class RabitTrackerHandler(numWorkers: Int, maxPortTrials: Int,
                          workerConnectionTimeout: Duration)
    extends Actor with ActorLogging {

  import context.system
  import RabitTrackerConnectionHandler._
  import RabitTrackerHelpers._

  val promisedWorkerEnvs = Promise[Map[String, String]]()
  val promisedShutdownWorkers = Promise[Int]()
  val tcpManager = IO(Tcp)

  // resolves worker connection dependency.
  val resolver = context.actorOf(Props(classOf[WorkerDependencyResolver], self), "Resolver")

  // workers that have sent "shutdown" signal
  private val shutdownWorkers = mutable.Set.empty[Int]
  private val jobToRankMap = mutable.HashMap.empty[String, Int]
  private val actorRefToHost = mutable.HashMap.empty[ActorRef, String]
  private val ranksToAssign = mutable.ListBuffer(0 until numWorkers: _*)
  private var portTrials = 0
  private val startedWorkers = mutable.Set.empty[Int]

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

  def receive: Actor.Receive = {
    case RabitTracker.StartTracker(addr) =>
      tcpManager ! Tcp.Bind(self, addr, backlog = 256)
      sender() ! true

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

    case RabitTracker.RequestBoundFuture =>
      sender() ! promisedWorkerEnvs.future

    case RabitTracker.RequestCompletionFuture =>
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
