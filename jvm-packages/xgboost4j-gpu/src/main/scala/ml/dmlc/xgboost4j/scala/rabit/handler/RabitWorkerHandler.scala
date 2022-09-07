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

import java.nio.{ByteBuffer, ByteOrder}

import akka.io.Tcp
import akka.actor._
import akka.util.ByteString
import ml.dmlc.xgboost4j.scala.rabit.util.{AssignedRank, RabitTrackerHelpers}

import scala.concurrent.{Await, Future}
import scala.concurrent.duration._
import scala.util.Try

/**
  * Actor to handle socket communication from worker node.
  * To handle fragmentation in received data, this class acts like a FSM
  * (finite-state machine) to keep track of the internal states.
  *
  * @param host IP address of the remote worker
  * @param worldSize number of total workers
  * @param tracker the RabitTrackerHandler actor reference
  */
private[scala] class RabitWorkerHandler(host: String, worldSize: Int, tracker: ActorRef,
                                        connection: ActorRef)
  extends FSM[RabitWorkerHandler.State, RabitWorkerHandler.DataStruct]
    with ActorLogging with Stash {

  import RabitWorkerHandler._
  import RabitTrackerHelpers._

  private[this] var rank: Int = 0
  private[this] var port: Int = 0

  // indicate if the connection is transient (like "print" or "shutdown")
  private[this] var transient: Boolean = false
  private[this] var peerClosed: Boolean = false

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

  def getNeighboringWorkers: Set[Int] = neighboringWorkers

  def decodeCommand(buffer: ByteBuffer): TrackerCommand = {
    val readBuffer = buffer.duplicate().order(ByteOrder.nativeOrder())
    readBuffer.flip()

    val rank = readBuffer.getInt()
    val worldSize = readBuffer.getInt()
    val jobId = readBuffer.getString

    val command = readBuffer.getString
    val trackerCommand = command match {
      case "start" => WorkerStart(rank, worldSize, jobId)
      case "shutdown" =>
        transient = true
        WorkerShutdown(rank, worldSize, jobId)
      case "recover" =>
        require(rank >= 0, "Invalid rank for recovering worker.")
        WorkerRecover(rank, worldSize, jobId)
      case "print" =>
        transient = true
        WorkerTrackerPrint(rank, worldSize, jobId, readBuffer.getString)
    }

    stashSpillOver(readBuffer)
    trackerCommand
  }

  startWith(AwaitingHandshake, DataStruct())

  when(AwaitingHandshake) {
    case Event(Tcp.Received(magic), _) =>
      assert(magic.length == 4)
      val purportedMagic = magic.asNativeOrderByteBuffer.getInt
      assert(purportedMagic == MAGIC_NUMBER, s"invalid magic number $purportedMagic from $host")

      // echo back the magic number
      connection ! Tcp.Write(magic)
      goto(AwaitingCommand) using StructTrackerCommand
  }

  when(AwaitingCommand) {
    case Event(Tcp.Received(bytes), validator) =>
      bytes.asByteBuffers.foreach { buf => readBuffer.put(buf) }
      if (validator.verify(readBuffer)) {
        Try(decodeCommand(readBuffer)) match {
          case scala.util.Success(decodedCommand) =>
            tracker ! decodedCommand
          case scala.util.Failure(th: java.nio.BufferUnderflowException) =>
            // BufferUnderflowException would occur if the message to print has not arrived yet.
            // Do nothing, wait for next Tcp.Received event
          case scala.util.Failure(th: Throwable) => throw th
        }
      }

      stay
    // when rank for a worker is assigned, send encoded rank information
    // back to worker over Tcp socket.
    case Event(aRank @ AssignedRank(assignedRank, neighbors, ring, parent), _) =>
      log.debug(s"Assigned rank [$assignedRank] for $host, T: $neighbors, R: $ring, P: $parent")

      rank = assignedRank
      // ranks from the ring
      val ringRanks = List(
        // ringPrev
        if (ring._1 != -1 && ring._1 != rank) ring._1 else -1,
        // ringNext
        if (ring._2 != -1 && ring._2 != rank) ring._2 else -1
      )

      // update the set of all linked workers to current worker.
      neighboringWorkers = neighbors.toSet ++ ringRanks.filterNot(_ == -1).toSet

      connection ! Tcp.Write(ByteString.fromByteBuffer(aRank.toByteBuffer(worldSize)))
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
              peerRef ! RequestWorkerHostPort
            }

            // a countdown for DivulgedHostPort messages.
            stay using DataStruct(Seq.empty[DataField], waitConnNodes.size - 1)
          }
      }

    case Event(DivulgedWorkerHostPort(peerRank, peerHost, peerPort), data) =>
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
          goto(AwaitingPortNumber)
        case _ =>
          stashSpillOver(buf)
          goto(BuildingLinkMap) using StructNodes
      }
  }

  when(AwaitingPortNumber) {
    case Event(Tcp.Received(assignedPort), _) =>
      assert(assignedPort.length == 4)
      port = assignedPort.asNativeOrderByteBuffer.getInt
      log.debug(s"Rank $rank listening @ $host:$port")
      // wait until the worker closes connection.
      if (peerClosed) goto(SetupComplete) else stay

    case Event(Tcp.PeerClosed, _) =>
      peerClosed = true
      if (port == 0) stay else goto(SetupComplete)
  }

  when(SetupComplete) {
    case Event(ReduceWaitCount(count: Int), _) =>
      awaitingAcceptance -= count
      // check peerClosed to avoid prematurely stopping this actor (which sends RST to worker)
      if (awaitingAcceptance == 0 && peerClosed) {
        tracker ! DropFromWaitingList(rank)
        // no longer needed.
        context.stop(self)
      }
      stay

    case Event(AcknowledgeAcceptance(peers, numBad), _) =>
      awaitingAcceptance = numBad
      tracker ! WorkerStarted(host, rank, awaitingAcceptance)
      peers.values.foreach { peer =>
        peer ! ReduceWaitCount(1)
      }

      if (awaitingAcceptance == 0 && peerClosed) self ! PoisonPill

      stay

    // can only divulge the complete host and port information
    // when this worker is declared fully connected (otherwise
    // port information is still missing.)
    case Event(RequestWorkerHostPort, _) =>
      sender() ! DivulgedWorkerHostPort(rank, host, port)
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
      peerClosed = true
      if (transient) context.stop(self)
      stay
  }
}

private[scala] object RabitWorkerHandler {
  val MAGIC_NUMBER = 0xff99

  // Finite states of this actor, which acts like a FSM.
  // The following states are defined in order as the FSM progresses.
  sealed trait State

  // [1] Initial state, awaiting worker to send magic number per protocol.
  case object AwaitingHandshake extends State
  // [2] Awaiting worker to send command (start/print/recover/shutdown etc.)
  case object AwaitingCommand extends State
  // [3] Brokers connections between workers per ring/tree/parent link map.
  case object BuildingLinkMap extends State
  // [4] A transient state in which the worker reports the number of errors in establishing
  // connections to other peer workers. If no errors, transition to next state.
  case object AwaitingErrorCount extends State
  // [5] Awaiting the worker to report its port number for accepting connections from peer workers.
  // This port number information is later forwarded to linked workers.
  case object AwaitingPortNumber extends State
  // [6] Final state after completing the setup with the connecting worker. At this stage, the
  // worker will have closed the Tcp connection. The actor remains alive to handle messages from
  // peer actors representing workers with pending setups.
  case object SetupComplete extends State

  sealed trait DataField
  case object IntField extends DataField
  // an integer preceding the actual string
  case object StringField extends DataField
  case object IntSeqField extends DataField

  object DataStruct {
    def apply(): DataStruct = DataStruct(Seq.empty[DataField], 0)
  }

  // Internal data pertaining to individual state, used to verify the validity of packets sent by
  // workers.
  case class DataStruct(fields: Seq[DataField], counter: Int) {
    /**
      * Validate whether the provided buffer is complete (i.e., contains
      * all data fields specified for this DataStruct.)
 *
      * @param buf a byte buffer containing received data.
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

  // ---- Messages between RabitTrackerHandler and RabitTrackerConnectionHandler ----

  // RabitWorkerHandler --> RabitTrackerHandler
  sealed trait RabitWorkerRequest
  // RabitWorkerHandler <-- RabitTrackerHandler
  sealed trait RabitWorkerResponse

  // Representations of decoded worker commands.
  abstract class TrackerCommand(val command: String) extends RabitWorkerRequest {
    def rank: Int
    def worldSize: Int
    def jobId: String

    def encode: ByteString = {
      val buf = ByteBuffer.allocate(4 * 4 + jobId.length + command.length)
        .order(ByteOrder.nativeOrder())

      buf.putInt(rank).putInt(worldSize).putInt(jobId.length).put(jobId.getBytes())
        .putInt(command.length).put(command.getBytes()).flip()

      ByteString.fromByteBuffer(buf)
    }
  }

  case class WorkerStart(rank: Int, worldSize: Int, jobId: String)
    extends TrackerCommand("start")
  case class WorkerShutdown(rank: Int, worldSize: Int, jobId: String)
    extends TrackerCommand("shutdown")
  case class WorkerRecover(rank: Int, worldSize: Int, jobId: String)
    extends TrackerCommand("recover")
  case class WorkerTrackerPrint(rank: Int, worldSize: Int, jobId: String, msg: String)
    extends TrackerCommand("print") {

    override def encode: ByteString = {
      val buf = ByteBuffer.allocate(4 * 5 + jobId.length + command.length + msg.length)
        .order(ByteOrder.nativeOrder())

      buf.putInt(rank).putInt(worldSize).putInt(jobId.length).put(jobId.getBytes())
        .putInt(command.length).put(command.getBytes())
        .putInt(msg.length).put(msg.getBytes()).flip()

      ByteString.fromByteBuffer(buf)
    }
  }

  // Request to remove the worker of given rank from the list of workers awaiting peer connections.
  case class DropFromWaitingList(rank: Int) extends RabitWorkerRequest
  // Notify the tracker that the worker of given rank has finished setup and started.
  case class WorkerStarted(host: String, rank: Int, awaitingAcceptance: Int)
    extends RabitWorkerRequest
  // Request the set of workers to connect to, according to the LinkMap structure.
  case class RequestAwaitConnWorkers(rank: Int, toConnectSet: Set[Int])
    extends RabitWorkerRequest

  // Request, from the tracker, the set of nodes to connect.
  case class AwaitingConnections(workers: Map[Int, ActorRef], numBad: Int)
    extends RabitWorkerResponse

  // ---- Messages between ConnectionHandler actors ----
  sealed trait IntraWorkerMessage

  // Notify neighboring workers to decrease the counter of awaiting workers by `count`.
  case class ReduceWaitCount(count: Int) extends IntraWorkerMessage
  // Request host and port information from peer ConnectionHandler actors (acting on behave of
  // connecting workers.) This message will be brokered by RabitTrackerHandler.
  case object RequestWorkerHostPort extends IntraWorkerMessage
  // Response to the above request
  case class DivulgedWorkerHostPort(rank: Int, host: String, port: Int) extends IntraWorkerMessage
  // A reminder to send ReduceWaitCount messages once the actor is in state "SetupComplete".
  case class AcknowledgeAcceptance(peers: Map[Int, ActorRef], numBad: Int)
    extends IntraWorkerMessage

  // ---- End of message definitions ----

  def props(host: String, worldSize: Int, tracker: ActorRef, connection: ActorRef): Props = {
    Props(new RabitWorkerHandler(host, worldSize, tracker, connection))
  }
}
