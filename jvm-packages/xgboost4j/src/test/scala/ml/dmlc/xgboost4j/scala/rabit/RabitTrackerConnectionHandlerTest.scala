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

import java.nio.{ByteBuffer, ByteOrder}

import akka.actor.{ActorRef, ActorSystem}
import akka.io.Tcp
import akka.testkit.{ImplicitSender, TestFSMRef, TestKit, TestProbe}
import akka.util.ByteString
import ml.dmlc.xgboost4j.scala.rabit.handler.RabitWorkerHandler
import ml.dmlc.xgboost4j.scala.rabit.handler.RabitWorkerHandler._
import ml.dmlc.xgboost4j.scala.rabit.util.LinkMap
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpecLike, Matchers}

import scala.concurrent.Promise

object RabitTrackerConnectionHandlerTest {
  def intSeqToByteString(seq: Seq[Int]): ByteString = {
    val buf = ByteBuffer.allocate(seq.length * 4).order(ByteOrder.nativeOrder())
    seq.foreach { i => buf.putInt(i) }
    buf.flip()
    ByteString.fromByteBuffer(buf)
  }
}

@RunWith(classOf[JUnitRunner])
class RabitTrackerConnectionHandlerTest
  extends TestKit(ActorSystem("RabitTrackerConnectionHandlerTest"))
    with FlatSpecLike with Matchers with ImplicitSender {

  import RabitTrackerConnectionHandlerTest._

  val magic = intSeqToByteString(List(0xff99))

  "RabitTrackerConnectionHandler" should "handle Rabit client 'start' command properly" in {
    val trackerProbe = TestProbe()
    val connProbe = TestProbe()

    val worldSize = 4

    val fsm = TestFSMRef(new RabitWorkerHandler("localhost", worldSize,
      trackerProbe.ref, connProbe.ref))
    fsm.stateName shouldEqual RabitWorkerHandler.AwaitingHandshake

    // send mock magic number
    fsm ! Tcp.Received(magic)
    connProbe.expectMsg(Tcp.Write(magic))

    fsm.stateName shouldEqual RabitWorkerHandler.AwaitingCommand
    fsm.stateData shouldEqual RabitWorkerHandler.StructTrackerCommand
    // ResumeReading should be seen once state transitions
    connProbe.expectMsg(Tcp.ResumeReading)

    // send mock tracker command in fragments: the handler should be able to handle it.
    val bufRank = ByteBuffer.allocate(8).order(ByteOrder.nativeOrder())
    bufRank.putInt(0).putInt(worldSize).flip()

    val bufJobId = ByteBuffer.allocate(5).order(ByteOrder.nativeOrder())
    bufJobId.putInt(1).put(Array[Byte]('0')).flip()

    val bufCmd = ByteBuffer.allocate(9).order(ByteOrder.nativeOrder())
    bufCmd.putInt(5).put("start".getBytes()).flip()

    fsm ! Tcp.Received(ByteString.fromByteBuffer(bufRank))
    fsm ! Tcp.Received(ByteString.fromByteBuffer(bufJobId))

    // the state should not change for incomplete command data.
    fsm.stateName shouldEqual RabitWorkerHandler.AwaitingCommand

    // send the last fragment, and expect message at tracker actor.
    fsm ! Tcp.Received(ByteString.fromByteBuffer(bufCmd))
    trackerProbe.expectMsg(WorkerStart(0, worldSize, "0"))

    val linkMap = new LinkMap(worldSize)
    val assignedRank = linkMap.assignRank(0)
    trackerProbe.reply(assignedRank)

    connProbe.expectMsg(Tcp.Write(ByteString.fromByteBuffer(
      assignedRank.toByteBuffer(worldSize)
    )))

    // reading should be suspended upon transitioning to BuildingLinkMap
    connProbe.expectMsg(Tcp.SuspendReading)
    // state should transition with according state data changes.
    fsm.stateName shouldEqual RabitWorkerHandler.BuildingLinkMap
    fsm.stateData shouldEqual RabitWorkerHandler.StructNodes
    connProbe.expectMsg(Tcp.ResumeReading)

    // since the connection handler in test has rank 0, it will not have any nodes to connect to.
    fsm ! Tcp.Received(intSeqToByteString(List(0)))
    trackerProbe.expectMsg(RequestAwaitConnWorkers(0, fsm.underlyingActor.getNeighboringWorkers))

    // return mock response to the connection handler
    val awaitConnPromise = Promise[AwaitingConnections]()
    awaitConnPromise.success(AwaitingConnections(Map.empty[Int, ActorRef],
      fsm.underlyingActor.getNeighboringWorkers.size
    ))
    fsm ! awaitConnPromise.future
    connProbe.expectMsg(Tcp.Write(
      intSeqToByteString(List(0, fsm.underlyingActor.getNeighboringWorkers.size))
    ))
    connProbe.expectMsg(Tcp.SuspendReading)
    fsm.stateName shouldEqual RabitWorkerHandler.AwaitingErrorCount
    connProbe.expectMsg(Tcp.ResumeReading)

    // send mock error count (0)
    fsm ! Tcp.Received(intSeqToByteString(List(0)))

    fsm.stateName shouldEqual RabitWorkerHandler.AwaitingPortNumber
    connProbe.expectMsg(Tcp.ResumeReading)

    // simulate Tcp.PeerClosed event first, then Tcp.Received to test handling of async events.
    fsm ! Tcp.PeerClosed
    // state should not transition
    fsm.stateName shouldEqual RabitWorkerHandler.AwaitingPortNumber
    fsm ! Tcp.Received(intSeqToByteString(List(32768)))

    fsm.stateName shouldEqual RabitWorkerHandler.SetupComplete
    connProbe.expectMsg(Tcp.ResumeReading)

    trackerProbe.expectMsg(RabitWorkerHandler.WorkerStarted("localhost", 0, 2))

    val handlerStopProbe = TestProbe()
    handlerStopProbe watch fsm

    // simulate connections from other workers by mocking ReduceWaitCount commands
    fsm ! RabitWorkerHandler.ReduceWaitCount(1)
    fsm.stateName shouldEqual RabitWorkerHandler.SetupComplete
    fsm ! RabitWorkerHandler.ReduceWaitCount(1)
    trackerProbe.expectMsg(RabitWorkerHandler.DropFromWaitingList(0))
    handlerStopProbe.expectTerminated(fsm)

    // all done.
  }

  it should "forward print command to tracker" in {
    val trackerProbe = TestProbe()
    val connProbe = TestProbe()

    val fsm = TestFSMRef(new RabitWorkerHandler("localhost", 4,
      trackerProbe.ref, connProbe.ref))
    fsm.stateName shouldEqual RabitWorkerHandler.AwaitingHandshake

    fsm ! Tcp.Received(magic)
    connProbe.expectMsg(Tcp.Write(magic))

    fsm.stateName shouldEqual RabitWorkerHandler.AwaitingCommand
    fsm.stateData shouldEqual RabitWorkerHandler.StructTrackerCommand
    // ResumeReading should be seen once state transitions
    connProbe.expectMsg(Tcp.ResumeReading)

    val printCmd = WorkerTrackerPrint(0, 4, "print", "hello world!")
    fsm ! Tcp.Received(printCmd.encode)

    trackerProbe.expectMsg(printCmd)
  }

  it should "handle fragmented print command without throwing exception" in {
    val trackerProbe = TestProbe()
    val connProbe = TestProbe()

    val fsm = TestFSMRef(new RabitWorkerHandler("localhost", 4,
      trackerProbe.ref, connProbe.ref))
    fsm.stateName shouldEqual RabitWorkerHandler.AwaitingHandshake

    fsm ! Tcp.Received(magic)
    connProbe.expectMsg(Tcp.Write(magic))

    fsm.stateName shouldEqual RabitWorkerHandler.AwaitingCommand
    fsm.stateData shouldEqual RabitWorkerHandler.StructTrackerCommand
    // ResumeReading should be seen once state transitions
    connProbe.expectMsg(Tcp.ResumeReading)

    val printCmd = WorkerTrackerPrint(0, 4, "0", "fragmented!")
    // 4 (rank: Int) + 4 (worldSize: Int) + (4+1) (jobId: String) + (4+5) (command: String) = 22
    val (partialMessage, remainder) = printCmd.encode.splitAt(22)

    // make sure that the partialMessage in itself is a valid command
    val partialMsgBuf = ByteBuffer.allocate(22).order(ByteOrder.nativeOrder())
    partialMsgBuf.put(partialMessage.asByteBuffer)
    RabitWorkerHandler.StructTrackerCommand.verify(partialMsgBuf) shouldBe true

    fsm ! Tcp.Received(partialMessage)
    fsm ! Tcp.Received(remainder)

    trackerProbe.expectMsg(printCmd)
  }

  it should "handle spill-over Tcp data correctly between state transition" in {
    val trackerProbe = TestProbe()
    val connProbe = TestProbe()

    val worldSize = 4

    val fsm = TestFSMRef(new RabitWorkerHandler("localhost", worldSize,
      trackerProbe.ref, connProbe.ref))
    fsm.stateName shouldEqual RabitWorkerHandler.AwaitingHandshake

    // send mock magic number
    fsm ! Tcp.Received(magic)
    connProbe.expectMsg(Tcp.Write(magic))

    fsm.stateName shouldEqual RabitWorkerHandler.AwaitingCommand
    fsm.stateData shouldEqual RabitWorkerHandler.StructTrackerCommand
    // ResumeReading should be seen once state transitions
    connProbe.expectMsg(Tcp.ResumeReading)

    // send mock tracker command in fragments: the handler should be able to handle it.
    val bufCmd = ByteBuffer.allocate(26).order(ByteOrder.nativeOrder())
    bufCmd.putInt(0).putInt(worldSize).putInt(1).put(Array[Byte]('0'))
      .putInt(5).put("start".getBytes())
      // spilled-over data
      .putInt(0).flip()

    // send data with 4 extra bytes corresponding to the next state.
    fsm ! Tcp.Received(ByteString.fromByteBuffer(bufCmd))

    trackerProbe.expectMsg(WorkerStart(0, worldSize, "0"))

    val linkMap = new LinkMap(worldSize)
    val assignedRank = linkMap.assignRank(0)
    trackerProbe.reply(assignedRank)

    connProbe.expectMsg(Tcp.Write(ByteString.fromByteBuffer(
      assignedRank.toByteBuffer(worldSize)
    )))

    // reading should be suspended upon transitioning to BuildingLinkMap
    connProbe.expectMsg(Tcp.SuspendReading)
    // state should transition with according state data changes.
    fsm.stateName shouldEqual RabitWorkerHandler.BuildingLinkMap
    fsm.stateData shouldEqual RabitWorkerHandler.StructNodes
    connProbe.expectMsg(Tcp.ResumeReading)

    // the handler should be able to handle spill-over data, and stash it until state transition.
    trackerProbe.expectMsg(RequestAwaitConnWorkers(0, fsm.underlyingActor.getNeighboringWorkers))
  }
}
