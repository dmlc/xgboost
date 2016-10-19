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

package ml.dmlc.xgboost4j.scala

import java.nio.{ByteBuffer, ByteOrder}

import akka.actor.ActorSystem
import akka.io.Tcp
import akka.testkit.{ImplicitSender, TestFSMRef, TestKit, TestProbe}
import akka.util.ByteString
import ml.dmlc.xgboost4j.scala.handler.RabitTrackerConnectionHandler
import ml.dmlc.xgboost4j.scala.handler.RabitTrackerConnectionHandler.WorkerStart
import ml.dmlc.xgboost4j.scala.util.{AssignedRank, LinkMap}
import org.junit.runner.RunWith
import org.scalatest.{FlatSpecLike, Matchers}
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class RabitTrackerConnectionHandlerTest extends TestKit(ActorSystem("RabitTrackerTest"))
  with FlatSpecLike with Matchers with ImplicitSender {

  val magicBuf = ByteBuffer.allocate(4).order(ByteOrder.nativeOrder()).putInt(0xff99)
  magicBuf.flip()
  val magic = ByteString.fromByteBuffer(magicBuf)

  "LinkMap" should "construct tree/parent and ring maps correctly" in {

  }

  "RabitTrackerConnectionHandler" should "handle Rabit client 'start' command properly" in {
    val trackerProbe = TestProbe()
    val connProbe = TestProbe()

    val fsm = TestFSMRef(new RabitTrackerConnectionHandler("localhost", 4,
      trackerProbe.ref, connProbe.ref))
    fsm.stateName shouldEqual RabitTrackerConnectionHandler.AwaitingHandshake

    // send mock magic number
    fsm ! Tcp.Received(magic)
    connProbe.expectMsg(Tcp.Write(magic))

    fsm.stateName shouldEqual RabitTrackerConnectionHandler.AwaitingCommand
    fsm.stateData shouldEqual RabitTrackerConnectionHandler.StructTrackerCommand
    // ResumeReading should be seen once state transitions
    connProbe.expectMsg(Tcp.ResumeReading)

    // send mock tracker command in fragments: the handler should be able to handle it.
    val bufRank = ByteBuffer.allocate(8).order(ByteOrder.nativeOrder())
    bufRank.putInt(0).putInt(4).flip()

    val bufJobId = ByteBuffer.allocate(5).order(ByteOrder.nativeOrder())
    bufJobId.putInt(1).put(Array[Byte]('0')).flip()

    val bufCmd = ByteBuffer.allocate(9).order(ByteOrder.nativeOrder())
    bufCmd.putInt(5).put("start".getBytes()).flip()

    fsm ! Tcp.Received(ByteString.fromByteBuffer(bufRank))
    fsm ! Tcp.Received(ByteString.fromByteBuffer(bufJobId))

    // the state should not change for incomplete command data.
    fsm.stateName shouldEqual RabitTrackerConnectionHandler.AwaitingCommand

    // send the last fragment, and expect message at tracker actor.
    fsm ! Tcp.Received(ByteString.fromByteBuffer(bufCmd))
    trackerProbe.expectMsg(WorkerStart(0, 4, "0"))

    val linkMap = new LinkMap(4)
    val assignedRank = linkMap.assignRank(0)
    trackerProbe.reply(assignedRank)

    // connProbe.expectMsg(Tcp.Write)
    // reading should be suspended upon transitioning to BuildingLinkMap
    // connProbe.expectMsg(Tcp.SuspendReading)
    fsm.stateName shouldEqual RabitTrackerConnectionHandler.BuildingLinkMap

  }

  it should "forward print command to tracker" in {

  }

  it should "handle spill-over Tcp data correctly between state transition" in {

  }
}
