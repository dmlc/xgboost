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

package ml.dmlc.xgboost4j.scala.rabit.util

import java.nio.{ByteBuffer, ByteOrder}

/**
  * The assigned rank to a connecting Rabit worker, along with the information of the ranks of
  * its linked peer workers, which are critical to perform Allreduce.
  * When RabitWorkerHandler delegates "start" or "recover" commands from the connecting worker
  * client, RabitTrackerHandler utilizes LinkMap to figure out linkage relationships, and respond
  * with this class as a message, which is later encoded to byte string, and sent over socket
  * connection to the worker client.
  *
  * @param rank assigned rank (ranked by worker connection order: first worker connecting to the
  *             tracker is assigned rank 0, second with rank 1, etc.)
  * @param neighbors ranks of neighboring workers in a tree map.
  * @param ring ranks of neighboring workers in a ring map.
  * @param parent rank of the parent worker.
  */
private[rabit] case class AssignedRank(rank: Int, neighbors: Seq[Int],
                                       ring: (Int, Int), parent: Int) {
  /**
    * Encode the AssignedRank message into byte sequence for socket communication with Rabit worker
    * client.
    * @param worldSize the number of total distributed workers. Must match `numWorkers` used in
    *                  LinkMap.
    * @return a ByteBuffer containing encoded data.
    */
  def toByteBuffer(worldSize: Int): ByteBuffer = {
    val buffer = ByteBuffer.allocate(4 * (neighbors.length + 6)).order(ByteOrder.nativeOrder())
    buffer.putInt(rank).putInt(parent).putInt(worldSize).putInt(neighbors.length)
    // neighbors in tree structure
    neighbors.foreach { n => buffer.putInt(n) }
    buffer.putInt(if (ring._1 != -1 && ring._1 != rank) ring._1 else -1)
    buffer.putInt(if (ring._2 != -1 && ring._2 != rank) ring._2 else -1)

    buffer.flip()
    buffer
  }
}

private[rabit] class LinkMap(numWorkers: Int) {
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
    * Construct a ring connection used to recover local data.
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
