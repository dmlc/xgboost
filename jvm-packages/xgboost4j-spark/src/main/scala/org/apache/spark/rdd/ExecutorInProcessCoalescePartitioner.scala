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

package org.apache.spark.rdd

import org.apache.commons.logging.LogFactory

import org.apache.spark.Partition
import org.apache.spark.SparkException
import org.apache.spark.scheduler.ExecutorCacheTaskLocation
import org.apache.spark.scheduler.TaskLocation

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

class ExecutorInProcessCoalescePartitioner(val balanceSlack: Double = 0.10)
  extends PartitionCoalescer with Serializable {
  private val logger = LogFactory.getLog("ExecutorInProcessCoalescePartitioner")

  def coalesce(maxPartitions: Int, prev: RDD[_]): Array[PartitionGroup] = {
    val map = new mutable.HashMap[String, mutable.HashSet[Partition]]()

    // System.out.println("xgbtck rddname " + prev.getClass.getName
    //  + "\n " + prev.toDebugString)
    // System.out.println("xgbtck rddname " + prev.prev().getClass.getName)
    // System.out.println("xgbtck rddname " + prev.prev().prev().getClass.getName)

    val groupArr = ArrayBuffer[PartitionGroup]()
    prev.partitions.foreach(p => {
      val loc = prev.context.getPreferredLocs(prev, p.index)
      loc.foreach{
      case location : ExecutorCacheTaskLocation =>
        // System.out.println("xgbtck partitionloc " + location.getClass.getName)
        val execLoc = "executor_" + location.host + "_" + location.executorId
        val partValue = map.getOrElse(execLoc, new mutable.HashSet[Partition]())
        partValue.add(p)
        map.put(execLoc, partValue)
        // System.out.println("xgbtck coalescePartitioner partid"
        //  +  String.valueOf(p.index)
        //  + " location = " + execLoc)

      case loc : TaskLocation =>
        System.out.println("xgbtck partitionloc " + loc.getClass.getName)
        System.out.println("xgbtck partitionloc " + loc.host)
        logger.error("Invalid location : ")
      }
    })
    map.foreach(x => {
      val pg = new PartitionGroup(Some(x._1))
      val list = x._2.toList.sortWith(_.index < _.index);
      list.foreach(part => pg.partitions += part)
      groupArr += pg
    })
    if (groupArr.length == 0) throw new SparkException("No partitions or" +
      " no locations for partitions found.")

    val sortedGroupArr = groupArr.sortWith(_.partitions(0).index < _.partitions(0).index)

    sortedGroupArr.foreach(pg => {
      System.out.print(" xgbtck executorcoalesce " + pg.prefLoc + " : ")
      pg.partitions.foreach(part => System.out.print(String.valueOf(part.index) + " "))
      System.out.print("\n")
    })
    return sortedGroupArr.toArray
  }
}

