/*
 Copyright (c) 2021 by Contributors

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

package ml.dmlc.xgboost4j.scala.nvidia.spark

import ai.rapids.cudf.Table
import com.nvidia.spark.rapids.ColumnarRdd

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame}
import org.apache.spark.TaskContext

private[spark] object GpuUtils {

  def toColumnarRdd(df: DataFrame): RDD[Table] = ColumnarRdd(df)

  def seqIntToSeqInteger(x: Seq[Int]): Seq[Integer] = x.map(new Integer(_))

  // APIs for gpu column data related
  def buildColumnDataBatch(featureNames: Seq[String],
      labelName: String,
      weightName: String,
      marginName: String,
      groupName: String,
      dataFrame: DataFrame): ColumnDataBatch = {
    // Some check first
    val schema = dataFrame.schema
    val featureNameSet = featureNames.distinct
    MLUtils.validateSchema(schema, featureNameSet, labelName, weightName, marginName)

    // group column
    val (opGroup, groupId) = if (groupName.isEmpty) {
      (None, None)
    } else {
      MLUtils.checkNumericType(schema, groupName)
      (Some(groupName), Some(schema.fieldIndex(groupName)))
    }
    // weight and base margin columns
    val Seq(weightId, marginId) = Seq(weightName, marginName).map {
      name =>
        if (name.isEmpty) None else Some(schema.fieldIndex(name))
    }

    val colsIndices = ColumnIndices(featureNameSet.map(schema.fieldIndex),
      schema.fieldIndex(labelName), weightId, marginId, groupId)
    ColumnDataBatch(dataFrame, colsIndices, opGroup)
  }

  // This method should be called on executor side
  def getGpuId(isLocal: Boolean): Int = {
    var gpuId = 0
    val context = TaskContext.get()
    if (!isLocal) {
      val resources = context.resources()
      val assignedGpuAddrs = resources.get("gpu").getOrElse(
        throw new RuntimeException("Spark could not allocate gpus for executor"))
      gpuId = if (assignedGpuAddrs.addresses.length < 1) {
        throw new RuntimeException("executor could not get specific address of gpu")
      } else assignedGpuAddrs.addresses(0).toInt
    }
    gpuId
  }

}

private[spark] case class ColumnIndices(
  featureIds: Seq[Int],
  labelId: Int,
  weightId: Option[Int],
  marginId: Option[Int],
  groupId: Option[Int])

private[spark] case class ColumnDataBatch(
  rawDF: DataFrame,
  colIndices: ColumnIndices,
  groupColName: Option[String])
