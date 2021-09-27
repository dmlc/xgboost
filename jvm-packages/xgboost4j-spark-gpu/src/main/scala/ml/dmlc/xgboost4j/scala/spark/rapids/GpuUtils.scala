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

package ml.dmlc.xgboost4j.scala.spark.rapids

import scala.collection.JavaConverters._

import ai.rapids.cudf.Table
import com.nvidia.spark.rapids.ColumnarRdd
import ml.dmlc.xgboost4j.gpu.java.CudfColumnBatch
import ml.dmlc.xgboost4j.java.spark.GpuColumnBatch
import ml.dmlc.xgboost4j.scala.DMatrix

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.TaskContext
import org.apache.spark.sql.types.StructType

private[spark] object GpuUtils {

  // APIs for plugin related
  def isRapidsEnabled(data: Dataset[_]): Boolean = {
    val pluginName = data.sparkSession.sparkContext.getConf.get("spark.sql.extensions", "")
    pluginName.split(",").contains("com.nvidia.spark.rapids.SQLExecPlugin")
  }

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

  // For transform
  def buildDMatrixAndColumnToRowIncrementally(missing: Float,
      iter: Iterator[GpuColumnBatch],
      featureIds: Seq[Int],
      rowSchema: StructType): (DMatrix, ColumnBatchToRow) = {
    // Create a convert first
    val columnToRow: ColumnBatchToRow = new ColumnBatchToRow(rowSchema)
    var dm: DMatrix = null

    while (iter.hasNext) {
      val colBatch = iter.next
      val cudfColumnBatch = new CudfColumnBatch(
        colBatch.slice(seqIntToSeqInteger(featureIds).asJava), null, null, null)
      if (dm == null) {
        // nthread now is useless, so assign it to 1
        dm = new DMatrix(cudfColumnBatch, missing, 1)
      } else {
        // TODO pending by native support, append data into a DMatrix
      }
      columnToRow.appendColumnBatch(colBatch)
      colBatch.close()
    }
    (dm, columnToRow)
  }

  // FIXME This is a WAR before native supports building DMatrix incrementally
  def buildDMatrixAndColumnToRow(missing: Float,
      iter: Iterator[GpuColumnBatch],
      featureIds: Seq[Int],
      rowSchema: StructType): (DMatrix, ColumnBatchToRow) = {
    // Merge all GpuColumnBatches
    if (iter.isEmpty) {
      (null, null)
    } else {
      val singleColBatch = GpuColumnBatch.merge(iter.toArray: _*)
      // Create ColumnBatchToRow
      val columnToRow = new ColumnBatchToRow(rowSchema)
        .appendColumnBatch(singleColBatch)
      // Create DMatrix
      val cudfColumnBatch = new CudfColumnBatch(
        singleColBatch.slice(seqIntToSeqInteger(featureIds).asJava), null, null, null)
      val dm = new DMatrix(cudfColumnBatch, missing, 1)
      singleColBatch.close()
      (dm, columnToRow)
    }
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

private[rapids] case class ColumnIndices(
  featureIds: Seq[Int],
  labelId: Int,
  weightId: Option[Int],
  marginId: Option[Int],
  groupId: Option[Int])

private[rapids] case class ColumnDataBatch(
  rawDF: DataFrame,
  colIndices: ColumnIndices,
  groupColName: Option[String])
