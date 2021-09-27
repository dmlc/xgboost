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

import ml.dmlc.xgboost4j.java.XGBoostSparkJNI
import ml.dmlc.xgboost4j.java.spark.GpuColumnBatch

import org.apache.spark.TaskContext
import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.expressions.UnsafeRow
import org.apache.spark.sql.types.StructType
import org.apache.spark.unsafe.Platform

private[rapids] class ColumnBatchToRow(rowSchema: StructType) {
  private var batches: Seq[ColumnBatchIter] = Seq()
  private lazy val batchIter = batches.toIterator
  private var currentBatchIter: ColumnBatchIter = _
  private var info: ColumnToRowInfo = _

  def appendColumnBatch(batch: GpuColumnBatch): this.type = {
    batches = batches :+ new ColumnBatchIter(batch, initAndCheck(batch))
    this
  }

  def toIterator: Iterator[Row] = {
    val iter = new Iterator[Row] with AutoCloseable {

      override def hasNext: Boolean = {
        (currentBatchIter != null && currentBatchIter.hasNext) || nextIterator()
      }

      override def next(): Row = {
        currentBatchIter.next()
      }

      override def close(): Unit = {
        if (currentBatchIter != null) {
          currentBatchIter.close()
        }
      }

      private def nextIterator(): Boolean = {
        if (batchIter.hasNext) {
          close()
          currentBatchIter = batchIter.next()
          try {
            hasNext
          } finally {
          }
        } else {
          false
        }
      }
    }
    TaskContext.get.addTaskCompletionListener[Unit](_ => iter.close())
    iter
  }

  private def initAndCheck(batch: GpuColumnBatch): ColumnToRowInfo = {
    val origSchema = batch.getSchema
    if (info == null) {
      // check schema relationship
      require(rowSchema.nonEmpty && rowSchema.forall(origSchema.contains(_)))
      // Indices of column to be converted to row
      val indicesToRow = rowSchema.names.map(origSchema.fieldIndex)
      // row shape
      // number of columns to be converted to row
      val numColsToRow = rowSchema.length
      val rowSize = UnsafeRow.calculateBitSetWidthInBytes(numColsToRow) + numColsToRow * 8
      // row parser and converter
      val row = new UnsafeRow(numColsToRow)
      val converter = new RowConverter(rowSchema,
        indicesToRow.map(batch.getColumnVector(_).getType()))

      // Finally create the info
      info = new ColumnToRowInfo(batch.getNumRows, indicesToRow, rowSize, row, converter,
        origSchema)
    } else {
      if (!origSchema.equals(info.origSchema)) {
        // All the column batches added to a ColumnBatchToRow should have the same schema.
        throw new RuntimeException(s"Schema of GpuColumnBatch at [${batches.length}]" +
          s" diverges from the first one.")
      }
    }
    info
  }

  private class ColumnToRowInfo(
    val numRows: Long,
    val indicesToRow: Array[Int],
    val rowSize: Int,
    val rowParser: UnsafeRow,
    val converter: RowConverter,
    private[ColumnBatchToRow] val origSchema: StructType)


  // Iterator to iterate one single GpuColmunBatch
  private class ColumnBatchIter(batch: GpuColumnBatch, info: ColumnToRowInfo)
      extends Iterator[Row] with AutoCloseable {
    private var nextRow = 0
    private var buffer = initBuffer()

    private def initBuffer(): Long = {
      val colDatas = batch.getAsColumnData(info.indicesToRow: _*)
      XGBoostSparkJNI.buildUnsafeRows(colDatas: _*)
    }

    override def hasNext: Boolean = nextRow < info.numRows

    override def next(): Row = {
      if (nextRow >= info.numRows) {
        throw new NoSuchElementException
      }
      info.rowParser.pointTo(null, buffer + info.rowSize * nextRow, info.rowSize)
      nextRow += 1
      info.converter.toExternalRow(info.rowParser)
    }

    override def close(): Unit = {
      if (buffer != 0) {
        Platform.freeMemory(buffer)
        buffer = 0
      }
    }
  }
}

private[rapids] object ColumnBatchToRow {

  def buildRowSchema(origSchema: StructType, toRowColNames: Seq[String],
    buildAllColumns: Boolean = true): StructType = {
    if (toRowColNames.nonEmpty) {
      val rowSchema = origSchema(toRowColNames.toSet)
      require(rowSchema.forall(f => RowConverter.isSupportedType(f.dataType)))
      rowSchema
    } else {
      if (buildAllColumns) {
        origSchema
      } else {
        val rowFields = origSchema.filter(f => RowConverter.isSupportedType(f.dataType))
        StructType(rowFields)
      }
    }
  }
}
