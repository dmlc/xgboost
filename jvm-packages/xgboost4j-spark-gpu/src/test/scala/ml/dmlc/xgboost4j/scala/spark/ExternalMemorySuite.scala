/*
 Copyright (c) 2025 by Contributors

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

package ml.dmlc.xgboost4j.scala.spark

import scala.collection.mutable.ArrayBuffer

import ai.rapids.cudf.Table

import ml.dmlc.xgboost4j.java.CudfColumnBatch
import ml.dmlc.xgboost4j.scala.rapids.spark.GpuTestSuite
import ml.dmlc.xgboost4j.scala.spark.Utils.withResource

class ExternalMemorySuite extends GpuTestSuite {

  private def assertColumnBatchEqual(lhs: Array[CudfColumnBatch],
                                     rhs: Array[CudfColumnBatch]): Unit = {
    def assertTwoTable(lhsTable: Table, rhsTable: Table): Unit = {
      assert(lhsTable.getNumberOfColumns === rhsTable.getNumberOfColumns)
      for (i <- 0 until lhsTable.getNumberOfColumns) {
        val lColumn = lhsTable.getColumn(i)
        val rColumn = rhsTable.getColumn(i)

        val lHost = lColumn.copyToHost()
        val rHost = rColumn.copyToHost()

        assert(lHost.getRowCount === rHost.getRowCount)
        for (j <- 0 until lHost.getRowCount.toInt) {
          assert(lHost.getFloat(j) === rHost.getFloat(j))
        }
      }
    }

    assert(lhs.length === rhs.length)
    for ((l, r) <- lhs.zip(rhs)) {
      assertTwoTable(l.getFeatureTable, r.getFeatureTable)
      assertTwoTable(l.getLabelTable, r.getLabelTable)
    }
  }

  def runExternalMemoryTest(buildExternalMemory: (Iterator[Table], ColumnIndices) =>
    ExternalMemoryIterator): Unit = {

    withResource(new Table.TestBuilder()
      .column(1.0f, 2.0f, 3.0f.asInstanceOf[java.lang.Float])
      .column(4.0f, 5.0f, 6.0f.asInstanceOf[java.lang.Float])
      .column(7.0f, 8.0f, 9.0f.asInstanceOf[java.lang.Float])
      .build) { table1 =>

      withResource(new Table.TestBuilder()
        .column(11.0f, 12.0f, 13.0f.asInstanceOf[java.lang.Float])
        .column(14.0f, 15.0f, 16.0f.asInstanceOf[java.lang.Float])
        .column(17.0f, 18.0f, 19.0f.asInstanceOf[java.lang.Float])
        .build) { table2 =>

        val tables = Seq(table1, table2)

        val indices = ColumnIndices(labelId = 0, featureIds = Some(Seq(1, 2)), featureId = None,
          weightId = None, marginId = None, groupId = None)
        val extMemIter = buildExternalMemory(tables.toIterator, indices)
        val expectTables = ArrayBuffer.empty[CudfColumnBatch]
        while (extMemIter.hasNext) {
          val table = extMemIter.next().asInstanceOf[CudfColumnBatch]
          expectTables.append(table)
        }
        // The hasNext has swap the iterator internally, so we can still get the
        // value for the next round of iteration

        val targetTables = ArrayBuffer.empty[CudfColumnBatch]
        while (extMemIter.hasNext) {
          val table = extMemIter.next().asInstanceOf[CudfColumnBatch]
          targetTables.append(table)
        }

        assertColumnBatchEqual(expectTables.toArray, targetTables.toArray)
      }
    }
  }

  test("DiskExternalMemory") {
    val buildIterator = (input: Iterator[Table], indices: ColumnIndices) => {
      val iter = new ExternalMemoryIterator(input, indices, Some("/tmp/"))
      assert(iter.externalMemory.isInstanceOf[DiskExternalMemoryIterator])
      iter
    }
    runExternalMemoryTest(buildIterator)
  }
}
