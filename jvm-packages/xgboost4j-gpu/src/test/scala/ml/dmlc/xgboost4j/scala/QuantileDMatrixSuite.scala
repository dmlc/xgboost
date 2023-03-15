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

package ml.dmlc.xgboost4j.scala

import scala.collection.mutable.ArrayBuffer

import ai.rapids.cudf.Table
import org.scalatest.funsuite.AnyFunSuite
import ml.dmlc.xgboost4j.gpu.java.CudfColumnBatch

class QuantileDMatrixSuite extends AnyFunSuite {

  test("QuantileDMatrix test") {

    val label1 = Array[java.lang.Float](25f, 21f, 22f, 20f, 24f)
    val weight1 = Array[java.lang.Float](1.3f, 2.31f, 0.32f, 3.3f, 1.34f)
    val baseMargin1 = Array[java.lang.Float](1.2f, 0.2f, 1.3f, 2.4f, 3.5f)

    val label2 = Array[java.lang.Float](9f, 5f, 4f, 10f, 12f)
    val weight2 = Array[java.lang.Float](3.0f, 1.3f, 3.2f, 0.3f, 1.34f)
    val baseMargin2 = Array[java.lang.Float](0.2f, 2.5f, 3.1f, 4.4f, 2.2f)

    withResource(new Table.TestBuilder()
      .column(1.2f, null.asInstanceOf[java.lang.Float], 5.2f, 7.2f, 9.2f)
      .column(0.2f, 0.4f, 0.6f, 2.6f, 0.10f.asInstanceOf[java.lang.Float])
      .build) { X_0 =>
      withResource(new Table.TestBuilder().column(label1: _*).build) { y_0 =>
        withResource(new Table.TestBuilder().column(weight1: _*).build) { w_0 =>
          withResource(new Table.TestBuilder().column(baseMargin1: _*).build) { m_0 =>
            withResource(new Table.TestBuilder()
              .column(11.2f, 11.2f, 15.2f, 17.2f, 19.2f.asInstanceOf[java.lang.Float])
              .column(1.2f, 1.4f, null.asInstanceOf[java.lang.Float], 12.6f, 10.10f).build)
            { X_1 =>
              withResource(new Table.TestBuilder().column(label2: _*).build) { y_1 =>
                withResource(new Table.TestBuilder().column(weight2: _*).build) { w_1 =>
                  withResource(new Table.TestBuilder().column(baseMargin2: _*).build) { m_1 =>
                    val batches = new ArrayBuffer[CudfColumnBatch]()
                    batches += new CudfColumnBatch(X_0, y_0, w_0, m_0)
                    batches += new CudfColumnBatch(X_1, y_1, w_1, m_1)
                    val dmatrix = new QuantileDMatrix(batches.toIterator, 0.0f, 8, 1)
                    assert(dmatrix.getLabel.sameElements(label1 ++ label2))
                    assert(dmatrix.getWeight.sameElements(weight1 ++ weight2))
                    assert(dmatrix.getBaseMargin.sameElements(baseMargin1 ++ baseMargin2))
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  /** Executes the provided code block and then closes the resource */
  private def withResource[T <: AutoCloseable, V](r: T)(block: T => V): V = {
    try {
      block(r)
    } finally {
      r.close()
    }
  }

}

