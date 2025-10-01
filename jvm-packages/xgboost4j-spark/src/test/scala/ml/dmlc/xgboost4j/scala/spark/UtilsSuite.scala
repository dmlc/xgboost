/*
 Copyright (c) 2014-2025 by Contributors

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

import ml.dmlc.xgboost4j.scala.spark.Utils.MLVectorToXGBLabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.scalatest.funsuite.AnyFunSuite

class UtilsSuite extends AnyFunSuite {

  test("MLVectorToXGBLabeledPoint.asXGB - dense vector conversion") {
    val denseValues = Array(1.0, 2.5, -1.0, 0.0, 3.7)
    val denseVector = Vectors.dense(denseValues)
    
    val xgbPoint = denseVector.asXGB
    
    assert(xgbPoint.label == 0.0f, "Label should be 0.0f for dummy label")
    assert(xgbPoint.size == denseValues.length, s"Size should be ${denseValues.length}")
    assert(xgbPoint.indices == null, "Indices should be null for dense vector")
    assert(xgbPoint.values.length == denseValues.length, "Values array length should match")
    
    assert(xgbPoint.weight == 1.0f, "Default weight should be 1.0f")
    assert(xgbPoint.group == -1, "Default group should be -1")
    assert(xgbPoint.baseMargin.isNaN, "Default baseMargin should be NaN")
  }

  test("MLVectorToXGBLabeledPoint.asXGB - sparse vector conversion") {
    val size = 10
    val indices = Array(0, 3, 7, 9)
    val values = Array(1.5, -2.0, 3.5, 0.8)
    val sparseVector = Vectors.sparse(size, indices, values)
    
    val xgbPoint = sparseVector.asXGB
    
    assert(xgbPoint.size == size, s"Size should be $size")
    assert(xgbPoint.indices != null, "Indices should not be null for sparse vector")
    assert(xgbPoint.indices.length == indices.length, "Indices array length should match")
    assert(xgbPoint.values.length == values.length, "Values array length should match")
    
    indices.zip(xgbPoint.indices).foreach { case (expected, actual) =>
      assert(expected == actual, s"Index mismatch: expected $expected, got $actual")
    }
    
    assert(xgbPoint.weight == 1.0f, "Default weight should be 1.0f")
    assert(xgbPoint.group == -1, "Default group should be -1")
    assert(xgbPoint.baseMargin.isNaN, "Default baseMargin should be NaN")
  }
}
