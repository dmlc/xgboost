/*
 Copyright (c) 2014-2023 by Contributors

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

import java.util.Arrays

import scala.util.Random

import org.scalatest.funsuite.AnyFunSuite
import ml.dmlc.xgboost4j.java.{DMatrix => JDMatrix}

class DMatrixSuite extends AnyFunSuite {
  test("create DMatrix from File") {
    val dmat = new DMatrix("../../demo/data/agaricus.txt.test?format=libsvm")
    // get label
    val labels: Array[Float] = dmat.getLabel
    // check length
    assert(dmat.rowNum === labels.length)
    // set weights
    val weights: Array[Float] = Arrays.copyOf(labels, labels.length)
    dmat.setWeight(weights)
    val dweights: Array[Float] = dmat.getWeight
    assert(weights === dweights)
  }

  test("create DMatrix from CSR") {
    // create Matrix from csr format sparse Matrix and labels
    /**
     * sparse matrix
     * 1 0 2 3 0
     * 4 0 2 3 5
     * 3 1 2 5 0
     */
    val data = List[Float](1, 2, 3, 4, 2, 3, 5, 3, 1, 2, 5).toArray
    val colIndex = List(0, 2, 3, 0, 2, 3, 4, 0, 1, 2, 3).toArray
    val rowHeaders = List[Long](0, 3, 7, 11).toArray
    val dmat1 = new DMatrix(rowHeaders, colIndex, data, JDMatrix.SparseType.CSR)
    assert(dmat1.rowNum === 3)
    val label1 = List[Float](1, 0, 1).toArray
    dmat1.setLabel(label1)
    val label2 = dmat1.getLabel
    assert(label2 === label1)

    val dmat2 = new DMatrix(rowHeaders, colIndex, data, JDMatrix.SparseType.CSR, 5, 1.0f, -1)
    assert(dmat2.nonMissingNum === 9);
  }

  test("create DMatrix from CSREx") {
    // create Matrix from csr format sparse Matrix and labels
    /**
     * sparse matrix
     * 1 0 2 3 0
     * 4 0 2 3 5
     * 3 1 2 5 0
     */
    val data = List[Float](1, 2, 3, 4, 2, 3, 5, 3, 1, 2, 5).toArray
    val colIndex = List(0, 2, 3, 0, 2, 3, 4, 0, 1, 2, 3).toArray
    val rowHeaders = List[Long](0, 3, 7, 11).toArray
    val dmat1 = new DMatrix(rowHeaders, colIndex, data, JDMatrix.SparseType.CSR, 5)
    assert(dmat1.rowNum === 3)
    val label1 = List[Float](1, 0, 1).toArray
    dmat1.setLabel(label1)
    val label2 = dmat1.getLabel
    assert(label2 === label1)
  }

  test("create DMatrix from CSC") {
    // create Matrix from csc format sparse Matrix and labels
    /**
     * sparse matrix
     * 1 0 2
     * 3 0 4
     * 0 2 3
     * 5 3 1
     * 2 5 0
     */
    val data = List[Float](1, 3, 5, 2, 2, 3, 5, 2, 4, 3, 1).toArray
    val rowIndex = List(0, 1, 3, 4, 2, 3, 4, 0, 1, 2, 3).toArray
    val colHeaders = List[Long](0, 4, 7, 11).toArray
    val dmat1 = new DMatrix(colHeaders, rowIndex, data, JDMatrix.SparseType.CSC)
    assert(dmat1.rowNum === 5)
    val label1 = List[Float](1, 0, 1, 1, 1).toArray
    dmat1.setLabel(label1)
    val label2 = dmat1.getLabel
    assert(label2 === label1)

    val dmat2 = new DMatrix(colHeaders, rowIndex, data, JDMatrix.SparseType.CSC, 5, 1.0f, -1)
    assert(dmat2.nonMissingNum === 9);
  }

  test("create DMatrix from CSCEx") {
    // create Matrix from csc format sparse Matrix and labels
    /**
     * sparse matrix
     * 1 0 2
     * 3 0 4
     * 0 2 3
     * 5 3 1
     * 2 5 0
     */
    val data = List[Float](1, 3, 5, 2, 2, 3, 5, 2, 4, 3, 1).toArray
    val rowIndex = List(0, 1, 3, 4, 2, 3, 4, 0, 1, 2, 3).toArray
    val colHeaders = List[Long](0, 4, 7, 11).toArray
    val dmat1 = new DMatrix(colHeaders, rowIndex, data, JDMatrix.SparseType.CSC, 5)
    assert(dmat1.rowNum === 5)
    val label1 = List[Float](1, 0, 1, 1, 1).toArray
    dmat1.setLabel(label1)
    val label2 = dmat1.getLabel
    assert(label2 === label1)
  }

  test("create DMatrix from DenseMatrix") {
    val nrow = 10
    val ncol = 5
    val data0 = new Array[Float](nrow * ncol)
    // put random nums
    for (i <- data0.indices) {
      data0(i) = Random.nextFloat()
    }
    // create label
    val label0 = new Array[Float](nrow)
    for (i <- label0.indices) {
      label0(i) = Random.nextFloat()
    }
    val dmat0 = new DMatrix(data0, nrow, ncol, Float.NaN)
    dmat0.setLabel(label0)
    // check
    assert(dmat0.rowNum === 10)
    assert(dmat0.getLabel.length === 10)
    // set weights for each instance
    val weights = new Array[Float](nrow)
    for (i <- weights.indices) {
      weights(i) = Random.nextFloat()
    }
    dmat0.setWeight(weights)
    assert(weights === dmat0.getWeight)
  }

  test("create DMatrix from DenseMatrix with missing value") {
    val nrow = 10
    val ncol = 5
    val data0 = new Array[Float](nrow * ncol)
    // put random nums
    for (i <- data0.indices) {
      if (i % 10 == 0) {
        data0(i) = -0.1f
      } else {
        data0(i) = Random.nextFloat()
      }
    }
    // create label
    val label0 = new Array[Float](nrow)
    for (i <- label0.indices) {
      label0(i) = Random.nextFloat()
    }
    val dmat0 = new DMatrix(data0, nrow, ncol, -0.1f)
    dmat0.setLabel(label0)
    // check
    assert(dmat0.rowNum === 10)
    assert(dmat0.getLabel.length === 10)
  }
}
