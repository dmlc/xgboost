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

import _root_.scala.collection.JavaConverters._

import ml.dmlc.xgboost4j.LabeledPoint
import ml.dmlc.xgboost4j.java.{Column, ColumnBatch, DataBatch, XGBoostError, DMatrix => JDMatrix}

class DMatrix private[scala](private[scala] val jDMatrix: JDMatrix) {
  /**
   * init DMatrix from file (svmlight format)
   *
   * @param dataPath path of data file
   * @throws XGBoostError native error
   */
  def this(dataPath: String) {
    this(new JDMatrix(dataPath))
  }

  /**
    *  init DMatrix from Iterator of LabeledPoint
    *
    * @param dataIter An iterator of LabeledPoint
    * @param cacheInfo  Cache path information, used for external memory setting, null by default.
    * @throws XGBoostError native error
    */
  def this(dataIter: Iterator[LabeledPoint], cacheInfo: String = null) {
    this(new JDMatrix(dataIter.asJava, cacheInfo))
  }

  /**
   * create DMatrix from sparse matrix
   *
   * @param headers index to headers (rowHeaders for CSR or colHeaders for CSC)
   * @param indices Indices (colIndexs for CSR or rowIndexs for CSC)
   * @param data    non zero values (sequence by row for CSR or by col for CSC)
   * @param st      sparse matrix type (CSR or CSC)
   */
  @throws(classOf[XGBoostError])
  @deprecated
  def this(headers: Array[Long], indices: Array[Int], data: Array[Float], st: JDMatrix.SparseType) {
    this(new JDMatrix(headers, indices, data, st, 0, Float.NaN, -1))
  }

  /**
   * create DMatrix from sparse matrix
   *
   * @param headers index to headers (rowHeaders for CSR or colHeaders for CSC)
   * @param indices Indices (colIndexs for CSR or rowIndexs for CSC)
   * @param data    non zero values (sequence by row for CSR or by col for CSC)
   * @param st      sparse matrix type (CSR or CSC)
   * @param shapeParam when st is CSR, it specifies the column number, otherwise it is taken as
   *                     row number
   */
  @throws(classOf[XGBoostError])
  def this(headers: Array[Long], indices: Array[Int], data: Array[Float], st: JDMatrix.SparseType,
           shapeParam: Int) {
    this(new JDMatrix(headers, indices, data, st, shapeParam, Float.NaN, -1))
  }

  /**
   * create DMatrix from sparse matrix
   *
   * @param headers index to headers (rowHeaders for CSR or colHeaders for CSC)
   * @param indices Indices (colIndexs for CSR or rowIndexs for CSC)
   * @param data    non zero values (sequence by row for CSR or by col for CSC)
   * @param st      sparse matrix type (CSR or CSC)
   * @param shapeParam when st is CSR, it specifies the column number, otherwise it is taken as
   *                     row number
   * @param missing missing value
   * @param nthread The number of threads used for constructing DMatrix
   */
  @throws(classOf[XGBoostError])
  def this(headers: Array[Long], indices: Array[Int], data: Array[Float], st: JDMatrix.SparseType,
           shapeParam: Int, missing: Float, nthread: Int) {
    this(new JDMatrix(headers, indices, data, st, shapeParam, missing, nthread))
  }

  /**
   * Create the normal DMatrix from column array interface
   * @param columnBatch the XGBoost ColumnBatch to provide the cuda array interface
   *                    of feature columns
   * @param missing missing value
   * @param nthread The number of threads used for constructing DMatrix
   */
  @throws(classOf[XGBoostError])
  def this(columnBatch: ColumnBatch, missing: Float, nthread: Int) {
    this(new JDMatrix(columnBatch, missing, nthread))
  }

  /**
   * create DMatrix from dense matrix
   *
   * @param data data values
   * @param nrow number of rows
   * @param ncol number of columns
   */
  @deprecated("Please specify the missing value explicitly", "XGBoost 1.5")
  @throws(classOf[XGBoostError])
  def this(data: Array[Float], nrow: Int, ncol: Int) {
    this(new JDMatrix(data, nrow, ncol))
  }

  /**
   * create DMatrix from dense matrix
   *
   * @param data data values
   * @param nrow number of rows
   * @param ncol number of columns
   * @param missing the specified value to represent the missing value
   */
  @throws(classOf[XGBoostError])
  def this(data: Array[Float], nrow: Int, ncol: Int, missing: Float) {
    this(new JDMatrix(data, nrow, ncol, missing))
  }

  /**
   * set label of dmatrix
   *
   * @param labels labels
   */
  @throws(classOf[XGBoostError])
  def setLabel(labels: Array[Float]): Unit = {
    jDMatrix.setLabel(labels)
  }

  /**
   * set weight of each instance
   *
   * @param weights weights
   */
  @throws(classOf[XGBoostError])
  def setWeight(weights: Array[Float]): Unit = {
    jDMatrix.setWeight(weights)
  }

  /**
   * if specified, xgboost will start from this init margin
   * can be used to specify initial prediction to boost from
   *
   * @param baseMargin base margin
   */
  @throws(classOf[XGBoostError])
  def setBaseMargin(baseMargin: Array[Float]): Unit = {
    jDMatrix.setBaseMargin(baseMargin)
  }

  /**
   * if specified, xgboost will start from this init margin
   * can be used to specify initial prediction to boost from
   *
   * @param baseMargin base margin
   */
  @throws(classOf[XGBoostError])
  def setBaseMargin(baseMargin: Array[Array[Float]]): Unit = {
    jDMatrix.setBaseMargin(baseMargin)
  }

  /**
   * Set group sizes of DMatrix (used for ranking)
   *
   * @param group group size as array
   */
  @throws(classOf[XGBoostError])
  def setGroup(group: Array[Int]): Unit = {
    jDMatrix.setGroup(group)
  }

  /**
   * Set label of DMatrix from cuda array interface
   */
  @throws(classOf[XGBoostError])
  def setLabel(column: Column): Unit = {
    jDMatrix.setLabel(column)
  }

  /**
   * set weight of dmatrix from column array interface
   */
  @throws(classOf[XGBoostError])
  def setWeight(column: Column): Unit = {
    jDMatrix.setWeight(column)
  }

  /**
   * set base margin of dmatrix from column array interface
   */
  @throws(classOf[XGBoostError])
  def setBaseMargin(column: Column): Unit = {
    jDMatrix.setBaseMargin(column)
  }

  /**
   * Get group sizes of DMatrix (used for ranking)
   */
  @throws(classOf[XGBoostError])
  def getGroup(): Array[Int] = {
    jDMatrix.getGroup()
  }

  /**
   * get label values
   *
   * @return label
   */
  @throws(classOf[XGBoostError])
  def getLabel: Array[Float] = {
    jDMatrix.getLabel
  }

  /**
   * get weight of the DMatrix
   *
   * @return weights
   */
  @throws(classOf[XGBoostError])
  def getWeight: Array[Float] = {
    jDMatrix.getWeight
  }

  /**
   * get base margin of the DMatrix
   *
   * @return base margin
   */
  @throws(classOf[XGBoostError])
  def getBaseMargin: Array[Float] = {
    jDMatrix.getBaseMargin
  }

  /**
   * Slice the DMatrix and return a new DMatrix that only contains `rowIndex`.
   *
   * @param rowIndex row index
   * @return sliced new DMatrix
   */
  @throws(classOf[XGBoostError])
  def slice(rowIndex: Array[Int]): DMatrix = {
    new DMatrix(jDMatrix.slice(rowIndex))
  }

  /**
   * get the row number of DMatrix
   *
   * @return number of rows
   */
  @throws(classOf[XGBoostError])
  def rowNum: Long = {
    jDMatrix.rowNum
  }

  /**
   * Get the number of non-missing values of DMatrix.
   *
   * @return The number of non-missing values
   */
  @throws(classOf[XGBoostError])
  def nonMissingNum: Long = {
    jDMatrix.nonMissingNum
  }

  /**
   * save DMatrix to filePath
   *
   * @param filePath file path
   */
  def saveBinary(filePath: String): Unit = {
    jDMatrix.saveBinary(filePath)
  }

  def getHandle: Long = {
    jDMatrix.getHandle
  }

  def delete(): Unit = {
    jDMatrix.dispose()
  }
}
