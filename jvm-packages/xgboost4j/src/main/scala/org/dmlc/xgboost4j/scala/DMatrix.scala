package org.dmlc.xgboost4j.scala

import org.dmlc.xgboost4j.{DMatrix => JDMatrix, XGBoostError}

class DMatrix private(private[scala] val jDMatrix: JDMatrix) {

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
   * create DMatrix from sparse matrix
   *
   * @param headers index to headers (rowHeaders for CSR or colHeaders for CSC)
   * @param indices Indices (colIndexs for CSR or rowIndexs for CSC)
   * @param data    non zero values (sequence by row for CSR or by col for CSC)
   * @param st      sparse matrix type (CSR or CSC)
   */
  @throws(classOf[XGBoostError])
  def this(headers: Array[Long], indices: Array[Int], data: Array[Float], st: JDMatrix.SparseType) {
    this(new JDMatrix(headers, indices, data, st))
  }

  /**
   * create DMatrix from dense matrix
   *
   * @param data data values
   * @param nrow number of rows
   * @param ncol number of columns
   */
  @throws(classOf[XGBoostError])
  def this(data: Array[Float], nrow: Int, ncol: Int) {
    this(new JDMatrix(data, nrow, ncol))
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
