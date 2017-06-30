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
package ml.dmlc.xgboost4j.java;

import java.util.Iterator;

import ml.dmlc.xgboost4j.LabeledPoint;

/**
 * DMatrix for xgboost.
 *
 * @author hzx
 */
public class DMatrix {
  protected long handle = 0;

  /**
   * sparse matrix type (CSR or CSC)
   */
  public static enum SparseType {
    CSR,
    CSC;
  }

  /**
   * Create DMatrix from iterator.
   *
   * @param iter The data iterator of mini batch to provide the data.
   * @param cacheInfo Cache path information, used for external memory setting, can be null.
   * @throws XGBoostError
   */
  public DMatrix(Iterator<LabeledPoint> iter, String cacheInfo) throws XGBoostError {
    if (iter == null) {
      throw new NullPointerException("iter: null");
    }
    // 32k as batch size
    int batchSize = 32 << 10;
    Iterator<DataBatch> batchIter = new DataBatch.BatchIterator(iter, batchSize);
    long[] out = new long[1];
    XGBoostJNI.checkCall(XGBoostJNI.XGDMatrixCreateFromDataIter(batchIter, cacheInfo, out));
    handle = out[0];
  }

  /**
   * Create DMatrix by loading libsvm file from dataPath
   *
   * @param dataPath The path to the data.
   * @throws XGBoostError
   */
  public DMatrix(String dataPath) throws XGBoostError {
    if (dataPath == null) {
      throw new NullPointerException("dataPath: null");
    }
    long[] out = new long[1];
    XGBoostJNI.checkCall(XGBoostJNI.XGDMatrixCreateFromFile(dataPath, 1, out));
    handle = out[0];
  }

  /**
   * Create DMatrix from Sparse matrix in CSR/CSC format.
   * @param headers The row index of the matrix.
   * @param indices The indices of presenting entries.
   * @param data The data content.
   * @param st  Type of sparsity.
   * @throws XGBoostError
   */
  @Deprecated
  public DMatrix(long[] headers, int[] indices, float[] data, SparseType st) throws XGBoostError {
    long[] out = new long[1];
    if (st == SparseType.CSR) {
      XGBoostJNI.checkCall(XGBoostJNI.XGDMatrixCreateFromCSREx(headers, indices, data, 0, out));
    } else if (st == SparseType.CSC) {
      XGBoostJNI.checkCall(XGBoostJNI.XGDMatrixCreateFromCSCEx(headers, indices, data, 0, out));
    } else {
      throw new UnknownError("unknow sparsetype");
    }
    handle = out[0];
  }

  /**
   * Create DMatrix from Sparse matrix in CSR/CSC format.
   * @param headers The row index of the matrix.
   * @param indices The indices of presenting entries.
   * @param data The data content.
   * @param st  Type of sparsity.
   * @param shapeParam   when st is CSR, it specifies the column number, otherwise it is taken as
   *                     row number
   * @throws XGBoostError
   */
  public DMatrix(long[] headers, int[] indices, float[] data, SparseType st, int shapeParam)
          throws XGBoostError {
    long[] out = new long[1];
    if (st == SparseType.CSR) {
      XGBoostJNI.checkCall(XGBoostJNI.XGDMatrixCreateFromCSREx(headers, indices, data,
              shapeParam, out));
    } else if (st == SparseType.CSC) {
      XGBoostJNI.checkCall(XGBoostJNI.XGDMatrixCreateFromCSCEx(headers, indices, data,
              shapeParam, out));
    } else {
      throw new UnknownError("unknow sparsetype");
    }
    handle = out[0];
  }

  /**
   * create DMatrix from dense matrix
   *
   * @param data data values
   * @param nrow number of rows
   * @param ncol number of columns
   * @throws XGBoostError native error
   */
  public DMatrix(float[] data, int nrow, int ncol) throws XGBoostError {
    long[] out = new long[1];
    XGBoostJNI.checkCall(XGBoostJNI.XGDMatrixCreateFromMat(data, nrow, ncol, 0.0f, out));
    handle = out[0];
  }

  /**
   * create DMatrix from dense matrix
   * @param data data values
   * @param nrow number of rows
   * @param ncol number of columns
   * @param missing the specified value to represent the missing value
   */
  public DMatrix(float[] data, int nrow, int ncol, float missing) throws XGBoostError {
    long[] out = new long[1];
    XGBoostJNI.checkCall(XGBoostJNI.XGDMatrixCreateFromMat(data, nrow, ncol, missing, out));
    handle = out[0];
  }

  /**
   * used for DMatrix slice
   */
  protected DMatrix(long handle) {
    this.handle = handle;
  }


  /**
   * set label of dmatrix
   *
   * @param labels labels
   * @throws XGBoostError native error
   */
  public void setLabel(float[] labels) throws XGBoostError {
    XGBoostJNI.checkCall(XGBoostJNI.XGDMatrixSetFloatInfo(handle, "label", labels));
  }

  /**
   * set weight of each instance
   *
   * @param weights weights
   * @throws XGBoostError native error
   */
  public void setWeight(float[] weights) throws XGBoostError {
    XGBoostJNI.checkCall(XGBoostJNI.XGDMatrixSetFloatInfo(handle, "weight", weights));
  }

  /**
   * Set base margin (initial prediction).
   *
   * The margin must have the same number of elements as the number of
   * rows in this matrix.
   */
  public void setBaseMargin(float[] baseMargin) throws XGBoostError {
    if (baseMargin.length != rowNum()) {
      throw new IllegalArgumentException(String.format(
              "base margin must have exactly %s elements, got %s",
              rowNum(), baseMargin.length));
    }

    XGBoostJNI.checkCall(XGBoostJNI.XGDMatrixSetFloatInfo(handle, "base_margin", baseMargin));
  }

  /**
   * Set base margin (initial prediction).
   */
  public void setBaseMargin(float[][] baseMargin) throws XGBoostError {
    setBaseMargin(flatten(baseMargin));
  }

  /**
   * Set group sizes of DMatrix (used for ranking)
   *
   * @param group group size as array
   * @throws XGBoostError native error
   */
  public void setGroup(int[] group) throws XGBoostError {
    XGBoostJNI.checkCall(XGBoostJNI.XGDMatrixSetGroup(handle, group));
  }

  private float[] getFloatInfo(String field) throws XGBoostError {
    float[][] infos = new float[1][];
    XGBoostJNI.checkCall(XGBoostJNI.XGDMatrixGetFloatInfo(handle, field, infos));
    return infos[0];
  }

  private int[] getIntInfo(String field) throws XGBoostError {
    int[][] infos = new int[1][];
    XGBoostJNI.checkCall(XGBoostJNI.XGDMatrixGetUIntInfo(handle, field, infos));
    return infos[0];
  }

  /**
   * get label values
   *
   * @return label
   * @throws XGBoostError native error
   */
  public float[] getLabel() throws XGBoostError {
    return getFloatInfo("label");
  }

  /**
   * get weight of the DMatrix
   *
   * @return weights
   * @throws XGBoostError native error
   */
  public float[] getWeight() throws XGBoostError {
    return getFloatInfo("weight");
  }

  /**
   * Get base margin of the DMatrix.
   */
  public float[] getBaseMargin() throws XGBoostError {
    return getFloatInfo("base_margin");
  }

  /**
   * Slice the DMatrix and return a new DMatrix that only contains `rowIndex`.
   *
   * @param rowIndex row index
   * @return sliced new DMatrix
   * @throws XGBoostError native error
   */
  public DMatrix slice(int[] rowIndex) throws XGBoostError {
    long[] out = new long[1];
    XGBoostJNI.checkCall(XGBoostJNI.XGDMatrixSliceDMatrix(handle, rowIndex, out));
    long sHandle = out[0];
    DMatrix sMatrix = new DMatrix(sHandle);
    return sMatrix;
  }

  /**
   * get the row number of DMatrix
   *
   * @return number of rows
   * @throws XGBoostError native error
   */
  public long rowNum() throws XGBoostError {
    long[] rowNum = new long[1];
    XGBoostJNI.checkCall(XGBoostJNI.XGDMatrixNumRow(handle, rowNum));
    return rowNum[0];
  }

  /**
   * save DMatrix to filePath
   */
  public void saveBinary(String filePath) {
    XGBoostJNI.XGDMatrixSaveBinary(handle, filePath, 1);
  }

  /**
   * Get the handle
   */
  public long getHandle() {
    return handle;
  }

  /**
   * flatten a mat to array
   */
  private static float[] flatten(float[][] mat) {
    int size = 0;
    for (float[] array : mat) size += array.length;
    float[] result = new float[size];
    int pos = 0;
    for (float[] ar : mat) {
      System.arraycopy(ar, 0, result, pos, ar.length);
      pos += ar.length;
    }

    return result;
  }

  @Override
  protected void finalize() {
    dispose();
  }

  public synchronized void dispose() {
    if (handle != 0) {
      XGBoostJNI.XGDMatrixFree(handle);
      handle = 0;
    }
  }
}
