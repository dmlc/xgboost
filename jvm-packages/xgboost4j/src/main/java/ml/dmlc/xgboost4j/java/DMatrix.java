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
package ml.dmlc.xgboost4j.java;

import java.util.Iterator;

import ml.dmlc.xgboost4j.LabeledPoint;
import ml.dmlc.xgboost4j.java.util.BigDenseMatrix;

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
  public DMatrix(long[] headers, int[] indices, float[] data,
                 DMatrix.SparseType st) throws XGBoostError {
    this(headers, indices, data, st, 0, Float.NaN, -1);
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
  public DMatrix(long[] headers, int[] indices, float[] data, DMatrix.SparseType st,
                 int shapeParam) throws XGBoostError {
    this(headers, indices, data, st, shapeParam, Float.NaN, -1);
  }

  public DMatrix(long[] headers, int[] indices, float[] data, DMatrix.SparseType st, int shapeParam,
                 float missing, int nthread) throws XGBoostError {
    long[] out = new long[1];
    if (st == SparseType.CSR) {
      XGBoostJNI.checkCall(XGBoostJNI.XGDMatrixCreateFromCSR(headers, indices, data,
                                                             shapeParam, missing, nthread, out));
    } else if (st == SparseType.CSC) {
      XGBoostJNI.checkCall(XGBoostJNI.XGDMatrixCreateFromCSC(headers, indices, data,
                                                             shapeParam, missing, nthread, out));
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
   *
   * @deprecated Please specify the missing value explicitly using
   * {@link DMatrix(float[], int, int, float)}
   */
  @Deprecated
  public DMatrix(float[] data, int nrow, int ncol) throws XGBoostError {
    long[] out = new long[1];
    XGBoostJNI.checkCall(XGBoostJNI.XGDMatrixCreateFromMat(data, nrow, ncol, 0.0f, out));
    handle = out[0];
  }

  /**
   * create DMatrix from a BigDenseMatrix
   *
   * @param matrix instance of BigDenseMatrix
   * @throws XGBoostError native error
   */
  public DMatrix(BigDenseMatrix matrix) throws XGBoostError {
    this(matrix, 0.0f);
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
   * create DMatrix from dense matrix
   * @param matrix instance of BigDenseMatrix
   * @param missing the specified value to represent the missing value
   */
  public DMatrix(BigDenseMatrix matrix, float missing) throws XGBoostError {
    long[] out = new long[1];
    XGBoostJNI.checkCall(XGBoostJNI.XGDMatrixCreateFromMatRef(matrix.address, matrix.nrow,
        matrix.ncol, missing, out));
    handle = out[0];
  }

  /**
   * used for DMatrix slice
   */
  protected DMatrix(long handle) {
    this.handle = handle;
  }

  /**
   * Create the normal DMatrix from column array interface
   * @param columnBatch the XGBoost ColumnBatch to provide the cuda array interface
   *                    of feature columns
   * @param missing missing value
   * @param nthread threads number
   * @throws XGBoostError
   */
  public DMatrix(ColumnBatch columnBatch, float missing, int nthread) throws XGBoostError {
    long[] out = new long[1];
    String json = columnBatch.getFeatureArrayInterface();
    if (json == null || json.isEmpty()) {
      throw new XGBoostError("Expecting non-empty feature columns' array interface");
    }
    XGBoostJNI.checkCall(XGBoostJNI.XGDMatrixCreateFromArrayInterfaceColumns(
        json, missing, nthread, out));
    handle = out[0];
  }

  /**
   * Set label of DMatrix from cuda array interface
   *
   * @param column the XGBoost Column to provide the cuda array interface
   *               of label column
   * @throws XGBoostError native error
   */
  public void setLabel(Column column) throws XGBoostError {
    setXGBDMatrixInfo("label", column.getArrayInterfaceJson());
  }

  /**
   * Set weight of DMatrix from cuda array interface
   *
   * @param column the XGBoost Column to provide the cuda array interface
   *               of weight column
   * @throws XGBoostError native error
   */
  public void setWeight(Column column) throws XGBoostError {
    setXGBDMatrixInfo("weight", column.getArrayInterfaceJson());
  }

  /**
   * Set base margin of DMatrix from cuda array interface
   *
   * @param column the XGBoost Column to provide the cuda array interface
   *               of base margin column
   * @throws XGBoostError native error
   */
  public void setBaseMargin(Column column) throws XGBoostError {
    setXGBDMatrixInfo("base_margin", column.getArrayInterfaceJson());
  }

  private void setXGBDMatrixInfo(String type, String json) throws XGBoostError {
    if (json == null || json.isEmpty()) {
      throw new XGBoostError("Empty " + type + " columns' array interface");
    }
    XGBoostJNI.checkCall(XGBoostJNI.XGDMatrixSetInfoFromInterface(handle, type, json));
  }

  private void setXGBDMatrixFeatureInfo(String type, String[] values) throws XGBoostError {
    if (type == null || type.isEmpty()) {
      throw new XGBoostError("Found empty type");
    }
    if (values == null || values.length == 0) {
      throw new XGBoostError("Found empty values");
    }
    XGBoostJNI.checkCall(XGBoostJNI.XGDMatrixSetStrFeatureInfo(handle, type, values));
  }

  private String[] getXGBDMatrixFeatureInfo(String type) throws XGBoostError {
    if (type == null || type.isEmpty()) {
      throw new XGBoostError("Found empty type");
    }
    long[] outLen = new long[1];
    String[][] outValue = new String[1][];
    XGBoostJNI.checkCall(XGBoostJNI.XGDMatrixGetStrFeatureInfo(handle, type, outLen, outValue));

    if (outLen[0] != outValue[0].length) {
      throw new RuntimeException("Failed to get " + type);
    }
    return outValue[0];
  }

  /**
   * Set feature names
   * @param values feature names to be set
   * @throws XGBoostError
   */
  public void setFeatureNames(String[] values) throws XGBoostError {
    setXGBDMatrixFeatureInfo("feature_name", values);
  }

  /**
   * Get feature names
   * @return an array of feature names to be returned
   * @throws XGBoostError
   */
  public String[] getFeatureNames() throws XGBoostError {
    return getXGBDMatrixFeatureInfo("feature_name");
  }

  /**
   * Set feature types
   * @param values feature types to be set
   * @throws XGBoostError
   */
  public void setFeatureTypes(String[] values) throws XGBoostError {
    setXGBDMatrixFeatureInfo("feature_type", values);
  }

  /**
   * Get feature types
   * @return an array of feature types to be returned
   * @throws XGBoostError
   */
  public String[] getFeatureTypes() throws XGBoostError {
    return getXGBDMatrixFeatureInfo("feature_type");
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
    XGBoostJNI.checkCall(XGBoostJNI.XGDMatrixSetUIntInfo(handle, "group", group));
  }

  /**
   * Get group sizes of DMatrix
   *
   * @throws XGBoostError native error
   * @return group size as array
   */
  public int[] getGroup() throws XGBoostError {
    return getIntInfo("group_ptr");
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
   * Get the number of non-missing values of DMatrix.
   *
   * @return The number of non-missing values
   * @throws XGBoostError native error
   */
  public long nonMissingNum() throws XGBoostError {
    long[] n = new long[1];
    XGBoostJNI.checkCall(XGBoostJNI.XGDMatrixNumNonMissing(handle, n));
    return n[0];
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
