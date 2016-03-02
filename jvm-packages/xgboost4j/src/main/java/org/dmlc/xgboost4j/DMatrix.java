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
package org.dmlc.xgboost4j;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.io.IOException;

/**
 * DMatrix for xgboost, similar to the python wrapper xgboost.py
 *
 * @author hzx
 */
public class DMatrix {
  private static final Log logger = LogFactory.getLog(DMatrix.class);
  private long handle = 0;

  //load native library
  static {
    try {
      NativeLibLoader.initXgBoost();
    } catch (IOException ex) {
      logger.error("load native library failed.");
      logger.error(ex);
    }
  }

  /**
   * sparse matrix type (CSR or CSC)
   */
  public static enum SparseType {
    CSR,
    CSC;
  }

  public DMatrix(String dataPath) throws XGBoostError {
    if (dataPath == null) {
      throw new NullPointerException("dataPath: null");
    }
    long[] out = new long[1];
    JNIErrorHandle.checkCall(XgboostJNI.XGDMatrixCreateFromFile(dataPath, 1, out));
    handle = out[0];
  }

  public DMatrix(long[] headers, int[] indices, float[] data, SparseType st) throws XGBoostError {
    long[] out = new long[1];
    if (st == SparseType.CSR) {
      JNIErrorHandle.checkCall(XgboostJNI.XGDMatrixCreateFromCSR(headers, indices, data, out));
    } else if (st == SparseType.CSC) {
      JNIErrorHandle.checkCall(XgboostJNI.XGDMatrixCreateFromCSC(headers, indices, data, out));
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
    JNIErrorHandle.checkCall(XgboostJNI.XGDMatrixCreateFromMat(data, nrow, ncol, 0.0f, out));
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
    JNIErrorHandle.checkCall(XgboostJNI.XGDMatrixSetFloatInfo(handle, "label", labels));
  }

  /**
   * set weight of each instance
   *
   * @param weights weights
   * @throws XGBoostError native error
   */
  public void setWeight(float[] weights) throws XGBoostError {
    JNIErrorHandle.checkCall(XgboostJNI.XGDMatrixSetFloatInfo(handle, "weight", weights));
  }

  /**
   * if specified, xgboost will start from this init margin
   * can be used to specify initial prediction to boost from
   *
   * @param baseMargin base margin
   * @throws XGBoostError native error
   */
  public void setBaseMargin(float[] baseMargin) throws XGBoostError {
    JNIErrorHandle.checkCall(XgboostJNI.XGDMatrixSetFloatInfo(handle, "base_margin", baseMargin));
  }

  /**
   * if specified, xgboost will start from this init margin
   * can be used to specify initial prediction to boost from
   *
   * @param baseMargin base margin
   * @throws XGBoostError native error
   */
  public void setBaseMargin(float[][] baseMargin) throws XGBoostError {
    float[] flattenMargin = flatten(baseMargin);
    setBaseMargin(flattenMargin);
  }

  /**
   * Set group sizes of DMatrix (used for ranking)
   *
   * @param group group size as array
   * @throws XGBoostError native error
   */
  public void setGroup(int[] group) throws XGBoostError {
    JNIErrorHandle.checkCall(XgboostJNI.XGDMatrixSetGroup(handle, group));
  }

  private float[] getFloatInfo(String field) throws XGBoostError {
    float[][] infos = new float[1][];
    JNIErrorHandle.checkCall(XgboostJNI.XGDMatrixGetFloatInfo(handle, field, infos));
    return infos[0];
  }

  private int[] getIntInfo(String field) throws XGBoostError {
    int[][] infos = new int[1][];
    JNIErrorHandle.checkCall(XgboostJNI.XGDMatrixGetUIntInfo(handle, field, infos));
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
   * get base margin of the DMatrix
   *
   * @return base margin
   * @throws XGBoostError native error
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
    JNIErrorHandle.checkCall(XgboostJNI.XGDMatrixSliceDMatrix(handle, rowIndex, out));
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
    JNIErrorHandle.checkCall(XgboostJNI.XGDMatrixNumRow(handle, rowNum));
    return rowNum[0];
  }

  /**
   * save DMatrix to filePath
   */
  public void saveBinary(String filePath) {
    XgboostJNI.XGDMatrixSaveBinary(handle, filePath, 1);
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
      XgboostJNI.XGDMatrixFree(handle);
      handle = 0;
    }
  }
}
