/*
 Copyright (c) 2014-2026 by Contributors

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

import java.nio.ByteBuffer;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

/**
 * xgboost JNI functions
 * change 2015-7-6: *use a long[] (length=1) as container of handle to get the
 * output DMatrix or Booster
 *
 * @author hzx
 */
public class XGBoostJNI {
  private static final Log logger = LogFactory.getLog(DMatrix.class);

  static {
    try {
      NativeLibLoader.initXGBoost();
    } catch (Exception ex) {
      logger.error("Failed to load native library", ex);
      throw new RuntimeException(ex);
    }
  }

  /**
   * Check the return code of the JNI call.
   *
   * @throws XGBoostError if the call failed.
   */
  static void checkCall(int ret) throws XGBoostError {
    if (ret != 0) {
      throw new XGBoostError(XGBGetLastError());
    }
  }

  /**
   * Gets the error message for non-zero return codes from JNI calls.
   *
   * @return Error message represented as a {@code String}.
   */
  public final static native String XGBGetLastError();

  /**
   * Create a DMatrix from a URI.
   *
   * @param uri The URI containing the data to create the DMatrix from.
   * @param silent The flag indicator deciding whether to print messages or not.
   * @param out The output array containing the handle to the created DMatrix.
   * @return 0 when success, -1 when failure happens
   */
  public final static native int XGDMatrixCreateFromURI(String uri, int silent, long[] out);

  /**
   * Create a DMatrix from a batch iteration.
   *
   * @param iter The URI containing the data to create the DMatrix from.
   * @param cache_info The flag indicator deciding whether to print messages or not.
   * @param missing The output array containing the handle to the created DMatrix.
   * @param out The output array containing the handle to the created DMatrix.
   * @return 0 when success, -1 when failure happens
   */
  final static native int XGDMatrixCreateFromDataIter(java.util.Iterator<DataBatch> iter,
      String cache_info, float missing, long[] out);

  /**
   * Create a DMatrix from a CSR matrix.
   *
   * @param indptr The index pointer array of the CSR matrix.
   * @param indices The indices array of the CSR matrix.
   * @param data The data array of the CSR matrix.
   * @param shapeParam The shape parameter of the CSR matrix.
   * @param missing The missing value indicator.
   * @param nthread The number of threads to use for creating the DMatrix.
   * @param out The output array containing the handle to the created DMatrix.
   * @return 0 when success, -1 when failure happens
   */
  public final static native int XGDMatrixCreateFromCSR(long[] indptr, int[] indices,
      float[] data, int shapeParam,
      float missing, int nthread,
      long[] out);

  /**
   * Create a DMatrix from a CSC matrix.
   *
   * @param colptr The column pointer array of the CSR matrix.
   * @param indices The indices array of the CSR matrix.
   * @param data The data array of the CSR matrix.
   * @param shapeParam The shape parameter of the CSR matrix.
   * @param missing The missing value indicator.
   * @param nthread The number of threads to use for creating the DMatrix.
   * @param out The output array containing the handle to the created DMatrix.
   * @return 0 when success, -1 when failure happens
   */
  public final static native int XGDMatrixCreateFromCSC(long[] colptr, int[] indices,
      float[] data, int shapeParam,
      float missing, int nthread,
      long[] out);

  /**
   * Create a DMatrix from a dense matrix.
   *
   * @param data The data array of the dense matrix.
   * @param nrow The number of rows in the dense matrix.
   * @param ncol The number of columns in the dense matrix.
   * @param missing The missing value indicator.
   * @param out The output array containing the handle to the created DMatrix.
   * @return 0 when success, -1 when failure happens
   */
  public final static native int XGDMatrixCreateFromMat(float[] data, int nrow, int ncol,
      float missing, long[] out);

  /**
   * Create a DMatrix from a dense matrix reference.
   *
   * @param dataRef The reference to the data array of the dense matrix.
   * @param nrow The number of rows in the dense matrix.
   * @param ncol The number of columns in the dense matrix.
   * @param missing The missing value indicator.
   * @param out The output array containing the handle to the created DMatrix.
   * @return 0 when success, -1 when failure happens
   */
  public final static native int XGDMatrixCreateFromMatRef(long dataRef, int nrow, int ncol,
      float missing, long[] out);

  /**
   * Slice a DMatrix to create a new DMatrix.
   *
   * @param handle The handle to the original DMatrix.
   * @param idxset The index set to slice the DMatrix.
   * @param out The output array containing the handle to the sliced DMatrix.
   * @return 0 when success, -1 when failure happens
   */
  public final static native int XGDMatrixSliceDMatrix(long handle, int[] idxset, long[] out);

  /**
   * Free the DMatrix.
   *
   * @param handle The handle to the DMatrix to be freed.
   * @return 0 when success, -1 when failure happens
   */
  public final static native int XGDMatrixFree(long handle);

  /**
   * Save the DMatrix to a binary file.
   *
   * @param handle The handle to the DMatrix to be saved.
   * @param fname The filename where the DMatrix will be saved.
   * @param silent The flag indicator deciding whether to print messages or not.
   * @return 0 when success, -1 when failure happens
   */
  public final static native int XGDMatrixSaveBinary(long handle, String fname, int silent);

  /**
   * Set the information of the DMatrix for floating point .
   *
   * @param handle The handle to the DMatrix.
   * @param field The field name to set the information for.
   * @param array The array of floating point values containing the information to set.
   * @return 0 when success, -1 when failure happens
   */
  public final static native int XGDMatrixSetFloatInfo(long handle, String field, float[] array);

  /**
   * Set the information of the DMatrix.
   *
   * @param handle The handle to the DMatrix.
   * @param field The field name to set the information for.
   * @param array The array of unsigned integer values containing the information to set.
   * @return 0 when success, -1 when failure happens
   */
  public final static native int XGDMatrixSetUIntInfo(long handle, String field, int[] array);

  /**
   * Get the information of the DMatrix for floating point values.
   *
   * @param handle The handle to the DMatrix.
   * @param field The field name to get the information for.
   * @param info The 2D array to store the retrieved information.
   * @return 0 when success, -1 when failure happens
   */
  public final static native int XGDMatrixGetFloatInfo(long handle, String field, float[][] info);

  /**
   * Get the information of the DMatrix for unsigned integer values.
   *
   * @param handle The handle to the DMatrix.
   * @param filed The field name to get the information for.
   * @param info The 2D array to store the retrieved information.
   * @return 0 when success, -1 when failure happens
   */
  public final static native int XGDMatrixGetUIntInfo(long handle, String filed, int[][] info);

  /**
   * Set the feature information
   *
   * @param handle the DMatrix native address
   * @param field  "feature_names" or "feature_types"
   * @param values an array of string
   * @return 0 when success, -1 when failure happens
   */
  public final static native int XGDMatrixSetStrFeatureInfo(long handle, String field,
      String[] values);

  public final static native int XGDMatrixGetStrFeatureInfo(long handle, String field,
      long[] outLength, String[][] outValues);

  public final static native int XGDMatrixNumRow(long handle, long[] row);

  public final static native int XGDMatrixNumNonMissing(long handle, long[] nonMissings);

  public final static native int XGBoosterCreate(long[] handles, long[] out);

  public final static native int XGBoosterFree(long handle);

  public final static native int XGBoosterSetParam(long handle, String name, String value);

  public final static native int XGBoosterSetParams(long handle, String config);

  public final static native int XGBoosterUpdateOneIter(long handle, int iter, long dtrain);

  public final static native int XGBoosterTrainOneIter(long handle, long dtrain, int iter, float[] grad,
      float[] hess);

  public final static native int XGBoosterEvalOneIter(long handle, int iter, long[] dmats,
      String[] evnames, String[] eval_info);

  public final static native int XGBoosterPredictFromDMatrix(long handle, long dmat,
      int predict_type, int iteration_end, float[][] predicts);

  public final static native int XGBoosterPredictFromDense(long handle, float[] data,
      long nrow, long ncol, float missing, int iteration_begin, int iteration_end, int predict_type, float[] margin,
      float[][] predicts);

  public final static native int XGBoosterLoadModel(long handle, String fname);

  public final static native int XGBoosterSaveModel(long handle, String fname);

  public final static native int XGBoosterLoadModelFromBuffer(long handle, byte[] bytes);

  public final static native int XGBoosterSaveModelToBuffer(long handle, String format, byte[][] out_bytes);

  public final static native int XGBoosterDumpModelEx(long handle, String fmap, int with_stats,
      String format, String[][] out_strings);

  public final static native int XGBoosterDumpModelExWithFeatures(
      long handle, String[] feature_names, int with_stats, String format, String[][] out_strings);

  public final static native int XGBoosterGetAttrNames(long handle, String[][] out_strings);

  public final static native int XGBoosterGetAttr(long handle, String key, String[] out_string);

  public final static native int XGBoosterSetAttr(long handle, String key, String value);

  public final static native int XGBoosterGetNumFeature(long handle, long[] feature);

  public final static native int XGBoosterGetNumBoostedRound(long handle, int[] rounds);

  // communicator functions
  public final static native int CommunicatorInit(String args);

  public final static native int CommunicatorFinalize();

  public final static native int CommunicatorPrint(String msg);

  public final static native int CommunicatorGetRank(int[] out);

  public final static native int CommunicatorGetWorldSize(int[] out);

  // Tracker functions
  public final static native int TrackerCreate(String host, int nWorkers, int port, int sortby, long timeout,
      long[] out);

  public final static native int TrackerRun(long handle);

  public final static native int TrackerWaitFor(long handle, long timeout);

  public final static native int TrackerWorkerArgs(long handle, long timeout, String[] out);

  public final static native int TrackerFree(long handle);

  // Perform Allreduce operation on data in sendrecvbuf.
  final static native int CommunicatorAllreduce(ByteBuffer sendrecvbuf, int count,
      int enum_dtype, int enum_op);

  public final static native int XGDMatrixSetInfoFromInterface(
      long handle, String field, String json);

  public final static native int XGQuantileDMatrixCreateFromCallback(
      java.util.Iterator<ColumnBatch> iter, long[] ref, String config, long[] out);

  public final static native int XGExtMemQuantileDMatrixCreateFromCallback(
      java.util.Iterator<ColumnBatch> iter, long[] ref, String config, long[] out);

  public final static native int XGDMatrixCreateFromArrayInterfaceColumns(
      String featureJson, float missing, int nthread, long[] out);

  public final static native int XGBoosterSetStrFeatureInfo(long handle, String field, String[] features);

  public final static native int XGBoosterGetStrFeatureInfo(long handle, String field, String[] out);

  public final static native int XGDMatrixGetQuantileCut(long handle, long[][] outIndptr, float[][] outValues);

  public final static native int XGBSetGlobalConfig(String config);

  public final static native int XGBGetGlobalConfig(String[] out);

  // CUDA device management functions
  public final static native int CudaSetDevice(int deviceId);
}
