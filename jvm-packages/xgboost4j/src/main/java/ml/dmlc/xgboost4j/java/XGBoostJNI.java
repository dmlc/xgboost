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

import java.nio.ByteBuffer;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

/**
 * xgboost JNI functions
 * change 2015-7-6: *use a long[] (length=1) as container of handle to get the output DMatrix or Booster
 *
 * @author hzx
 */
class XGBoostJNI {
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

  public final static native String XGBGetLastError();

  public final static native int XGDMatrixCreateFromFile(String fname, int silent, long[] out);

  final static native int XGDMatrixCreateFromDataIter(java.util.Iterator<DataBatch> iter,
                                                             String cache_info, long[] out);

  public final static native int XGDMatrixCreateFromCSR(long[] indptr, int[] indices,
                                                        float[] data, int shapeParam,
                                                        float missing, int nthread,
                                                        long[] out);

  public final static native int XGDMatrixCreateFromCSC(long[] colptr, int[] indices,
                                                        float[] data, int shapeParam,
                                                        float missing, int nthread,
                                                        long[] out);

  public final static native int XGDMatrixCreateFromMat(float[] data, int nrow, int ncol,
                                                        float missing, long[] out);

  public final static native int XGDMatrixCreateFromMatRef(long dataRef, int nrow, int ncol,
                                                           float missing, long[] out);

  public final static native int XGDMatrixSliceDMatrix(long handle, int[] idxset, long[] out);

  public final static native int XGDMatrixFree(long handle);

  public final static native int XGDMatrixSaveBinary(long handle, String fname, int silent);

  public final static native int XGDMatrixSetFloatInfo(long handle, String field, float[] array);

  public final static native int XGDMatrixSetUIntInfo(long handle, String field, int[] array);

  public final static native int XGDMatrixGetFloatInfo(long handle, String field, float[][] info);

  public final static native int XGDMatrixGetUIntInfo(long handle, String filed, int[][] info);

  /**
   * Set the feature information
   * @param handle the DMatrix native address
   * @param field "feature_names" or "feature_types"
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

  public final static native int XGBoosterUpdateOneIter(long handle, int iter, long dtrain);

  public final static native int XGBoosterBoostOneIter(long handle, long dtrain, float[] grad,
                                                       float[] hess);

  public final static native int XGBoosterEvalOneIter(long handle, int iter, long[] dmats,
                                                      String[] evnames, String[] eval_info);

  public final static native int XGBoosterPredict(long handle, long dmat, int option_mask,
                                                  int ntree_limit, float[][] predicts);

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
  public final static native int XGBoosterLoadRabitCheckpoint(long handle, int[] out_version);
  public final static native int XGBoosterSaveRabitCheckpoint(long handle);
  public final static native int XGBoosterGetNumFeature(long handle, long[] feature);

  // communicator functions
  public final static native int CommunicatorInit(String[] args);
  public final static native int CommunicatorFinalize();
  public final static native int CommunicatorPrint(String msg);
  public final static native int CommunicatorGetRank(int[] out);
  public final static native int CommunicatorGetWorldSize(int[] out);

  // Perform Allreduce operation on data in sendrecvbuf.
  final static native int CommunicatorAllreduce(ByteBuffer sendrecvbuf, int count,
    int enum_dtype, int enum_op);

  public final static native int XGDMatrixSetInfoFromInterface(
    long handle, String field, String json);

  @Deprecated
  public final static native int XGDeviceQuantileDMatrixCreateFromCallback(
    java.util.Iterator<ColumnBatch> iter, float missing, int nthread, int maxBin, long[] out);

  public final static native int XGQuantileDMatrixCreateFromCallback(
    java.util.Iterator<ColumnBatch> iter, java.util.Iterator<ColumnBatch> ref, String config, long[] out);

  public final static native int XGDMatrixCreateFromArrayInterfaceColumns(
    String featureJson, float missing, int nthread, long[] out);

}
