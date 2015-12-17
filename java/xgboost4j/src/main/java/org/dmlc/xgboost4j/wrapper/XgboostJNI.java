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
package org.dmlc.xgboost4j.wrapper;

/**
 * xgboost jni wrapper functions for xgboost_wrapper.h
 * change 2015-7-6: *use a long[] (length=1) as container of handle to get the output DMatrix or Booster
 * @author hzx
 */
public class XgboostJNI {
  public final static native String XGBGetLastError();
  public final static native int XGDMatrixCreateFromFile(String fname, int silent, long[] out);
  public final static native int XGDMatrixCreateFromCSR(long[] indptr, int[] indices, float[] data, long[] out);
  public final static native int XGDMatrixCreateFromCSC(long[] colptr, int[] indices, float[] data, long[] out);
  public final static native int XGDMatrixCreateFromMat(float[] data, int nrow, int ncol, float missing, long[] out);
  public final static native int XGDMatrixSliceDMatrix(long handle, int[] idxset, long[] out);
  public final static native int XGDMatrixFree(long handle);
  public final static native int XGDMatrixSaveBinary(long handle, String fname, int silent);
  public final static native int XGDMatrixSetFloatInfo(long handle, String field, float[] array);
  public final static native int XGDMatrixSetUIntInfo(long handle, String field, int[] array);
  public final static native int XGDMatrixSetGroup(long handle, int[] group);
  public final static native int XGDMatrixGetFloatInfo(long handle, String field, float[][] info);
  public final static native int XGDMatrixGetUIntInfo(long handle, String filed, int[][] info);
  public final static native int XGDMatrixNumRow(long handle, long[] row);
  public final static native int XGBoosterCreate(long[] handles, long[] out);
  public final static native int XGBoosterFree(long handle);
  public final static native int XGBoosterSetParam(long handle, String name, String value);
  public final static native int XGBoosterUpdateOneIter(long handle, int iter, long dtrain);
  public final static native int XGBoosterBoostOneIter(long handle, long dtrain, float[] grad, float[] hess);
  public final static native int XGBoosterEvalOneIter(long handle, int iter, long[] dmats, String[] evnames, String[] eval_info);
  public final static native int XGBoosterPredict(long handle, long dmat, int option_mask, long ntree_limit, float[][] predicts);
  public final static native int XGBoosterLoadModel(long handle, String fname);
  public final static native int XGBoosterSaveModel(long handle, String fname);
  public final static native int XGBoosterLoadModelFromBuffer(long handle, long buf, long len);
  public final static native int XGBoosterGetModelRaw(long handle, String[] out_string);
  public final static native int XGBoosterDumpModel(long handle, String fmap, int with_stats, String[][] out_strings);
}
