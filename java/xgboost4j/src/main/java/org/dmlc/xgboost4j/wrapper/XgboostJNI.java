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
 * xgboost jni wrapper class
 * @author hzx
 */
public class XgboostJNI {
  public final static native long XGDMatrixCreateFromFile(String fname, int silent);
  public final static native long XGDMatrixCreateFromCSR(long[] indptr, long[] indices, float[] data);
  public final static native long XGDMatrixCreateFromCSC(long[] col_ptr, long[] indices, float[] data);
  public final static native long XGDMatrixCreateFromMat(float[] data, int nrow, int ncol, float missing);
  public final static native long XGDMatrixSliceDMatrix(long handle, int[] idxset);
  public final static native void XGDMatrixFree(long handle);
  public final static native void XGDMatrixSaveBinary(long handle, String fname, int silent);
  public final static native void XGDMatrixSetFloatInfo(long handle, String field, float[] array);
  public final static native void XGDMatrixSetUIntInfo(long handle, String field, int[] array);
  public final static native void XGDMatrixSetGroup(long handle, int[] group);
  public final static native float[] XGDMatrixGetFloatInfo(long handle, String field);
  public final static native int[] XGDMatrixGetUIntInfo(long handle, String filed);
  public final static native long XGDMatrixNumRow(long handle);
  public final static native long XGBoosterCreate(long[] handles);
  public final static native void XGBoosterFree(long handle);
  public final static native void XGBoosterSetParam(long handle, String name, String value);
  public final static native void XGBoosterUpdateOneIter(long handle, int iter, long dtrain);
  public final static native void XGBoosterBoostOneIter(long handle, long dtrain, float[] grad, float[] hess);
  public final static native String XGBoosterEvalOneIter(long handle, int iter, long[] dmats, String[] evnames);
  public final static native float[] XGBoosterPredict(long handle, long dmat, int option_mask, long ntree_limit);
  public final static native void XGBoosterLoadModel(long handle, String fname);
  public final static native void XGBoosterSaveModel(long handle, String fname);
  public final static native void XGBoosterLoadModelFromBuffer(long handle, long buf, long len);
  public final static native String XGBoosterGetModelRaw(long handle);
  public final static native String[] XGBoosterDumpModel(long handle, String fmap, int with_stats);
}
