/*
 Copyright (c) 2021 by Contributors

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

package ml.dmlc.xgboost4j.gpu.java;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import ml.dmlc.xgboost4j.java.NativeLibLoader;
import ml.dmlc.xgboost4j.java.XGBoostError;

class GpuXGBoostJNI {
  private static final Log logger = LogFactory.getLog(GpuXGBoostJNI.class);

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
      throw new XGBoostError(XGBGpuGetLastError());
    }
  }

  private final static native String XGBGpuGetLastError();

  public final static native int XGDMatrixSetInfoFromInterface(
    long handle, String field, String json);

  public final static native int XGDeviceQuantileDMatrixCreateFromCallback(
    java.util.Iterator<TableBatch> iter, float missing, int nthread, int maxBin, long[] out);

  public final static native int XGDMatrixCreateFromArrayInterfaceColumns(
    String columnJosn, float missing, int nthread, long[] out);

}
