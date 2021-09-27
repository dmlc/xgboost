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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import ml.dmlc.xgboost4j.gpu.java.CudfColumn;


/**
 * JNI functions for XGBoost4J-Spark
 */
public class XGBoostSparkJNI {
  private static final Log logger = LogFactory.getLog(XGBoostSparkJNI.class);

  static {
    try {
      NativeLibLoader.initXGBoost();
    } catch (Exception ex) {
      logger.error("Failed to load native library", ex);
      throw new RuntimeException(ex);
    }
  }

  public static long buildUnsafeRows(final CudfColumn... cds) {
    if (cds == null || cds.length <= 0) return 0L;
    long[] dataPtrs  = new long[cds.length];
    long[] validPtrs = new long[cds.length];
    int[] dTypeSizes = new int[cds.length];
    int i = 0;
    for(CudfColumn cd: cds) {
      dataPtrs[i]  = cd.getDataPtr();
      validPtrs[i] = cd.getValidPtr();
      dTypeSizes[i] = cd.getTypeSize();
      i++;
    }
    return buildUnsafeRows(dataPtrs, validPtrs, dTypeSizes, cds[0].getShape());
  }

  /**
   * Build an array of fixed-length Spark UnsafeRow using the GPU.
   * @param dataPtrs native address of cudf column data pointers
   * @param validPtrs native address of cudf column valid pointers
   * @param dTypeSizes
   * @param rows
   * @return native address of the UnsafeRow array
   * NOTE: It is the responsibility of the caller to free the native memory
   *       returned by this function (e.g.: using Platform.freeMemory).
   */
  private static native long buildUnsafeRows(long[] dataPtrs, long[] validPtrs,
                                            int[] dTypeSizes, long rows);

  public static native int getGpuDevice();

  public static native int allocateGpuDevice(int gpuId);
}
