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

import java.util.Iterator;

import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;

/**
 * DMatrix based on GPU
 */
class GpuDMatrix extends DMatrix {

  /**
   * Create DeviceQuantileDMatrix from iterator based on the cuda array interface
   * @param iter     the cudf table batch to provide the corresponding cuda array interface
   * @param missing  the missing value
   * @param maxBin   the max bin
   * @param nthread  the parallelism
   * @throws XGBoostError
   */
  public GpuDMatrix(Iterator<GpuTable> iter, float missing, int maxBin, int nthread)
      throws XGBoostError {
    super(0);
    long[] out = new long[1];
    Iterator<TableBatch> batchIter = new TableBatch.ColumnBatchIterator(iter);
    GpuXGBoostJNI.checkCall(GpuXGBoostJNI.XGDeviceQuantileDMatrixCreateFromCallback(
        batchIter, missing, maxBin, nthread, out));
    handle = out[0];
  }

  /**
   * Create the normal DMatrix from column array interface
   * @param featureArrayInterfaces the cuda array interface of feature columns
   * @param missing missing value
   * @param nthread threads number
   * @throws XGBoostError
   */
  public GpuDMatrix(String featureArrayInterfaces, float missing, int nthread) throws XGBoostError {
    super(0);
    long[] out = new long[1];
    GpuXGBoostJNI.checkCall(GpuXGBoostJNI.XGDMatrixCreateFromArrayInterfaceColumns(
        featureArrayInterfaces, missing, nthread, out));
    handle = out[0];
  }

  /**
   * Set label of DMatrix from cuda array interface
   *
   * @param labelJson the cuda array interface of label column
   * @throws XGBoostError native error
   */
  public void setLabel(String labelJson) throws XGBoostError {
    GpuXGBoostJNI.checkCall(
        GpuXGBoostJNI.XGDMatrixSetInfoFromInterface(handle, "label", labelJson));
  }

  /**
   * Set weight of DMatrix from cuda array interface
   *
   * @param weightJson the cuda array interface of weight column
   * @throws XGBoostError native error
   */
  public void setWeight(String weightJson) throws XGBoostError {
    GpuXGBoostJNI.checkCall(
        GpuXGBoostJNI.XGDMatrixSetInfoFromInterface(handle, "weight", weightJson));
  }

  /**
   * Set base margin of DMatrix from cuda array interface
   *
   * @param baseMarginJson the cuda array interface of base margin column
   * @throws XGBoostError native error
   */
  public void setBaseMargin(String baseMarginJson) throws XGBoostError {
    GpuXGBoostJNI.checkCall(
        GpuXGBoostJNI.XGDMatrixSetInfoFromInterface(handle, "base_margin", baseMarginJson));
  }

}
