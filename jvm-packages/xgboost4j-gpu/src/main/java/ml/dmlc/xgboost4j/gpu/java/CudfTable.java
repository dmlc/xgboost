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

import java.util.Arrays;

import ai.rapids.cudf.BaseDeviceMemoryBuffer;
import ai.rapids.cudf.BufferType;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.Table;

/**
 * Class to wrap CUDF Table to generate the cuda array interface.
 */
public class CudfTable extends XGBoostTable {
  private final Table table;          // The CUDF Table
  private final int[] featureIndices; // The feature columns
  private final int[] labelIndices;   // The label columns
  private final int[] weightIndices;  // The weight columns
  private final int[] baseMarginIndices; // The base margin columns

  /**
   * CudfTable constructor
   * @param table             the CUDF table
   * @param featureIndices    must-have, specify the feature's indices in the table
   * @param labelIndices      must-have, specify the label's indices in the table
   */
  public CudfTable(Table table, int[] featureIndices, int[] labelIndices) {
    this(table, featureIndices, labelIndices, null, null);
  }

  /**
   * CudfTable constructor
   * @param table             the CUDF table
   * @param featureIndices    must-have, specify the feature's indices in the table
   * @param labelIndices      must-have, specify the label's indices in the table
   * @param weightIndices     optional, specify the weight's indices in the table
   * @param baseMarginIndices optional, specify the base marge's indices in the table
   */
  public CudfTable(Table table, int[] featureIndices, int[] labelIndices, int[] weightIndices,
                   int[] baseMarginIndices) {
    this.table = table;
    this.featureIndices = featureIndices;
    this.labelIndices = labelIndices;
    this.weightIndices = weightIndices;
    this.baseMarginIndices = baseMarginIndices;

    validate();
  }

  private void validate() {
    if (labelIndices == null) {
      throw new RuntimeException("CudfTable requires label column");
    } else {
      validateArrayIndex(labelIndices, "label");
    }


    if (featureIndices == null) {
      throw new RuntimeException("CudfTable requires feature columns");
    } else {
      validateArrayIndex(featureIndices, "feature");
    }

    if (weightIndices != null) {
      validateArrayIndex(weightIndices, "weight");
    }

    if (baseMarginIndices != null) {
      validateArrayIndex(baseMarginIndices, "base_margin");
    }
  }

  private void validateArrayIndex(int[] array, String category) {
    assert array != null;
    int min = array[0];
    int max = array[0];
    for (int i = 1; i < array.length; i++) {
      if (array[i] > max) {
        max = array[i];
      }

      if (array[i] < min) {
        min = array[i];
      }
    }

    if (min < 0 || max >= table.getNumberOfColumns()) {
      throw new IllegalArgumentException("Wrong " + category + " indices, Out of boundary");
    }
  }

  public ColumnVector getColumnVector(int index) {
    return table.getColumn(index);
  }

  /**
   * Get the Json string for cuda array interfaces.
   *
   * @return Json string
   */
  @Override
  public String getArrayInterfaceJson() {
    StringBuilder builder = new StringBuilder();

    builder.append("{");

    String featureStr = getArrayInterface(featureIndices);
    if (featureStr == null || featureStr.isEmpty()) {
      throw new RuntimeException("Feature json must not be empty");
    } else {
      builder.append("\"features_str\":" + featureStr);
    }

    String labelStr = getArrayInterface(labelIndices);
    if (labelStr == null || labelStr.isEmpty()) {
      throw new RuntimeException("label json must not be empty");
    } else {
      builder.append(",\"label_str\":" + labelStr);
    }

    String weightStr = getArrayInterface(weightIndices);
    if (weightStr != null && ! weightStr.isEmpty()) {
      builder.append(",\"weight_str\":" + weightStr);
    }

    String baseMarginStr = getArrayInterface(baseMarginIndices);
    if (baseMarginStr != null && ! baseMarginStr.isEmpty()) {
      builder.append(",\"basemargin_str\":" + baseMarginStr);
    }

    builder.append("}");

    return builder.toString();
  }

  @Override
  public String getFeatureArrayInterface() {
    return getArrayInterface(featureIndices);
  }

  public String getLabelArrayInterface() {
    return getArrayInterface(labelIndices);
  }

  public String getWeightArrayInterface() {
    return getArrayInterface(weightIndices);
  }

  public String getBaseMarginArrayInterface() {
    return getArrayInterface(baseMarginIndices);
  }

  @Override
  public void close() {
    if (table != null) table.close();
  }

  private String getArrayInterface(int... indices) {
    if (indices == null || indices.length == 0) return "";
    return ColumnData.getArrayInterface(getAsColumnData(indices));
  }

  private ColumnData[] getAsColumnData(int... indices) {
    if (indices == null || indices.length == 0) return new ColumnData[]{};
    return Arrays.stream(indices)
      .mapToObj(this::getColumnVector)
      .map(this::getColumnData)
      .toArray(ColumnData[]::new);
  }

  private ColumnData getColumnData(ColumnVector columnVector) {
    BaseDeviceMemoryBuffer dataBuffer = columnVector.getDeviceBufferFor(BufferType.DATA);
    BaseDeviceMemoryBuffer validBuffer = columnVector.getDeviceBufferFor(BufferType.VALIDITY);
    long validPtr = 0;
    if (validBuffer != null) {
      validPtr = validBuffer.getAddress();
    }
    DType dType = columnVector.getType();
    String typeStr = "";
    if (dType == DType.FLOAT32 || dType == DType.FLOAT64 ||
          dType == DType.TIMESTAMP_DAYS || dType == DType.TIMESTAMP_MICROSECONDS ||
          dType == DType.TIMESTAMP_MILLISECONDS || dType == DType.TIMESTAMP_NANOSECONDS ||
          dType == DType.TIMESTAMP_SECONDS) {
      typeStr = "<f" + dType.getSizeInBytes();
    } else if (dType == DType.BOOL8 || dType == DType.INT8 || dType == DType.INT16 ||
          dType == DType.INT32 || dType == DType.INT64) {
      typeStr = "<i" + dType.getSizeInBytes();
    } else {
      // not supporting type.
      throw new IllegalArgumentException("Not supporting data type: " + dType);
    }

    return new ColumnData(dataBuffer.getAddress(), columnVector.getRowCount(), validPtr,
      dType.getSizeInBytes(), typeStr, columnVector.getNullCount());
  }

}
