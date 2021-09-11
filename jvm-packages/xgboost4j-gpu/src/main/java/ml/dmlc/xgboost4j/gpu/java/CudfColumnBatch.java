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

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.Table;

import ml.dmlc.xgboost4j.java.ColumnBatch;

/**
 * Class to wrap CUDF Table to generate the cuda array interface.
 */
public class CudfColumnBatch extends ColumnBatch {
  private final Table table;          // The CUDF Table
  private final int[] featureIndices; // The feature columns
  private final int[] labelIndices;   // The label columns
  private final int[] weightIndices;  // The weight columns
  private final int[] baseMarginIndices; // The base margin columns

  /**
   * CudfColumnBatch constructor
   * @param table             the CUDF table
   * @param featureIndices    must-have, specify the feature's indices in the table
   */
  public CudfColumnBatch(Table table, int[] featureIndices) {
    this(table, featureIndices, null, null, null);
  }

  /**
   * CudfColumnBatch constructor
   * @param table             the CUDF table
   * @param featureIndices    must-have, specify the feature's indices in the table
   * @param labelIndices      optional, specify the label's indices in the table
   */
  public CudfColumnBatch(Table table, int[] featureIndices, int[] labelIndices) {
    this(table, featureIndices, labelIndices, null, null);
  }

  /**
   * CudfColumnBatch constructor
   * @param table             the CUDF table
   * @param featureIndices    must-have, specify the feature's indices in the table
   * @param labelIndices      must-have, specify the label's indices in the table
   * @param weightIndices     optional, specify the weight's indices in the table
   * @param baseMarginIndices optional, specify the base marge's indices in the table
   */
  public CudfColumnBatch(Table table, int[] featureIndices, int[] labelIndices, int[] weightIndices,
                         int[] baseMarginIndices) {
    this.table = table;
    this.featureIndices = featureIndices;
    this.labelIndices = labelIndices;
    this.weightIndices = weightIndices;
    this.baseMarginIndices = baseMarginIndices;

    validate();
  }

  private void validate() {
    if (featureIndices == null) {
      throw new RuntimeException("CudfColumnBatch requires feature columns");
    } else {
      validateArrayIndex(featureIndices, "feature");
    }

    if (labelIndices != null) {
      validateArrayIndex(labelIndices, "label");
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

  @Override
  public void close() {
    if (table != null) table.close();
  }

  private String getArrayInterface(int... indices) {
    if (indices == null || indices.length == 0) return "";
    return CudfUtils.buildArrayInterface(getAsCudfColumn(indices));
  }

  private CudfColumn[] getAsCudfColumn(int... indices) {
    if (indices == null || indices.length == 0) return new CudfColumn[]{};
    return Arrays.stream(indices)
      .mapToObj(this::getColumnVector)
      .map(CudfColumn::from)
      .toArray(CudfColumn[]::new);
  }

}
