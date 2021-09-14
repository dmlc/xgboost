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
  private final Table labels;
  private final Table weights;
  private final Table baseMargins;

  public CudfColumnBatch(Table data, Table labels, Table weights, Table baseMargins) {
    this.table = data;
    this.labels = labels;
    this.weights = weights;
    this.baseMargins = baseMargins;
  }

  public ColumnVector getColumnVector(int index) {
    return table.getColumn(index);
  }

  private int[] buildIndices(Table val) {
    int[] out = new int[val.getNumberOfColumns()];
    for (int i = 0; i < out.length; ++i) {
      out[i] = i;
    }
    return out;
  }

  @Override
  public String getArrayInterfaceJson() {
    StringBuilder builder = new StringBuilder();
    String featureStr = getArrayInterface(this.table);
    if (featureStr == null || featureStr.isEmpty()) {
      throw new RuntimeException("Feature json must not be empty");
    } else {
      builder.append("\"features_str\":" + featureStr);
    }

    String labelStr = this.getLabelsArrayInterface();
    if (labelStr != null && ! labelStr.isEmpty()) {
      builder.append(",\"label_str\":" + labelStr);
    }

    String weightStr = getWeightsArrayInterface();
    if (weightStr != null && ! weightStr.isEmpty()) {
      builder.append(",\"weight_str\":" + weightStr);
    }

    String baseMarginStr = getBaseMarginsArrayInterface();
    if (baseMarginStr != null && ! baseMarginStr.isEmpty()) {
      builder.append(",\"basemargin_str\":" + baseMarginStr);
    }

    builder.append("}");
    return builder.toString();
  }

  @Override
  public String getFeatureArrayInterface() {
    return getArrayInterface(table);
  }

  @Override
  public String getLabelsArrayInterface() {
    return getArrayInterface(this.labels);
  }

  @Override
  public String getWeightsArrayInterface() {
    return getArrayInterface(this.weights);
  }
  @Override
  public String getBaseMarginsArrayInterface() {
    return getArrayInterface(this.baseMargins);
  }

  @Override
  public void close() {
    if (table != null) table.close();
  }

  private String getArrayInterface(Table data) {
    if (data == null) {
      return "";
    }
    return CudfUtils.buildArrayInterface(getAsCudfColumn(data));
  }

  private CudfColumn[] getAsCudfColumn(Table data) {
    if (data == null) {
      return new CudfColumn[]{};
    }
    int[] indices = this.buildIndices(data);
    return Arrays.stream(indices)
      .mapToObj((i) -> data.getColumn(i))
      .map(CudfColumn::from)
      .toArray(CudfColumn[]::new);
  }
}
