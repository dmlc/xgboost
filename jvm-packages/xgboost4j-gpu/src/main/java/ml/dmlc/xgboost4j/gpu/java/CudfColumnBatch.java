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

import java.util.stream.IntStream;

import ai.rapids.cudf.Table;

import ml.dmlc.xgboost4j.java.ColumnBatch;

/**
 * Class to wrap CUDF Table to generate the cuda array interface.
 */
public class CudfColumnBatch extends ColumnBatch {
  private final Table feature;
  private final Table label;
  private final Table weight;
  private final Table baseMargin;

  public CudfColumnBatch(Table feature, Table labels, Table weights, Table baseMargins) {
    this.feature = feature;
    this.label = labels;
    this.weight = weights;
    this.baseMargin = baseMargins;
  }

  @Override
  public String getFeatureArrayInterface() {
    return getArrayInterface(this.feature);
  }

  @Override
  public String getLabelsArrayInterface() {
    return getArrayInterface(this.label);
  }

  @Override
  public String getWeightsArrayInterface() {
    return getArrayInterface(this.weight);
  }

  @Override
  public String getBaseMarginsArrayInterface() {
    return getArrayInterface(this.baseMargin);
  }

  @Override
  public void close() {
    if (feature != null) feature.close();
    if (label != null) label.close();
    if (weight != null) weight.close();
    if (baseMargin != null) baseMargin.close();
  }

  private String getArrayInterface(Table table) {
    if (table == null || table.getNumberOfColumns() == 0) {
      return "";
    }
    return CudfUtils.buildArrayInterface(getAsCudfColumn(table));
  }

  private CudfColumn[] getAsCudfColumn(Table table) {
    if (table == null || table.getNumberOfColumns() == 0) {
      // This will never happen.
      return new CudfColumn[]{};
    }

    return IntStream.range(0, table.getNumberOfColumns())
      .mapToObj((i) -> table.getColumn(i))
      .map(CudfColumn::from)
      .toArray(CudfColumn[]::new);
  }

}
