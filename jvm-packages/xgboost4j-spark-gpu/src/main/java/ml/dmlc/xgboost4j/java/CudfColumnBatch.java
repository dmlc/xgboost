/*
 Copyright (c) 2021-2024 by Contributors

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

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import ai.rapids.cudf.Table;
import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * CudfColumnBatch wraps multiple CudfColumns to provide the cuda
 * array interface json string for all columns.
 */
public class CudfColumnBatch extends ColumnBatch {
  @JsonIgnore
  private final Table featureTable;
  @JsonIgnore
  private final Table labelTable;
  @JsonIgnore
  private final Table weightTable;
  @JsonIgnore
  private final Table baseMarginTable;
  @JsonIgnore
  private final Table qidTable;

  private List<CudfColumn> features;
  private List<CudfColumn> label;
  private List<CudfColumn> weight;
  private List<CudfColumn> baseMargin;
  private List<CudfColumn> qid;

  public CudfColumnBatch(Table featureTable, Table labelTable, Table weightTable,
                         Table baseMarginTable, Table qidTable) {
    this.featureTable = featureTable;
    this.labelTable = labelTable;
    this.weightTable = weightTable;
    this.baseMarginTable = baseMarginTable;
    this.qidTable = qidTable;

    features = initializeCudfColumns(featureTable);
    if (labelTable != null) {
      assert labelTable.getNumberOfColumns() == 1;
      label = initializeCudfColumns(labelTable);
    }

    if (weightTable != null) {
      assert weightTable.getNumberOfColumns() == 1;
      weight = initializeCudfColumns(weightTable);
    }

    if (baseMarginTable != null) {
      baseMargin = initializeCudfColumns(baseMarginTable);
    }

    if (qidTable != null) {
      qid = initializeCudfColumns(qidTable);
    }

  }

  private List<CudfColumn> initializeCudfColumns(Table table) {
    assert table != null && table.getNumberOfColumns() > 0;

    return IntStream.range(0, table.getNumberOfColumns())
      .mapToObj(table::getColumn)
      .map(CudfColumn::from)
      .collect(Collectors.toList());
  }

  public List<CudfColumn> getFeatures() {
    return features;
  }

  public List<CudfColumn> getLabel() {
    return label;
  }

  public List<CudfColumn> getWeight() {
    return weight;
  }

  public List<CudfColumn> getBaseMargin() {
    return baseMargin;
  }

  public List<CudfColumn> getQid() {
    return qid;
  }

  public String toJson() {
    ObjectMapper mapper = new ObjectMapper();
    mapper.setSerializationInclusion(JsonInclude.Include.NON_NULL);
    try {
      return mapper.writeValueAsString(this);
    } catch (JsonProcessingException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public String toFeaturesJson() {
    ObjectMapper mapper = new ObjectMapper();
    try {
      return mapper.writeValueAsString(features);
    } catch (JsonProcessingException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public void close() {
    if (featureTable != null) featureTable.close();
    if (labelTable != null) labelTable.close();
    if (weightTable != null) weightTable.close();
    if (baseMarginTable != null) baseMarginTable.close();
    if (qidTable != null) qidTable.close();
  }
}
