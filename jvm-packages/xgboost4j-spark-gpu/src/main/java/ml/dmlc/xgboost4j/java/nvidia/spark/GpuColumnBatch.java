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

package ml.dmlc.xgboost4j.java.nvidia.spark;

import java.util.List;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.Table;
import org.apache.spark.sql.types.*;

/**
 * Wrapper of CudfTable with schema for scala
 */
public class GpuColumnBatch implements AutoCloseable {
  private final StructType schema;
  private Table table; // the original Table

  public GpuColumnBatch(Table table, StructType schema) {
    this.table = table;
    this.schema = schema;
  }

  @Override
  public void close() {
    if (table != null) {
      table.close();
      table = null;
    }
  }

  /** Slice the columns indicated by indices into a Table*/
  public Table slice(List<Integer> indices) {
    if (indices == null || indices.size() == 0) {
      return null;
    }

    int len = indices.size();
    ColumnVector[] cv = new ColumnVector[len];
    for (int i = 0; i < len; i++) {
      int index = indices.get(i);
      if (index >= table.getNumberOfColumns()) {
        throw new RuntimeException("Wrong index");
      }
      cv[i] = table.getColumn(index);
    }

    return new Table(cv);
  }

  public StructType getSchema() {
    return schema;
  }

}
