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

package ml.dmlc.xgboost4j.java;

import java.util.Iterator;

/**
 * A mini-batch of ColumnBatch that can be converted to DMatrix.
 *
 * This class is used to support advanced creation of DMatrix from Iterator of DataFrameBatch,
 */
class DataFrameBatch implements AutoCloseable {

  private String arrayInterfaceJson;
  private ColumnBatch columnBatch;

  public DataFrameBatch(ColumnBatch columnBatch, String arrayInterfaceJson) {
    this.columnBatch = columnBatch;
    this.arrayInterfaceJson = arrayInterfaceJson;
  }

  // Called from native
  public String getArrayInterfaceJson() {
    return arrayInterfaceJson;
  }

  // Called from native
  @Override
  public void close() throws Exception {
    columnBatch.close();
  }

  static class BatchIterator implements Iterator<DataFrameBatch> {
    private Iterator<ColumnBatch> base;

    public BatchIterator(Iterator<ColumnBatch> base) {
      this.base = base;
    }

    @Override
    public boolean hasNext() {
      return base.hasNext();
    }

    @Override
    public DataFrameBatch next() {
      ColumnBatch columnBatch = base.next();
      return new DataFrameBatch(columnBatch, columnBatch.getArrayInterfaceJson());
    }
  }
}
