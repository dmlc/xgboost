/*
 Copyright (c) 2021-2022 by Contributors

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

import java.util.ArrayList;

/**
 * Cudf utilities to build cuda array interface against {@link CudfColumn}
 */
class CudfUtils {

  /**
   * Build the cuda array interface based on CudfColumn(s)
   * @param cudfColumns the CudfColumn(s) to be built
   * @return the json format of cuda array interface
   */
  public static String buildArrayInterface(CudfColumn... cudfColumns) {
    return new Builder().add(cudfColumns).build();
  }

  // Helper class to build array interface string
  private static class Builder {
    private ArrayList<String> colArrayInterfaces = new ArrayList<String>();

    private Builder add(CudfColumn... columns) {
      if (columns == null || columns.length <= 0) {
        throw new IllegalArgumentException("At least one ColumnData is required.");
      }
      for (CudfColumn cd : columns) {
        colArrayInterfaces.add(buildColumnObject(cd));
      }
      return this;
    }

    private String build() {
      StringBuilder builder = new StringBuilder();
      builder.append("[");
      for (int i = 0; i < colArrayInterfaces.size(); i++) {
        builder.append(colArrayInterfaces.get(i));
        if (i != colArrayInterfaces.size() - 1) {
          builder.append(",");
        }
      }
      builder.append("]");
      return builder.toString();
    }

    /** build the whole column information including data and valid info */
    private String buildColumnObject(CudfColumn column) {
      if (column.getDataPtr() == 0) {
        throw new IllegalArgumentException("Empty column data is NOT accepted!");
      }
      if (column.getTypeStr() == null || column.getTypeStr().isEmpty()) {
        throw new IllegalArgumentException("Empty type string is NOT accepted!");
      }

      StringBuilder builder = new StringBuilder();
      String colData = buildMetaObject(column.getDataPtr(), column.getShape(),
          column.getTypeStr());
      builder.append("{");
      builder.append(colData);
      if (column.getValidPtr() != 0 && column.getNullCount() != 0) {
        String validString = buildMetaObject(column.getValidPtr(), column.getShape(), "<t1");
        builder.append(",\"mask\":");
        builder.append("{");
        builder.append(validString);
        builder.append("}");
      }
      builder.append("}");
      return builder.toString();
    }

    /** build the base information of a column */
    private String buildMetaObject(long ptr, long shape, final String typeStr) {
      StringBuilder builder = new StringBuilder();
      builder.append("\"shape\":[" + shape + "],");
      builder.append("\"data\":[" + ptr + "," + "false" + "],");
      builder.append("\"typestr\":\"" + typeStr + "\",");
      builder.append("\"version\":" + 1);
      return builder.toString();
    }
  }

}
