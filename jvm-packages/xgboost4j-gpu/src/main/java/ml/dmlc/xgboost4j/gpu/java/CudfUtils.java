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

import java.io.ByteArrayOutputStream;
import java.io.IOException;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;

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
    private JsonNodeFactory nodeFactory = new JsonNodeFactory(false);
    private ArrayNode rootArrayNode = nodeFactory.arrayNode();

    private Builder add(CudfColumn... columns) {
      if (columns == null || columns.length <= 0) {
        throw new IllegalArgumentException("At least one ColumnData is required.");
      }
      for (CudfColumn cd : columns) {
        rootArrayNode.add(buildColumnObject(cd));
      }
      return this;
    }

    private String build() {
      try {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        JsonGenerator jsonGen = new JsonFactory().createGenerator(bos);
        new ObjectMapper().writeTree(jsonGen, rootArrayNode);
        return bos.toString();
      } catch (IOException ie) {
        ie.printStackTrace();
        throw new RuntimeException("Failed to build array interface. Error: " + ie);
      }
    }

    private ObjectNode buildColumnObject(CudfColumn column) {
      if (column.getDataPtr() == 0) {
        throw new IllegalArgumentException("Empty column data is NOT accepted!");
      }
      if (column.getTypeStr() == null || column.getTypeStr().isEmpty()) {
        throw new IllegalArgumentException("Empty type string is NOT accepted!");
      }
      ObjectNode colDataObj = buildMetaObject(column.getDataPtr(), column.getShape(),
          column.getTypeStr());

      if (column.getValidPtr() != 0 && column.getNullCount() != 0) {
        ObjectNode validObj = buildMetaObject(column.getValidPtr(), column.getShape(), "<t1");
        colDataObj.set("mask", validObj);
      }
      return colDataObj;
    }

    private ObjectNode buildMetaObject(long ptr, long shape, final String typeStr) {
      ObjectNode objNode = nodeFactory.objectNode();
      ArrayNode shapeNode = objNode.putArray("shape");
      shapeNode.add(shape);
      ArrayNode dataNode = objNode.putArray("data");
      dataNode.add(ptr)
        .add(false);
      objNode.put("typestr", typeStr)
        .put("version", 1);
      return objNode;
    }
  }

}
