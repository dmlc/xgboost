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
 * This class is composing of base data with Apache Arrow format from Gpu ColumnVector.
 * It will be used to generate the cuda array interface.
 */
public class ColumnData {
  private long dataPtr; //  gpu data buffer address
  private long shape;   // row count
  private long validPtr; // gpu valid buffer address
  private int typeSize; // type size in bytes
  private String typeStr; // follow array interface spec
  private long nullCount; // null count

  public ColumnData(long dataPtr, long shape, long validPtr, int typeSize, String typeStr,
                    long nullCount) {
    this.dataPtr = dataPtr;
    this.shape = shape;
    this.validPtr = validPtr;
    this.typeSize = typeSize;
    this.typeStr = typeStr;
    this.nullCount = nullCount;
  }

  public static String getArrayInterface(ColumnData... columnDataList) {
    return new Builder().add(columnDataList).build();
  }

  // Helper class to build array interface string
  private static class Builder {
    private JsonNodeFactory nodeFactory = new JsonNodeFactory(false);
    private ArrayNode rootArrayNode = nodeFactory.arrayNode();

    private Builder add(ColumnData... cds) {
      if (cds == null || cds.length <= 0) {
        throw new IllegalArgumentException("At least one ColumnData is required.");
      }
      for (ColumnData cd : cds) {
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

    private ObjectNode buildColumnObject(ColumnData cd) {
      if (cd.dataPtr == 0) {
        throw new IllegalArgumentException("Empty column data is NOT accepted!");
      }
      if (cd.typeStr == null || cd.typeStr.isEmpty()) {
        throw new IllegalArgumentException("Empty type string is NOT accepted!");
      }
      ObjectNode colDataObj = buildMetaObject(cd.dataPtr, cd.shape, cd.typeStr);

      if (cd.validPtr != 0 && cd.nullCount != 0) {
        ObjectNode validObj = buildMetaObject(cd.validPtr, cd.shape, "<t1");
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
