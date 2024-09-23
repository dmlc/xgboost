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

import java.util.ArrayList;
import java.util.List;

import ai.rapids.cudf.BaseDeviceMemoryBuffer;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.DType;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * CudfColumn is the CUDF column representing, providing the cuda array interface
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class CudfColumn extends Column {
  private List<Long> shape = new ArrayList<>();   // row count
  private List<Object> data = new ArrayList<>(); //  gpu data buffer address
  private String typestr;
  private int version = 1;
  private CudfColumn mask = null;

  public CudfColumn(long shape, long data, String typestr, int version) {
    this.shape.add(shape);
    this.data.add(data);
    this.data.add(false);
    this.typestr = typestr;
    this.version = version;
  }

  /**
   * Create CudfColumn according to ColumnVector
   */
  public static CudfColumn from(ColumnVector cv) {
    BaseDeviceMemoryBuffer dataBuffer = cv.getData();
    assert dataBuffer != null;

    DType dType = cv.getType();
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
      // Unsupported type.
      throw new IllegalArgumentException("Unsupported data type: " + dType);
    }

    CudfColumn data = new CudfColumn(cv.getRowCount(), dataBuffer.getAddress(), typeStr, 1);

    BaseDeviceMemoryBuffer validBuffer = cv.getValid();
    if (validBuffer != null && cv.getNullCount() != 0) {
      CudfColumn mask = new CudfColumn(cv.getRowCount(), validBuffer.getAddress(), "<t1", 1);
      data.setMask(mask);
    }
    return data;
  }

  public List<Long> getShape() {
    return shape;
  }

  public List<Object> getData() {
    return data;
  }

  public String getTypestr() {
    return typestr;
  }

  public int getVersion() {
    return version;
  }

  public CudfColumn getMask() {
    return mask;
  }

  public void setMask(CudfColumn mask) {
    this.mask = mask;
  }

  @Override
  public String toJson() {
    ObjectMapper mapper = new ObjectMapper();
    mapper.setSerializationInclusion(JsonInclude.Include.NON_NULL);
    try {
      List<CudfColumn> objects = new ArrayList<>(1);
      objects.add(this);
      return mapper.writeValueAsString(objects);
    } catch (JsonProcessingException e) {
      throw new RuntimeException(e);
    }
  }

}
