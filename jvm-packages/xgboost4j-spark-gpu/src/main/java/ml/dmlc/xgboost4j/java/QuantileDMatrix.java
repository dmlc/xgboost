/*
 Copyright (c) 2021-2025 by Contributors

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

import java.io.IOException;
import java.util.Iterator;
import java.util.Map;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.module.SimpleModule;

class F64NaNSerializer extends JsonSerializer<Double> {
  @Override
  public void serialize(Double value, JsonGenerator gen,
                        SerializerProvider serializers) throws IOException {
    if (value.isNaN()) {
      gen.writeRawValue("NaN"); // Write NaN without quotes
    } else {
      gen.writeNumber(value);
    }
  }
}

class F32NaNSerializer extends JsonSerializer<Float> {
  @Override
  public void serialize(Float value, JsonGenerator gen,
                        SerializerProvider serializers) throws IOException {
    if (value.isNaN()) {
      gen.writeRawValue("NaN"); // Write NaN without quotes
    } else {
      gen.writeNumber(value);
    }
  }
}

/**
 * QuantileDMatrix will only be used to train
 */
public class QuantileDMatrix extends DMatrix {
  /**
   * Create QuantileDMatrix from iterator based on the cuda array interface
   *
   * @param iter    the XGBoost ColumnBatch batch to provide the corresponding cuda array interface
   * @param missing the missing value
   * @param maxBin  the max bin
   * @param nthread the parallelism
   * @throws XGBoostError
   */
  public QuantileDMatrix(
      Iterator<ColumnBatch> iter,
      float missing,
      int maxBin,
      int nthread) throws XGBoostError {
    this(iter, null, missing, maxBin, nthread);
  }

  /**
   * Create QuantileDMatrix from iterator based on the cuda array interface
   *
   * @param iter       the XGBoost ColumnBatch batch to provide the corresponding cuda array
   *                   interface
   * @param refDMatrix The reference QuantileDMatrix that provides quantile information, needed
   *                   when creating validation/test dataset with QuantileDMatrix. Supplying the
   *                   training DMatrix as a reference means that the same quantisation
   *                   applied to the training data is applied to the validation/test data
   * @param missing    the missing value
   * @param maxBin     the max bin
   * @param nthread    the parallelism
   * @throws XGBoostError
   */
  public QuantileDMatrix(
      Iterator<ColumnBatch> iter,
      QuantileDMatrix refDMatrix,
      float missing,
      int maxBin,
      int nthread) throws XGBoostError {
    super(0);
    long[] out = new long[1];
    String conf = getConfig(missing, maxBin, nthread);
    long[] ref = null;
    if (refDMatrix != null) {
      ref = new long[1];
      ref[0] = refDMatrix.getHandle();
    }
    XGBoostJNI.checkCall(XGBoostJNI.XGQuantileDMatrixCreateFromCallback(
        iter, ref, conf, out));
    handle = out[0];
  }

  @Override
  public void setLabel(Column column) throws XGBoostError {
    throw new XGBoostError("QuantileDMatrix does not support setLabel.");
  }

  @Override
  public void setWeight(Column column) throws XGBoostError {
    throw new XGBoostError("QuantileDMatrix does not support setWeight.");
  }

  @Override
  public void setBaseMargin(Column column) throws XGBoostError {
    throw new XGBoostError("QuantileDMatrix does not support setBaseMargin.");
  }

  @Override
  public void setLabel(float[] labels) throws XGBoostError {
    throw new XGBoostError("QuantileDMatrix does not support setLabel.");
  }

  @Override
  public void setWeight(float[] weights) throws XGBoostError {
    throw new XGBoostError("QuantileDMatrix does not support setWeight.");
  }

  @Override
  public void setBaseMargin(float[] baseMargin) throws XGBoostError {
    throw new XGBoostError("QuantileDMatrix does not support setBaseMargin.");
  }

  @Override
  public void setBaseMargin(float[][] baseMargin) throws XGBoostError {
    throw new XGBoostError("QuantileDMatrix does not support setBaseMargin.");
  }

  @Override
  public void setGroup(int[] group) throws XGBoostError {
    throw new XGBoostError("QuantileDMatrix does not support setGroup.");
  }

  private String getConfig(float missing, int maxBin, int nthread) {
    Map<String, Object> conf = new java.util.HashMap<>();
    conf.put("missing", missing);
    conf.put("max_bin", maxBin);
    conf.put("nthread", nthread);
    ObjectMapper mapper = new ObjectMapper();

    // Handle NaN values. Jackson by default serializes NaN values into strings.
    SimpleModule module = new SimpleModule();
    module.addSerializer(Double.class, new F64NaNSerializer());
    module.addSerializer(Float.class, new F32NaNSerializer());
    mapper.registerModule(module);

    try {
      String config = mapper.writeValueAsString(conf);
      return config;
    } catch (JsonProcessingException e) {
      throw new RuntimeException("Failed to serialize configuration", e);
    }
  }
}
