/*
 Copyright (c) 2025 by Contributors

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
import java.util.Map;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.module.SimpleModule;

public class ExtMemQuantileDMatrix extends QuantileDMatrix {
  // on_host is set to true by default as we only support GPU at the moment
  // cache_prefix is not used yet since we have on_host=true.
  public ExtMemQuantileDMatrix(Iterator<ColumnBatch> iter,
      float missing,
      int maxBin,
      DMatrix ref,
      int nthread,
      int maxQuantileBatches,
      long minCachePageBytes,
      float cacheHostRatio) throws XGBoostError {
    long[] out = new long[1];
    long[] refHandle = null;
    if (ref != null) {
      refHandle = new long[1];
      refHandle[0] = ref.getHandle();
    }
    String conf = this.getConfig(missing, maxBin, nthread,
                                 maxQuantileBatches, minCachePageBytes, cacheHostRatio);
    XGBoostJNI.checkCall(XGBoostJNI.XGExtMemQuantileDMatrixCreateFromCallback(
        iter, refHandle, conf, out));
    handle = out[0];
  }

  public ExtMemQuantileDMatrix(
      Iterator<ColumnBatch> iter,
      float missing,
      int maxBin,
      DMatrix ref) throws XGBoostError {
    this(iter, missing, maxBin, ref, 0, -1, -1, Float.NaN);
  }

  public ExtMemQuantileDMatrix(
      Iterator<ColumnBatch> iter,
      float missing,
      int maxBin) throws XGBoostError {
    this(iter, missing, maxBin, null);
  }

  private String getConfig(float missing, int maxBin, int nthread,
                           int maxQuantileBatches, long minCachePageBytes, float cacheHostRatio) {
    Map<String, Object> conf = new java.util.HashMap<>();
    conf.put("missing", missing);
    conf.put("max_bin", maxBin);
    conf.put("nthread", nthread);

    if (maxQuantileBatches > 0) {
      conf.put("max_quantile_blocks", maxQuantileBatches);
    }
    if (minCachePageBytes > 0) {
      conf.put("min_cache_page_bytes", minCachePageBytes);
    }

    if (cacheHostRatio >= 0.0 && cacheHostRatio <= 1.0) {
      conf.put("cache_host_ratio", cacheHostRatio);
    }

    conf.put("on_host", true);
    conf.put("cache_prefix", ".");
    ObjectMapper mapper = new ObjectMapper();

    // Handle NaN values. Jackson by default serializes NaN values into strings.
    SimpleModule module = new SimpleModule();
    module.addSerializer(Double.class, new F64NaNSerializer());
    module.addSerializer(Float.class, new F32NaNSerializer());
    mapper.registerModule(module);

    try {
      return mapper.writeValueAsString(conf);
    } catch (JsonProcessingException e) {
      throw new RuntimeException("Failed to serialize configuration", e);
    }
  }
};
