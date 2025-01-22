/*
 * Copyright (c) 2025, XGBoost Contributors
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
      int max_num_device_pages,
      int max_quantile_batches,
      int min_cache_page_bytes) throws XGBoostError {
    long[] out = new long[1];
    long[] ref_handle = null;
    if (ref != null) {
      ref_handle = new long[1];
      ref_handle[0] = ref.getHandle();
    }
    String conf = this.getConfig(missing, maxBin, nthread, max_num_device_pages,
        max_quantile_batches, min_cache_page_bytes);
    XGBoostJNI.checkCall(XGBoostJNI.XGExtMemQuantileDMatrixCreateFromCallback(
        iter, ref_handle, conf, out));
    handle = out[0];
  }

  public ExtMemQuantileDMatrix(
      Iterator<ColumnBatch> iter,
      float missing,
      int maxBin,
      DMatrix ref) throws XGBoostError {
    this(iter, missing, maxBin, ref, 1, -1, -1, -1);
  }

  public ExtMemQuantileDMatrix(
      Iterator<ColumnBatch> iter,
      float missing,
      int maxBin) throws XGBoostError {
    this(iter, missing, maxBin, null);
  }

  private String getConfig(float missing, int maxBin, int nthread, int max_num_device_pages,
      int max_quantile_batches,
      int min_cache_page_bytes) {
    Map<String, Object> conf = new java.util.HashMap<>();
    conf.put("missing", missing);
    conf.put("max_bin", maxBin);
    conf.put("nthread", nthread);

    if (max_num_device_pages > 0) {
      conf.put("max_num_device_pages", max_num_device_pages);
    }
    if (max_quantile_batches > 0) {
      conf.put("max_quantile_batches", max_quantile_batches);
    }
    if (min_cache_page_bytes > 0) {
      conf.put("min_cache_page_bytes", min_cache_page_bytes);
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
      String config = mapper.writeValueAsString(conf);
      return config;
    } catch (JsonProcessingException e) {
      throw new RuntimeException("Failed to serialize configuration", e);
    }
  }
};
