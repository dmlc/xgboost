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

import java.util.HashMap;
import java.util.Map;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Global configuration context for XGBoost.
 *
 * @version 3.0.0
 *
 * See the parameter document for supported global configuration. The configuration is
 * restored upon close.
 */
public class ConfigContext implements AutoCloseable {
  String orig;

  ConfigContext() throws XGBoostError {
    this.orig = getImpl();
  }

  static String getImpl() throws XGBoostError {
    String[] config = new String[1];
    XGBoostJNI.checkCall(XGBoostJNI.XGBGetGlobalConfig(config));
    return config[0];
  }

  public static Map<String, Object> get() throws XGBoostError {
    String jconfig = getImpl();
    ObjectMapper mapper = new ObjectMapper();
    try {
      Map<String, Object> config = mapper.readValue(jconfig,
          new TypeReference<Map<String, Object>>() {
          });
      return config;
    } catch (JsonProcessingException ex) {
      throw new XGBoostError("Failed to get the global config due to a decode error.", ex);
    }
  }

  public <T> ConfigContext set(String key, T value) throws XGBoostError {
    HashMap<String, Object> map = new HashMap<String, Object>();
    map.put(key, value);
    ObjectMapper mapper = new ObjectMapper();
    try {
      String config = mapper.writeValueAsString(map);
      XGBoostJNI.checkCall(XGBoostJNI.XGBSetGlobalConfig(config));
    } catch (JsonProcessingException ex) {
      throw new XGBoostError("Failed to set the global config due to an encode error.", ex);
    }
    return this;
  }

  @Override
  public void close() throws XGBoostError {
    XGBoostJNI.checkCall(XGBoostJNI.XGBSetGlobalConfig(this.orig));
  }
};
